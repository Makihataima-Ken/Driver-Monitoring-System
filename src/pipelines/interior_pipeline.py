"""
InteriorPipeline — processes a single frame through:
  1. FaceMeshDetector   → FaceResult
  2. YOLODetector       → List[Detection]
  3. Behavior analyzers → List[DmsEvent]
  4. Drawing overlays

Returns annotated frame + list of events.
"""

from __future__ import annotations

import logging
import time
from typing import List, Tuple, Optional

import cv2
import numpy as np

from src.config.settings import SystemConfig
from src.detectors.face_mesh_detector import FaceMeshDetector, FaceResult
from src.detectors.yolo_detector import YOLODetector, Detection
from src.behaviors.driver_behaviors import (
    FatigueAnalyzer, DrowsinessAnalyzer, YawnAnalyzer,
    DistractionAnalyzer, NoDriverAnalyzer,
)
from src.detectors.drowsiness_detector import DrowsinessDetector
from src.behaviors.yolo_behaviors import PhoneCallAnalyzer, SmokingAnalyzer, SeatbeltAnalyzer
from src.alerts.event_types import DmsEvent
from src.utils.drawing import (
    draw_face_box, draw_bounding_box, draw_ear_mar, draw_text_box, COLORS
)

logger = logging.getLogger("dms.interior")


class InteriorPipeline:
    """
    Full interior driver monitoring pipeline.

    Usage:
        pipeline = InteriorPipeline(config)
        pipeline.start()
        events, frame = pipeline.process(raw_frame)
        pipeline.stop()
    """

    def __init__(self, config: SystemConfig):
        self._cfg = config
        self._face_detector: Optional[FaceMeshDetector] = None
        self._yolo: Optional[YOLODetector] = None
        self._drowsiness_detector: Optional[DrowsinessDetector] = None

        # Behavior state machines
        self._fatigue = FatigueAnalyzer(config.mediapipe)
        self._drowsiness = DrowsinessAnalyzer(config.drowsiness)
        self._yawn = YawnAnalyzer(config.mediapipe)
        self._distraction = DistractionAnalyzer(config.mediapipe)
        self._no_driver = NoDriverAnalyzer()
        self._phone = PhoneCallAnalyzer()
        self._smoke = SmokingAnalyzer()
        self._seatbelt = SeatbeltAnalyzer()

        # YOLO skip logic — run YOLO every N frames to save CPU
        self._yolo_frame_interval = 3   # Run YOLO every 3 frames
        self._yolo_frame_counter = 0
        self._last_yolo_dets: List[Detection] = []

    def start(self):
        logger.info("Initialising InteriorPipeline...")

        logger.info("Loading FaceMesh...")
        self._face_detector = FaceMeshDetector(self._cfg.mediapipe)

        logger.info("Loading YOLO...")
        try:
            self._yolo = YOLODetector(
                model_path=self._cfg.yolo.model_path,
                conf_threshold=self._cfg.yolo.conf_threshold,
                iou_threshold=self._cfg.yolo.iou_threshold,
                imgsz=self._cfg.yolo.imgsz,
                device=self._cfg.yolo.device,
                classes_of_interest=self._cfg.yolo.classes_of_interest,
            )
        except Exception as e:
            logger.warning(f"YOLO failed to load: {e} — running without YOLO")
            self._yolo = None

        if self._cfg.drowsiness.enabled:
            logger.info("Loading drowsiness model...")
            try:
                self._drowsiness_detector = DrowsinessDetector(self._cfg.drowsiness)
                self._drowsiness_detector.load()
            except Exception as e:
                logger.warning(f"Drowsiness model failed to load: {e}")
                self._drowsiness_detector = None
                if not self._cfg.drowsiness.use_ear_fallback:
                    logger.warning("EAR fallback disabled — fatigue detection unavailable")

        logger.info("InteriorPipeline ready.")

    def process(self, frame: np.ndarray) -> Tuple[List[DmsEvent], np.ndarray]:
        """
        Process one frame through the full interior pipeline.

        Returns:
            events: list of DmsEvent triggered this frame
            annotated_frame: frame with overlays drawn
        """
        events: List[DmsEvent] = []
        annotated = frame.copy()

        # ── MediaPipe face analysis ──────────────────────────────────────
        face: Optional[FaceResult] = None
        if self._face_detector:
            face = self._face_detector.process(frame)

        # ── Drowsiness model (vector DNN) ────────────────────────────────
        drowsiness_result = None
        if self._drowsiness_detector and face is not None:
            drowsiness_result = self._drowsiness_detector.process(face)

        # ── Behavior analysis (face) ─────────────────────────────────────
        fatigue_evt = None
        if self._drowsiness_detector and drowsiness_result is not None:
            fatigue_evt = self._drowsiness.analyze(drowsiness_result)
        elif self._cfg.drowsiness.use_ear_fallback or not self._cfg.drowsiness.enabled:
            fatigue_evt = self._fatigue.analyze(face)

        for evt in [
            fatigue_evt,
            self._yawn.analyze(face),
            self._distraction.analyze(face),
            self._no_driver.analyze(face),
        ]:
            if evt:
                events.append(evt)

        # ── YOLO (throttled) ─────────────────────────────────────────────
        self._yolo_frame_counter += 1
        if self._yolo and (self._yolo_frame_counter % self._yolo_frame_interval == 0):
            self._last_yolo_dets = self._yolo.detect(frame)

        dets = self._last_yolo_dets
        for evt in [
            self._phone.analyze(dets),
            self._smoke.analyze(dets),
            self._seatbelt.analyze(dets),
        ]:
            if evt:
                events.append(evt)

        # ── Drawing ──────────────────────────────────────────────────────
        if self._cfg.display.show:
            self._draw(annotated, face, dets)

        return events, annotated

    def _draw(self, frame: np.ndarray, face: Optional[FaceResult], dets: List[Detection]):
        cfg = self._cfg.display

        # Face bounding box + state label
        if face:
            state = "OK"
            if self._drowsiness.is_drowsy or self._fatigue.is_fatigued:
                state = "FATIGUE"
            elif self._yawn.is_yawning:
                state = "YAWN"
            elif self._distraction.is_distracted:
                state = "DISTRACTED"

            x1, y1, x2, y2 = face.face_bbox
            draw_face_box(frame, x1, y1, x2, y2, state)

            # EAR / MAR values
            if cfg.show_landmarks:
                draw_ear_mar(frame, face.ear, face.mar)

            if self._drowsiness_detector:
                if not self._drowsiness_detector.is_calibrated:
                    draw_text_box(
                        frame, "DNN calibrating EAR...", (8, 188),
                        color=COLORS["yellow"], scale=0.42,
                    )
                elif not self._drowsiness_detector.buffer_ready:
                    draw_text_box(
                        frame, self._drowsiness.status or "DNN buffering...",
                        (8, 188), color=COLORS["yellow"], scale=0.42,
                    )
                else:
                    prob_txt = f"DNN:{self._drowsiness.probability:.2f}"
                    draw_text_box(frame, prob_txt, (8, 188), color=COLORS["cyan"], scale=0.48)
                    if self._drowsiness.status and "Normal" not in self._drowsiness.status:
                        draw_text_box(
                            frame, self._drowsiness.status[:40], (8, 210),
                            color=COLORS["orange"], scale=0.40,
                        )

            # Head pose info
            pose_txt = f"Y:{face.yaw:.1f} P:{face.pitch:.1f} R:{face.roll:.1f}"
            draw_text_box(frame, pose_txt, (8, 165), color=COLORS["gray"], scale=0.42)

            # Fatigue progress bar (EAR fallback only)
            if not self._drowsiness_detector and self._fatigue.consec_frames > 0:
                frac = min(
                    self._fatigue.consec_frames / self._cfg.mediapipe.ear_consec_frames,
                    1.0,
                )
                h, w = frame.shape[:2]
                bar_w = int(w * frac)
                cv2.rectangle(frame, (0, h - 8), (bar_w, h), COLORS["alert"], -1)
        else:
            # No face detected
            h, w = frame.shape[:2]
            draw_text_box(frame, "NO FACE DETECTED", (w // 2 - 80, 30),
                          color=COLORS["red"], scale=0.65, thickness=2)

        # YOLO detections
        for det in dets:
            color = COLORS["orange"] if det.label == "cell_phone" else COLORS["green"]
            draw_bounding_box(frame, det.x1, det.y1, det.x2, det.y2, det.label, det.confidence, color)

    def stop(self):
        if self._face_detector:
            self._face_detector.release()
        logger.info("InteriorPipeline stopped.")
