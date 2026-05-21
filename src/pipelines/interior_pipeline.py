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
    FatigueAnalyzer, YawnAnalyzer, DistractionAnalyzer, NoDriverAnalyzer
)
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

        # Behavior state machines
        self._fatigue = FatigueAnalyzer(config.mediapipe)
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

        # ── Behavior analysis (face) ─────────────────────────────────────
        for evt in [
            self._fatigue.analyze(face),
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
            if self._fatigue.is_fatigued:
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

            # Head pose info
            pose_txt = f"Y:{face.yaw:.1f} P:{face.pitch:.1f} R:{face.roll:.1f}"
            draw_text_box(frame, pose_txt, (8, 165), color=COLORS["gray"], scale=0.42)

            # Fatigue countdown bar
            if self._fatigue.consec_frames > 0:
                frac = min(self._fatigue.consec_frames / self._cfg.mediapipe.ear_consec_frames, 1.0)
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
