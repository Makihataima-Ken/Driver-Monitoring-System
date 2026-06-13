"""
InteriorResultProcessor — converts InferenceResult into DMS events.

Holds the stateful behaviour analysers so that consecutive-frame
counters persist across individual inference results.  Draws overlays
on a copy of the raw frame and returns the annotated frame together
with any triggered events.
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Optional

import cv2
import numpy as np

from src.config.settings import SystemConfig
from src.detectors.face_mesh_detector import FaceResult
from src.detectors.yolo_detector import Detection
from src.alerts.event_types import DmsEvent

from src.behaviors.driver_behaviors import (
    FatigueAnalyzer,
    YawnAnalyzer,
    DistractionAnalyzer,
    NoDriverAnalyzer,
)
from src.behaviors.yolo_behaviors import (
    PhoneCallAnalyzer,
    SmokingAnalyzer,
    SeatbeltAnalyzer,
)
from src.utils.drawing import (
    draw_face_box,
    draw_bounding_box,
    draw_ear_mar,
    draw_text_box,
    COLORS,
)

logger = logging.getLogger("dms.interior.processor")


class InteriorResultProcessor:
    """
    Stateful processor for interior inference results.

    Usage::

        processor = InteriorResultProcessor(config)
        events, annotated = processor.process(result)
    """

    def __init__(self, config: SystemConfig):
        self._cfg = config

        # Stateful analysers — their internal counters must survive
        # across individual calls to process().
        self._fatigue = FatigueAnalyzer(config.mediapipe)
        self._yawn = YawnAnalyzer(config.mediapipe)
        self._distraction = DistractionAnalyzer(config.mediapipe)
        self._no_driver = NoDriverAnalyzer()
        self._phone = PhoneCallAnalyzer()
        self._smoke = SmokingAnalyzer()
        self._seatbelt = SeatbeltAnalyzer()

    def process(self, result) -> Tuple[List[DmsEvent], np.ndarray]:
        """
        Analyse a single inference result and draw overlays.

        Args:
            result: :class:`InferenceResult` (or anything with
                ``face_result``, ``yolo_detections``, and ``frame``
                attributes).

        Returns:
            ``(events, annotated_frame)``
        """
        events: List[DmsEvent] = []
        display_frame = result.frame.copy()

        face: Optional[FaceResult] = result.face_result
        dets: List[Detection] = result.yolo_detections

        # -- Face-based behaviours -------------------------------------
        for evt in [
            self._fatigue.analyze(face),
            self._yawn.analyze(face),
            self._distraction.analyze(face),
            self._no_driver.analyze(face),
        ]:
            if evt:
                events.append(evt)

        # -- YOLO-based behaviours -------------------------------------
        for evt in [
            self._phone.analyze(dets),
            self._smoke.analyze(dets),
            self._seatbelt.analyze(dets),
        ]:
            if evt:
                events.append(evt)

        # -- Drawing ---------------------------------------------------
        if self._cfg.display.show:
            self._draw(display_frame, face, dets)

        return events, display_frame

    def _draw(
        self,
        frame: np.ndarray,
        face: Optional[FaceResult],
        dets: List[Detection],
    ):
        cfg = self._cfg.display

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

            if cfg.show_landmarks:
                draw_ear_mar(frame, face.ear, face.mar)

            pose_txt = f"Y:{face.yaw:.1f} P:{face.pitch:.1f} R:{face.roll:.1f}"
            draw_text_box(frame, pose_txt, (8, 165), color=COLORS["gray"], scale=0.42)

            if self._fatigue.consec_frames > 0:
                frac = min(self._fatigue.consec_frames / self._cfg.mediapipe.ear_consec_frames, 1.0)
                h, w = frame.shape[:2]
                bar_w = int(w * frac)
                cv2.rectangle(frame, (0, h - 8), (bar_w, h), COLORS["alert"], -1)
        else:
            h, w = frame.shape[:2]
            draw_text_box(
                frame,
                "NO FACE DETECTED",
                (w // 2 - 80, 30),
                color=COLORS["red"],
                scale=0.65,
                thickness=2,
            )

        for det in dets:
            color = COLORS["orange"] if det.label == "cell_phone" else COLORS["green"]
            draw_bounding_box(frame, det.x1, det.y1, det.x2, det.y2, det.label, det.confidence, color)
