"""
ExteriorPipeline — road-facing camera analysis.

DEMO v1: Placeholder stubs for exterior features.
Full implementation targets v2.

Stubs present:
  - LANE_DEPARTURE        (optical flow + Hough lines)
  - FRONT_CAR_COLLISION   (YOLO vehicle + TTC estimation)
  - PEDESTRIAN_COLLISION  (YOLO person + TTC)
  - DISTANCE_ALARM        (bounding box area heuristic)
  - MOTION_DETECTION      (frame differencing)
"""

from __future__ import annotations

import logging
import cv2
import numpy as np
from typing import List, Tuple, Optional

from src.alerts.event_types import DmsEvent, EventType, Severity
from src.config.settings import SystemConfig
from src.detectors.yolo_detector import YOLODetector, Detection
from src.utils.drawing import draw_bounding_box, draw_text_box, COLORS

logger = logging.getLogger("dms.exterior")


class ExteriorPipeline:
    """
    Exterior (road-facing) monitoring pipeline.
    v1: Motion detection active. All others are stubs.
    """

    def __init__(self, config: SystemConfig):
        self._cfg = config
        self._yolo: Optional[YOLODetector] = None
        self._prev_gray: Optional[np.ndarray] = None
        self._motion_threshold = 500_000  # sum of abs diff

    def start(self):
        logger.info("Initialising ExteriorPipeline...")
        try:
            self._yolo = YOLODetector(
                model_path=self._cfg.yolo.model_path,
                conf_threshold=self._cfg.yolo.conf_threshold,
                imgsz=self._cfg.yolo.imgsz,
                device=self._cfg.yolo.device,
                classes_of_interest=[0, 2, 5, 7],  # person, car, bus, truck
            )
        except Exception as e:
            logger.warning(f"Exterior YOLO failed: {e}")
            self._yolo = None
        logger.info("ExteriorPipeline ready (v1 stub).")

    def process(self, frame: np.ndarray) -> Tuple[List[DmsEvent], np.ndarray]:
        events: List[DmsEvent] = []
        annotated = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Motion detection
        evt = self._motion_detect(gray)
        if evt:
            events.append(evt)

        self._prev_gray = gray

        # YOLO detections — pedestrian/vehicle proximity
        dets: List[Detection] = []
        if self._yolo:
            dets = self._yolo.detect(frame)
            for det in dets:
                color = COLORS["red"] if det.label == "person" else COLORS["orange"]
                draw_bounding_box(annotated, det.x1, det.y1, det.x2, det.y2, det.label, det.confidence, color)

        # Stub: Lane departure
        self._lane_stub(annotated)

        draw_text_box(annotated, "[EXTERIOR MODE - v1]", (8, 22), color=COLORS["cyan"])
        return events, annotated

    def _motion_detect(self, gray: np.ndarray) -> Optional[DmsEvent]:
        if self._prev_gray is None:
            return None
        diff = cv2.absdiff(self._prev_gray, gray)
        score = int(diff.sum())
        if score > self._motion_threshold:
            return DmsEvent.make(
                EventType.MOTION_DETECTION,
                Severity.INFO,
                f"Motion detected (score={score})",
                confidence=min(1.0, score / (self._motion_threshold * 3)),
            )
        return None

    def _lane_stub(self, frame: np.ndarray):
        """Placeholder: Hough-based lane detection."""
        h, w = frame.shape[:2]
        roi_y = int(h * 0.55)
        cv2.line(frame, (0, roi_y), (w, roi_y), COLORS["gray"], 1)
        draw_text_box(frame, "LANE DETECTION: v2", (8, roi_y + 15), color=COLORS["gray"], scale=0.4)

    def stop(self):
        logger.info("ExteriorPipeline stopped.")
