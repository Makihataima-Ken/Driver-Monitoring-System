"""
YOLO-based behavior analyzers.

Uses object detection results to emit DMS events for:
  - DRIVER_CALL (phone detected near face)
  - DRIVER_SMOKE (cigarette detected)
  - SEAT_BELT_DETECTION (no seatbelt)
"""

from __future__ import annotations

import time
import logging
from typing import Optional, List

from src.alerts.event_types import DmsEvent, EventType, Severity
from src.detectors.yolo_detector import Detection

logger = logging.getLogger("dms.yolo_behaviors")


class PhoneCallAnalyzer:
    """
    Detects DRIVER_CALL when a cell phone is detected near the driver's face.

    Uses COCO class 67 (cell phone) from the base YOLOv8n model.
    For higher accuracy, swap in fine-tuned weights.
    """
    PHONE_CLASS_ID = 67
    PHONE_LABEL = "cell_phone"
    _CONSEC_THRESHOLD = 5

    def __init__(self):
        self._consec = 0
        self._alert_active = False

    def analyze(self, detections: List[Detection]) -> Optional[DmsEvent]:
        phones = [
            d for d in detections
            if d.class_id == self.PHONE_CLASS_ID or d.label == self.PHONE_LABEL
        ]

        if phones:
            self._consec += 1
        else:
            self._consec = max(0, self._consec - 1)
            if self._consec == 0:
                self._alert_active = False
            return None

        if self._consec >= self._CONSEC_THRESHOLD and not self._alert_active:
            self._alert_active = True
            best = max(phones, key=lambda d: d.confidence)
            return DmsEvent.make(
                EventType.DRIVER_CALL,
                Severity.WARNING,
                f"Phone use detected (conf={best.confidence:.2f})",
                confidence=best.confidence,
                bbox=(best.x1, best.y1, best.x2, best.y2),
            )
        return None

    @property
    def is_on_phone(self) -> bool:
        return self._alert_active


class SmokingAnalyzer:
    """
    Detects DRIVER_SMOKE.

    With base YOLOv8n: no cigarette class — placeholder that fires
    when custom fine-tuned weights supply class label 'cigarette'.
    """
    SMOKE_LABELS = {"cigarette", "smoke", "smoking"}
    _CONSEC_THRESHOLD = 5

    def __init__(self):
        self._consec = 0
        self._alert_active = False

    def analyze(self, detections: List[Detection]) -> Optional[DmsEvent]:
        smokes = [d for d in detections if d.label in self.SMOKE_LABELS]

        if smokes:
            self._consec += 1
        else:
            self._consec = max(0, self._consec - 1)
            if self._consec == 0:
                self._alert_active = False
            return None

        if self._consec >= self._CONSEC_THRESHOLD and not self._alert_active:
            self._alert_active = True
            best = max(smokes, key=lambda d: d.confidence)
            return DmsEvent.make(
                EventType.DRIVER_SMOKE,
                Severity.WARNING,
                f"Smoking detected (conf={best.confidence:.2f})",
                confidence=best.confidence,
            )
        return None

    @property
    def is_smoking(self) -> bool:
        return self._alert_active


class SeatbeltAnalyzer:
    """
    Detects SEAT_BELT_DETECTION (missing seatbelt).

    With base YOLOv8n: placeholder.
    Fine-tuned weights should supply labels: 'seatbelt' / 'no_seatbelt'.
    """
    NO_BELT_LABELS = {"no_seatbelt", "no_belt"}
    BELT_LABELS = {"seatbelt", "belt"}
    _CONSEC_THRESHOLD = 8

    def __init__(self):
        self._consec_missing = 0
        self._alert_active = False
        self._belt_seen = False

    def analyze(self, detections: List[Detection]) -> Optional[DmsEvent]:
        has_belt = any(d.label in self.BELT_LABELS for d in detections)
        missing_belt = any(d.label in self.NO_BELT_LABELS for d in detections)

        if has_belt:
            self._belt_seen = True
            self._consec_missing = 0
            self._alert_active = False
            return None

        if missing_belt:
            self._consec_missing += 1
        else:
            self._consec_missing = max(0, self._consec_missing - 1)
            if self._consec_missing == 0:
                self._alert_active = False
            return None

        if self._consec_missing >= self._CONSEC_THRESHOLD and not self._alert_active:
            self._alert_active = True
            return DmsEvent.make(
                EventType.SEAT_BELT_DETECTION,
                Severity.WARNING,
                "Seatbelt not detected — please fasten seatbelt",
                confidence=0.75,
            )
        return None

    @property
    def belt_missing(self) -> bool:
        return self._alert_active
