"""
Behavior analyzers — stateful counters that convert per-frame
detector outputs into meaningful DMS events.

Each analyzer maintains its own frame counters and emits
DmsEvent objects when thresholds are crossed.
"""

from __future__ import annotations

import time
import logging
from typing import Optional, List

from src.alerts.event_types import DmsEvent, EventType, Severity
from src.config.settings import MediaPipeConfig
from src.detectors.face_mesh_detector import FaceResult

logger = logging.getLogger("dms.behaviors")


class FatigueAnalyzer:
    """
    Detects FATIGUE_DRIVING via Eye Aspect Ratio (EAR).

    When EAR stays below threshold for N consecutive frames,
    emits a FATIGUE_DRIVING event.
    """

    def __init__(self, cfg: MediaPipeConfig):
        self._cfg = cfg
        self._consec = 0
        self._alert_active = False
        self._total_microsleep_events = 0

    def analyze(self, face: Optional[FaceResult]) -> Optional[DmsEvent]:
        if face is None:
            self._consec = 0
            self._alert_active = False
            return None

        if face.ear < self._cfg.ear_threshold:
            self._consec += 1
        else:
            if self._alert_active:
                logger.debug(f"Fatigue cleared after {self._consec} frames")
            self._consec = 0
            self._alert_active = False
            return None

        if self._consec >= self._cfg.ear_consec_frames:
            if not self._alert_active:
                self._alert_active = True
                self._total_microsleep_events += 1
                logger.warning(f"FATIGUE detected — EAR={face.ear:.3f}, frames={self._consec}")
                return DmsEvent.make(
                    EventType.FATIGUE_DRIVING,
                    Severity.CRITICAL,
                    f"Driver fatigue detected (EAR={face.ear:.3f})",
                    confidence=min(1.0, self._consec / (self._cfg.ear_consec_frames * 2)),
                    ear=face.ear,
                    consec_frames=self._consec,
                )

        return None

    @property
    def is_fatigued(self) -> bool:
        return self._alert_active

    @property
    def consec_frames(self) -> int:
        return self._consec


class YawnAnalyzer:
    """
    Detects DRIVER_YAWNS via Mouth Aspect Ratio (MAR).
    """

    def __init__(self, cfg: MediaPipeConfig):
        self._cfg = cfg
        self._consec = 0
        self._alert_active = False

    def analyze(self, face: Optional[FaceResult]) -> Optional[DmsEvent]:
        if face is None:
            self._consec = 0
            self._alert_active = False
            return None

        if face.mar > self._cfg.mar_threshold:
            self._consec += 1
        else:
            self._consec = 0
            self._alert_active = False
            return None

        if self._consec >= self._cfg.mar_consec_frames:
            if not self._alert_active:
                self._alert_active = True
                return DmsEvent.make(
                    EventType.DRIVER_YAWNS,
                    Severity.WARNING,
                    f"Driver yawning detected (MAR={face.mar:.3f})",
                    confidence=0.85,
                    mar=face.mar,
                )

        return None

    @property
    def is_yawning(self) -> bool:
        return self._alert_active


class DistractionAnalyzer:
    """
    Detects DRIVER_UNDER_DISTRACTION via head pose (yaw/pitch).

    Large head turn or pitch-down (looking at phone) triggers alert.
    """

    def __init__(self, cfg: MediaPipeConfig):
        self._cfg = cfg
        self._consec = 0
        self._alert_active = False

    def analyze(self, face: Optional[FaceResult]) -> Optional[DmsEvent]:
        if face is None:
            self._consec = 0
            self._alert_active = False
            return None

        distracted = (
            abs(face.yaw) > self._cfg.yaw_threshold or
            abs(face.pitch) > self._cfg.pitch_threshold
        )

        if distracted:
            self._consec += 1
        else:
            self._consec = 0
            self._alert_active = False
            return None

        if self._consec >= self._cfg.distraction_consec_frames:
            if not self._alert_active:
                self._alert_active = True
                return DmsEvent.make(
                    EventType.DRIVER_UNDER_DISTRACTION,
                    Severity.WARNING,
                    f"Driver distracted (yaw={face.yaw:.1f}°, pitch={face.pitch:.1f}°)",
                    confidence=0.80,
                    yaw=face.yaw,
                    pitch=face.pitch,
                )

        return None

    @property
    def is_distracted(self) -> bool:
        return self._alert_active


class NoDriverAnalyzer:
    """
    Emits NO_DRIVER when face disappears for N frames.
    """
    _ABSENT_THRESHOLD = 30  # frames

    def __init__(self):
        self._absent_frames = 0
        self._alert_active = False

    def analyze(self, face: Optional[FaceResult]) -> Optional[DmsEvent]:
        if face is not None:
            self._absent_frames = 0
            self._alert_active = False
            return None

        self._absent_frames += 1
        if self._absent_frames >= self._ABSENT_THRESHOLD and not self._alert_active:
            self._alert_active = True
            return DmsEvent.make(
                EventType.NO_DRIVER,
                Severity.CRITICAL,
                "No driver detected in frame",
                confidence=1.0,
            )
        return None

    @property
    def is_absent(self) -> bool:
        return self._alert_active
