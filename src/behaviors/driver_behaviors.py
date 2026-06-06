"""
Behavior analyzers — stateful counters that convert per-frame
detector outputs into meaningful DMS events.

Each analyzer maintains its own frame counters and emits
DmsEvent objects when thresholds are crossed.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional

from src.alerts.event_types import DmsEvent, EventType, Severity
from src.config.settings import MediaPipeConfig, DrowsinessConfig
from src.detectors.face_mesh_detector import FaceResult
from src.detectors.drowsiness_detector import DrowsinessResult

logger = logging.getLogger("dms.behaviors")


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * percentile / 100.0))
    idx = max(0, min(idx, len(ordered) - 1))
    return ordered[idx]


class FatigueAnalyzer:
    """
    Detects FATIGUE_DRIVING via calibrated EAR and rolling PERCLOS.

    Consecutive low EAR catches microsleeps. PERCLOS catches repeated partial
    closures over a moving window and filters out isolated blink noise.
    """
    _MIN_DYNAMIC_EAR_THRESHOLD = 0.12
    _MAX_DYNAMIC_EAR_THRESHOLD = 0.32

    def __init__(self, cfg: MediaPipeConfig):
        self._cfg = cfg
        self._consec = 0
        self._alert_active = False
        self._total_microsleep_events = 0
        self._calibration_ears: list[float] = []
        self._baseline_ear = 0.0
        self._is_calibrated = cfg.ear_calibration_frames <= 0
        self._perclos_window = deque(
            maxlen=max(1, int(cfg.perclos_window_frames))
        )

    def analyze(self, face: Optional[FaceResult]) -> Optional[DmsEvent]:
        if face is None:
            self._consec = 0
            self._alert_active = False
            self._perclos_window.clear()
            return None

        self._update_calibration(face.ear)

        threshold = self.effective_ear_threshold
        eye_closed = face.ear < threshold
        self._perclos_window.append(eye_closed)

        if eye_closed:
            self._consec += 1
        else:
            self._consec = 0

        perclos = self.perclos
        perclos_ready = len(self._perclos_window) == self._perclos_window.maxlen
        perclos_alarm = (
            perclos_ready and
            self._cfg.perclos_threshold > 0 and
            perclos >= self._cfg.perclos_threshold
        )
        consec_alarm = self._consec >= self._cfg.ear_consec_frames

        if consec_alarm or perclos_alarm:
            if not self._alert_active:
                self._alert_active = True
                self._total_microsleep_events += 1
                method = "PERCLOS" if perclos_alarm and not consec_alarm else "EAR"
                logger.warning(
                    "FATIGUE detected — method=%s EAR=%.3f threshold=%.3f "
                    "frames=%d perclos=%.2f",
                    method,
                    face.ear,
                    threshold,
                    self._consec,
                    perclos,
                )
                return DmsEvent.make(
                    EventType.FATIGUE_DRIVING,
                    Severity.CRITICAL,
                    f"Driver fatigue detected ({method}, EAR={face.ear:.3f})",
                    confidence=min(
                        1.0,
                        max(
                            self._consec / (self._cfg.ear_consec_frames * 2),
                            perclos,
                        ),
                    ),
                    ear=face.ear,
                    ear_threshold=threshold,
                    perclos=perclos,
                    method=method,
                    consec_frames=self._consec,
                )
            return None

        if (
            self._alert_active and
            not eye_closed and
            (not perclos_ready or perclos < self._cfg.perclos_threshold * 0.5)
        ):
            logger.debug("Fatigue cleared after %d frames", self._consec)
            self._alert_active = False

        return None

    def _update_calibration(self, ear: float):
        if self._is_calibrated or ear <= 0:
            return

        self._calibration_ears.append(ear)
        if len(self._calibration_ears) < self._cfg.ear_calibration_frames:
            return

        self._baseline_ear = max(_percentile(self._calibration_ears, 90), 0.05)
        self._is_calibrated = True
        logger.info(
            "EAR fallback calibration done — baseline=%.3f threshold=%.3f",
            self._baseline_ear,
            self.effective_ear_threshold,
        )

    @property
    def effective_ear_threshold(self) -> float:
        if not self._is_calibrated or self._baseline_ear <= 0:
            return self._cfg.ear_threshold

        dynamic_threshold = self._baseline_ear * self._cfg.ear_baseline_ratio
        min_threshold = self._cfg.ear_threshold * 0.75
        return max(
            min_threshold,
            min(
                self._MAX_DYNAMIC_EAR_THRESHOLD,
                max(self._MIN_DYNAMIC_EAR_THRESHOLD, dynamic_threshold),
            ),
        )

    @property
    def perclos(self) -> float:
        if not self._perclos_window:
            return 0.0
        return sum(1 for closed in self._perclos_window if closed) / len(
            self._perclos_window
        )

    @property
    def is_fatigued(self) -> bool:
        return self._alert_active

    @property
    def consec_frames(self) -> int:
        return self._consec


class DrowsinessAnalyzer:
    """
    Emits FATIGUE_DRIVING when the Vector DNN engine raises alarm.

    Alarm timing (probability > threshold for alarm_seconds) is handled
    inside DrowsinessEngine per the SafeDrive AI integration guide.
    """

    def __init__(self, cfg: DrowsinessConfig):
        self._cfg = cfg
        self._alert_active = False
        self._last_probability = 0.0
        self._last_status = ""

    def analyze(self, result: Optional[DrowsinessResult]) -> Optional[DmsEvent]:
        if result is None or not result.calibrated:
            self._alert_active = False
            return None

        self._last_probability = result.probability
        self._last_status = result.status

        if result.alarm:
            if not self._alert_active:
                self._alert_active = True
                logger.warning(
                    "DROWSINESS detected — prob=%.3f",
                    result.probability,
                )
                return DmsEvent.make(
                    EventType.FATIGUE_DRIVING,
                    Severity.CRITICAL,
                    f"Driver drowsiness detected (model={result.probability:.2f})",
                    confidence=result.probability,
                    drowsiness_prob=result.probability,
                )
            return None

        self._alert_active = False
        return None

    @property
    def is_drowsy(self) -> bool:
        return self._alert_active

    @property
    def probability(self) -> float:
        return self._last_probability

    @property
    def status(self) -> str:
        return self._last_status


class YawnAnalyzer:
    """
    Detects DRIVER_YAWNS via personalized Mouth Aspect Ratio (MAR).
    """

    def __init__(self, cfg: MediaPipeConfig):
        self._cfg = cfg
        self._consec = 0
        self._alert_active = False
        self._calibration_mars: list[float] = []
        self._baseline_mar = 0.0
        self._is_calibrated = cfg.mar_calibration_frames <= 0

    def analyze(self, face: Optional[FaceResult]) -> Optional[DmsEvent]:
        if face is None:
            self._consec = 0
            self._alert_active = False
            return None

        self._update_calibration(face.mar)
        if not self._is_calibrated:
            return None

        threshold = self.effective_mar_threshold
        if face.mar > threshold:
            self._consec += 1
        else:
            self._consec = 0
            self._alert_active = False
            return None

        if self._consec >= self._cfg.mar_consec_frames:
            if not self._alert_active:
                self._alert_active = True
                confidence = min(
                    1.0,
                    0.65 + max(0.0, face.mar - threshold) * 4.0,
                )
                return DmsEvent.make(
                    EventType.DRIVER_YAWNS,
                    Severity.WARNING,
                    f"Driver yawning detected (MAR={face.mar:.3f})",
                    confidence=confidence,
                    mar=face.mar,
                    mar_threshold=threshold,
                    baseline_mar=self._baseline_mar,
                    consec_frames=self._consec,
                )

        return None

    def _update_calibration(self, mar: float):
        if self._is_calibrated or mar <= 0:
            return

        self._calibration_mars.append(mar)
        if len(self._calibration_mars) < self._cfg.mar_calibration_frames:
            return

        self._baseline_mar = max(_percentile(self._calibration_mars, 30), 0.05)
        self._is_calibrated = True
        logger.info(
            "MAR calibration done — baseline=%.3f threshold=%.3f",
            self._baseline_mar,
            self.effective_mar_threshold,
        )

    @property
    def effective_mar_threshold(self) -> float:
        if not self._is_calibrated or self._baseline_mar <= 0:
            return self._cfg.mar_threshold
        return max(
            self._cfg.mar_threshold,
            self._baseline_mar + self._cfg.mar_baseline_margin,
        )

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
