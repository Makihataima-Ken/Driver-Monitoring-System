"""
DrowsinessDetector — bridges FaceMesh FaceResult to DrowsinessEngine.

Handles EAR calibration, per-frame metric extraction, and delegates
60-frame buffering / vectorization / inference to DrowsinessEngine.
"""

from __future__ import annotations

import logging
import os
from typing import NamedTuple, Optional

import numpy as np

from src.config.settings import DrowsinessConfig
from src.detectors.drowsiness_engine import DrowsinessEngine
from src.detectors.face_mesh_detector import FaceResult
from src.utils.landmark_features import eye_aspect_ratio, frame_metrics, LEFT_EYE, RIGHT_EYE

logger = logging.getLogger("dms.drowsiness")


class DrowsinessResult(NamedTuple):
    probability: float
    alarm: bool
    status: str
    calibrated: bool
    buffer_ready: bool
    relative_ear: float = 0.0
    mar: float = 0.0
    head_pitch: float = 0.0


class DrowsinessDetector:
    def __init__(self, cfg: DrowsinessConfig):
        self._cfg = cfg
        self._engine: Optional[DrowsinessEngine] = None
        self._calibration_ears: list[float] = []
        self._baseline_ear: float = 0.0
        self._is_calibrated = False

    def load(self):
        self._engine = DrowsinessEngine(
            model_path=self._cfg.model_path,
            csv_path=self._cfg.csv_path,
            window_size=self._cfg.window_size,
            threshold=self._cfg.threshold,
            alarm_seconds=self._cfg.alarm_seconds,
        )

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    @property
    def buffer_ready(self) -> bool:
        return self._engine is not None and self._engine.buffer_ready

    def process(self, face: Optional[FaceResult]) -> Optional[DrowsinessResult]:
        if face is None or self._engine is None:
            return None

        avg_ear = (
            eye_aspect_ratio(face.landmarks_px, LEFT_EYE)
            + eye_aspect_ratio(face.landmarks_px, RIGHT_EYE)
        ) / 2.0

        if not self._is_calibrated:
            self._calibration_ears.append(avg_ear)
            if len(self._calibration_ears) >= self._cfg.calibration_frames:
                self._baseline_ear = float(np.percentile(self._calibration_ears, 90))
                if self._baseline_ear < 0.05:
                    self._baseline_ear = 0.05
                self._is_calibrated = True
                logger.info("Drowsiness EAR calibration done — baseline=%.3f", self._baseline_ear)
            return DrowsinessResult(
                probability=0.0,
                alarm=False,
                status=f"Calibrating ({len(self._calibration_ears)}/{self._cfg.calibration_frames})",
                calibrated=False,
                buffer_ready=False,
            )

        relative_ear, mar, head_pitch = frame_metrics(face.landmarks_px, self._baseline_ear)
        alarm, confidence, status = self._engine.process_frame(
            relative_ear, mar, head_pitch
        )

        return DrowsinessResult(
            probability=confidence,
            alarm=alarm,
            status=status,
            calibrated=True,
            buffer_ready=self._engine.buffer_ready,
            relative_ear=relative_ear,
            mar=mar,
            head_pitch=head_pitch,
        )
