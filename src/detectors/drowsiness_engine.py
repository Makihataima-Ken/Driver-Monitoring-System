"""
SafeDrive AI Vector DNN engine — 60-frame window → 13 statistical features → DNN.

Matches the official integration guide: buffer [relative_ear, mar, head_pitch],
vectorize with rolling statistics, scale with StandardScaler fit on training CSV.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque

import numpy as np

logger = logging.getLogger("dms.drowsiness.engine")

# Exact feature order fed to vector_dnn_drowsiness_model.keras (after StandardScaler)
VECTOR_FEATURE_NAMES = [
  # 1. Eye Aspect Ratio (EAR) — from relative_ear over 60 frames
    "EAR_Mean",
    "EAR_Std",
    "EAR_Min",
    "EAR_25th_Percentile",
  # 2. Mouth Aspect Ratio (MAR)
    "MAR_Mean",
    "MAR_Max",
    "MAR_Std",
  # 3. Head Pitch
    "Pitch_Mean",
    "Pitch_Min",
    "Pitch_Std",
  # 4. Advanced temporal
    "PERCLOS",
    "EAR_Trend",
    "Pitch_Trend",
]


def vectorize_window(window: np.ndarray, window_size: int) -> list[float]:
    """
    Convert a (window_size, 3) window into the 13 statistical features.

    Input columns per frame: relative_ear, mar, head_pitch
    Output order matches VECTOR_FEATURE_NAMES (training / SafeDrive AI spec).
    """
    ear_window = window[:, 0]
    mar_window = window[:, 1]
    pitch_window = window[:, 2]

    return [
        float(np.mean(ear_window)),                    # EAR_Mean
        float(np.std(ear_window)),                     # EAR_Std
        float(np.min(ear_window)),                     # EAR_Min
        float(np.percentile(ear_window, 25)),          # EAR_25th_Percentile
        float(np.mean(mar_window)),                    # MAR_Mean
        float(np.max(mar_window)),                     # MAR_Max
        float(np.std(mar_window)),                     # MAR_Std
        float(np.mean(pitch_window)),                  # Pitch_Mean
        float(np.min(pitch_window)),                   # Pitch_Min
        float(np.std(pitch_window)),                   # Pitch_Std
        float(np.sum(ear_window < 0.5) / window_size),  # PERCLOS
        float(ear_window[-1] - ear_window[0]),         # EAR_Trend
        float(pitch_window[-1] - pitch_window[0]),      # Pitch_Trend
    ]


class DrowsinessEngine:
    """Vector DNN drowsiness engine (official SafeDrive AI integration)."""

    def __init__(
        self,
        model_path: str,
        csv_path: str,
        window_size: int = 60,
        threshold: float = 0.5,
        alarm_seconds: float = 1.5,
    ):
        try:
            import tensorflow as tf  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "tensorflow is required. Install with: pip install tensorflow"
            ) from e
        try:
            from sklearn.preprocessing import StandardScaler
        except ImportError as e:
            raise ImportError(
                "scikit-learn is required. Install with: pip install scikit-learn"
            ) from e

        import tensorflow as tf

        self.window_size = window_size
        self.threshold = threshold
        self.alarm_seconds = alarm_seconds
        self.buffer: deque = deque(maxlen=window_size)
        self.drowsy_start_time: float | None = None

        model_path = self._resolve_path(model_path)
        csv_path = self._resolve_path(csv_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Drowsiness model not found: {model_path}")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Training CSV not found: {csv_path}. "
                "Copy robust_driver_features.csv into data/ (see README)."
            )

        logger.info("Loading Vector DNN model from %s", model_path)
        self.model = tf.keras.models.load_model(model_path)

        logger.info("Fitting StandardScaler from %s (may take a few seconds)...", csv_path)
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self._init_scaler(csv_path)
        logger.info("DrowsinessEngine ready (window=%d frames)", window_size)

    @staticmethod
    def _resolve_path(path: str) -> str:
        if os.path.isabs(path):
            return path
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        return os.path.join(root, path)

    def _init_scaler(self, csv_path: str):
        import pandas as pd
        from sklearn.preprocessing import StandardScaler

        df = pd.read_csv(csv_path)
        df["relative_ear"] = df["relative_ear"].clip(lower=0.0, upper=2.0)
        df["mar"] = df["mar"].clip(lower=0.0, upper=1.0)
        df["head_pitch"] = df["head_pitch"].clip(lower=0.0, upper=2.0)

        vectors = []
        for _, group in df.groupby("video_id"):
            data = group[["relative_ear", "mar", "head_pitch"]].values
            if len(data) < self.window_size:
                continue
            for i in range(0, len(data) - self.window_size, 5):
                window = data[i : i + self.window_size]
                vectors.append(vectorize_window(window, self.window_size))

        if not vectors:
            raise ValueError(
                f"No training windows built from {csv_path}. "
                "Check CSV columns: video_id, relative_ear, mar, head_pitch"
            )
        self.scaler.fit(vectors)

    def process_frame(
        self,
        relative_ear: float,
        mar: float,
        head_pitch: float,
    ) -> tuple[bool, float, str]:
        """
        Process one frame's raw features.

        Returns:
            alarm: True if drowsiness persisted > alarm_seconds
            confidence: model probability (0 until buffer is full)
            status: human-readable status string
        """
        relative_ear = float(np.clip(relative_ear, 0.0, 2.0))
        mar = float(np.clip(mar, 0.0, 1.0))
        head_pitch = float(np.clip(head_pitch, 0.0, 2.0))

        self.buffer.append([relative_ear, mar, head_pitch])

        prediction_confidence = 0.0
        alarm = False
        status = "Normal"

        if len(self.buffer) < self.window_size:
            status = f"Buffering ({len(self.buffer)}/{self.window_size})"
            return alarm, prediction_confidence, status

        window_data = np.array(list(self.buffer))
        feature_vector = vectorize_window(window_data, self.window_size)
        scaled_vector = self.scaler.transform([feature_vector])
        prediction_confidence = float(self.model.predict(scaled_vector, verbose=0)[0][0])

        current_time = time.time()
        if prediction_confidence > self.threshold:
            if self.drowsy_start_time is None:
                self.drowsy_start_time = current_time
            duration = current_time - self.drowsy_start_time
            if duration > self.alarm_seconds:
                alarm = True
                status = "CRITICAL: DROWSINESS DETECTED!"
            else:
                status = f"Warning: Drowsiness suspected ({duration:.1f}s)"
        else:
            self.drowsy_start_time = None

        return alarm, prediction_confidence, status

    @property
    def buffer_ready(self) -> bool:
        return len(self.buffer) >= self.window_size
