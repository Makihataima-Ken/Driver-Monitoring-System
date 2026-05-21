"""
CameraManager — unified camera abstraction.

Supports:
  - OpenCV VideoCapture (PC + Pi USB webcam)
  - Picamera2 (Pi CSI camera)

Uses a background thread for frame capture so the main inference
loop always gets the freshest frame without I/O blocking.
"""

from __future__ import annotations

import threading
import time
import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from src.config.settings import CameraConfig
from src.utils.platform_detect import is_raspberry_pi, picamera2_available

logger = logging.getLogger("dms.camera")


class _OpenCVBackend:
    """VideoCapture wrapper with optimized settings."""

    def __init__(self, cfg: CameraConfig):
        self._cfg = cfg
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self):
        idx = self._cfg.index
        logger.info(f"Opening OpenCV camera index={idx}")
        self._cap = cv2.VideoCapture(idx)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {idx}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.height)
        self._cap.set(cv2.CAP_PROP_FPS, self._cfg.fps_target)
        # Minimize internal buffer to reduce latency
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Camera opened at {actual_w}x{actual_h}")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            return False, None
        return self._cap.read()

    def release(self):
        if self._cap:
            self._cap.release()
            self._cap = None


class _Picamera2Backend:
    """Picamera2 backend for Pi CSI cameras."""

    def __init__(self, cfg: CameraConfig):
        self._cfg = cfg
        self._cam = None

    def open(self):
        from picamera2 import Picamera2  # type: ignore
        logger.info("Opening Picamera2 (CSI camera)")
        self._cam = Picamera2()
        config = self._cam.create_preview_configuration(
            main={"size": (self._cfg.width, self._cfg.height), "format": "RGB888"}
        )
        self._cam.configure(config)
        self._cam.start()
        time.sleep(0.5)  # Camera warm-up

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cam is None:
            return False, None
        try:
            frame = self._cam.capture_array()
            # Picamera2 gives RGB — convert to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return True, frame_bgr
        except Exception as e:
            logger.warning(f"Picamera2 read error: {e}")
            return False, None

    def release(self):
        if self._cam:
            self._cam.stop()
            self._cam = None


class CameraManager:
    """
    Thread-safe camera manager.

    The capture thread continuously reads frames into a slot.
    The inference thread reads from that slot — always getting
    the latest frame without blocking on camera I/O.
    """

    def __init__(self, cfg: CameraConfig):
        self._cfg = cfg
        self._backend = self._choose_backend()
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_count = 0
        self._drop_count = 0

    def _choose_backend(self):
        if self._cfg.use_picamera2:
            if picamera2_available():
                return _Picamera2Backend(self._cfg)
            else:
                logger.warning("Picamera2 requested but not available — falling back to OpenCV")

        if self._cfg.auto_detect_pi and is_raspberry_pi() and picamera2_available():
            logger.info("Raspberry Pi detected — trying Picamera2 backend")
            return _Picamera2Backend(self._cfg)

        return _OpenCVBackend(self._cfg)

    def start(self):
        self._backend.open()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="camera-capture")
        self._thread.start()
        logger.info("Camera capture thread started")

    def _capture_loop(self):
        interval = 1.0 / max(self._cfg.fps_target, 1)
        while self._running:
            t0 = time.perf_counter()
            ok, frame = self._backend.read()
            if ok and frame is not None:
                if self._cfg.flip_horizontal:
                    frame = cv2.flip(frame, 1)
                if self._cfg.flip_vertical:
                    frame = cv2.flip(frame, 0)
                with self._lock:
                    self._frame = frame
                    self._frame_count += 1
            else:
                self._drop_count += 1
                if self._drop_count % 30 == 0:
                    logger.warning(f"Camera: {self._drop_count} dropped frames")

            elapsed = time.perf_counter() - t0
            sleep = interval - elapsed
            if sleep > 0:
                time.sleep(sleep)

    def read(self) -> Optional[np.ndarray]:
        """Return the latest captured frame (or None if none yet)."""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._backend.release()
        logger.info(f"Camera stopped. Captured={self._frame_count}, Dropped={self._drop_count}")

    @property
    def frame_count(self) -> int:
        return self._frame_count
