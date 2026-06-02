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
import shutil
import subprocess
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np

from src.config.settings import CameraConfig
from src.utils.platform_detect import (
    is_raspberry_pi,
    picamera2_available,
    rpicam_camera_available,
)

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


class _RpicamVidBackend:
    """CSI camera backend using the rpicam-vid command-line application."""

    def __init__(self, cfg: CameraConfig):
        self._cfg = cfg
        self._process: Optional[subprocess.Popen] = None
        self._buffer = b""

    def open(self):
        command = shutil.which("rpicam-vid")
        if command is None:
            raise RuntimeError(
                "rpicam-vid is not installed. Install it with: "
                "sudo apt install rpicam-apps-lite"
            )

        args = [
            command,
            "--camera", str(self._cfg.index),
            "--nopreview",
            "--codec", "mjpeg",
            "--width", str(self._cfg.width),
            "--height", str(self._cfg.height),
            "--framerate", str(self._cfg.fps_target),
            "--timeout", "0",
            "--output", "-",
        ]
        logger.info(
            "Opening rpicam-vid CSI camera index=%d at %dx%d",
            self._cfg.index,
            self._cfg.width,
            self._cfg.height,
        )
        self._process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        time.sleep(0.5)
        if self._process.poll() is not None:
            self.release()
            raise RuntimeError(
                "rpicam-vid could not start. Check the CSI camera with: "
                "rpicam-hello --list-cameras"
            )

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._process is None or self._process.stdout is None:
            return False, None

        while self._process.poll() is None:
            frame = self._extract_frame()
            if frame is not None:
                return True, frame

            chunk = self._process.stdout.read(4096)
            if not chunk:
                return False, None
            self._buffer += chunk

        return False, None

    def _extract_frame(self) -> Optional[np.ndarray]:
        start = self._buffer.find(b"\xff\xd8")
        if start < 0:
            self._buffer = self._buffer[-1:]
            return None

        end = self._buffer.find(b"\xff\xd9", start + 2)
        if end < 0:
            if start > 0:
                self._buffer = self._buffer[start:]
            return None

        jpeg = self._buffer[start:end + 2]
        self._buffer = self._buffer[end + 2:]
        encoded = np.frombuffer(jpeg, dtype=np.uint8)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    def release(self):
        if self._process is None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=2)
        self._process = None
        self._buffer = b""


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
        self._capture_timestamps = deque(maxlen=60)

    def _choose_backend(self):
        backend = self._cfg.backend.lower()
        if backend not in {"auto", "opencv", "picamera2", "rpicam"}:
            raise ValueError(
                f"Unsupported camera backend '{self._cfg.backend}'. "
                "Choose: auto, opencv, picamera2, or rpicam"
            )

        if backend == "opencv":
            return _OpenCVBackend(self._cfg)

        if backend == "rpicam":
            return _RpicamVidBackend(self._cfg)

        if backend == "picamera2" or self._cfg.use_picamera2:
            if picamera2_available():
                return _Picamera2Backend(self._cfg)
            if backend == "picamera2":
                raise RuntimeError(
                    "Picamera2 requested but not available. "
                    "Use camera backend 'rpicam' with an isolated Python environment."
                )
            logger.warning("Picamera2 requested but not available - trying another backend")

        if self._cfg.auto_detect_pi and is_raspberry_pi():
            if picamera2_available():
                logger.info("Raspberry Pi detected - using Picamera2 backend")
                return _Picamera2Backend(self._cfg)
            if rpicam_camera_available():
                logger.info("Raspberry Pi CSI camera detected - using rpicam-vid backend")
                return _RpicamVidBackend(self._cfg)

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
                    self._capture_timestamps.append(time.perf_counter())
            else:
                with self._lock:
                    self._drop_count += 1
                    drop_count = self._drop_count
                if drop_count % 30 == 0:
                    logger.warning(f"Camera: {drop_count} dropped frames")

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
        was_active = self._running or self._thread is not None
        self._running = False
        self._backend.release()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        if not was_active:
            return
        logger.info(f"Camera stopped. Captured={self._frame_count}, Dropped={self._drop_count}")

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def get_stats(self) -> dict:
        """Return a thread-safe snapshot for the web status endpoint."""
        with self._lock:
            timestamps = list(self._capture_timestamps)
            frame_count = self._frame_count
            drop_count = self._drop_count
            if self._frame is None:
                width, height = 0, 0
            else:
                height, width = self._frame.shape[:2]

        if len(timestamps) >= 2:
            span = timestamps[-1] - timestamps[0]
            capture_fps = (len(timestamps) - 1) / max(span, 1e-6)
        else:
            capture_fps = 0.0

        return {
            "capture_fps": round(capture_fps, 1),
            "captured_frames": frame_count,
            "dropped_frames": drop_count,
            "width": width,
            "height": height,
        }
