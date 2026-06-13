"""
SystemPipeline — top-level orchestrator.

Coordinates:
  - CameraManager (threaded frame capture)
  - InteriorPipeline and/or ExteriorPipeline
  - AlertManager
  - PerformanceMonitor
  - OpenCV display window
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

import cv2
import numpy as np

from src.alerts.event_types import DmsEvent
from src.config.settings import SystemConfig
from src.camera.camera_manager import CameraManager
from src.pipelines.interior_pipeline import InteriorPipeline
from src.pipelines.exterior_pipeline import ExteriorPipeline
from src.alerts.alert_manager import AlertManager
from src.utils.metrics import FrameMetrics, PerformanceMonitor
from src.utils.drawing import draw_hud

logger = logging.getLogger("dms.system")


class SystemPipeline:
    def __init__(
        self,
        config: SystemConfig,
        alert_manager: Optional[AlertManager] = None,
        event_callback: Optional[Callable[[DmsEvent], None]] = None,
    ):
        self._cfg = config
        self._camera = CameraManager(config.camera)
        self._alert_mgr = alert_manager or AlertManager(config.alert)
        self._metrics = PerformanceMonitor(window=30)
        self._event_callback = event_callback

        self._interior: InteriorPipeline | None = None
        self._exterior: ExteriorPipeline | None = None
        self._running = False
        self._stopped = False
        self._stop_lock = threading.Lock()
        self._stream_condition = threading.Condition()
        self._stream_frame: np.ndarray | None = None
        self._stream_sequence = 0
        self._latest_perf = FrameMetrics()
        self._start_time: Optional[float] = None

        mode = config.pipeline.mode
        if mode in ("interior", "both"):
            self._interior = InteriorPipeline(config)
        if mode in ("exterior", "both"):
            self._exterior = ExteriorPipeline(config)

    def start(self):
        logger.info("Starting SystemPipeline...")
        self._camera.start()

        if self._interior:
            self._interior.start()
        if self._exterior:
            self._exterior.start()

        # Wait for first frame
        logger.info("Waiting for first camera frame...")
        for _ in range(50):
            if self._camera.read() is not None:
                break
            time.sleep(0.05)
        else:
            logger.warning(
                "No camera frame received after 2.5s. "
                "The web dashboard will report camera diagnostics."
            )

        self._running = True
        self._start_time = time.time()
        logger.info("SystemPipeline running.")

    def run(self):
        """Main inference loop — blocks until stopped."""
        fps_target = self._cfg.camera.fps_target
        frame_interval = 1.0 / fps_target

        while self._running:
            self._metrics.frame_start()
            t0 = time.perf_counter()

            frame = self._camera.read()
            if frame is None:
                time.sleep(0.01)
                continue

            all_events = []
            display_frame = frame

            # ── Interior pipeline ────────────────────────────────────────
            if self._interior:
                int_events, display_frame = self._interior.process(frame)
                all_events.extend(int_events)

            # ── Exterior pipeline ────────────────────────────────────────
            if self._exterior:
                ext_events, ext_frame = self._exterior.process(frame)
                all_events.extend(ext_events)
                # Side-by-side if both pipelines
                if self._interior:
                    small = cv2.resize(ext_frame, (display_frame.shape[1] // 3, display_frame.shape[0] // 3))
                    h, w = display_frame.shape[:2]
                    display_frame[h - small.shape[0]:h, w - small.shape[1]:w] = small
                else:
                    display_frame = ext_frame

            # ── Dispatch alerts ──────────────────────────────────────────
            for evt in all_events:
                if self._event_callback:
                    self._event_callback(evt)
                self._alert_mgr.dispatch(evt)

            # ── Performance metrics ──────────────────────────────────────
            perf = self._metrics.frame_end()

            # ── HUD overlay ──────────────────────────────────────────────
            if self._cfg.display.show_metrics:
                active_labels = self._alert_mgr.get_active_event_labels()
                elapsed_seconds = time.time() - self._start_time if self._start_time else 0
                draw_hud(
                    display_frame,
                    fps=perf.fps,
                    latency_ms=perf.latency_ms,
                    cpu=perf.cpu_percent,
                    ram_mb=perf.ram_mb,
                    events=active_labels,
                    elapsed_seconds=elapsed_seconds,
                )

            if self._cfg.web.enabled:
                self._publish_stream_frame(display_frame, perf)

            # ── Display window ────────────────────────────────────────────
            if self._cfg.display.show:
                cv2.imshow(self._cfg.display.window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # q or ESC
                    logger.info("Quit key pressed.")
                    break
                elif key == ord("s"):
                    # Screenshot
                    fname = f"screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(fname, display_frame)
                    logger.info(f"Screenshot saved: {fname}")

                # If the user closed the window (clicking the X), OpenCV
                # reports the window as not visible via getWindowProperty.
                # Detect that and exit the main loop so the program stops.
                try:
                    if cv2.getWindowProperty(self._cfg.display.window_name, cv2.WND_PROP_VISIBLE) < 1:
                        logger.info("Display window closed by user.")
                        break
                except Exception:
                    # Some OpenCV builds/contexts may raise if property is invalid;
                    # ignore and continue so we don't crash on unsupported platforms.
                    pass

            # ── FPS throttle ─────────────────────────────────────────────
            elapsed = time.perf_counter() - t0
            sleep = frame_interval - elapsed
            if sleep > 0:
                time.sleep(sleep)

        if self._cfg.display.show:
            cv2.destroyAllWindows()

    def _publish_stream_frame(self, frame: np.ndarray, perf: FrameMetrics):
        with self._stream_condition:
            self._stream_frame = frame.copy()
            self._stream_sequence += 1
            self._latest_perf = perf
            self._stream_condition.notify_all()

    def wait_for_stream_frame(
        self,
        last_sequence: int,
        timeout: float = 2.0,
    ) -> tuple[int, np.ndarray | None]:
        """Wait for a new annotated frame without reading the camera twice."""
        with self._stream_condition:
            self._stream_condition.wait_for(
                lambda: self._stream_sequence > last_sequence or not self._running,
                timeout=timeout,
            )
            if self._stream_frame is None:
                return self._stream_sequence, None
            return self._stream_sequence, self._stream_frame.copy()

    def get_status(self) -> dict:
        camera = self._camera.get_stats()
        with self._stream_condition:
            perf = self._latest_perf
            stream_sequence = self._stream_sequence

        if camera["width"] and camera["height"]:
            resolution = f'{camera["width"]}x{camera["height"]}'
        else:
            resolution = "waiting for camera"

        if not self._running:
            state = "stopped"
        elif camera["captured_frames"] == 0:
            state = "waiting_for_camera"
        else:
            state = "running"

        return {
            "state": state,
            "resolution": resolution,
            "processing_fps": perf.fps,
            "camera_fps": camera["capture_fps"],
            "target_fps": float(self._cfg.camera.fps_target),
            "latency_ms": perf.latency_ms,
            "cpu_percent": perf.cpu_percent,
            "ram_mb": perf.ram_mb,
            "captured_frames": camera["captured_frames"],
            "dropped_frames": camera["dropped_frames"],
            "stream_frames": stream_sequence,
            "events": self._alert_mgr.get_active_event_labels(),
        }

    @property
    def is_running(self) -> bool:
        return self._running

    def request_stop(self):
        """Ask the main loop to exit; resource cleanup happens in stop()."""
        self._running = False
        with self._stream_condition:
            self._stream_condition.notify_all()

    def stop(self):
        with self._stop_lock:
            if self._stopped:
                return
            self._stopped = True

        self.request_stop()
        if self._interior:
            self._interior.stop()
        if self._exterior:
            self._exterior.stop()
        self._camera.stop()
        if self._cfg.display.show:
            cv2.destroyAllWindows()
        logger.info("SystemPipeline stopped.")
