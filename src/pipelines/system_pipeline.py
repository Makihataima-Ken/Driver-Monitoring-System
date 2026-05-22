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
from src.utils.metrics import PerformanceMonitor
from src.utils.drawing import draw_hud

logger = logging.getLogger("dms.system")


class SystemPipeline:
    def __init__(
        self,
        config: SystemConfig,
        event_callback: Optional[Callable[[DmsEvent], None]] = None,
    ):
        self._cfg = config
        self._camera = CameraManager(config.camera)
        self._alert_mgr = AlertManager(config.alert)
        self._metrics = PerformanceMonitor(window=30)
        self._event_callback = event_callback

        self._interior: InteriorPipeline | None = None
        self._exterior: ExteriorPipeline | None = None
        self._running = False

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

        self._running = True
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
                draw_hud(
                    display_frame,
                    fps=perf.fps,
                    latency_ms=perf.latency_ms,
                    cpu=perf.cpu_percent,
                    ram_mb=perf.ram_mb,
                    events=active_labels,
                )

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

        cv2.destroyAllWindows()

    def stop(self):
        self._running = False
        if self._interior:
            self._interior.stop()
        if self._exterior:
            self._exterior.stop()
        self._camera.stop()
        cv2.destroyAllWindows()
        logger.info("SystemPipeline stopped.")
