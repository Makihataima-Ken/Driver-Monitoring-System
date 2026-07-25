#!/usr/bin/env python3
"""
Driver Monitoring and Vehicle Safety System
Entry point for real-time inference on PC or Raspberry Pi.
"""

import argparse
import signal
import sys
import logging

from src.alerts.alert_manager import AlertManager
from src.api.server import ApiServer
from src.config.settings import SystemConfig, apply_runtime_tuning
from src.pipelines.system_pipeline import SystemPipeline
from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-Time Driver Monitoring System"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--config", type=str, default="src/config/default.yaml",
        help="Path to config YAML"
    )
    parser.add_argument(
        "--pipeline", choices=["interior", "exterior", "both"],
        default="interior",
        help="Which pipeline(s) to run"
    )
    parser.add_argument(
        "--show", action="store_true", default=True,
        help="Show OpenCV display window"
    )
    parser.add_argument(
        "--no-show", dest="show", action="store_false",
        help="Disable display window (headless mode)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--http-api", action="store_true", default=False,
        help="Start FastAPI HTTP alert API"
    )
    parser.add_argument(
        "--api-host", type=str, default="0.0.0.0",
        help="HTTP API host"
    )
    parser.add_argument(
        "--api-port", type=int, default=8000,
        help="HTTP API port"
    )
    parser.add_argument(
        "--width", type=int, default=640,
        help="Frame width"
    )
    parser.add_argument(
        "--height", type=int, default=480,
        help="Frame height"
    )
    parser.add_argument(
        "--fps-target", type=int, default=15,
        help="Target FPS (lower = less CPU)"
    )
    # ---- Performance tuning flags (Raspberry Pi) ----
    parser.add_argument(
        "--yolo-interval", type=int, default=None,
        help="Run YOLO every Nth frame (higher = less CPU). Default from config."
    )
    parser.add_argument(
        "--no-yolo", action="store_true",
        help="Disable YOLO entirely (FaceMesh-only fatigue/yawn/distraction)"
    )
    parser.add_argument(
        "--no-roi", action="store_true",
        help="Disable face-ROI cropping and run YOLO on the full frame"
    )
    parser.add_argument(
        "--process-width", type=int, default=None,
        help="Downscale width fed to FaceMesh (0 = native). Default from config."
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="Log a per-stage latency breakdown every few seconds"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger("dms", log_level)
    logger.info("=== Driver Monitoring System Starting ===")

    config = SystemConfig.from_yaml(args.config)
    config.camera.index = args.camera
    config.camera.width = args.width
    config.camera.height = args.height
    config.camera.fps_target = args.fps_target
    config.display.show = args.show
    config.pipeline.mode = args.pipeline

    # ---- Performance overrides ----
    if args.yolo_interval is not None:
        config.yolo.interval = max(1, args.yolo_interval)
    if args.no_roi:
        config.yolo.roi_enabled = False
    if args.process_width is not None:
        config.mediapipe.process_width = args.process_width
    if args.profile:
        config.runtime.profile = True
    if args.no_yolo:
        # Cheapest possible interior pipeline: FaceMesh only.
        config.yolo.model_path = ""

    # PERF: must run before OpenCV / MediaPipe / NCNN spin up their thread
    # pools, otherwise they each grab all 4 Pi cores and fight each other.
    apply_runtime_tuning(config.runtime)

    alert_manager = AlertManager(config.alert)
    pipeline = SystemPipeline(config, alert_manager=alert_manager)

    api_server = None
    if args.http_api:
        logger.info("Starting HTTP alert API...")
        api_server = ApiServer(
            alert_manager=alert_manager,
            host=args.api_host,
            port=args.api_port,
        )
        api_server.start()

    # Graceful shutdown
    def _shutdown(sig, frame):
        logger.info("Shutdown signal received.")
        if api_server:
            api_server.stop()
        pipeline.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        pipeline.start()
        pipeline.run()
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
    finally:
        if api_server:
            api_server.stop()
        pipeline.stop()
        logger.info("System stopped cleanly.")


if __name__ == "__main__":
    main()
