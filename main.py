#!/usr/bin/env python3
"""
Driver Monitoring and Vehicle Safety System
Entry point for real-time inference on PC or Raspberry Pi.
"""

import argparse
import signal
import logging

from src.config.settings import SystemConfig
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
        "--camera-backend",
        choices=["auto", "opencv", "picamera2", "rpicam"],
        default=None,
        help="Camera capture backend (default: config value)"
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
        "--log-file", type=str, default="logs/dms.log",
        help="Application log file path (default: logs/dms.log)"
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
        "--fps-target", type=int, default=20,
        help="Target FPS (lower = less CPU)"
    )
    parser.add_argument(
        "--web", action="store_true",
        help="Serve the annotated video and metrics in a browser"
    )
    parser.add_argument(
        "--web-host", type=str, default=None,
        help="Web dashboard bind address (default: config value)"
    )
    parser.add_argument(
        "--web-port", type=int, default=None,
        help="Web dashboard port (default: config value)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger("dms", log_level, args.log_file)
    logger.info("=== Driver Monitoring System Starting ===")

    config = SystemConfig.from_yaml(args.config)
    config.camera.index = args.camera
    if args.camera_backend is not None:
        config.camera.backend = args.camera_backend
    config.camera.width = args.width
    config.camera.height = args.height
    config.camera.fps_target = args.fps_target
    config.display.show = args.show
    config.pipeline.mode = args.pipeline
    if args.web:
        config.web.enabled = True
    if args.web_host is not None:
        config.web.host = args.web_host
    if args.web_port is not None:
        config.web.port = args.web_port

    pipeline = SystemPipeline(config)
    web_server = None

    # Graceful shutdown
    def _shutdown(sig, frame):
        logger.info("Shutdown signal received.")
        pipeline.request_stop()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        pipeline.start()
        if config.web.enabled:
            try:
                from src.web.server import WebServer
            except ImportError as e:
                raise RuntimeError(
                    "Flask is required for --web. Install it with: "
                    "uv pip install flask"
                ) from e
            web_server = WebServer(config.web, pipeline)
            web_server.start()
        pipeline.run()
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
    finally:
        pipeline.request_stop()
        if web_server:
            web_server.stop()
        pipeline.stop()
        logger.info("System stopped cleanly.")


if __name__ == "__main__":
    main()
