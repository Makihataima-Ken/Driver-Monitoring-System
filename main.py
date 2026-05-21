#!/usr/bin/env python3
"""
Driver Monitoring and Vehicle Safety System
Entry point for real-time inference on PC or Raspberry Pi.
"""

import argparse
import signal
import sys
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

    pipeline = SystemPipeline(config)

    # Graceful shutdown
    def _shutdown(sig, frame):
        logger.info("Shutdown signal received.")
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
        pipeline.stop()
        logger.info("System stopped cleanly.")


if __name__ == "__main__":
    main()
