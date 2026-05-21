#!/usr/bin/env python3
"""
demo_test.py — Quick functionality test without a camera.

Uses a static image or synthetic frame to verify the pipeline
loads correctly. Useful for CI and first-time setup checks.

Usage:
    python demo_test.py
    python demo_test.py --image path/to/face.jpg
"""

import argparse
import sys
import logging
import time

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("demo_test")


def make_synthetic_frame(w=640, h=480) -> np.ndarray:
    """Create a coloured frame with a simple face-like oval for testing."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = (40, 40, 60)  # Dark background

    # Skin-tone oval (approximate face region)
    cx, cy = w // 2, h // 2
    cv2.ellipse(frame, (cx, cy), (90, 115), 0, 0, 360, (120, 160, 200), -1)

    # Eyes
    cv2.ellipse(frame, (cx - 30, cy - 20), (18, 10), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(frame, (cx + 30, cy - 20), (18, 10), 0, 0, 360, (255, 255, 255), -1)
    cv2.circle(frame, (cx - 30, cy - 20), 7, (40, 30, 20), -1)
    cv2.circle(frame, (cx + 30, cy - 20), 7, (40, 30, 20), -1)

    # Mouth
    cv2.ellipse(frame, (cx, cy + 45), (28, 14), 0, 0, 180, (80, 60, 120), -1)

    cv2.putText(frame, "SYNTHETIC TEST FRAME", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    return frame


def test_pipeline(image_path=None):
    logger.info("=== DMS Demo Test ===")

    # Import check
    try:
        from src.config.settings import SystemConfig
        from src.pipelines.interior_pipeline import InteriorPipeline
        logger.info("✓ Imports OK")
    except ImportError as e:
        logger.error(f"Import failed: {e}")
        logger.error("Run: pip install -r requirements.txt")
        sys.exit(1)

    cfg = SystemConfig()
    cfg.display.show = False  # Headless for test

    pipeline = InteriorPipeline(cfg)

    try:
        pipeline.start()
        logger.info("✓ Pipeline started")
    except Exception as e:
        logger.error(f"Pipeline start failed: {e}")
        sys.exit(1)

    if image_path:
        frame = cv2.imread(image_path)
        if frame is None:
            logger.error(f"Cannot read image: {image_path}")
            sys.exit(1)
    else:
        frame = make_synthetic_frame()
        logger.info("Using synthetic test frame (no image provided)")

    # Process several frames and measure timing
    latencies = []
    for i in range(10):
        t0 = time.perf_counter()
        events, annotated = pipeline.process(frame)
        dt = (time.perf_counter() - t0) * 1000
        latencies.append(dt)
        if events:
            for e in events:
                logger.info(f"  Event: {e.event_type.value} [{e.severity.value}]")

    avg_lat = sum(latencies) / len(latencies)
    logger.info(f"✓ 10 frames processed, avg latency: {avg_lat:.1f}ms ({1000/avg_lat:.1f} FPS equivalent)")

    pipeline.stop()
    logger.info("✓ Pipeline stopped cleanly")
    logger.info("=== Test PASSED ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="Path to test image")
    args = parser.parse_args()
    test_pipeline(args.image)
