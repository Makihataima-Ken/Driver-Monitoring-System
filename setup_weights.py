#!/usr/bin/env python3
"""
setup_weights.py — Download model weights for the DMS.

Downloads:
  - YOLOv8n (COCO base model)
  - Optionally prompts for custom fine-tuned weights

Run once before first launch:
    python setup_weights.py
"""

import os
import sys
import urllib.request
import shutil


WEIGHTS_DIR = "weights"
YOLOV8N_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
YOLOV8N_PATH = os.path.join(WEIGHTS_DIR, "yolov8n.pt")


def download_file(url: str, dest: str):
    print(f"Downloading {url} → {dest}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct = min(downloaded / total_size * 100, 100)
        bar = "#" * int(pct / 2)
        print(f"\r  [{bar:<50}] {pct:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


def main():
    print("=== DMS Weight Setup ===\n")
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    # YOLOv8n
    if os.path.exists(YOLOV8N_PATH):
        size_mb = os.path.getsize(YOLOV8N_PATH) / 1e6
        print(f"✓ YOLOv8n already present ({size_mb:.1f} MB)")
    else:
        print("Downloading YOLOv8n (~6 MB)...")
        try:
            # Let ultralytics handle it automatically on first run
            from ultralytics import YOLO
            model = YOLO("yolov8n.pt")
            # Move to weights/
            src = "yolov8n.pt"
            if os.path.exists(src):
                shutil.move(src, YOLOV8N_PATH)
                print(f"✓ Saved to {YOLOV8N_PATH}")
        except Exception as e:
            print(f"Auto-download failed: {e}")
            print(f"Manual download from: {YOLOV8N_URL}")

    print("\nSetup complete. Run the system with:")
    print("  python main.py --pipeline interior")
    print("  python main.py --pipeline interior --no-show  # headless")
    print("  python main.py --help")


if __name__ == "__main__":
    main()
