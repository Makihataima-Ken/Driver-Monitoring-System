#!/usr/bin/env python3
"""
bench_interior.py - measure where the milliseconds actually go.

Run this ON THE PI before and after tuning. It isolates each stage of the
interior pipeline so you optimise facts instead of guesses.

    python3 tools/bench_interior.py --frames 100
    python3 tools/bench_interior.py --frames 100 --source video.mp4
    python3 tools/bench_interior.py --frames 60 --no-yolo

It reports, per stage: mean / p50 / p95 latency and the implied FPS ceiling.
"""

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.config.settings import SystemConfig, apply_runtime_tuning


def summarize(name, samples):
    if not samples:
        print(f"  {name:<22} (no samples)")
        return 0.0
    mean = statistics.mean(samples)
    p50 = statistics.median(samples)
    p95 = sorted(samples)[int(len(samples) * 0.95) - 1] if len(samples) > 1 else mean
    print(f"  {name:<22} mean={mean:6.1f} ms  p50={p50:6.1f}  p95={p95:6.1f}  "
          f"-> {1000.0 / max(mean, 1e-6):5.1f} FPS ceiling")
    return mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="src/config/default.yaml")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--source", default=None,
                    help="Video file or camera index. Default: synthetic noise frames.")
    ap.add_argument("--no-yolo", action="store_true")
    ap.add_argument("--no-facemesh", action="store_true")
    args = ap.parse_args()

    cfg = SystemConfig.from_yaml(args.config)
    apply_runtime_tuning(cfg.runtime)

    import cv2
    from src.detectors.face_mesh_detector import FaceMeshDetector
    from src.detectors.yolo_detector import YOLODetector

    w, h = cfg.camera.width, cfg.camera.height

    # ---- frame source -------------------------------------------------
    cap = None
    if args.source is not None:
        src = int(args.source) if str(args.source).isdigit() else args.source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            sys.exit(f"Could not open source: {args.source}")

    def next_frame():
        if cap is not None:
            ok, f = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, f = cap.read()
            return cv2.resize(f, (w, h)) if ok else None
        return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

    # ---- models -------------------------------------------------------
    face = None if args.no_facemesh else FaceMeshDetector(cfg.mediapipe)
    yolo = None
    if not args.no_yolo:
        try:
            yolo = YOLODetector(
                model_path=cfg.yolo.model_path,
                conf_threshold=cfg.yolo.conf_threshold,
                iou_threshold=cfg.yolo.iou_threshold,
                imgsz=cfg.yolo.imgsz,
                device=cfg.yolo.device,
                classes_of_interest=cfg.yolo.classes_of_interest,
                backend=cfg.yolo.backend,
                warmup_iterations=cfg.yolo.warmup_iterations,
            )
        except Exception as exc:
            print(f"YOLO unavailable ({exc}) - benchmarking FaceMesh only")

    roi_size = int(cfg.yolo.imgsz)
    face_ms, yolo_full_ms, yolo_roi_ms, copy_ms = [], [], [], []

    print(f"\nBenchmarking {args.frames} frames at {w}x{h} "
          f"(facemesh process_width={cfg.mediapipe.process_width}, "
          f"yolo imgsz={cfg.yolo.imgsz}, backend={yolo.backend if yolo else 'none'})\n")

    for i in range(args.frames):
        frame = next_frame()
        if frame is None:
            break

        t0 = time.perf_counter()
        _ = frame.copy()
        copy_ms.append((time.perf_counter() - t0) * 1000)

        if face is not None:
            t0 = time.perf_counter()
            face.process(frame)
            face_ms.append((time.perf_counter() - t0) * 1000)

        if yolo is not None:
            t0 = time.perf_counter()
            yolo.detect(frame)
            yolo_full_ms.append((time.perf_counter() - t0) * 1000)

            crop = frame[0:roi_size, 0:roi_size]
            t0 = time.perf_counter()
            yolo.detect(crop)
            yolo_roi_ms.append((time.perf_counter() - t0) * 1000)

    print("Per-stage latency:")
    f_mean = summarize("frame.copy()", copy_ms)
    fm_mean = summarize("FaceMesh", face_ms)
    yf_mean = summarize("YOLO full frame", yolo_full_ms)
    yr_mean = summarize(f"YOLO ROI ({roi_size}px)", yolo_roi_ms)

    interval = max(1, cfg.yolo.interval)
    budget = fm_mean + (yr_mean / interval) + f_mean
    print(f"\nEstimated pipeline latency with yolo.interval={interval} and ROI: "
          f"{budget:.1f} ms -> {1000.0 / max(budget, 1e-6):.1f} FPS")
    if yf_mean and yr_mean:
        print(f"ROI speedup vs full frame: {yf_mean / max(yr_mean, 1e-6):.1f}x")

    if face is not None:
        face.release()
    if cap is not None:
        cap.release()


if __name__ == "__main__":
    main()
