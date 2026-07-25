#!/usr/bin/env python3
"""
export_ncnn.py - build a Raspberry-Pi-friendly YOLO model.

RUN THIS ON YOUR PC (not the Pi). Exporting needs PyTorch; running the
result does not.

Why NCNN?
    Ultralytics' .pt path executes PyTorch on the Pi's Cortex-A72 cores.
    NCNN is Tencent's ARM-native inference engine with hand-written NEON
    kernels: 3-5x faster, ~40 MB of RAM instead of ~1 GB, and ~1 s model
    load instead of ~15 s.

Usage:
    python tools/export_ncnn.py --model yolo11n.pt --imgsz 192
    python tools/export_ncnn.py --model yolo11n.pt --imgsz 256 --also-onnx

The exported directory (e.g. weights/yolo11n_ncnn_model) is what you copy
to the Pi and point `yolo.model_path` at.

IMPORTANT: `imgsz` is baked into the exported graph. The value here MUST
match `yolo.imgsz` in default.yaml.
  - imgsz 192 -> use with roi_enabled: true  (detect on a face crop)
  - imgsz 256 -> use with roi_enabled: false (detect on the full frame)
"""

import argparse
import os
import shutil
import sys


def main():
    ap = argparse.ArgumentParser(description="Export YOLO to NCNN for Raspberry Pi")
    ap.add_argument("--model", default="yolo11n.pt",
                    help="Source .pt weights (yolo11n.pt, yolov8n.pt, or your fine-tune)")
    ap.add_argument("--imgsz", type=int, default=192,
                    help="Export input size. Must match yolo.imgsz in the config.")
    ap.add_argument("--out-dir", default="weights",
                    help="Where to place the exported model directory")
    ap.add_argument("--also-onnx", action="store_true",
                    help="Additionally export ONNX (useful for onnxruntime benchmarking)")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics is required on the export machine: pip install ultralytics")

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading {args.model} ...")
    model = YOLO(args.model)

    print(f"Exporting NCNN at imgsz={args.imgsz} ...")
    ncnn_path = model.export(format="ncnn", imgsz=args.imgsz, half=False)
    dest = os.path.join(args.out_dir, os.path.basename(str(ncnn_path).rstrip("/\\")))
    if os.path.abspath(str(ncnn_path)) != os.path.abspath(dest):
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.move(str(ncnn_path), dest)
    print(f"NCNN model ready: {dest}")

    if args.also_onnx:
        print(f"Exporting ONNX at imgsz={args.imgsz} ...")
        onnx_path = model.export(format="onnx", imgsz=args.imgsz, simplify=True, opset=12)
        onnx_dest = os.path.join(args.out_dir, os.path.basename(str(onnx_path)))
        if os.path.abspath(str(onnx_path)) != os.path.abspath(onnx_dest):
            shutil.move(str(onnx_path), onnx_dest)
        print(f"ONNX model ready: {onnx_dest}")

    print("\nNow set in src/config/default.yaml:")
    print(f"  yolo:")
    print(f"    model_path: {dest}")
    print(f"    imgsz: {args.imgsz}")
    print(f"    backend: auto")
    print("\nOn the Pi, install the runtime only:  pip install ncnn")


if __name__ == "__main__":
    main()
