"""
YOLODetector - low-latency YOLO inference wrapper for Raspberry Pi.

PERF NOTES (Pi 4, interior pipeline)
------------------------------------
The original implementation loaded ``yolov8n.pt`` through Ultralytics,
which runs the full PyTorch stack. On a Pi 4 that costs ~250-450 ms per
frame and ~1 GB RSS. This version:

  * Prefers an exported **NCNN** model directory (``*_ncnn_model``) or an
    **ONNX** file. NCNN uses hand-written ARM NEON kernels and is 3-5x
    faster than PyTorch on a Cortex-A72.
  * Warms the model up at load time so the first real frame is not stalled.
  * Supports **ROI inference** via ``offset`` so the caller can run
    detection on a face crop and get boxes back in full-frame coordinates.
  * Tracks per-inference latency so the pipeline can adapt its throttling.

IMPORTANT: for exported backends (NCNN / ONNX) ``imgsz`` is baked into the
exported graph. It must match the value used at export time, otherwise
Ultralytics either errors or silently re-letterboxes and you lose accuracy.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger("dms.yolo")

# COCO class names relevant to DMS
COCO_CLASSES = {
    0:  "person",
    2:  "car",
    5:  "bus",
    7:  "truck",
    67: "cell_phone",
}

# Additional custom class names (when using fine-tuned weights)
CUSTOM_CLASSES = {
    0: "phone",
    1: "cigarette",
    2: "no_seatbelt",
    3: "seatbelt",
}


@dataclass
class Detection:
    class_id: int
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center(self):
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def area(self):
        return (self.x2 - self.x1) * (self.y2 - self.y1)


def resolve_backend(model_path: str, requested: str = "auto") -> str:
    """Infer the inference backend from the weights path."""
    if requested and requested != "auto":
        return requested
    p = model_path.rstrip("/\\")
    if p.endswith("_ncnn_model") or os.path.isdir(p):
        return "ncnn"
    if p.endswith(".onnx"):
        return "onnx"
    if p.endswith("_openvino_model"):
        return "openvino"
    return "pytorch"


class YOLODetector:
    """Wraps Ultralytics YOLO with Pi-oriented defaults."""

    def __init__(
        self,
        model_path: str = "weights/yolov8n_ncnn_model",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        imgsz: int = 192,
        device: str = "cpu",
        classes_of_interest: Optional[List[int]] = None,
        class_map: Optional[dict] = None,
        backend: str = "auto",
        warmup_iterations: int = 2,
    ):
        self._conf = conf_threshold
        self._iou = iou_threshold
        self._imgsz = imgsz
        self._device = device
        self._classes = classes_of_interest
        self._class_map = class_map or COCO_CLASSES
        self._backend = resolve_backend(model_path, backend)
        self._model = None
        self._last_latency_ms = 0.0
        self._infer_count = 0
        self._total_latency_ms = 0.0

        self._load_model(model_path)
        if warmup_iterations > 0:
            self._warmup(warmup_iterations)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_model(self, path: str):
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError:
            raise ImportError(
                "ultralytics is not installed.\nRun: pip install ultralytics"
            )

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"YOLO weights not found at '{path}'.\n"
                "Export a Pi-friendly model first:\n"
                "    python tools/export_ncnn.py --model yolo11n.pt --imgsz 192"
            )

        if self._backend == "pytorch":
            logger.warning(
                "Loading a PyTorch .pt model (%s). On a Raspberry Pi 4 this is "
                "3-5x slower than NCNN and uses ~1 GB RAM. Export with "
                "tools/export_ncnn.py for real-time performance.",
                path,
            )

        t0 = time.perf_counter()
        # task="detect" is required for exported formats, which carry no
        # task metadata of their own.
        self._model = YOLO(path, task="detect")
        logger.info(
            "YOLO loaded | backend=%s | imgsz=%s | path=%s | %.2fs",
            self._backend, self._imgsz, path, time.perf_counter() - t0,
        )

    def _warmup(self, iterations: int):
        """PERF: first inference allocates buffers and JITs kernels; doing it
        here keeps the first live frame from stalling for 1-2 seconds."""
        dummy = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        for _ in range(iterations):
            try:
                self._predict(dummy)
            except Exception as exc:
                logger.warning("Warmup inference failed: %s", exc)
                return
        logger.info(
            "YOLO warmup done (%s iters, %.0f ms total)",
            iterations, (time.perf_counter() - t0) * 1000,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _predict(self, image: np.ndarray):
        return self._model.predict(
            source=image,
            conf=self._conf,
            iou=self._iou,
            imgsz=self._imgsz,
            device=self._device,
            verbose=False,
            classes=self._classes,
            max_det=8,          # PERF: interior scenes never need 300 boxes
        )

    def detect(
        self,
        frame: np.ndarray,
        offset: Tuple[int, int] = (0, 0),
    ) -> List[Detection]:
        """Run inference on ``frame``.

        Args:
            frame: BGR image. May be a crop of a larger frame.
            offset: ``(dx, dy)`` of the crop's top-left corner within the
                original frame. Returned boxes are shifted by this so the
                caller always gets full-frame coordinates.
        """
        if self._model is None or frame is None or frame.size == 0:
            return []

        dx, dy = offset
        t0 = time.perf_counter()
        results = self._predict(frame)
        self._last_latency_ms = (time.perf_counter() - t0) * 1000.0
        self._infer_count += 1
        self._total_latency_ms += self._last_latency_ms

        detections: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cid = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(Detection(
                    class_id=cid,
                    label=self._class_map.get(cid, str(cid)),
                    confidence=conf,
                    x1=x1 + dx, y1=y1 + dy,
                    x2=x2 + dx, y2=y2 + dy,
                ))

        return detections

    # ------------------------------------------------------------------
    # Metrics / helpers
    # ------------------------------------------------------------------

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def last_latency_ms(self) -> float:
        return self._last_latency_ms

    @property
    def avg_latency_ms(self) -> float:
        if self._infer_count == 0:
            return 0.0
        return self._total_latency_ms / self._infer_count

    def filter_by_label(self, detections: List[Detection], label: str) -> List[Detection]:
        return [d for d in detections if d.label == label]

    def filter_by_class_id(self, detections: List[Detection], class_id: int) -> List[Detection]:
        return [d for d in detections if d.class_id == class_id]
