"""
YOLODetector — lightweight YOLOv8n inference wrapper.

Uses Ultralytics YOLO for object detection.
Targets the following classes for DMS:
  - 0:  person / pedestrian
  - 2:  car
  - 5:  bus
  - 7:  truck
  - 67: cell phone

Custom fine-tuned weights for phone/cigarette/seatbelt can be
dropped in as a separate model (see SeatbeltPhoneDetector).

Optimised for Pi:
  - imgsz=320 (vs 640)
  - FP32 CPU inference
  - Subset of classes only
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional

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


class YOLODetector:
    """
    Wraps YOLOv8n for real-time object detection.

    model_path: path to .pt or .onnx weights.
    If weights don't exist, downloads YOLOv8n automatically.
    """

    def __init__(
        self,
        model_path: str = "weights/yolov8n.pt",
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        imgsz: int = 320,
        device: str = "cpu",
        classes_of_interest: Optional[List[int]] = None,
        class_map: Optional[dict] = None,
    ):
        self._conf = conf_threshold
        self._iou = iou_threshold
        self._imgsz = imgsz
        self._device = device
        self._classes = classes_of_interest
        self._class_map = class_map or COCO_CLASSES
        self._model = None
        self._load_model(model_path)

    def _load_model(self, path: str):
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError:
            raise ImportError(
                "ultralytics is not installed.\n"
                "Run: pip install ultralytics"
            )

        if not os.path.exists(path):
            logger.warning(f"Weights not found at '{path}' — downloading YOLOv8n...")
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        self._model = YOLO(path)
        logger.info(f"YOLO model loaded from: {path}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run inference and return list of Detection objects."""
        if self._model is None:
            return []

        results = self._model.predict(
            source=frame,
            conf=self._conf,
            iou=self._iou,
            imgsz=self._imgsz,
            device=self._device,
            verbose=False,
            classes=self._classes,
        )

        detections: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cid = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = self._class_map.get(cid, str(cid))
                detections.append(Detection(
                    class_id=cid,
                    label=label,
                    confidence=conf,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                ))

        return detections

    def filter_by_label(self, detections: List[Detection], label: str) -> List[Detection]:
        return [d for d in detections if d.label == label]

    def filter_by_class_id(self, detections: List[Detection], class_id: int) -> List[Detection]:
        return [d for d in detections if d.class_id == class_id]
