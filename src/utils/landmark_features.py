"""
Per-frame facial metrics for the Vector DNN pipeline.

Outputs the three raw features expected by DrowsinessEngine:
  relative_ear, mar, head_pitch — matching robust_driver_features.csv / training.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

# MediaPipe FaceMesh indices (aligned with training / project2)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH_INDICES = [78, 81, 13, 311, 308, 402, 14, 178]

NOSE_TIP = 1
FOREHEAD = 10
CHIN = 152


def _pt(landmarks_px: Sequence[Tuple[int, int]], idx: int) -> Tuple[float, float]:
    return float(landmarks_px[idx][0]), float(landmarks_px[idx][1])


def _dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def eye_aspect_ratio(landmarks_px: Sequence[Tuple[int, int]], indices: List[int]) -> float:
    pts = [_pt(landmarks_px, i) for i in indices]
    v1 = _dist(pts[1], pts[5])
    v2 = _dist(pts[2], pts[4])
    h = _dist(pts[0], pts[3])
    return (v1 + v2) / (2.0 * h + 1e-6)


def mouth_aspect_ratio(landmarks_px: Sequence[Tuple[int, int]]) -> float:
    """8-point MAR (same formula as training pipeline)."""
    pts = [_pt(landmarks_px, i) for i in MOUTH_INDICES]
    v1 = _dist(pts[1], pts[7])
    v2 = _dist(pts[2], pts[6])
    v3 = _dist(pts[3], pts[5])
    h = _dist(pts[0], pts[4])
    return (v1 + v2 + v3) / (3.0 * h + 1e-6)


def head_pitch_ratio(landmarks_px: Sequence[Tuple[int, int]]) -> float:
    """Nose–chin / forehead–nose ratio (drops when head nods down)."""
    nose = _pt(landmarks_px, NOSE_TIP)
    chin = _pt(landmarks_px, CHIN)
    forehead = _pt(landmarks_px, FOREHEAD)
    nose_to_chin = abs(chin[1] - nose[1])
    forehead_to_nose = abs(nose[1] - forehead[1])
    if forehead_to_nose < 1e-6:
        return 1.0
    return nose_to_chin / forehead_to_nose


def frame_metrics(
    landmarks_px: Sequence[Tuple[int, int]],
    baseline_ear: float,
) -> tuple[float, float, float]:
    """
    Compute (relative_ear, mar, head_pitch) for one frame.

    Returns clipped values ready for DrowsinessEngine.process_frame().
    """
    left = eye_aspect_ratio(landmarks_px, LEFT_EYE)
    right = eye_aspect_ratio(landmarks_px, RIGHT_EYE)
    avg_ear = (left + right) / 2.0
    baseline = max(baseline_ear, 0.05)
    relative_ear = min(max(avg_ear / baseline, 0.0), 2.0)
    mar = min(max(mouth_aspect_ratio(landmarks_px), 0.0), 1.0)
    head_pitch = min(max(head_pitch_ratio(landmarks_px), 0.0), 2.0)
    return relative_ear, mar, head_pitch
