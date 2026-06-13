"""
OpenCV drawing helpers for overlays, bounding boxes, and HUD.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional

# Color palette (BGR)
COLORS = {
    "green":   (50, 205, 50),
    "red":     (0, 0, 220),
    "orange":  (0, 140, 255),
    "yellow":  (0, 220, 220),
    "white":   (255, 255, 255),
    "black":   (0, 0, 0),
    "cyan":    (220, 220, 0),
    "gray":    (160, 160, 160),
    "alert":   (0, 60, 220),
}

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_text_box(
    frame: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    color: Tuple[int, int, int] = COLORS["white"],
    bg_color: Optional[Tuple[int, int, int]] = COLORS["black"],
    scale: float = 0.55,
    thickness: int = 1,
    alpha: float = 0.6,
):
    """Draw text with optional translucent background box."""
    x, y = pos
    (tw, th), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    pad = 4

    if bg_color is not None:
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad), bg_color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def draw_hud(
    frame: np.ndarray,
    fps: float,
    latency_ms: float,
    cpu: float,
    ram_mb: float,
    events: List[str],
    elapsed_seconds: float = 0,
):
    """Draw top-left performance HUD and active events."""
    h, w = frame.shape[:2]

    # Format elapsed time as HH:MM:SS
    hours = int(elapsed_seconds // 3600)
    minutes = int((elapsed_seconds % 3600) // 60)
    seconds = int(elapsed_seconds % 60)
    elapsed_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    lines = [
        f"Timer: {elapsed_str}",
        f"FPS: {fps:.1f}",
        f"Latency: {latency_ms:.1f}ms",
        f"CPU: {cpu:.1f}%",
        f"RAM: {ram_mb:.0f}MB",
    ]
    y = 22
    for line in lines:
        draw_text_box(frame, line, (8, y), color=COLORS["cyan"], scale=0.5)
        y += 20

    # Active events — right-aligned, red/orange
    y_e = 22
    for evt in events:
        color = COLORS["alert"] if "ALARM" in evt or "COLLISION" in evt or "FATIGUE" in evt else COLORS["orange"]
        (tw, _), _ = cv2.getTextSize(evt, FONT, 0.55, 1)
        draw_text_box(frame, evt, (w - tw - 12, y_e), color=color, scale=0.55, thickness=2)
        y_e += 26


def draw_bounding_box(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    label: str,
    conf: float,
    color: Tuple[int, int, int] = COLORS["green"],
):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    tag = f"{label} {conf:.2f}"
    draw_text_box(frame, tag, (x1, max(y1 - 6, 14)), color=COLORS["white"], bg_color=color, scale=0.48, alpha=0.8)


def draw_face_box(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, state: str):
    color = COLORS["red"] if state in ("FATIGUE", "YAWN", "DISTRACTED") else COLORS["green"]
    draw_bounding_box(frame, x1, y1, x2, y2, state, 1.0, color)


def draw_ear_mar(frame: np.ndarray, ear: float, mar: float):
    draw_text_box(frame, f"EAR:{ear:.3f}", (8, 120), color=COLORS["yellow"], scale=0.48)
    draw_text_box(frame, f"MAR:{mar:.3f}", (8, 142), color=COLORS["yellow"], scale=0.48)
