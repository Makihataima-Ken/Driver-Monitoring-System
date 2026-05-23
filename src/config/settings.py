"""
System-wide configuration using dataclasses.
Supports loading from YAML and runtime overrides.
"""

from __future__ import annotations
import os
import yaml
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480
    fps_target: int = 20
    use_picamera2: bool = False          # Auto-detected if not set
    auto_detect_pi: bool = True
    buffer_size: int = 2                 # Threaded capture queue depth
    flip_horizontal: bool = False
    flip_vertical: bool = False


@dataclass
class MediaPipeConfig:
    # FaceMesh
    max_faces: int = 1
    refine_landmarks: bool = False       # Saves ~20% CPU on Pi
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.5

    # Thresholds
    ear_threshold: float = 0.22          # Eye Aspect Ratio → fatigue
    ear_consec_frames: int = 20          # Frames below EAR → alert
    mar_threshold: float = 0.6           # Mouth Aspect Ratio → yawn
    mar_consec_frames: int = 15

    # Head pose distraction angles (degrees)
    yaw_threshold: float = 30.0
    pitch_threshold: float = 20.0
    distraction_consec_frames: int = 25


@dataclass
class YOLOConfig:
    model_path: str = "weights/yolov8n.pt"
    conf_threshold: float = 0.45
    iou_threshold: float = 0.45
    imgsz: int = 320                     # Smaller → faster on Pi
    device: str = "cpu"
    half: bool = False                   # FP16 only on CUDA
    classes_of_interest: list = field(default_factory=lambda: [
        0,   # person / pedestrian
        2,   # car
        5,   # bus
        7,   # truck
        67,  # cell phone
    ])


@dataclass
class DrowsinessConfig:
    """Vector DNN settings (SafeDrive AI integration guide)."""
    enabled: bool = True
    model_path: str = "vector_dnn_drowsiness_model.keras"
    csv_path: str = "data/robust_driver_features.csv"
    window_size: int = 60               # Frames buffered before inference (~2s @ 30fps)
    threshold: float = 0.5
    alarm_seconds: float = 1.5          # Seconds above threshold → CRITICAL alarm
    calibration_frames: int = 50        # Frames to learn open-eye baseline EAR
    use_ear_fallback: bool = True


@dataclass
class AlertConfig:
    sound_enabled: bool = False
    sound_file: str = "assets/alert.wav"
    overlay_enabled: bool = True
    console_enabled: bool = True
    cooldown_seconds: float = 3.0        # Prevent alert spam


@dataclass
class DisplayConfig:
    show: bool = True
    window_name: str = "Driver Monitor"
    show_fps: bool = True
    show_landmarks: bool = True
    show_metrics: bool = True
    overlay_alpha: float = 0.6


@dataclass
class PipelineConfig:
    mode: str = "interior"               # interior | exterior | both
    interior_enabled: bool = True
    exterior_enabled: bool = False


@dataclass
class SystemConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    mediapipe: MediaPipeConfig = field(default_factory=MediaPipeConfig)
    drowsiness: DrowsinessConfig = field(default_factory=DrowsinessConfig)
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "SystemConfig":
        if not os.path.exists(path):
            return cls()  # Use all defaults
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        cfg = cls()
        for section, values in data.items():
            if hasattr(cfg, section) and isinstance(values, dict):
                section_obj = getattr(cfg, section)
                for k, v in values.items():
                    if hasattr(section_obj, k):
                        setattr(section_obj, k, v)
        return cfg

    def to_yaml(self, path: str):
        import dataclasses
        data = dataclasses.asdict(self)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
