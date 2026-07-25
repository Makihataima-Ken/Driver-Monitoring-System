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

    # PERF: width of the image actually fed to FaceMesh. The frame is
    # downscaled to this width before inference and landmarks are mapped
    # back to full-resolution pixel coordinates (FaceMesh returns
    # normalised 0..1 coords, so this is lossless for our metrics).
    # 0 or None disables downscaling. 256 is the sweet spot on a Pi 4.
    process_width: int = 256

    # Thresholds
    ear_threshold: float = 0.22          # Eye Aspect Ratio -> fatigue
    ear_consec_frames: int = 20          # Frames below EAR -> alert
    mar_threshold: float = 0.6           # Mouth Aspect Ratio -> yawn
    mar_consec_frames: int = 15

    # Head pose distraction angles (degrees)
    yaw_threshold: float = 30.0
    pitch_threshold: float = 20.0
    distraction_consec_frames: int = 25


@dataclass
class YOLOConfig:
    # PERF: point this at an exported NCNN directory (…_ncnn_model) or an
    # .onnx file. A raw .pt drags in the whole PyTorch runtime, which is
    # the single biggest cost on a Pi 4.
    model_path: str = "weights/yolov8n_ncnn_model"
    backend: str = "auto"                # auto | ncnn | onnx | pytorch
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45
    imgsz: int = 192                     # MUST match the export size
    device: str = "cpu"
    half: bool = False                   # FP16 only on CUDA

    # PERF: frame throttling. YOLO runs once every `interval` frames.
    interval: int = 3
    adaptive_interval: bool = True       # Back off when YOLO is slow
    max_interval: int = 10
    latency_budget_ms: float = 45.0      # Per-inference target on Pi 4
    warmup_iterations: int = 2           # Avoid a 1-2 s stall on frame 1

    # PERF: run YOLO only on a crop around the driver's face instead of
    # the whole frame. Cheaper AND more accurate for small objects
    # (phone / cigarette) because they occupy far more pixels.
    roi_enabled: bool = True
    roi_scale: float = 1.6               # Face bbox expansion factor
    roi_min_size: int = 96               # Ignore absurdly small crops
    roi_fallback_full_frame: bool = False  # Skip YOLO entirely with no face

    classes_of_interest: list = field(default_factory=lambda: [
        67,  # cell phone (interior pipeline only needs this from COCO)
    ])


@dataclass
class RuntimeConfig:
    """PERF: thread budget. A Pi 4 has 4 cores that must be shared by the
    camera thread, MediaPipe, YOLO and the render loop. Letting OpenCV or
    PyTorch grab all 4 causes contention and *lowers* throughput."""
    cv_num_threads: int = 2
    torch_num_threads: int = 2
    omp_num_threads: int = 2
    ncnn_num_threads: int = 3
    profile: bool = False                # Log per-stage latency breakdown
    profile_interval_s: float = 5.0


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
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
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
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)


def apply_runtime_tuning(cfg: RuntimeConfig) -> None:
    """PERF: must be called BEFORE cv2 / torch / ultralytics do any work.

    Sets thread-count environment variables and library thread pools so the
    four Pi 4 cores are not oversubscribed.
    """
    import logging

    log = logging.getLogger("dms.runtime")

    os.environ.setdefault("OMP_NUM_THREADS", str(cfg.omp_num_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(cfg.omp_num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(cfg.omp_num_threads))
    # Silence MediaPipe / TF logging noise, which also costs a little I/O.
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    try:
        import cv2
        cv2.setNumThreads(cfg.cv_num_threads)
        log.info("OpenCV threads set to %s", cfg.cv_num_threads)
    except Exception as exc:  # pragma: no cover
        log.debug("Could not set OpenCV threads: %s", exc)

    try:
        import torch  # Only present if the pytorch backend is used
        torch.set_num_threads(cfg.torch_num_threads)
        log.info("Torch threads set to %s", cfg.torch_num_threads)
    except Exception:
        pass  # Torch is not required once you move to NCNN
