"""
InferenceRunner — dedicated inference worker thread.

Separates heavy model inference (FaceMesh + YOLO) from the camera
capture thread and the main loop.  Frames are consumed from a shared
``queue.Queue(maxsize=1)`` so the worker always processes the freshest
available frame.  Results are pushed to a thread-safe results queue.

Design goals for Raspberry Pi:
  * Low latency — never process stale frames.
  * No backlog — single-frame queue drops old frames automatically.
  * Stable shutdown — Event signals stop; models are released from the
    inference thread.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from src.config.settings import SystemConfig
from src.detectors.face_mesh_detector import FaceMeshDetector, FaceResult
from src.detectors.yolo_detector import YOLODetector, Detection

logger = logging.getLogger("dms.inference")


# ---------------------------------------------------------------------------
# Result container (immutable, safe to pass across threads)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InferenceResult:
    """Output of a single inference cycle."""

    frame: np.ndarray
    """The *original* frame that was processed (not annotated)."""

    face_result: Optional[FaceResult] = None
    """FaceMesh output, or ``None`` if no face detected."""

    yolo_detections: List[Detection] = field(default_factory=list)
    """YOLO detections for this frame."""

    yolo_skipped: bool = False
    """True when YOLO was throttled this frame."""


# ---------------------------------------------------------------------------
# Inference worker
# ---------------------------------------------------------------------------

class InferenceRunner:
    """
    Runs FaceMesh and YOLO in a dedicated background thread.

    Usage::

        runner = InferenceRunner(config, camera_queue)
        runner.start()

        # In the main loop:
        result = runner.get_result(timeout=0.05)
        if result:
            ...

        runner.stop()
    """

    def __init__(
        self,
        config: SystemConfig,
        frame_queue: "queue.Queue[np.ndarray]",
    ):
        self._cfg = config
        self._frame_queue = frame_queue

        # Results produced by the inference thread
        self._results_queue: queue.Queue[InferenceResult] = queue.Queue(maxsize=1)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Detectors (initialised inside the worker thread)
        self._face_detector: Optional[FaceMeshDetector] = None
        self._yolo: Optional[YOLODetector] = None

        # YOLO throttling state
        self._yolo_interval = 3
        self._yolo_counter = 0
        self._last_yolo_dets: List[Detection] = []

        # Metrics
        self._processed = 0
        self._dropped_frames = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Start the inference worker thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="inference-worker",
        )
        self._thread.start()
        logger.info("InferenceRunner thread started")

    def _run(self):
        """Main loop: initialise models, then consume frames."""
        try:
            self._init_models()
            self._inference_loop()
        except Exception as exc:
            logger.exception("Inference thread encountered an error: %s", exc)
        finally:
            self._release_models()
            logger.info("Inference thread exiting")

    def _init_models(self):
        """Create detectors inside the worker thread."""
        logger.info("Inference thread: initialising FaceMeshDetector...")
        self._face_detector = FaceMeshDetector(self._cfg.mediapipe)

        logger.info("Inference thread: initialising YOLODetector...")
        try:
            self._yolo = YOLODetector(
                model_path=self._cfg.yolo.model_path,
                conf_threshold=self._cfg.yolo.conf_threshold,
                iou_threshold=self._cfg.yolo.iou_threshold,
                imgsz=self._cfg.yolo.imgsz,
                device=self._cfg.yolo.device,
                classes_of_interest=self._cfg.yolo.classes_of_interest,
            )
        except Exception as exc:
            logger.warning(f"YOLO failed to load: {exc} — running without YOLO")
            self._yolo = None

    def _release_models(self):
        """Release MediaPipe / detector resources."""
        if self._face_detector is not None:
            try:
                self._face_detector.release()
            except Exception as exc:
                logger.warning("Error releasing FaceMeshDetector: %s", exc)
            self._face_detector = None

        self._yolo = None
        logger.info("Inference models released")

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _inference_loop(self):
        """Consume frames and run inference until stopped."""
        while not self._stop_event.is_set():
            try:
                # Block briefly to avoid burning CPU when idle.
                frame = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            result = self._process_frame(frame)
            self._push_result(result)

    def _process_frame(self, frame: np.ndarray) -> InferenceResult:
        """Run FaceMesh + throttled YOLO on a single frame."""
        face: Optional[FaceResult] = None
        yolo_skipped = False

        # FaceMesh
        if self._face_detector is not None:
            try:
                face = self._face_detector.process(frame)
            except Exception as exc:
                logger.warning("FaceMesh processing error: %s", exc)

        # YOLO (throttled)
        self._yolo_counter += 1
        if self._yolo and (self._yolo_counter % self._yolo_interval == 0):
            try:
                self._last_yolo_dets = self._yolo.detect(frame)
            except Exception as exc:
                logger.warning("YOLO processing error: %s", exc)
        else:
            yolo_skipped = True

        self._processed += 1
        return InferenceResult(
            frame=frame,
            face_result=face,
            yolo_detections=self._last_yolo_dets.copy(),
            yolo_skipped=yolo_skipped,
        )

    def _push_result(self, result: InferenceResult):
        """Push result to the results queue, discarding the old one if full."""
        if self._results_queue.full():
            try:
                self._results_queue.get_nowait()
                self._dropped_frames += 1
            except queue.Empty:
                pass
        self._results_queue.put(result)

    # ------------------------------------------------------------------
    # Consumer API
    # ------------------------------------------------------------------

    def get_result(self, timeout: Optional[float] = 0.05) -> Optional[InferenceResult]:
        """
        Retrieve the latest inference result.

        Args:
            timeout: Max seconds to wait for a result.  Use ``None``
                to block indefinitely.

        Returns:
            :class:`InferenceResult` or ``None`` if nothing is available.
        """
        try:
            return self._results_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def stop(self):
        """Signal the inference thread to stop and wait for it to exit."""
        logger.info("InferenceRunner stop() requested")
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("Inference thread did not stop within 5s")
            self._thread = None

        logger.info(
            "InferenceRunner stopped. Processed=%s, DroppedResults=%s",
            self._processed,
            self._dropped_frames,
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def processed_count(self) -> int:
        return self._processed

    @property
    def dropped_results_count(self) -> int:
        return self._dropped_frames
