"""
InferenceRunner - dedicated inference worker thread.

Separates heavy model inference (FaceMesh + YOLO) from the camera
capture thread and the main loop. Frames are consumed from a shared
``queue.Queue(maxsize=1)`` so the worker always processes the freshest
available frame. Results are pushed to a thread-safe results queue.

PERF CHANGES (Raspberry Pi 4)
-----------------------------
1. **ROI YOLO**: instead of detecting on the whole frame, the worker crops
   a box around the face returned by FaceMesh and runs YOLO on that. A
   1.6x face crop is ~6-10x fewer pixels than a 640x480 frame, and a phone
   held to the ear goes from ~20 px to ~90 px, which massively improves
   recall for small objects.
2. **No-face short circuit**: if there is no face, there is no driver
   behaviour to detect, so YOLO can be skipped entirely.
3. **Adaptive throttling**: the YOLO interval grows automatically when
   measured inference latency exceeds the configured budget, so the
   FaceMesh/fatigue path (the safety-critical one) keeps its frame rate
   even under thermal throttling.
4. **Per-stage profiling**: optional latency breakdown so you can see
   exactly where the milliseconds go instead of guessing.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

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
    """YOLO detections for this frame, in full-frame coordinates."""

    yolo_skipped: bool = False
    """True when YOLO was throttled this frame."""

    face_ms: float = 0.0
    yolo_ms: float = 0.0
    total_ms: float = 0.0


# ---------------------------------------------------------------------------
# Inference worker
# ---------------------------------------------------------------------------

class InferenceRunner:
    """Runs FaceMesh and YOLO in a dedicated background thread."""

    def __init__(
        self,
        config: SystemConfig,
        frame_queue: "queue.Queue[np.ndarray]",
    ):
        self._cfg = config
        self._frame_queue = frame_queue

        self._results_queue: queue.Queue[InferenceResult] = queue.Queue(maxsize=1)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._face_detector: Optional[FaceMeshDetector] = None
        self._yolo: Optional[YOLODetector] = None

        # YOLO throttling state (now config-driven)
        ycfg = config.yolo
        self._yolo_interval = max(1, int(ycfg.interval))
        self._base_interval = self._yolo_interval
        self._yolo_counter = 0
        self._last_yolo_dets: List[Detection] = []

        # Metrics
        self._processed = 0
        self._dropped_frames = 0
        self._face_ms_acc = 0.0
        self._yolo_ms_acc = 0.0
        self._yolo_runs = 0
        self._last_profile_log = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="inference-worker",
        )
        self._thread.start()
        logger.info("InferenceRunner thread started")

    def _run(self):
        try:
            self._init_models()
            self._inference_loop()
        except Exception as exc:
            logger.exception("Inference thread encountered an error: %s", exc)
        finally:
            self._release_models()
            logger.info("Inference thread exiting")

    def _init_models(self):
        logger.info("Inference thread: initialising FaceMeshDetector...")
        self._face_detector = FaceMeshDetector(self._cfg.mediapipe)

        logger.info("Inference thread: initialising YOLODetector...")
        ycfg = self._cfg.yolo
        try:
            self._yolo = YOLODetector(
                model_path=ycfg.model_path,
                conf_threshold=ycfg.conf_threshold,
                iou_threshold=ycfg.iou_threshold,
                imgsz=ycfg.imgsz,
                device=ycfg.device,
                classes_of_interest=ycfg.classes_of_interest,
                backend=ycfg.backend,
                warmup_iterations=ycfg.warmup_iterations,
            )
        except Exception as exc:
            logger.warning("YOLO failed to load: %s - running without YOLO", exc)
            self._yolo = None

    def _release_models(self):
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
        while not self._stop_event.is_set():
            try:
                frame = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            result = self._process_frame(frame)
            self._push_result(result)
            self._maybe_log_profile()

    # ------------------------------------------------------------------
    # ROI helper
    # ------------------------------------------------------------------

    def _face_roi(
        self,
        frame: np.ndarray,
        face: FaceResult,
    ) -> Optional[Tuple[np.ndarray, Tuple[int, int]]]:
        """Return ``(crop, (dx, dy))`` around the face, or None if too small.

        The crop is squared and expanded by ``roi_scale`` so that objects
        near the head (phone, cigarette, hand) stay inside it.
        """
        ycfg = self._cfg.yolo
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = face.face_bbox

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        side = int(max(x2 - x1, y2 - y1) * ycfg.roi_scale)
        if side < ycfg.roi_min_size:
            return None
        half = side // 2

        rx1 = max(cx - half, 0)
        ry1 = max(cy - half, 0)
        rx2 = min(cx + half, w)
        ry2 = min(cy + half, h)
        if rx2 - rx1 < ycfg.roi_min_size or ry2 - ry1 < ycfg.roi_min_size:
            return None

        # Slicing a numpy array is a view, not a copy - effectively free.
        return frame[ry1:ry2, rx1:rx2], (rx1, ry1)

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    def _process_frame(self, frame: np.ndarray) -> InferenceResult:
        t_start = time.perf_counter()
        face: Optional[FaceResult] = None
        yolo_skipped = True
        face_ms = 0.0
        yolo_ms = 0.0

        # ---- FaceMesh (every frame - it drives the safety events) ----
        if self._face_detector is not None:
            t0 = time.perf_counter()
            try:
                face = self._face_detector.process(frame)
            except Exception as exc:
                logger.warning("FaceMesh processing error: %s", exc)
            face_ms = (time.perf_counter() - t0) * 1000.0
            self._face_ms_acc += face_ms

        # ---- YOLO (throttled + ROI-cropped) --------------------------
        self._yolo_counter += 1
        should_run = (
            self._yolo is not None
            and self._yolo_counter % self._yolo_interval == 0
        )

        if should_run:
            ycfg = self._cfg.yolo
            target = frame
            offset = (0, 0)
            run = True

            if ycfg.roi_enabled:
                if face is not None:
                    roi = self._face_roi(frame, face)
                    if roi is not None:
                        target, offset = roi
                elif not ycfg.roi_fallback_full_frame:
                    # No face -> no driver -> nothing for YOLO to say.
                    run = False
                    self._last_yolo_dets = []

            if run:
                t0 = time.perf_counter()
                try:
                    self._last_yolo_dets = self._yolo.detect(target, offset=offset)
                    yolo_skipped = False
                except Exception as exc:
                    logger.warning("YOLO processing error: %s", exc)
                yolo_ms = (time.perf_counter() - t0) * 1000.0
                self._yolo_ms_acc += yolo_ms
                self._yolo_runs += 1
                self._adapt_interval(yolo_ms)

        self._processed += 1
        total_ms = (time.perf_counter() - t_start) * 1000.0

        return InferenceResult(
            frame=frame,
            face_result=face,
            yolo_detections=self._last_yolo_dets,
            yolo_skipped=yolo_skipped,
            face_ms=face_ms,
            yolo_ms=yolo_ms,
            total_ms=total_ms,
        )

    def _adapt_interval(self, yolo_ms: float):
        """Grow or shrink the YOLO interval to protect the FaceMesh rate.

        On a Pi 4 sustained load causes thermal throttling; this keeps the
        fatigue detector responsive instead of letting everything degrade.
        """
        ycfg = self._cfg.yolo
        if not ycfg.adaptive_interval:
            return

        budget = ycfg.latency_budget_ms
        if yolo_ms > budget * 1.5 and self._yolo_interval < ycfg.max_interval:
            self._yolo_interval += 1
            logger.info(
                "YOLO slow (%.0f ms > %.0f ms budget) - interval raised to %s",
                yolo_ms, budget, self._yolo_interval,
            )
        elif yolo_ms < budget * 0.6 and self._yolo_interval > self._base_interval:
            self._yolo_interval -= 1
            logger.info(
                "YOLO fast (%.0f ms) - interval lowered to %s",
                yolo_ms, self._yolo_interval,
            )

    def _maybe_log_profile(self):
        rcfg = self._cfg.runtime
        if not rcfg.profile:
            return
        now = time.perf_counter()
        if now - self._last_profile_log < rcfg.profile_interval_s:
            return
        self._last_profile_log = now
        n = max(self._processed, 1)
        logger.info(
            "[profile] frames=%s | facemesh avg=%.1f ms | yolo avg=%.1f ms "
            "(runs=%s, interval=%s) | backend=%s",
            self._processed,
            self._face_ms_acc / n,
            self._yolo_ms_acc / max(self._yolo_runs, 1),
            self._yolo_runs,
            self._yolo_interval,
            self._yolo.backend if self._yolo else "none",
        )

    def _push_result(self, result: InferenceResult):
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
        try:
            return self._results_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def stop(self):
        logger.info("InferenceRunner stop() requested")
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("Inference thread did not stop within 5s")
            self._thread = None

        n = max(self._processed, 1)
        logger.info(
            "InferenceRunner stopped. Processed=%s, DroppedResults=%s, "
            "FaceMeshAvg=%.1fms, YoloAvg=%.1fms",
            self._processed, self._dropped_frames,
            self._face_ms_acc / n,
            self._yolo_ms_acc / max(self._yolo_runs, 1),
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

    @property
    def yolo_interval(self) -> int:
        return self._yolo_interval
