"""
Real-time performance metrics: FPS, latency, CPU, RAM.
Lightweight — safe to run every frame on Pi.
"""

import time
import collections
import psutil
import os
from dataclasses import dataclass, field
from typing import Deque


@dataclass
class FrameMetrics:
    fps: float = 0.0
    latency_ms: float = 0.0
    cpu_percent: float = 0.0
    ram_mb: float = 0.0


class PerformanceMonitor:
    def __init__(self, window: int = 30):
        self._timestamps: Deque[float] = collections.deque(maxlen=window)
        self._latencies: Deque[float] = collections.deque(maxlen=window)
        self._process = psutil.Process(os.getpid())
        self._frame_start: float = 0.0
        self._cpu_sample_interval = 1.0  # seconds
        self._last_cpu_time = 0.0
        self._cached_cpu = 0.0

    def frame_start(self):
        self._frame_start = time.perf_counter()

    def frame_end(self) -> FrameMetrics:
        now = time.perf_counter()
        elapsed = now - self._frame_start
        self._timestamps.append(now)
        self._latencies.append(elapsed * 1000)

        # FPS over rolling window
        if len(self._timestamps) >= 2:
            span = self._timestamps[-1] - self._timestamps[0]
            fps = (len(self._timestamps) - 1) / max(span, 1e-6)
        else:
            fps = 0.0

        # CPU — sampled at interval to avoid overhead
        if now - self._last_cpu_time > self._cpu_sample_interval:
            self._cached_cpu = self._process.cpu_percent(interval=None)
            self._last_cpu_time = now

        ram = self._process.memory_info().rss / (1024 * 1024)

        return FrameMetrics(
            fps=round(fps, 1),
            latency_ms=round(sum(self._latencies) / len(self._latencies), 1),
            cpu_percent=round(self._cached_cpu, 1),
            ram_mb=round(ram, 1),
        )
