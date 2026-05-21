"""
AlertManager — receives DmsEvent objects and dispatches them to:
  - Console (always)
  - OpenCV overlay (optional)
  - Sound (optional, platform-dependent)

Uses a per-event-type cooldown to prevent alert spam.
"""

from __future__ import annotations

import logging
import time
from typing import List, Dict
from collections import deque

from src.alerts.event_types import DmsEvent, EventType, Severity
from src.config.settings import AlertConfig

logger = logging.getLogger("dms.alerts")

_SEVERITY_COLOR = {
    Severity.INFO: "\033[36m",      # Cyan
    Severity.WARNING: "\033[33m",   # Yellow
    Severity.CRITICAL: "\033[31m",  # Red
}
_RESET = "\033[0m"


class AlertManager:
    def __init__(self, cfg: AlertConfig):
        self._cfg = cfg
        self._cooldowns: Dict[EventType, float] = {}
        self._active_events: deque = deque(maxlen=8)  # For overlay
        self._sound_available = False

        if cfg.sound_enabled:
            self._init_sound()

    def _init_sound(self):
        try:
            import pygame  # type: ignore
            pygame.mixer.init()
            pygame.mixer.music.load(self._cfg.sound_file)
            self._sound_available = True
            logger.info("Sound alerts enabled")
        except Exception as e:
            logger.warning(f"Sound unavailable: {e}")

    def _is_cooling_down(self, event_type: EventType) -> bool:
        last = self._cooldowns.get(event_type, 0.0)
        return time.time() - last < self._cfg.cooldown_seconds

    def dispatch(self, event: DmsEvent):
        if self._is_cooling_down(event.event_type):
            return

        self._cooldowns[event.event_type] = time.time()
        self._active_events.append(event)

        if self._cfg.console_enabled:
            self._console_alert(event)

        if self._cfg.sound_enabled and self._sound_available:
            self._sound_alert(event)

    def _console_alert(self, event: DmsEvent):
        color = _SEVERITY_COLOR.get(event.severity, "")
        ts = time.strftime("%H:%M:%S", time.localtime(event.timestamp))
        print(
            f"{color}[{ts}] [{event.severity.value}] "
            f"{event.event_type.value}: {event.message}{_RESET}"
        )

    def _sound_alert(self, event: DmsEvent):
        try:
            import pygame
            pygame.mixer.music.play()
        except Exception:
            pass

    def get_active_event_labels(self) -> List[str]:
        """Return event label strings for overlay rendering."""
        now = time.time()
        labels = []
        for evt in reversed(self._active_events):
            if now - evt.timestamp < self._cfg.cooldown_seconds * 2:
                labels.append(evt.event_type.value)
        return labels[:5]  # Max 5 on screen
