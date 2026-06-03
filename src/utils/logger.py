"""Centralized console, rotating file, and browser logging setup."""

from __future__ import annotations

from collections import deque
import logging
from logging.handlers import RotatingFileHandler
import sys
from pathlib import Path
from threading import Lock


_DEFAULT_LOG_PATH = "logs/dms.log"
_ACTIVE_LOG_PATH = Path(_DEFAULT_LOG_PATH)
_RECENT_LOGS = deque(maxlen=250)
_RECENT_LOGS_LOCK = Lock()


class _RecentLogHandler(logging.Handler):
    """Keep recent formatted records in memory for the browser dashboard."""

    def emit(self, record: logging.LogRecord):
        try:
            message = self.format(record)
        except Exception:
            self.handleError(record)
            return

        with _RECENT_LOGS_LOCK:
            _RECENT_LOGS.append(message)


def get_recent_logs(limit: int = 120) -> list[str]:
    """Return the newest application log lines for the web dashboard."""
    limit = max(1, min(limit, _RECENT_LOGS.maxlen))
    with _RECENT_LOGS_LOCK:
        return list(_RECENT_LOGS)[-limit:]


def get_log_path() -> str:
    """Return the active application log file path."""
    return str(_ACTIVE_LOG_PATH)


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_path: str = _DEFAULT_LOG_PATH,
) -> logging.Logger:
    global _ACTIVE_LOG_PATH

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        log_file = Path(log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        _ACTIVE_LOG_PATH = log_file.resolve()

        console_format = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S"
        )
        file_format = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(console_format)

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(file_format)

        recent_handler = _RecentLogHandler()
        recent_handler.setLevel(level)
        recent_handler.setFormatter(console_format)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        logger.addHandler(recent_handler)
        logger.info("Logging to %s", log_file.resolve())

    return logger
