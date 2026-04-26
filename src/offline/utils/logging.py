from __future__ import annotations

import logging
from pathlib import Path

from offline.utils.io import LOGS_DIR


_LOGGING_READY = False
_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_CONFIGURED_LOGGERS: set[str] = set()


def _sanitize_logger_name(name: str) -> str:
    return name.replace("/", "_").replace(".", "_")


def get_logger(name: str) -> logging.Logger:
    global _LOGGING_READY
    if not _LOGGING_READY:
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.handlers.clear()

        formatter = logging.Formatter(_LOG_FORMAT)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

        session_handler = logging.FileHandler(LOGS_DIR / "session.log", mode="w", encoding="utf-8")
        session_handler.setFormatter(formatter)
        root_logger.addHandler(session_handler)

        _LOGGING_READY = True

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = True

    if name not in _CONFIGURED_LOGGERS:
        logger_path = LOGS_DIR / f"{_sanitize_logger_name(name)}.log"
        handler = logging.FileHandler(logger_path, mode="w", encoding="utf-8")
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        _CONFIGURED_LOGGERS.add(name)
    return logger


def format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    total_seconds = max(0, int(seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
