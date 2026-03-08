"""
logger.py
---------
Centralised logging configuration for the AI Interview Analyzer.

Usage
-----
    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Pipeline started")

Outputs
-------
* Console  – INFO and above, human-readable format.
* logs/app.log – DEBUG and above, timestamped, auto-rotates at 5 MB (3 backups).
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_LOGS_DIR     = os.path.join(_PROJECT_ROOT, "logs")
_LOG_FILE     = os.path.join(_LOGS_DIR, "app.log")

# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------
_CONSOLE_FMT = logging.Formatter(
    fmt="[%(asctime)s] %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
_FILE_FMT = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)-40s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger configured with console + rotating-file handlers.

    Calling ``get_logger`` multiple times with the same *name* is safe and
    returns the same logger instance without adding duplicate handlers.

    Parameters
    ----------
    name:
        Logger name — pass ``__name__`` for module-level loggers.
    """
    logger = logging.getLogger(name)

    # Prevent adding handlers more than once (e.g. during Flask debug reloads)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # ── Console handler ────────────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(_CONSOLE_FMT)

    # ── Rotating file handler ──────────────────────────────────────────────
    os.makedirs(_LOGS_DIR, exist_ok=True)
    file_handler = RotatingFileHandler(
        _LOG_FILE,
        maxBytes=5 * 1024 * 1024,   # 5 MB per file
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_FILE_FMT)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Prevent log records from propagating to the root logger to avoid
    # duplicate output when the root logger also has handlers configured.
    logger.propagate = False

    return logger
