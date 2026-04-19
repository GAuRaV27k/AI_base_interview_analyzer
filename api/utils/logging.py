from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_configured = False
_NAMESPACE = "ai_interview_analyzer"


def configure_logging(log_dir: str | Path, level: str = "INFO") -> None:
    global _configured
    if _configured:
        return

    level_value = getattr(logging, level.upper(), logging.INFO)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level_value)
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    app_logger = logging.getLogger(_NAMESPACE)
    app_logger.setLevel(logging.DEBUG)
    app_logger.handlers.clear()
    app_logger.addHandler(console_handler)
    app_logger.addHandler(file_handler)
    app_logger.propagate = False

    _configured = True


def get_logger(name: str) -> logging.Logger:
    if not _configured:
        configure_logging(log_dir=Path("logs"), level="INFO")
    suffix = name if name.startswith(_NAMESPACE) else f"{_NAMESPACE}.{name}"
    return logging.getLogger(suffix)

