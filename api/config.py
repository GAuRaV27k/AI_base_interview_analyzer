from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _parse_csv_set(value: str | None, default: set[str]) -> set[str]:
    if not value:
        return set(default)
    return {part.strip().lower() for part in value.split(",") if part.strip()}


class AppConfig:
    def __init__(self) -> None:
        self.BASE_DIR = Path(__file__).resolve().parent.parent
        load_dotenv(self.BASE_DIR / ".env")

        self.DEBUG = _parse_bool(os.getenv("FLASK_DEBUG"), False)
        self.SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

        self.TEMPLATE_FOLDER = self.BASE_DIR / "templates"
        self.STATIC_FOLDER = self.BASE_DIR / "static"
        self.UPLOAD_FOLDER = Path(
            os.getenv("UPLOAD_FOLDER", str(self.BASE_DIR / "uploads"))
        )
        self.LOG_DIR = Path(os.getenv("LOG_DIR", str(self.BASE_DIR / "logs")))
        self.AUDIO_TMP_ROOT = Path(
            os.getenv("AUDIO_TMP_ROOT", str(self.BASE_DIR / "data" / "processed" / "audio_tmp"))
        )

        self.MAX_UPLOAD_MB = _parse_int(os.getenv("MAX_UPLOAD_MB"), 500)
        self.MAX_CONTENT_LENGTH = self.MAX_UPLOAD_MB * 1024 * 1024
        self.PIPELINE_MAX_WORKERS = _parse_int(os.getenv("PIPELINE_MAX_WORKERS"), 2)
        self.FRAME_SKIP = _parse_int(os.getenv("FRAME_SKIP"), 5)
        self.MAX_FRAMES = _parse_int(os.getenv("MAX_FRAMES"), 150)

        self.ALLOWED_EXTENSIONS = _parse_csv_set(
            os.getenv("ALLOWED_EXTENSIONS"),
            {"mp4", "avi", "mov", "mkv", "webm"},
        )
        self.ALLOWED_MIME_TYPES = _parse_csv_set(
            os.getenv("ALLOWED_MIME_TYPES"),
            {
                "video/mp4",
                "video/x-msvideo",
                "video/quicktime",
                "video/x-matroska",
                "video/webm",
                "application/octet-stream",
            },
        )

        self.RF_MODEL_PATH = Path(
            os.getenv("RF_MODEL_PATH", str(self.BASE_DIR / "models" / "tuned_randomforest_model.joblib"))
        )
        self.RF_MODEL_URL = os.getenv("RF_MODEL_URL", "").strip()
        self.LANDMARKER_MODEL_PATH = Path(
            os.getenv("LANDMARKER_MODEL_PATH", str(self.BASE_DIR / "face_landmarker.task"))
        )
        self.LANDMARKER_MODEL_URL = os.getenv("LANDMARKER_MODEL_URL", "").strip()
        self.WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
        self.WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")
        self.MODEL_PREFLIGHT_ON_STARTUP = _parse_bool(os.getenv("MODEL_PREFLIGHT_ON_STARTUP"), True)
        self.MODEL_PREFLIGHT_STRICT = _parse_bool(os.getenv("MODEL_PREFLIGHT_STRICT"), False)
        self.ANALYSIS_ASYNC_ENABLED = _parse_bool(os.getenv("ANALYSIS_ASYNC_ENABLED"), True)
        self.ANALYSIS_QUEUE_WORKERS = _parse_int(os.getenv("ANALYSIS_QUEUE_WORKERS"), 2)
        self.JOB_RESULT_TTL_MINUTES = _parse_int(os.getenv("JOB_RESULT_TTL_MINUTES"), 120)
        self.DELETE_UPLOAD_AFTER_ANALYSIS = _parse_bool(os.getenv("DELETE_UPLOAD_AFTER_ANALYSIS"), False)

    def to_flask_config(self) -> dict:
        return {
            "DEBUG": self.DEBUG,
            "SECRET_KEY": self.SECRET_KEY,
            "UPLOAD_FOLDER": str(self.UPLOAD_FOLDER),
            "LOG_DIR": str(self.LOG_DIR),
            "MAX_UPLOAD_MB": self.MAX_UPLOAD_MB,
            "MAX_CONTENT_LENGTH": self.MAX_CONTENT_LENGTH,
            "ALLOWED_EXTENSIONS": self.ALLOWED_EXTENSIONS,
            "ALLOWED_MIME_TYPES": self.ALLOWED_MIME_TYPES,
            "RF_MODEL_PATH": str(self.RF_MODEL_PATH),
            "RF_MODEL_URL": self.RF_MODEL_URL,
            "LANDMARKER_MODEL_PATH": str(self.LANDMARKER_MODEL_PATH),
            "LANDMARKER_MODEL_URL": self.LANDMARKER_MODEL_URL,
            "AUDIO_TMP_ROOT": str(self.AUDIO_TMP_ROOT),
            "WHISPER_MODEL": self.WHISPER_MODEL,
            "WHISPER_LANGUAGE": self.WHISPER_LANGUAGE,
            "PIPELINE_MAX_WORKERS": self.PIPELINE_MAX_WORKERS,
            "FRAME_SKIP": self.FRAME_SKIP,
            "MAX_FRAMES": self.MAX_FRAMES,
            "MODEL_PREFLIGHT_ON_STARTUP": self.MODEL_PREFLIGHT_ON_STARTUP,
            "MODEL_PREFLIGHT_STRICT": self.MODEL_PREFLIGHT_STRICT,
            "ANALYSIS_ASYNC_ENABLED": self.ANALYSIS_ASYNC_ENABLED,
            "ANALYSIS_QUEUE_WORKERS": self.ANALYSIS_QUEUE_WORKERS,
            "JOB_RESULT_TTL_MINUTES": self.JOB_RESULT_TTL_MINUTES,
            "DELETE_UPLOAD_AFTER_ANALYSIS": self.DELETE_UPLOAD_AFTER_ANALYSIS,
            "JSON_SORT_KEYS": False,
        }

