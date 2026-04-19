from __future__ import annotations

from pathlib import Path

from src.pipeline.interview_pipeline import analyze_interview


def analyze_uploaded_video(video_path: str | Path, config: dict) -> dict:
    return analyze_interview(
        video_path=str(video_path),
        rf_model_path=config["RF_MODEL_PATH"],
        landmarker_path=config["LANDMARKER_MODEL_PATH"],
        audio_tmp_root=config["AUDIO_TMP_ROOT"],
        whisper_model=config["WHISPER_MODEL"],
        whisper_language=config["WHISPER_LANGUAGE"],
        max_workers=int(config.get("PIPELINE_MAX_WORKERS", 2)),
    )

