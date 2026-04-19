from __future__ import annotations

from pathlib import Path

from src.pipeline.interview_pipeline import analyze_interview


def build_analysis_config(config: dict) -> dict:
    return {
        "RF_MODEL_PATH": config["RF_MODEL_PATH"],
        "RF_MODEL_URL": config.get("RF_MODEL_URL") or None,
        "LANDMARKER_MODEL_PATH": config["LANDMARKER_MODEL_PATH"],
        "LANDMARKER_MODEL_URL": config.get("LANDMARKER_MODEL_URL") or None,
        "AUDIO_TMP_ROOT": config["AUDIO_TMP_ROOT"],
        "WHISPER_MODEL": config["WHISPER_MODEL"],
        "WHISPER_LANGUAGE": config["WHISPER_LANGUAGE"],
        "PIPELINE_MAX_WORKERS": int(config.get("PIPELINE_MAX_WORKERS", 2)),
        "FRAME_SKIP": int(config.get("FRAME_SKIP", 5)),
        "MAX_FRAMES": int(config.get("MAX_FRAMES", 150)),
    }


def analyze_uploaded_video(video_path: str | Path, config: dict) -> dict:
    runtime_config = build_analysis_config(config)
    return analyze_interview(
        video_path=str(video_path),
        rf_model_path=runtime_config["RF_MODEL_PATH"],
        rf_model_url=runtime_config["RF_MODEL_URL"],
        landmarker_path=runtime_config["LANDMARKER_MODEL_PATH"],
        landmarker_url=runtime_config["LANDMARKER_MODEL_URL"],
        audio_tmp_root=runtime_config["AUDIO_TMP_ROOT"],
        whisper_model=runtime_config["WHISPER_MODEL"],
        whisper_language=runtime_config["WHISPER_LANGUAGE"],
        max_workers=runtime_config["PIPELINE_MAX_WORKERS"],
        frame_skip=runtime_config["FRAME_SKIP"],
        max_frames=runtime_config["MAX_FRAMES"],
    )

