"""
interview_pipeline.py
---------------------
Full ML pipeline for the AI Interview Analyzer.

Flow
----
Input video
 ├── [Thread A] video_processor  →  iter_frames
 │               facial_features →  MediaPipe landmarks  →  10 features / frame
 │               models/tuned_randomforest_model.joblib  →  emotion per frame
 │               Aggregate: dominant emotion, eye-contact %, confidence score
 │
 └── [Thread B] audio_pipeline   →  extract_audio (moviepy)
                                 →  transcribe_audio (Whisper)
                                 →  extract_audio_features (librosa)
                                 →  speech_rate (WPM)

Both threads run in parallel via concurrent.futures.ThreadPoolExecutor.
Results are merged and returned as AnalysisResult to the Flask API.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import concurrent.futures
from urllib.request import urlopen
from typing import TypedDict

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level model cache (loaded once per process)
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict[tuple[str, str], tuple[object | None, object | None]] = {}
_MODEL_LOCK = threading.Lock()


def _download_if_missing(
    local_path: str,
    remote_url: str | None,
    label: str,
    url_env_name: str,
) -> None:
    if os.path.isfile(local_path):
        return

    if not remote_url:
        raise FileNotFoundError(
            f"{label} not found: {local_path}. Set {url_env_name} to download it in production."
        )

    parent_dir = os.path.dirname(local_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    log.info("%s missing, downloading from %s", label, remote_url)

    try:
        with urlopen(remote_url, timeout=600) as response, open(local_path, "wb") as target:
            shutil.copyfileobj(response, target)
    except Exception as exc:
        raise RuntimeError(f"Failed to download {label} from {remote_url}: {exc}") from exc

    if not os.path.isfile(local_path):
        raise FileNotFoundError(f"{label} download finished but file is missing: {local_path}")


def _maybe_prepare_asset(
    local_path: str,
    remote_url: str | None,
    label: str,
    url_env_name: str,
) -> tuple[bool, str]:
    abs_path = os.path.abspath(local_path)
    if os.path.isfile(abs_path):
        return True, ""
    if remote_url:
        try:
            _download_if_missing(abs_path, remote_url, label, url_env_name)
            return True, ""
        except Exception as exc:
            return False, f"{label} unavailable (download failed): {exc}"
    return False, f"{label} unavailable: {abs_path}"


def _get_models(
    rf_model_path: str,
    landmarker_path: str,
    rf_model_url: str | None = None,
    landmarker_url: str | None = None,
):
    cache_key = (os.path.abspath(rf_model_path), os.path.abspath(landmarker_path))
    with _MODEL_LOCK:
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]

        log.info("Loading ML models for the first time...")
        from src.feature_engineering.facial_features import load_landmarker, load_rf_model

        rf_ok, rf_msg = _maybe_prepare_asset(
            cache_key[0], rf_model_url, "RandomForest model", "RF_MODEL_URL"
        )
        landmarker_ok, landmarker_msg = _maybe_prepare_asset(
            cache_key[1], landmarker_url, "MediaPipe face landmarker", "LANDMARKER_MODEL_URL"
        )

        rf_model = load_rf_model(cache_key[0]) if rf_ok else None
        landmarker = load_landmarker(cache_key[1]) if landmarker_ok else None

        if rf_msg:
            log.warning("%s", rf_msg)
        if landmarker_msg:
            log.warning("%s", landmarker_msg)
        if not rf_ok and not landmarker_ok:
            log.warning(
                "Both facial models are unavailable; continuing with audio-only fallback."
            )

        _MODEL_CACHE[cache_key] = (rf_model, landmarker)
        log.info(
            "Model load result: rf_model=%s, landmarker=%s",
            "ready" if rf_model is not None else "missing",
            "ready" if landmarker is not None else "missing",
        )
        return rf_model, landmarker


def ensure_model_assets(
    rf_model_path: str,
    landmarker_path: str,
    rf_model_url: str | None = None,
    landmarker_url: str | None = None,
) -> None:
    """Ensure model files exist locally, downloading them if URLs are provided."""
    rf_abs = os.path.abspath(rf_model_path)
    landmarker_abs = os.path.abspath(landmarker_path)
    _download_if_missing(rf_abs, rf_model_url, "RandomForest model", "RF_MODEL_URL")
    _download_if_missing(
        landmarker_abs,
        landmarker_url,
        "MediaPipe face landmarker",
        "LANDMARKER_MODEL_URL",
    )


# ---------------------------------------------------------------------------
# Extended result type
# ---------------------------------------------------------------------------

class AnalysisResult(TypedDict):
    # Core metrics (used by the original result.html)
    confidence_score:      float   # 0–100
    speech_rate:           float   # words per minute
    eye_contact_score:     float   # 0–100
    emotion_prediction:    str     # dominant display-name emotion
    final_interview_score: float   # 0–100
    # Extended metrics
    transcript:            str
    word_count:            int
    emotion_breakdown:     dict    # {emotion_name: percentage}
    emotion_timeline:      list    # per-frame emotion class indices (0-6)
    frames_analyzed:       int
    faces_detected:        int
    audio_energy:          float   # voice_energy_mean from librosa
    audio_duration:        float   # seconds
    audio_error:           str     # non-empty if audio pipeline failed


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def analyze_interview(
    video_path: str,
    rf_model_path: str | None = None,
    rf_model_url: str | None = None,
    landmarker_path: str | None = None,
    landmarker_url: str | None = None,
    audio_tmp_root: str | None = None,
    whisper_model: str = "base",
    whisper_language: str | None = "en",
    max_workers: int = 2,
    frame_skip: int = 5,
    max_frames: int = 150,
) -> AnalysisResult:
    """Run the full interview analysis pipeline on *video_path*.

    Parameters
    ----------
    video_path:
        Absolute or relative path to the uploaded video file.

    Returns
    -------
    AnalysisResult
        Dictionary containing all metrics.

    Raises
    ------
    FileNotFoundError
        If *video_path* does not exist.
    RuntimeError
        If any pipeline stage fails unexpectedly.
    """
    log.info("Starting interview analysis for: %s", os.path.basename(video_path))

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    rf_model_path = rf_model_path or os.path.join(_PROJECT_ROOT, "models", "tuned_randomforest_model.joblib")
    landmarker_path = landmarker_path or os.path.join(_PROJECT_ROOT, "face_landmarker.task")
    audio_tmp_root = audio_tmp_root or os.path.join(_PROJECT_ROOT, "data", "processed", "audio_tmp")

    rf_model, landmarker = _get_models(
        rf_model_path=rf_model_path,
        landmarker_path=landmarker_path,
        rf_model_url=rf_model_url,
        landmarker_url=landmarker_url,
    )

    os.makedirs(audio_tmp_root, exist_ok=True)
    audio_tmp = tempfile.mkdtemp(
        prefix=f"{os.path.splitext(os.path.basename(video_path))[0]}_",
        dir=audio_tmp_root,
    )

    # -----------------------------------------------------------------------
    # Run video and audio pipelines in parallel
    # -----------------------------------------------------------------------
    log.info("Launching parallel video + audio pipelines")
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
            video_future = pool.submit(
                _run_video_pipeline,
                video_path,
                rf_model,
                landmarker,
                frame_skip,
                max_frames,
            )
            audio_future = pool.submit(
                _run_audio_pipeline_safe,
                video_path,
                audio_tmp,
                whisper_model,
                whisper_language,
            )

            video_result = video_future.result()
            audio_result = audio_future.result()
    except Exception as exc:
        log.error("Video pipeline failed: %s", exc, exc_info=True)
        raise RuntimeError(f"Video analysis failed: {exc}") from exc
    finally:
        shutil.rmtree(audio_tmp, ignore_errors=True)

    # -----------------------------------------------------------------------
    # Merge & compute final score
    # -----------------------------------------------------------------------
    log.info("Aggregating results — frames_analyzed=%d  faces_detected=%d",
             video_result.get("frames_analyzed", 0),
             video_result.get("faces_detected", 0))

    confidence_score   = video_result["confidence_score"]
    eye_contact_score  = video_result["eye_contact_score"]
    emotion_prediction = video_result["emotion_prediction"]

    # Blend audio energy into confidence (40 % weight)
    audio_energy = audio_result["audio_features"].get("voice_energy_mean", 0.0)
    energy_score = min(float(audio_energy) * 800.0, 100.0)
    confidence_score = round(0.60 * confidence_score + 0.40 * energy_score, 1)
    confidence_score = float(min(max(confidence_score, 0.0), 100.0))

    speech_rate = audio_result.get("speech_rate_wpm", 0.0)

    if audio_result.get("error"):
        log.warning("Audio pipeline warning: %s", audio_result["error"])
    else:
        log.info("Audio pipeline complete — speech_rate=%.1f wpm  words=%d",
                 speech_rate, audio_result.get("word_count", 0))

    log.info("Running emotion prediction — dominant=%s", emotion_prediction)

    try:
        final_score = _compute_final_score(
            confidence_score, speech_rate, eye_contact_score, emotion_prediction
        )
    except Exception as exc:
        log.error("Score calculation failed: %s", exc)
        raise RuntimeError(f"Score calculation failed: {exc}") from exc

    log.info("Final interview score calculated: %.1f  (confidence=%.1f  eye_contact=%.1f  wpm=%.1f  emotion=%s)",
             final_score, confidence_score, eye_contact_score, speech_rate, emotion_prediction)

    return AnalysisResult(
        confidence_score      = round(confidence_score, 1),
        speech_rate           = round(speech_rate, 1),
        eye_contact_score     = round(eye_contact_score, 1),
        emotion_prediction    = emotion_prediction,
        final_interview_score = round(final_score, 1),
        transcript            = audio_result.get("transcript", ""),
        word_count            = audio_result.get("word_count", 0),
        emotion_breakdown     = video_result.get("emotion_breakdown", {}),
        emotion_timeline      = video_result.get("emotion_timeline", []),
        frames_analyzed       = video_result.get("frames_analyzed", 0),
        faces_detected        = video_result.get("faces_detected", 0),
        audio_energy          = round(audio_energy, 5),
        audio_duration        = round(
            audio_result.get("audio_features", {}).get("audio_duration_seconds", 0.0), 1
        ),
        audio_error           = audio_result.get("error", ""),
    )


# ---------------------------------------------------------------------------
# Private pipeline stages
# ---------------------------------------------------------------------------

def _run_video_pipeline(
    video_path: str,
    rf_model,
    landmarker,
    frame_skip: int,
    max_frames: int,
) -> dict:
    """Extract frames → MediaPipe features → RF emotion predictions."""
    log.info("Frame extraction started")
    try:
        from src.feature_engineering.facial_features import run_video_analysis
        result = run_video_analysis(
            video_path = video_path,
            rf_model   = rf_model,
            landmarker = landmarker,
            frame_skip = frame_skip,
            max_frames = max_frames,
        )
        log.info("Emotion prediction complete — dominant=%s  faces=%d/%d",
                 result.get("emotion_prediction"), result.get("faces_detected"),
                 result.get("frames_analyzed"))
        return result
    except Exception as exc:
        log.error("Video pipeline stage failed: %s", exc, exc_info=True)
        raise RuntimeError(f"Video feature extraction failed: {exc}") from exc


def _run_audio_pipeline_safe(
    video_path: str,
    output_dir: str,
    whisper_model: str,
    whisper_language: str | None,
) -> dict:
    """Run the audio pipeline, capturing errors gracefully.

    Always returns a dict with the expected keys; populates ``'error'`` on
    failure so the caller can surface a warning without crashing.
    """
    _empty: dict = {
        "transcript":      "",
        "word_count":      0,
        "speech_rate_wpm": 0.0,
        "audio_features":  {"voice_energy_mean": 0.0, "audio_duration_seconds": 0.0},
        "error":           "",
    }
    log.info("Audio extraction started")
    try:
        from src.audio_processing.audio_pipeline import run_audio_pipeline
        result = run_audio_pipeline(
            video_path=video_path,
            output_dir=output_dir,
            whisper_model=whisper_model,
            language=whisper_language,
        )
        result["error"] = ""
        log.info("Audio extraction complete — duration=%.1fs",
                 result.get("audio_features", {}).get("audio_duration_seconds", 0))
        return result
    except Exception as exc:
        log.warning("Audio pipeline failed (non-fatal): %s", exc)
        _empty["error"] = str(exc)
        return _empty


def _compute_final_score(
    confidence: float,
    speech_rate: float,
    eye_contact: float,
    emotion: str,
) -> float:
    """Weighted combination of sub-scores → single interview score 0-100."""
    from src.feature_engineering.facial_features import EMOTION_SCORE_BONUS

    optimal_wpm    = 145.0
    speech_penalty = (abs(speech_rate - optimal_wpm) / optimal_wpm * 100
                      if speech_rate > 0 else 50.0)
    speech_score   = max(0.0, 100.0 - speech_penalty)

    emotion_bonus  = EMOTION_SCORE_BONUS.get(emotion, 0.0)
    emotion_score  = (emotion_bonus / 5.0) * 100.0

    score = (
        0.35 * confidence
        + 0.30 * eye_contact
        + 0.25 * speech_score
        + 0.10 * emotion_score
    )
    return float(min(max(score, 0.0), 100.0))
