"""
audio_pipeline.py
-----------------
End-to-end audio analysis: video → WAV → transcript → features → speech rate.

All logic is extracted from src/model_training/audio_extraction.ipynb so it
can be imported as a regular Python module.

Public API
----------
run_audio_pipeline(video_path, output_dir, ...) -> dict
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# FFmpeg bootstrap (ensures Whisper can locate the binary)
# ---------------------------------------------------------------------------

def _ensure_ffmpeg(output_dir: str) -> None:
    """Add a bundled ffmpeg binary to PATH so Whisper can find it."""
    try:
        import imageio_ffmpeg  # type: ignore
        ffmpeg_src = imageio_ffmpeg.get_ffmpeg_exe()
        shim_dir = Path(output_dir)
        shim_dir.mkdir(parents=True, exist_ok=True)
        ffmpeg_dst = shim_dir / "ffmpeg.exe"
        if not ffmpeg_dst.exists():
            shutil.copy2(ffmpeg_src, str(ffmpeg_dst))
        path_env = os.environ.get("PATH", "")
        if str(shim_dir) not in path_env:
            os.environ["PATH"] = str(shim_dir) + os.pathsep + path_env
    except Exception:
        # imageio_ffmpeg not installed — hope ffmpeg is already on PATH
        pass


# ---------------------------------------------------------------------------
# Step 1 — extract WAV from video
# ---------------------------------------------------------------------------

def extract_audio(video_path: str, output_dir: str, sample_rate: int = 16_000,
                  overwrite: bool = True) -> str:
    """Extract a mono 16-kHz WAV file from *video_path*.

    Returns the path to the WAV file.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / f"{video_path.stem}.wav"

    if audio_path.exists() and not overwrite:
        return str(audio_path)

    # Try moviepy v2 API first, then fall back to v1 API
    try:
        from moviepy import VideoFileClip  # type: ignore  (moviepy v2)
    except ImportError:
        from moviepy.editor import VideoFileClip  # type: ignore  (moviepy v1)

    with VideoFileClip(str(video_path)) as video:
        if video.audio is None:
            raise ValueError(f"No audio track in: {video_path}")
        video.audio.write_audiofile(
            str(audio_path),
            fps=sample_rate,
            codec="pcm_s16le",
            ffmpeg_params=["-ac", "1"],
            logger=None,
        )

    return str(audio_path)


# ---------------------------------------------------------------------------
# Step 2 — Whisper transcription
# ---------------------------------------------------------------------------

_WHISPER_CACHE: dict = {}


def _get_whisper_model(model_name: str = "base"):
    """Load and cache a Whisper model."""
    if model_name not in _WHISPER_CACHE:
        import whisper  # type: ignore
        _WHISPER_CACHE[model_name] = whisper.load_model(model_name)
    return _WHISPER_CACHE[model_name]


def transcribe_audio(audio_path: str, model_name: str = "base",
                     language: Optional[str] = "en") -> dict:
    """Transcribe *audio_path* with Whisper and return transcript + metadata."""
    model = _get_whisper_model(model_name)
    result = model.transcribe(audio_path, language=language, fp16=False)
    return {
        "transcript": result.get("text", "").strip(),
        "segments": result.get("segments", []),
        "language": result.get("language", language),
    }


# ---------------------------------------------------------------------------
# Step 3 — librosa audio features
# ---------------------------------------------------------------------------

def extract_audio_features(audio_path: str, sample_rate: int = 16_000,
                            n_mfcc: int = 13) -> dict:
    """Extract time/frequency-domain features from a WAV file using librosa."""
    import librosa  # type: ignore
    import numpy as np

    y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    if y.size == 0:
        raise ValueError(f"Empty audio file: {audio_path}")

    duration = float(librosa.get_duration(y=y, sr=sr))
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])

    features: dict = {
        "audio_duration_seconds": duration,
        "sample_rate": int(sr),
        "voice_energy_mean": float(np.mean(rms)),
        "voice_energy_std": float(np.std(rms)),
        "zero_crossing_rate_mean": float(np.mean(zcr)),
        "spectral_centroid_mean": float(np.mean(centroid)),
        "spectral_bandwidth_mean": float(np.mean(bandwidth)),
        "tempo_bpm": tempo,
    }
    for i in range(n_mfcc):
        features[f"mfcc_{i + 1}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{i + 1}_std"] = float(np.std(mfcc[i]))

    return features


# ---------------------------------------------------------------------------
# Step 4 — speech rate
# ---------------------------------------------------------------------------

def compute_speech_rate(transcript: str, duration_seconds: float) -> dict:
    """Return word count and words-per-minute from transcript + duration."""
    words = re.findall(r"[A-Za-z0-9']+", transcript or "")
    word_count = len(words)
    wpm = float((word_count / duration_seconds) * 60.0) if duration_seconds > 0 else 0.0
    return {"word_count": word_count, "words_per_minute": wpm}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_audio_pipeline(
    video_path: str,
    output_dir: str,
    whisper_model: str = "base",
    language: Optional[str] = "en",
) -> dict:
    """Full audio pipeline: video → audio → transcript → features → rate.

    Parameters
    ----------
    video_path:    Path to the uploaded video file.
    output_dir:    Working directory for intermediate WAV files.
    whisper_model: Whisper model size (``'tiny'``, ``'base'``, ``'small'``, …).
    language:      ISO language code passed to Whisper (``None`` = auto-detect).

    Returns
    -------
    dict with keys:
        audio_path, transcript, language, word_count,
        speech_rate_wpm, audio_features, segments
    """
    _ensure_ffmpeg(output_dir)

    audio_path = extract_audio(video_path, output_dir)
    transcript_result = transcribe_audio(audio_path, model_name=whisper_model,
                                         language=language)
    audio_features = extract_audio_features(audio_path)
    rate = compute_speech_rate(
        transcript_result["transcript"],
        audio_features["audio_duration_seconds"],
    )

    return {
        "audio_path": audio_path,
        "transcript": transcript_result["transcript"],
        "language": transcript_result["language"],
        "word_count": rate["word_count"],
        "speech_rate_wpm": rate["words_per_minute"],
        "audio_features": audio_features,
        "segments": transcript_result["segments"],
    }


def save_result(result: dict, output_dir: str,
                filename: str = "audio_pipeline_result.json") -> str:
    """Persist pipeline output as JSON for debugging / reproducibility."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    with open(path, "w", encoding="utf-8") as f:
        # segments can contain non-serialisable numpy floats
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    return str(path)
