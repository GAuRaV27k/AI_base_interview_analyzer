"""
video_processor.py
------------------
Efficient frame extraction from interview video files.

Public API
----------
extract_frames(video_path, output_dir, ...) -> dict
    Save frames to disk and return metadata.

iter_frames(video_path, frame_skip, max_frames, resize)
    Yield (frame_index, bgr_frame) without touching disk.
"""

from __future__ import annotations

import os
from typing import Generator, Optional, Tuple

import cv2 as cv
import numpy as np


def extract_frames(
    video_path: str,
    output_dir: str,
    frame_skip: int = 5,
    max_frames: Optional[int] = 150,
    image_format: str = "jpg",
    resize: Optional[Tuple[int, int]] = (640, 480),
    verbose: bool = False,
) -> dict:
    """Extract frames from a video file and save them to disk.

    Parameters
    ----------
    video_path:   Path to the input video.
    output_dir:   Directory where extracted frames are saved.
    frame_skip:   Extract every Nth frame (1 = every frame).
    max_frames:   Upper limit on frames saved (None = no limit).
    image_format: Output image extension, e.g. ``'jpg'``.
    resize:       ``(width, height)`` to resize frames, or None.
    verbose:      Print progress information.

    Returns
    -------
    dict with keys: total_frames_in_video, frames_extracted, fps,
                    original_resolution, output_dir
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    if verbose:
        print(f"Video: {total_frames} frames @ {fps:.1f} fps  ({width}x{height})")

    frame_count = 0
    saved_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                if resize is not None:
                    frame = cv.resize(frame, resize)

                out_path = os.path.join(output_dir, f"frame_{saved_count:06d}.{image_format}")
                cv.imwrite(out_path, frame)
                saved_count += 1

                if verbose and saved_count % 50 == 0:
                    print(f"  Extracted {saved_count} frames…")

                if max_frames and saved_count >= max_frames:
                    break

            frame_count += 1
    finally:
        cap.release()

    if verbose:
        print(f"Done — {saved_count} frames saved to {output_dir}")

    return {
        "total_frames_in_video": total_frames,
        "frames_extracted": saved_count,
        "fps": fps,
        "original_resolution": (width, height),
        "output_dir": output_dir,
    }


def iter_frames(
    video_path: str,
    frame_skip: int = 5,
    max_frames: Optional[int] = 150,
    resize: Optional[Tuple[int, int]] = (640, 480),
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """Yield ``(saved_index, bgr_frame)`` without writing to disk.

    More memory-efficient than saving all frames first when the goal is
    immediate per-frame feature extraction.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    frame_count = 0
    yielded = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                if resize is not None:
                    frame = cv.resize(frame, resize)
                yield yielded, frame
                yielded += 1

                if max_frames and yielded >= max_frames:
                    break

            frame_count += 1
    finally:
        cap.release()
