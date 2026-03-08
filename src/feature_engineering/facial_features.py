"""
facial_features.py
------------------
Per-frame facial feature extraction (MediaPipe Face Landmarker) and
emotion classification using the pre-trained RandomForest model.

Feature vector (10 values, same order as training CSV):
    avg_ear, left_ear, right_ear, blink_flag,
    smile_ratio, eye_sync,
    head_pitch, head_yaw, head_roll,
    eye_gaze_direction

Emotion label map (LabelEncoder alphabetical order):
    0=angry  1=disgust  2=fear  3=happy  4=neutral  5=sad  6=surprise

Public API
----------
load_landmarker(model_path) -> FaceLandmarker
load_rf_model(model_path)   -> RandomForestClassifier
extract_features_from_frame(frame_bgr, landmarker) -> dict | None
run_video_analysis(video_path, rf_model, landmarker, ...) -> dict
"""

from __future__ import annotations

import math
import os
from typing import Optional

import cv2 as cv
import mediapipe as mp
import numpy as np
from joblib import load as joblib_load

# ---------------------------------------------------------------------------
# Emotion label map (LabelEncoder sorts alphabetically)
# ---------------------------------------------------------------------------
EMOTION_LABELS: dict[int, str] = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised",
}

# Interview-friendly display names
EMOTION_DISPLAY: dict[str, str] = {
    "Angry":     "Frustrated",
    "Disgust":   "Displeased",
    "Fear":      "Nervous",
    "Happy":     "Enthusiastic",
    "Neutral":   "Calm",
    "Sad":       "Disengaged",
    "Surprised": "Surprised",
}

# Interview score bonus per dominant emotion (used in final score)
EMOTION_SCORE_BONUS: dict[str, float] = {
    "Enthusiastic": 5.0,
    "Calm":         4.0,
    "Surprised":    3.0,
    "Disengaged":   1.0,
    "Nervous":      0.0,
    "Frustrated":   0.0,
    "Displeased":   0.0,
}

# ---------------------------------------------------------------------------
# Landmark index constants (MediaPipe 468-point mesh)
# ---------------------------------------------------------------------------
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
MOUTH     = [61, 291, 13, 14]   # left, right, upper, lower

# Head pose reference landmarks
_NOSE_TIP         = 1
_CHIN             = 152
_LEFT_EYE_OUTER   = 33
_RIGHT_EYE_OUTER  = 263
_LEFT_MOUTH       = 61
_RIGHT_MOUTH      = 291

# 3-D model points for PnP (standard face geometry)
_MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0),         # Nose tip
    (0.0,  -330.0,  -65.0),        # Chin
    (-225.0, 170.0, -135.0),       # Left eye outer corner
    (225.0,  170.0, -135.0),       # Right eye outer corner
    (-150.0, -150.0, -125.0),      # Left mouth corner
    (150.0,  -150.0, -125.0),      # Right mouth corner
], dtype=np.float64)


# ---------------------------------------------------------------------------
# Low-level geometry helpers
# ---------------------------------------------------------------------------

def _pt(landmarks, idx: int, w: int, h: int) -> np.ndarray:
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def _eye_aspect_ratio(landmarks, eye_idx: list[int], w: int, h: int) -> float:
    """Eye Aspect Ratio: (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)."""
    p = [_pt(landmarks, i, w, h) for i in eye_idx]
    vert1 = float(np.linalg.norm(p[1] - p[5]))
    vert2 = float(np.linalg.norm(p[2] - p[4]))
    horiz = float(np.linalg.norm(p[0] - p[3]))
    return (vert1 + vert2) / (2.0 * horiz) if horiz > 0 else 0.0


def _smile_ratio(landmarks, w: int, h: int) -> float:
    """Mouth width divided by mouth height."""
    left  = _pt(landmarks, MOUTH[0], w, h)
    right = _pt(landmarks, MOUTH[1], w, h)
    upper = _pt(landmarks, MOUTH[2], w, h)
    lower = _pt(landmarks, MOUTH[3], w, h)
    width  = float(np.linalg.norm(left - right))
    height = float(np.linalg.norm(upper - lower))
    return width / height if height > 0 else 0.0


def _head_pose(landmarks, w: int, h: int) -> tuple[float, float, float]:
    """Return (pitch, yaw, roll) in degrees via PnP + RQ decomposition."""
    image_pts = np.array([
        [landmarks[_NOSE_TIP].x * w,       landmarks[_NOSE_TIP].y * h],
        [landmarks[_CHIN].x * w,           landmarks[_CHIN].y * h],
        [landmarks[_LEFT_EYE_OUTER].x * w, landmarks[_LEFT_EYE_OUTER].y * h],
        [landmarks[_RIGHT_EYE_OUTER].x * w,landmarks[_RIGHT_EYE_OUTER].y * h],
        [landmarks[_LEFT_MOUTH].x * w,     landmarks[_LEFT_MOUTH].y * h],
        [landmarks[_RIGHT_MOUTH].x * w,    landmarks[_RIGHT_MOUTH].y * h],
    ], dtype=np.float64)

    focal   = float(w)
    cam_mat = np.array([[focal, 0, w / 2],
                        [0, focal, h / 2],
                        [0, 0, 1]], dtype=np.float64)
    dist    = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, _ = cv.solvePnP(_MODEL_POINTS, image_pts, cam_mat, dist,
                               flags=cv.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0

    rmat, _ = cv.Rodrigues(rvec)
    angles, *_ = cv.RQDecomp3x3(rmat)
    pitch, yaw, roll = angles
    return float(pitch), float(yaw), float(roll)


def _gaze_direction(landmarks, w: int, h: int) -> int:
    """Return -1 (left), 0 (centre), or 1 (right)."""
    left_cx  = float(np.mean([landmarks[i].x * w for i in LEFT_EYE]))
    right_cx = float(np.mean([landmarks[i].x * w for i in RIGHT_EYE]))
    face_cx  = float(landmarks[1].x * w)
    offset   = (left_cx + right_cx) / 2.0 - face_cx
    if offset > 10:
        return 1
    elif offset < -10:
        return -1
    return 0


# ---------------------------------------------------------------------------
# Model / landmarker loaders
# ---------------------------------------------------------------------------

def load_landmarker(model_path: str):
    """Initialise and return a MediaPipe FaceLandmarker (IMAGE mode)."""
    BaseOptions         = mp.tasks.BaseOptions
    FaceLandmarker      = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode   = mp.tasks.vision.RunningMode

    opts = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
    )
    return FaceLandmarker.create_from_options(opts)


def load_rf_model(model_path: str):
    """Load the pre-trained RandomForestClassifier from a joblib file."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib_load(model_path)


# ---------------------------------------------------------------------------
# Per-frame feature extraction
# ---------------------------------------------------------------------------

def extract_features_from_frame(frame_bgr: np.ndarray, landmarker) -> Optional[dict]:
    """Extract the 10 facial features from a single BGR frame.

    Returns None when no face is detected in the frame.
    """
    h, w = frame_bgr.shape[:2]
    rgb   = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect(mp_img)
    if not result.face_landmarks:
        return None

    lms = result.face_landmarks[0]

    left_ear  = _eye_aspect_ratio(lms, LEFT_EYE, w, h)
    right_ear = _eye_aspect_ratio(lms, RIGHT_EYE, w, h)
    avg_ear   = (left_ear + right_ear) / 2.0
    eye_sync  = abs(left_ear - right_ear)
    smile     = _smile_ratio(lms, w, h)
    pitch, yaw, roll = _head_pose(lms, w, h)
    gaze      = _gaze_direction(lms, w, h)

    return {
        "avg_ear":            avg_ear,
        "left_ear":           left_ear,
        "right_ear":          right_ear,
        "blink_flag":         0,           # static frame — no temporal blink logic
        "smile_ratio":        smile,
        "eye_sync":           eye_sync,
        "head_pitch":         pitch,
        "head_yaw":           yaw,
        "head_roll":          roll,
        "eye_gaze_direction": float(gaze),
    }


# ---------------------------------------------------------------------------
# Full video analysis
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "avg_ear", "left_ear", "right_ear", "blink_flag",
    "smile_ratio", "eye_sync",
    "head_pitch", "head_yaw", "head_roll",
    "eye_gaze_direction",
]


def run_video_analysis(
    video_path: str,
    rf_model,
    landmarker,
    frame_skip: int = 5,
    max_frames: int = 150,
) -> dict:
    """Process the video, extract features, run the RF model, and aggregate.

    Parameters
    ----------
    video_path:  Path to the uploaded video.
    rf_model:    Loaded RandomForestClassifier.
    landmarker:  Loaded MediaPipe FaceLandmarker.
    frame_skip:  Process every Nth frame.
    max_frames:  Maximum frames to process.

    Returns
    -------
    dict with keys:
        emotion_prediction  – dominant display-name emotion (str)
        emotion_breakdown   – {display_name: percentage} for all emotions
        eye_contact_score   – 0-100 float
        confidence_score    – 0-100 float (head stability)
        head_pitches        – list[float] (for downstream aggregation)
        head_yaws           – list[float]
        frames_analyzed     – int
        faces_detected      – int
    """
    # -- import here to avoid top-level circular / heavy imports -----------
    import cv2 as cv  # noqa: F401 (already imported at top)
    from src.video_processing.video_processor import iter_frames

    all_features: list[dict] = []
    gazes:  list[int]   = []
    pitches: list[float] = []
    yaws:    list[float] = []

    for _idx, frame in iter_frames(video_path, frame_skip=frame_skip,
                                   max_frames=max_frames, resize=(640, 480)):
        feats = extract_features_from_frame(frame, landmarker)
        if feats is None:
            continue
        all_features.append(feats)
        gazes.append(int(feats["eye_gaze_direction"]))
        pitches.append(feats["head_pitch"])
        yaws.append(feats["head_yaw"])

    frames_analyzed = _idx + 1 if '_idx' in dir() else 0  # total frames iterated
    faces_detected  = len(all_features)

    if faces_detected == 0:
        # Fallback — no face detected in any frame
        return {
            "emotion_prediction": "Neutral",
            "emotion_breakdown":  {"Calm": 100.0},
            "emotion_timeline":   [],
            "eye_contact_score":  50.0,
            "confidence_score":   50.0,
            "head_pitches":       [],
            "head_yaws":          [],
            "frames_analyzed":    frames_analyzed,
            "faces_detected":     0,
        }

    # -- Build feature matrix and run RF model ----------------------------
    X = np.array([[f[c] for c in FEATURE_COLS] for f in all_features], dtype=np.float64)
    preds = rf_model.predict(X)           # array of int labels [0-6]
    emotion_timeline = [int(p) for p in preds]  # per-frame predictions for charts

    # -- Emotion aggregation ----------------------------------------------
    from collections import Counter
    counts   = Counter(int(p) for p in preds)
    total    = len(preds)
    raw_mode = counts.most_common(1)[0][0]
    raw_name = EMOTION_LABELS.get(raw_mode, "Neutral")
    dominant = EMOTION_DISPLAY.get(raw_name, raw_name)

    emotion_breakdown: dict[str, float] = {}
    for label_id, display in EMOTION_DISPLAY.items():
        raw_key = {v: k for k, v in EMOTION_DISPLAY.items()}.get(display, display)
        count = counts.get(
            next((k for k, v in EMOTION_LABELS.items() if v == raw_key), -1), 0
        )
        pct = round(count / total * 100, 1)
        if pct > 0:
            emotion_breakdown[display] = pct

    # -- Eye contact score ------------------------------------------------
    center_frames    = sum(1 for g in gazes if g == 0)
    eye_contact_score = round(center_frames / len(gazes) * 100, 1)

    # -- Confidence score (head stability + pose) -------------------------
    arr_pitch = np.array(pitches)
    arr_yaw   = np.array(yaws)
    forward_pct   = float(np.mean(np.abs(arr_yaw) < 20)) * 100
    upright_pct   = float(np.mean(np.abs(arr_pitch) < 20)) * 100
    confidence_score = round((forward_pct + upright_pct) / 2.0, 1)

    return {
        "emotion_prediction": dominant,
        "emotion_breakdown":  emotion_breakdown,
        "emotion_timeline":   emotion_timeline,
        "eye_contact_score":  eye_contact_score,
        "confidence_score":   confidence_score,
        "head_pitches":       pitches,
        "head_yaws":          yaws,
        "frames_analyzed":    frames_analyzed,
        "faces_detected":     faces_detected,
    }
