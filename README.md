# AI Interview Analyzer

> **Evaluate your interview performance with AI** — upload a video recording and receive instant, data-driven feedback on eye contact, speech clarity, emotional state, and overall confidence.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Tech Stack](#tech-stack)
5. [Project Structure](#project-structure)
6. [Setup Instructions](#setup-instructions)
7. [Demo](#demo)
8. [Future Improvements](#future-improvements)

---

## Project Overview

The AI Interview Analyzer is a Flask-based web application that evaluates mock interview videos using a combination of **computer vision**, **speech recognition**, and **machine learning**. A candidate uploads their interview recording; the system simultaneously processes the visual and audio tracks in parallel, then presents an interactive dashboard with scores, charts, and personalised recommendations.

**Core capabilities:**
- Facial landmark tracking via **MediaPipe** (468-point mesh) to measure eye gaze, head pose, and facial expressions.
- Emotion classification using a **Random Forest** model trained on the FER-7 dataset (angry, disgust, fear, happy, neutral, sad, surprised).
- Speech transcription with **OpenAI Whisper** and acoustic feature extraction with **Librosa**.
- A Bootstrap 5 dashboard with **Chart.js** visualisations for real-time analytics.

---

## Features

| Feature | Description |
|---|---|
| 👁️ **Eye Contact Detection** | Measures the percentage of frames in which the candidate's gaze is directed at the camera using EAR and gaze-offset geometry. |
| 🎤 **Speech Rate Analysis** | Transcribes the audio with Whisper and calculates words per minute, flagging deviations from the optimal 120–160 WPM range. |
| 😊 **Emotion Recognition** | Predicts per-frame emotions using MediaPipe facial landmarks fed into a tuned Random Forest classifier. |
| 🛡️ **Confidence Scoring** | Combines head-pose stability (yaw/pitch variance) with vocal energy (librosa RMS) into a single 0–100 confidence score. |
| 📊 **Interview Performance Score** | Weighted aggregation of all sub-scores into one final 0–100 score with contextual performance labels. |
| 📈 **Interactive Dashboard** | Emotion distribution doughnut chart, per-frame emotion timeline, animated progress bars, and speech rate indicator — all rendered with Chart.js. |
| 📝 **Speech Transcript** | Displays the full Whisper transcript alongside word count and recording duration. |
| 💡 **Personalised Recommendations** | Context-aware coaching tips based on each candidate's unique score profile. |
| 🎬 **Video Preview** | In-browser HTML5 video preview before upload, with filename, file size, and a "Change" button. |

---

## Architecture

```
Upload Video (POST /analyze)
        │
        ├── [Thread A — Visual Pipeline] ─────────────────────────────────
        │     Video Processor        →  iter_frames()  (every 5th, max 150)
        │     MediaPipe FaceLandmarker  →  468 facial landmarks per frame
        │     Feature Extractor      →  10 geometric features per frame
        │                                (EAR, smile ratio, head pose, gaze)
        │     Random Forest Model    →  emotion class per frame  [0–6]
        │     Aggregation            →  dominant emotion
        │                               emotion breakdown %
        │                               eye contact score
        │                               confidence score (head stability)
        │
        └── [Thread B — Audio Pipeline] ──────────────────────────────────
              MoviePy                →  extract 16 kHz WAV
              OpenAI Whisper         →  speech transcript + segments
              Librosa                →  MFCC, RMS energy, tempo, spectral
              Speech Rate            →  words per minute

        Merge results
        │   confidence += 0.40 × voice_energy_score
        │   _compute_final_score(confidence, wpm, eye_contact, emotion)
        │
        Flask API  →  render result.html
        │
        Bootstrap 5 Dashboard
              Animated progress bars (confidence, eye contact)
              Speech rate indicator with optimal zone
              Emotion distribution doughnut (Chart.js)
              Per-frame emotion timeline (Chart.js stepped line)
              Transcript panel
              Recommendations
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3, Flask, Werkzeug |
| **Computer Vision** | OpenCV, MediaPipe Face Landmarker |
| **ML Model** | Scikit-learn Random Forest (tuned, 757 MB) |
| **Speech Recognition** | OpenAI Whisper (`base` model) |
| **Audio Features** | Librosa (MFCC, RMS, spectral centroid, tempo) |
| **Video I/O** | MoviePy, imageio-ffmpeg |
| **Frontend** | Bootstrap 5, Chart.js 4, Bootstrap Icons |
| **Logging** | Python `logging` + `RotatingFileHandler` |
| **Concurrency** | `concurrent.futures.ThreadPoolExecutor` |

---

## Project Structure

```
ai-interview-analyzer/
│
├── api/
│   └── app.py                        # Flask routes, validation, error handlers
│
├── src/
│   ├── pipeline/
│   │   └── interview_pipeline.py     # Orchestrator — parallel video + audio
│   ├── feature_engineering/
│   │   ├── facial_features.py        # MediaPipe extraction + RF inference
│   │   └── tuned_randomforest_model.joblib   # Pre-trained emotion classifier
│   ├── audio_processing/
│   │   └── audio_pipeline.py         # Audio extraction, Whisper, Librosa
│   ├── video_processing/
│   │   └── video_processor.py        # Frame extraction utilities
│   └── utils/
│       └── logger.py                 # Structured logging (console + file)
│
├── templates/
│   ├── index.html                    # Upload page with video preview
│   └── result.html                   # Results dashboard with Chart.js
│
├── static/
│   ├── css/style.css                 # Custom styles
│   └── js/main.js                    # Upload UX, preview, progress bars
│
├── data/
│   ├── raw/train/{emotion}/          # Training images (FER-7 categories)
│   └── processed/                    # CSV features, audio intermediates
│
├── notebooks/                        # Jupyter notebooks (training & EDA)
├── logs/                             # Auto-created — app.log (rotating)
├── uploads/                          # Uploaded video files
├── face_landmarker.task              # MediaPipe model asset (3.6 MB)
├── requirements.txt
└── run.bat                           # One-click launcher (Windows)
```

---

## Setup Instructions

### Prerequisites

- **Anaconda** with the `deep_learning` conda environment (contains all required packages).
- **Windows** — the provided `run.bat` targets Windows paths. Linux/macOS users should call Python directly.

### 1. Clone / download the repository

```bash
git clone https://github.com/your-username/ai-interview-analyzer.git
cd ai-interview-analyzer
```

### 2. Create and activate the conda environment

```bash
conda create -n deep_learning python=3.10 -y
conda activate deep_learning
pip install -r requirements.txt
```

> **Key packages installed:** `flask`, `opencv-python`, `mediapipe`, `openai-whisper`, `librosa`, `moviepy`, `scikit-learn`, `torch`, `imageio-ffmpeg`, `joblib`

### 3. Verify model assets

Ensure the following files exist in the project root:

| File | Size | Description |
|---|---|---|
| `face_landmarker.task` | ~3.6 MB | MediaPipe Face Landmarker model |
| `src/feature_engineering/tuned_randomforest_model.joblib` | ~757 MB | Trained emotion classifier |

### 4. Run the server

**Windows (recommended):**

```bat
run.bat
```

**Manual:**

```bash
# Activate the environment first
conda activate deep_learning
# Suppress TensorFlow noise from MediaPipe
set TF_ENABLE_ONEDNN_OPTS=0
python api/app.py
```

The server starts at **http://localhost:5000**.

---

## Demo

1. Open **http://localhost:5000** in your browser.
2. Drag and drop an interview video (MP4, AVI, MOV, MKV, or WEBM — up to 500 MB) onto the upload zone, or click to browse.
3. A live **video preview** appears — review the clip and click **Analyze Interview**.
4. The loading overlay appears while the pipeline runs (typically 30–120 seconds depending on video length and hardware).
5. The **Results Dashboard** displays:
   - Final interview score ring (0–100)
   - Animated confidence and eye-contact bars
   - Speech rate indicator with optimal-zone highlight
   - Emotion distribution doughnut chart
   - Per-frame emotion timeline
   - Full speech transcript
   - Personalised coaching recommendations

---

## Future Improvements

- **Real-time analysis** — process live webcam feed using WebSockets or WebRTC.
- **Deep learning emotion model** — replace the Random Forest with a CNN or Vision Transformer trained end-to-end on facial images for higher accuracy.
- **Body language analysis** — extend MediaPipe to full-body pose estimation (shoulders, posture, hand gestures).
- **Multi-language support** — leverage Whisper's multilingual capability and surface language-specific recommendations.
- **Session history** — store past analyses in a database (SQLite/PostgreSQL) so candidates can track progress over time.
- **PDF export** — generate a downloadable performance report with charts.
- **Docker deployment** — containerise the app for one-command cloud deployment.
- **A/B testing recommendations** — rank coaching tips by impact using historical data.

---

*Built with ❤️ using Flask, OpenCV, MediaPipe, Whisper, and Chart.js.*
# GAuRaV27k-AI_base_interview_analyzer

---

## 💼 LinkedIn Project Description

> *Copy the text below directly into the **Projects** section of your LinkedIn profile.*

---

**AI-Based Interview Analyzer** | Python · Flask · OpenCV · MediaPipe · Whisper · Scikit-learn · Chart.js

Built an end-to-end AI-powered web application that evaluates mock interview recordings and delivers instant, data-driven feedback — helping candidates understand and improve their interview performance before the real thing.

**What it does:**
Upload an interview video (MP4/AVI/MOV/MKV/WEBM, up to 500 MB) and the system simultaneously analyses the visual and audio tracks using a multi-threaded pipeline, then renders a rich analytics dashboard in seconds.

**Key technical highlights:**
• **Computer Vision** — Extracts 468 facial landmarks per frame with MediaPipe Face Landmarker to compute Eye Aspect Ratio (EAR), gaze offset, smile ratio, and head-pose angles (yaw/pitch/roll).
• **Emotion Recognition** — A tuned Random Forest classifier trained on the FER-7 dataset predicts the candidate's emotion (angry, disgust, fear, happy, neutral, sad, surprised) for every sampled frame.
• **Speech Analysis** — OpenAI Whisper transcribes the audio and Librosa extracts acoustic features (MFCC, RMS energy, spectral centroid, tempo) to calculate speech rate and vocal energy.
• **Confidence Scoring** — Fuses head-pose stability and vocal energy into a single 0–100 confidence index.
• **Interactive Dashboard** — Bootstrap 5 frontend with Chart.js visualisations: emotion-distribution doughnut, per-frame emotion timeline, animated progress bars, and a speech-rate indicator with an optimal-zone highlight.
• **Personalised Recommendations** — Context-aware coaching tips generated from each candidate's unique score profile.

**Tech stack:** Python 3, Flask, OpenCV, MediaPipe, Scikit-learn, OpenAI Whisper, Librosa, MoviePy, PyTorch, Bootstrap 5, Chart.js, concurrent.futures

This project demonstrates practical experience in computer vision, NLP, audio signal processing, machine learning model integration, REST API design, and full-stack web development.
