@echo off
:: ============================================================
:: AI Interview Analyzer — Launch Script
:: Runs the Flask app using the deep_learning conda environment
:: which contains all required dependencies:
::   opencv-python, mediapipe, openai-whisper, librosa,
::   moviepy, flask, scikit-learn, joblib, torch
:: ============================================================

set CONDA_PYTHON=C:\Users\gk480\anaconda3\envs\deep_learning\python.exe
set PROJECT_DIR=%~dp0

echo.
echo  ============================================
echo   AI Interview Analyzer
echo  ============================================
echo   Python  : %CONDA_PYTHON%
echo   Project : %PROJECT_DIR%
echo  ============================================
echo.

cd /d "%PROJECT_DIR%"

:: Suppress TensorFlow oneDNN noise from MediaPipe
set TF_ENABLE_ONEDNN_OPTS=0
set TF_CPP_MIN_LOG_LEVEL=3

echo  Starting server at http://localhost:5000
echo  Press Ctrl+C to stop.
echo.

"%CONDA_PYTHON%" api\app.py
