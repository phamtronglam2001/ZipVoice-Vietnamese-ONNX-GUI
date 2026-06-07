@echo off
title ZipVoice ONNX TTS
setlocal
cd /d "%~dp0"

set ZIPVOICE_FORCE_CPU=1
set CUDA_VISIBLE_DEVICES=

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Run install_cpu.bat first.
    pause
    exit /b 1
)

echo Starting ZipVoice ONNX GUI on http://127.0.0.1:7862
".venv\Scripts\python.exe" app.py
if errorlevel 1 pause
