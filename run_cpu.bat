@echo off

title ZipVoice ONNX TTS - GUI (CPU)

setlocal EnableDelayedExpansion

cd /d "%~dp0"

set ZIPVOICE_FORCE_CPU=1
set ZIPVOICE_ONNX_GPU=0
set CUDA_VISIBLE_DEVICES=

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Chua cai dat. Chay install_cpu.bat truoc.
    pause
    exit /b 1
)

echo Dang khoi dong GUI ZipVoice ONNX (CPU-only) tai http://127.0.0.1:7862

".venv\Scripts\python.exe" app.py

if errorlevel 1 pause
