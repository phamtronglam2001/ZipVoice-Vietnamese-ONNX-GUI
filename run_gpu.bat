@echo off

title ZipVoice ONNX TTS - GUI (GPU)

setlocal EnableDelayedExpansion

cd /d "%~dp0"

set INSTALL_MODE=cpu
if exist ".install_mode" set /p INSTALL_MODE=<.install_mode

if /i not "!INSTALL_MODE!"=="gpu" (
    echo [WARN] .install_mode=cpu — chay install_gpu.bat de cai onnxruntime-gpu + CUDA DLL.
    echo        Van khoi dong voi ZIPVOICE_ONNX_GPU=1 ^(co the fallback CPU^).
    echo.
)

set "ZIPVOICE_FORCE_CPU="
set ZIPVOICE_ONNX_GPU=1
if /i "!CUDA_VISIBLE_DEVICES!"=="" set "CUDA_VISIBLE_DEVICES=0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Chua cai dat. Chay install_cpu.bat hoac install_gpu.bat truoc.
    pause
    exit /b 1
)

echo Kiem tra CUDA runtime...
".venv\Scripts\python.exe" scripts\diagnose_gpu.py
echo.

echo Dang khoi dong GUI ZipVoice ONNX (GPU) tai http://127.0.0.1:7862

".venv\Scripts\python.exe" app.py

if errorlevel 1 pause
