@echo off

title ZipVoice ONNX TTS - GUI

setlocal EnableDelayedExpansion

cd /d "%~dp0"

set INSTALL_MODE=cpu
if exist ".install_mode" set /p INSTALL_MODE=<.install_mode

if /i "!INSTALL_MODE!"=="cpu" (
    set ZIPVOICE_FORCE_CPU=1
    set CUDA_VISIBLE_DEVICES=
) else (
    rem GPU install — allow CUDA; enable via GUI or ZIPVOICE_ONNX_GPU=1
    if not defined ZIPVOICE_ONNX_GPU set ZIPVOICE_ONNX_GPU=0
)

if not exist ".venv\Scripts\python.exe" (

    echo [ERROR] Chua cai dat. Chay install_cpu.bat truoc.

    pause

    exit /b 1

)



echo Dang khoi dong GUI ZipVoice ONNX tai http://127.0.0.1:7862

".venv\Scripts\python.exe" app.py

if errorlevel 1 pause

