@echo off

title ZipVoice ONNX TTS - GUI (Auto)

setlocal EnableDelayedExpansion

cd /d "%~dp0"

set INSTALL_MODE=cpu
if exist ".install_mode" set /p INSTALL_MODE=<.install_mode

if /i "!INSTALL_MODE!"=="gpu" (
    call "%~dp0run_gpu.bat" %*
) else (
    call "%~dp0run_cpu.bat" %*
)
