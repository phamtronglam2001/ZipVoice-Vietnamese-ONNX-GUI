@echo off
title ZipVoice ONNX TTS - CPU Install
setlocal
cd /d "%~dp0"

echo.
echo  ============================================
echo   ZipVoice Vietnamese ONNX TTS - CPU Install
echo  ============================================
echo   ONNX weights already in models/onnx/
echo   Setup downloads vocoder only (~50 MB)
echo  ============================================
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_cpu.ps1"
if errorlevel 1 (
    echo.
    echo [FAILED] Setup did not complete.
    pause
    exit /b 1
)

echo.
set /p LAUNCH="Launch ONNX GUI now? (Y/n): "
if /i "%LAUNCH%"=="n" (
    echo Done. Double-click run_cpu.bat when ready.
    pause
    exit /b 0
)

call "%~dp0run_cpu.bat"
