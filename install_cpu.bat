@echo off
title ZipVoice ONNX TTS - Install (uv)
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set ZIPVOICE_FORCE_CPU=1
set CUDA_VISIBLE_DEVICES=

echo.
echo  ============================================================
echo   ZipVoice Vietnamese ONNX TTS - Install (uv)
echo  ============================================================
echo   ZipVoice model     = ONNX Runtime (no PyTorch)
echo   Vocos vocoder      = ONNX wetdog + librosa ISTFT
echo   Model weights      = bundled in models/ (Git LFS)
echo   Download           = none (offline after clone)
echo  ============================================================
echo.

where uv >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Missing `uv`. Install: https://docs.astral.sh/uv/
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Creating venv with uv...
    uv venv .venv --python 3.11
    if errorlevel 1 (
        echo [WARN] Python 3.11 not found, trying default python...
        uv venv .venv
    )
) else (
    echo [1/4] .venv exists.
)

set PY=%CD%\.venv\Scripts\python.exe
if not exist "%PY%" (
    echo [ERROR] venv python not found.
    pause
    exit /b 1
)

echo [2/4] Runtime (onnxruntime, gradio, librosa, scipy...)...
uv pip install --python "%PY%" -r requirements-cpu.txt
if errorlevel 1 goto :fail

echo [3/4] piper_phonemize (Espeak Vietnamese)...
uv pip install --python "%PY%" piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html
if errorlevel 1 goto :fail

echo [4/4] Normalizers (vinorm / sea-g2p — optional for pipeline)...
uv pip install --python "%PY%" -r requirements-normalize.txt
if errorlevel 1 goto :fail

echo.
echo Verifying bundled model weights...
"%PY%" -c "from config import models_ready; import sys; sys.exit(0 if models_ready() else 1)"
if errorlevel 1 (
    echo [WARN] Some model files missing. Try: git lfs pull
    echo        Or run: python download_models.py  ^(vocoder fallback only^)
    goto :fail
)

echo cpu>"%CD%\.install_mode"

"%PY%" -c "import onnxruntime; import onnx_engine; print('OK | onnxruntime', onnxruntime.__version__, '| vocoder=onnx+librosa (no torch)')"

echo.
echo  === Install complete ===
echo  Run: run_cpu.bat  -^>  http://127.0.0.1:7862
echo.
set /p LAUNCH="Launch GUI now? (Y/n): "
if /i "!LAUNCH!"=="n" goto :done
call "%~dp0run_cpu.bat"
goto :eof

:fail
echo.
echo [FAILED] See errors above.
pause
exit /b 1

:done
pause
