@echo off

title ZipVoice ONNX TTS - Install GPU (uv)

setlocal EnableDelayedExpansion

cd /d "%~dp0"

echo.
echo  ============================================================
echo   ZipVoice Vietnamese ONNX TTS - GPU install (onnxruntime-gpu)
echo  ============================================================
echo   Requires NVIDIA GPU + CUDA/cuDNN matching the ORT wheel.
echo   onnxruntime and onnxruntime-gpu cannot coexist — CPU package
echo   will be removed before installing the GPU build.
echo.
echo   ORT 1.26.x (typical): CUDA 12.x + cuDNN 9
echo   See: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
echo  ============================================================
echo.

where uv >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Missing `uv`. Install: https://docs.astral.sh/uv/
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/5] Creating venv with uv...
    uv venv .venv --python 3.11
    if errorlevel 1 uv venv .venv
) else (
    echo [1/5] .venv exists.
)

set PY=%CD%\.venv\Scripts\python.exe
if not exist "%PY%" (
    echo [ERROR] venv python not found.
    pause
    exit /b 1
)

echo [2/5] Base deps (gradio, librosa...) without onnxruntime CPU...
uv pip install --python "%PY%" gradio scipy soundfile pydub "numpy>=1.24.0,<=1.26.4" librosa
if errorlevel 1 goto :fail

echo [3/5] onnxruntime-gpu (replaces onnxruntime if present)...
uv pip uninstall --python "%PY%" onnxruntime 2>nul
uv pip install --python "%PY%" -r requirements-gpu.txt
if errorlevel 1 goto :fail

echo [4/5] piper_phonemize...
uv pip install --python "%PY%" piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html
if errorlevel 1 goto :fail

echo [5/5] Normalizer (optional)...
uv pip install --python "%PY%" -r requirements-normalize.txt
if errorlevel 1 goto :fail

echo gpu>"%CD%\.install_mode"

"%PY%" -c "import onnxruntime as ort; eps=ort.get_available_providers(); print('OK | onnxruntime', ort.__version__, '| EPs:', eps); assert 'CUDAExecutionProvider' in eps or 'DmlExecutionProvider' in eps, 'No GPU EP — check CUDA/driver'"

echo.
echo  === GPU install complete ===
echo  Enable GPU: GUI checkbox or set ZIPVOICE_ONNX_GPU=1
echo  run_gui.bat will NOT force CPU when .install_mode=gpu
echo  Chay GUI: run_gui.bat  -^>  http://127.0.0.1:7862
echo.
pause
exit /b 0

:fail
echo.
echo [FAILED] See errors above.
pause
exit /b 1
