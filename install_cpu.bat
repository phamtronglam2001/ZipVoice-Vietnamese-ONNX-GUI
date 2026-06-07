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

echo   ZipVoice model     = ONNX Runtime

echo   Vocos vocoder      = PyTorch Vocos (default, khuyen dung)

echo   Vocoder ONNX       = optional fallback (wetdog + librosa ISTFT)

echo   Model weights      = bundled in models/ (Git LFS)

echo   PyTorch vocoder DL = python download_models.py --pytorch-vocoder

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



echo [2/4] Runtime (onnxruntime, gradio, torch/vocos, librosa...)...

set /p USE_GPU="Co GPU NVIDIA CUDA? (y/N): "

if /i "!USE_GPU!"=="y" (
    echo Installing onnxruntime-gpu ^(CUDA^)...
    uv pip install --python "%PY%" gradio scipy soundfile pydub "numpy>=1.24.0,<=1.26.4" librosa
    if errorlevel 1 goto :fail
    uv pip uninstall --python "%PY%" onnxruntime 2>nul
    uv pip install --python "%PY%" -r requirements-gpu.txt
    if errorlevel 1 goto :fail
    set INSTALL_MODE=gpu
) else (
    uv pip install --python "%PY%" -r requirements-cpu.txt
    if errorlevel 1 goto :fail
    set INSTALL_MODE=cpu
)



echo [3/4] piper_phonemize (Espeak Vietnamese)...

uv pip install --python "%PY%" piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html

if errorlevel 1 goto :fail



echo [4/4] Normalizer (sea-g2p — optional for pipeline)...

uv pip install --python "%PY%" -r requirements-normalize.txt

if errorlevel 1 goto :fail



echo.

echo Verifying bundled model weights...

"%PY%" -c "from config import models_ready, pytorch_vocoder_ready; import sys; sys.exit(0 if models_ready() else 1)"

if errorlevel 1 (

    echo [WARN] Some model files missing. Try: git lfs pull

    echo        PyTorch vocoder: python download_models.py --pytorch-vocoder

    echo        ONNX vocoder fallback: python download_models.py --onnx-vocoder

    goto :fail

)



echo !INSTALL_MODE!>"%CD%\.install_mode"



"%PY%" -c "import onnxruntime; import onnx_engine; print('OK | onnxruntime', onnxruntime.__version__, '| vocoder=PyTorch Vocos (default)')"



echo.

echo  === Install complete ===
echo  Chay GUI: run_gui.bat  -^>  http://127.0.0.1:7862
if /i "!INSTALL_MODE!"=="gpu" (
    echo  GPU: bat dau GUI, bat "Dung GPU" hoac set ZIPVOICE_ONNX_GPU=1
) else (
    echo  CPU-only. GPU: chay install_gpu.bat hoac install_cpu.bat ^(Y^) hoac pip install onnxruntime-gpu
)
echo  (run_cpu.bat = alias cu, tuong duong run_gui.bat)

echo.

set /p LAUNCH="Chay giao dien GUI ngay? (Y/n): "

if /i "!LAUNCH!"=="n" goto :done

call "%~dp0run_gui.bat"

goto :eof



:fail

echo.

echo [FAILED] See errors above.

pause

exit /b 1



:done

pause

