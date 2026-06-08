@echo off

title ZipVoice ONNX TTS - Install GPU (uv)

setlocal EnableDelayedExpansion

cd /d "%~dp0"

echo.
echo  ============================================================
echo   ZipVoice Vietnamese ONNX TTS - GPU install (onnxruntime-gpu)
echo  ============================================================
echo   ZipVoice model     = ONNX Runtime
echo   Vocos vocoder      = Vocos ONNX 100 mel + librosa ISTFT (export PyTorch GUI)
echo   Model weights      = bundled in models/ (Git LFS)
echo   Vocoder verify     = local models/vocoder/mel_spec_24khz.onnx (100 mel)
echo  ============================================================
echo   Requires NVIDIA GPU + CUDA/cuDNN matching the ORT wheel.
echo   onnxruntime and onnxruntime-gpu cannot coexist — CPU package
echo   will be removed before installing the GPU build.
echo.
echo   ORT 1.26.x (typical): CUDA 12.x + cuDNN 9
echo   Runtime DLLs: nvidia-cublas/cudnn/cuda-runtime/cufft-cu12 (pip, in requirements-gpu.txt)
echo   Prompt mel: librosa only (vocos_fbank.py), not PyTorch vocoder
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
    set "RECREATE_VENV=1"
) else if not exist ".venv\pyvenv.cfg" (
    echo [WARN] Broken .venv ^(missing pyvenv.cfg^) — recreating...
    rmdir /s /q ".venv" 2>nul
    set "RECREATE_VENV=1"
) else (
    .venv\Scripts\python.exe --version >nul 2>&1
    if errorlevel 1 (
        echo [WARN] Broken .venv ^(python not runnable^) — recreating...
        rmdir /s /q ".venv" 2>nul
        set "RECREATE_VENV=1"
    ) else (
        set "RECREATE_VENV=0"
    )
)

if "!RECREATE_VENV!"=="1" (
    if exist ".venv" rmdir /s /q ".venv" 2>nul
    echo [1/5] Creating venv with uv...
    uv venv .venv --python 3.11
    if errorlevel 1 (
        echo [WARN] Python 3.11 not found, trying default python...
        uv venv .venv
    )
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

echo [3/5] onnxruntime-gpu + CUDA DLLs (replaces onnxruntime if present)...
uv pip uninstall --python "%PY%" onnxruntime 2>nul
uv pip install --python "%PY%" -r requirements-gpu.txt
if errorlevel 1 goto :fail

echo [4/5] piper_phonemize (Espeak Vietnamese)...
uv pip install --python "%PY%" piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html
if errorlevel 1 goto :fail

echo [5/5] Normalizer (sea-g2p — optional for pipeline)...
uv pip install --python "%PY%" -r requirements-normalize.txt
if errorlevel 1 goto :fail

echo.
echo Kiem tra file ONNX bundled ^(ZipVoice int4/int8 + vocoder 100 mel^)...
echo   - models/onnx/: text_encoder_*.onnx, fm_decoder_*.onnx, model.json, tokens.txt
echo   - models/vocoder/mel_spec_24khz.onnx ^(100 mel, export ZipVoice-Vietnamese-GUI^)
echo   - Phat hien stub Git LFS ^(chua git lfs pull^) cung tinh la thieu
"%PY%" -c "from config import models_ready, models_ready_report; import sys; missing=models_ready_report(); [print('  MISSING:', x) for x in missing]; sys.exit(0 if models_ready() else 1)"
if errorlevel 1 (
    echo [WARN] Thieu hoac chua pull file model ^(xem dong MISSING phia tren^).
    echo        ZipVoice ONNX: git lfs pull  ^(hoac copy export tu PyTorch GUI^)
    echo        Vocoder 100 mel: dat tai models/vocoder/mel_spec_24khz.onnx
    echo        ^(export ZipVoice-Vietnamese-GUI — Tab Export -^> Export Vocos ONNX^)
    goto :fail
)
echo [OK] File ONNX bundled da san sang.

echo gpu>"%CD%\.install_mode"

"%PY%" -c "import onnxruntime; import onnx_engine; print('OK | onnxruntime', onnxruntime.__version__, '| vocoder=Vocos ONNX 100 mel + librosa ISTFT')"
if errorlevel 1 goto :fail

"%PY%" -c "import onnxruntime as ort; from onnx_providers import ensure_cuda_runtime_on_path, is_cuda_execution_provider_loadable, provider_status_message; ensure_cuda_runtime_on_path(); eps=ort.get_available_providers(); print('OK | EPs:', eps); assert 'CUDAExecutionProvider' in eps or 'DmlExecutionProvider' in eps, 'No GPU EP — check CUDA/driver'"
if errorlevel 1 goto :fail
"%PY%" -c "from onnx_providers import ensure_cuda_runtime_on_path, is_cuda_execution_provider_loadable, provider_status_message; ensure_cuda_runtime_on_path(); ok=is_cuda_execution_provider_loadable(); print('CUDA loadable:', ok, '|', provider_status_message(True)); print('[WARN] CUDA DLL chua san sang — app van chay CPU khi bat GPU.' if not ok else '[OK] CUDA runtime DLL san sang.')"

echo.
echo  === GPU install complete ===
echo  GPU install: run_gpu.bat hoac run_gui.bat  -^>  http://127.0.0.1:7862
echo  ^(GPU tu bat ZIPVOICE_ONNX_GPU=1^)
echo  CPU-only: run_cpu.bat
echo.
pause
exit /b 0

:fail
echo.
echo [FAILED] See errors above.
pause
exit /b 1
