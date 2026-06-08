@echo off

title ZipVoice ONNX TTS - Slint GUI (Auto)

setlocal EnableDelayedExpansion

cd /d "%~dp0"

set "PYTHONPATH=%~dp0src"

set INSTALL_MODE=cpu
if exist ".install_mode" set /p INSTALL_MODE=<.install_mode

if /i "!INSTALL_MODE!"=="gpu" (
    set "ZIPVOICE_FORCE_CPU="
    set ZIPVOICE_ONNX_GPU=1
    if /i "!CUDA_VISIBLE_DEVICES!"=="" set "CUDA_VISIBLE_DEVICES=0"
) else (
    set ZIPVOICE_FORCE_CPU=1
    set ZIPVOICE_ONNX_GPU=0
    set CUDA_VISIBLE_DEVICES=
)

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Chua cai dat. Chay install_cpu.bat truoc.
    pause
    exit /b 1
)

echo Kiem tra Slint...
".venv\Scripts\python.exe" -c "import slint" 2>nul
if errorlevel 1 (
    echo Cai dat Slint tu requirements-slint.txt...
    ".venv\Scripts\python.exe" -m pip install -r requirements-slint.txt 2>nul
    if errorlevel 1 (
        where uv >nul 2>&1
        if errorlevel 1 (
            echo [ERROR] Khong cai duoc slint. Thu: uv pip install --python .venv\Scripts\python.exe -r requirements-slint.txt
            pause
            exit /b 1
        )
        uv pip install --python ".venv\Scripts\python.exe" -r requirements-slint.txt
        if errorlevel 1 (
            echo [ERROR] Khong cai duoc slint. Xem requirements-slint.txt
            pause
            exit /b 1
        )
    )
    ".venv\Scripts\python.exe" -c "import slint" 2>nul
    if errorlevel 1 (
        echo [ERROR] Slint chua import duoc sau khi cai.
        pause
        exit /b 1
    )
)

if /i "!INSTALL_MODE!"=="gpu" (
    echo Dang khoi dong Slint GUI ZipVoice ONNX ^(GPU^)...
) else (
    echo Dang khoi dong Slint GUI ZipVoice ONNX ^(CPU-only^)...
)

echo.
echo [CANH BAO] Slint GUI CHUA HOAN THANH - co the crash im lang.
echo            Dung run_gui.bat ^(Gradio^) cho den khi README TODO xong.
echo.

".venv\Scripts\python.exe" "%~dp0src\slint_gui\main.py"

if errorlevel 1 pause
