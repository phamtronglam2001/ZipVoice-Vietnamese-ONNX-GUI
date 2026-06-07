# ZipVoice Vietnamese ONNX TTS — minimal CPU setup
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Write-Host "=== ZipVoice ONNX TTS — Minimal CPU Install ===" -ForegroundColor Cyan
Write-Host "ONNX bundled | downloads vocoder only (~50 MB)" -ForegroundColor Gray

$env:ZIPVOICE_FORCE_CPU = "1"
$env:CUDA_VISIBLE_DEVICES = ""

$venv = Join-Path $Root ".venv"
if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) {
    Write-Host "[1/5] Creating virtual environment..." -ForegroundColor Yellow
    python -m venv $venv
} else {
    Write-Host "[1/5] Virtual environment exists." -ForegroundColor Green
}

$python = Join-Path $venv "Scripts\python.exe"
$pip = Join-Path $venv "Scripts\pip.exe"

Write-Host "[2/5] CPU PyTorch + torchaudio..." -ForegroundColor Yellow
& $python -m pip install --upgrade pip wheel
& $pip install "torch==2.6.0" "torchaudio==2.6.0" --index-url https://download.pytorch.org/whl/cpu

Write-Host "[3/5] Runtime dependencies..." -ForegroundColor Yellow
& $pip install -r requirements-cpu.txt
& $pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html

Write-Host "[4/5] Setup tools + vocoder download..." -ForegroundColor Yellow
& $pip install -r requirements-setup.txt
& $python download_models.py

"cpu" | Out-File -FilePath (Join-Path $Root ".install_mode") -Encoding ascii -NoNewline

Write-Host "[5/5] Verify..." -ForegroundColor Yellow
& $python -c "from config import models_ready; import sys; sys.exit(0 if models_ready() else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Setup incomplete." -ForegroundColor Red
    exit 1
}

& $python -c "import onnxruntime, torch, vocos; print('onnxruntime + torch + vocos OK')"

Write-Host ""
Write-Host "=== Done ===" -ForegroundColor Green
Write-Host "Run: run_cpu.bat  ->  http://127.0.0.1:7862" -ForegroundColor Cyan
