# ZipVoice Vietnamese ONNX TTS — CPU-only setup (requires internet once)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Write-Host "=== ZipVoice Vietnamese ONNX TTS — CPU Install ===" -ForegroundColor Cyan
Write-Host "ONNX weights bundled; downloads vocoder + ZipVoice tokenizer only." -ForegroundColor Gray

$env:ZIPVOICE_FORCE_CPU = "1"
$env:CUDA_VISIBLE_DEVICES = ""

$venv = Join-Path $Root ".venv"
if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) {
    Write-Host "[1/5] Creating virtual environment..." -ForegroundColor Yellow
    python -m venv $venv
} else {
    Write-Host "[1/5] Virtual environment already exists." -ForegroundColor Green
}

$python = Join-Path $venv "Scripts\python.exe"
$pip = Join-Path $venv "Scripts\pip.exe"

Write-Host "[2/5] Installing CPU PyTorch..." -ForegroundColor Yellow
& $python -m pip install --upgrade pip wheel
& $pip install "torch==2.6.0" "torchaudio==2.6.0" --index-url https://download.pytorch.org/whl/cpu

Write-Host "[3/5] Installing Python dependencies..." -ForegroundColor Yellow
& $pip install -r requirements-cpu.txt
& $pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html

Write-Host "[4/5] Downloading vocoder + ZipVoice source..." -ForegroundColor Yellow
& $python download_models.py

"cpu" | Out-File -FilePath (Join-Path $Root ".install_mode") -Encoding ascii -NoNewline

Write-Host "[5/5] Verifying installation..." -ForegroundColor Yellow
& $python -c "from config import models_ready; import sys; sys.exit(0 if models_ready() else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Setup incomplete. Check errors above." -ForegroundColor Red
    exit 1
}

& $python -c "import torch, onnxruntime; print('torch', torch.__version__, '| onnxruntime ok')"

Write-Host ""
Write-Host "=== ONNX CPU setup complete ===" -ForegroundColor Green
Write-Host "Run the GUI with:  run_cpu.bat" -ForegroundColor Cyan
