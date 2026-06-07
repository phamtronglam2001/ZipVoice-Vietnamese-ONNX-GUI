"""
Download Vocos vocoder into ./models/vocoder/ for offline use.
ONNX weights are bundled; Espeak tokenizer is built-in (espeak_tokenizer.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from config import HF_VOCODER_REPO, MODELS_DIR, ROOT, VOCODER_DIR


def download_vocoder() -> None:
    VOCODER_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {HF_VOCODER_REPO} -> {VOCODER_DIR}")
    snapshot_download(
        repo_id=HF_VOCODER_REPO,
        local_dir=str(VOCODER_DIR),
        local_dir_use_symlinks=False,
    )


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    download_vocoder()
    print("\n[OK] Vocoder ready for offline ONNX inference.")
    print(f"     ONNX weights: {ROOT / 'models' / 'onnx'}")
    print(f"     Project root: {ROOT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
