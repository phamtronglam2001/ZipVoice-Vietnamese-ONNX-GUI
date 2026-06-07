"""
Optional fallback: fetch Vocos ONNX vocoder if missing from a partial clone (e.g. Git LFS not pulled).

All model weights are normally bundled in ./models/ — no Hugging Face library required.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

from config import MODELS_DIR, ROOT, VOCODER_DIR, VOCODER_ONNX, VOCODER_ONNX_FILENAME

# Direct file URL (no huggingface_hub dependency)
HF_VOCODER_URL = (
    "https://huggingface.co/wetdog/vocos-mel-24khz-onnx/resolve/main/"
    f"{VOCODER_ONNX_FILENAME}"
)


def download_vocoder() -> None:
    if VOCODER_ONNX.is_file():
        print(f"[OK] Vocoder already present: {VOCODER_ONNX}")
        return

    VOCODER_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {HF_VOCODER_URL}")
    print(f"         -> {VOCODER_ONNX}")
    urllib.request.urlretrieve(HF_VOCODER_URL, VOCODER_ONNX)
    if not VOCODER_ONNX.is_file() or VOCODER_ONNX.stat().st_size < 1_000_000:
        raise RuntimeError(f"Download failed or file too small: {VOCODER_ONNX}")


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
        print(
            "Tip: clone with Git LFS — git lfs install && git lfs pull",
            file=sys.stderr,
        )
        sys.exit(1)
