"""
Optional fallback: fetch model weights if missing from a partial clone (e.g. Git LFS not pulled).

Default path uses PyTorch Vocos (charactr/vocos-mel-24khz).
ONNX wetdog vocoder is optional fallback when --onnx-vocoder is used.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

from config import (
    HF_VOCODER_ONNX_REPO,
    HF_VOCODER_PYTORCH_REPO,
    MODELS_DIR,
    ROOT,
    VOCODER_DIR,
    VOCODER_ONNX,
    VOCODER_ONNX_FILENAME,
    VOCODER_PYTORCH_CONFIG,
    VOCODER_PYTORCH_WEIGHTS,
    pytorch_vocoder_ready,
    vocoder_onnx_ready,
)

HF_VOCODER_ONNX_URL = (
    f"https://huggingface.co/{HF_VOCODER_ONNX_REPO}/resolve/main/"
    f"{VOCODER_ONNX_FILENAME}"
)
HF_VOCODER_PYTORCH_FILES = {
    "config.yaml": (
        f"https://huggingface.co/{HF_VOCODER_PYTORCH_REPO}/resolve/main/config.yaml"
    ),
    "pytorch_model.bin": (
        f"https://huggingface.co/{HF_VOCODER_PYTORCH_REPO}/resolve/main/pytorch_model.bin"
    ),
}


def _download_url(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    print(f"         -> {dest}")
    urllib.request.urlretrieve(url, dest)


def download_vocoder_onnx() -> None:
    if vocoder_onnx_ready():
        print(f"[OK] ONNX vocoder already present: {VOCODER_ONNX}")
        return
    _download_url(HF_VOCODER_ONNX_URL, VOCODER_ONNX)
    if not VOCODER_ONNX.is_file() or VOCODER_ONNX.stat().st_size < 1_000_000:
        raise RuntimeError(f"Download failed or file too small: {VOCODER_ONNX}")


def download_vocoder_pytorch() -> None:
    if pytorch_vocoder_ready():
        print(f"[OK] PyTorch vocoder already present: {VOCODER_DIR}")
        return
    VOCODER_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in HF_VOCODER_PYTORCH_FILES.items():
        dest = VOCODER_DIR / name
        if dest.is_file():
            print(f"[OK] {name} already present")
            continue
        _download_url(url, dest)
    if not pytorch_vocoder_ready():
        raise RuntimeError(
            f"PyTorch vocoder download incomplete — expected under `{VOCODER_DIR}`"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download missing vocoder weights")
    parser.add_argument(
        "--onnx-vocoder",
        action="store_true",
        help="Download ONNX wetdog vocoder only (fallback path)",
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if args.onnx_vocoder:
        download_vocoder_onnx()
        print("\n[OK] ONNX vocoder ready.")
    else:
        download_vocoder_pytorch()
        print("\n[OK] PyTorch Vocos vocoder ready for default inference path.")
    print(f"     ONNX ZipVoice weights: {ROOT / 'models' / 'onnx'}")
    print(f"     Vocoder dir: {VOCODER_DIR}")


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
