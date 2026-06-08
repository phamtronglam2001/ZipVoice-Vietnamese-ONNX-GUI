"""
Verify bundled model weights (local files only — no network download).

Vocoder: models/vocoder/mel_spec_24khz.onnx (100 mel, export ZipVoice-Vietnamese-GUI).
ZipVoice ONNX: models/onnx/ (int4/int8 per onnx_quant_mode). Git LFS stubs count as missing.
"""
from __future__ import annotations

import argparse
import sys

from config import (
    MODELS_DIR,
    ROOT,
    VOCODER_DIR,
    VOCODER_ONNX,
    models_ready,
    models_ready_report,
    vocoder_deploy_instructions,
    vocoder_onnx_ready,
)


def ensure_vocoder_onnx() -> None:
    if vocoder_onnx_ready():
        print(f"[OK] ONNX vocoder present: {VOCODER_ONNX}")
        return
    print("[ERROR] Missing ONNX vocoder.", file=sys.stderr)
    print(vocoder_deploy_instructions(), file=sys.stderr)
    raise FileNotFoundError(f"Missing {VOCODER_ONNX}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify bundled ZipVoice ONNX + local 100-mel vocoder "
            "(no download — export from ZipVoice-Vietnamese-GUI)"
        )
    )
    parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    VOCODER_DIR.mkdir(parents=True, exist_ok=True)

    missing = models_ready_report()
    if missing:
        print("[ERROR] Missing or invalid model files:", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        if not vocoder_onnx_ready():
            print(file=sys.stderr)
            print(vocoder_deploy_instructions(), file=sys.stderr)
        raise FileNotFoundError("Bundled models not ready")

    ensure_vocoder_onnx()
    print("\n[OK] Bundled models ready.")
    print(f"     ZipVoice ONNX: {ROOT / 'models' / 'onnx'}")
    print(f"     Vocoder:       {VOCODER_ONNX}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
