"""
Portable path configuration for ZipVoice Vietnamese ONNX TTS.
All inference weights are bundled under models/ (Git LFS) — no runtime HF download.
See models/THIRD_PARTY_LICENSES.md for license terms.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

ASSETS_DIR = ROOT / "assets"
OUTPUT_DIR = ROOT / "output"
REF_INFO_JSON = ASSETS_DIR / "ref_info.json"

FFMPEG_BIN = ROOT / "ffmpeg" / "bin"
SYSTEM_FFMPEG_BIN = Path(r"C:\ffmpeg\bin")

MODELS_DIR = ROOT / "models"
ONNX_DIR = MODELS_DIR / "onnx"
VOCODER_DIR = MODELS_DIR / "vocoder"

ONNX_MODEL_JSON = ONNX_DIR / "model.json"
ONNX_TOKENS = ONNX_DIR / "tokens.txt"

HF_VOCODER_REPO = "wetdog/vocos-mel-24khz-onnx"  # attribution only; bundled in repo
VOCODER_ONNX = VOCODER_DIR / "mel_spec_24khz.onnx"
VOCODER_ONNX_FILENAME = "mel_spec_24khz.onnx"


def ensure_ffmpeg_on_path() -> None:
    candidates = [FFMPEG_BIN, SYSTEM_FFMPEG_BIN]
    for bin_dir in candidates:
        ffmpeg_exe = bin_dir / "ffmpeg.exe"
        if ffmpeg_exe.exists():
            path = os.environ.get("PATH", "")
            bin_str = str(bin_dir)
            if bin_str not in path.split(os.pathsep):
                os.environ["PATH"] = bin_str + os.pathsep + path
            return


def is_force_cpu() -> bool:
    return os.environ.get("ZIPVOICE_FORCE_CPU", "").strip() in {"1", "true", "yes"}


def apply_cpu_env() -> None:
    if is_force_cpu():
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ.setdefault("ZIPVOICE_FORCE_CPU", "1")


def set_offline_env() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def use_onnx_int8() -> bool:
    return os.environ.get("ZIPVOICE_ONNX_INT8", "1").strip() in {"1", "true", "yes"}


def onnx_files(use_int8: bool | None = None) -> tuple[str, str]:
    int8 = use_onnx_int8() if use_int8 is None else use_int8
    if int8:
        return "text_encoder_int8.onnx", "fm_decoder_int8.onnx"
    return "text_encoder.onnx", "fm_decoder.onnx"


def onnx_ready(use_int8: bool | None = None) -> bool:
    te, fm = onnx_files(use_int8)
    return all(
        (ONNX_DIR / name).is_file()
        for name in (te, fm, "model.json", "tokens.txt")
    )


def vocoder_ready() -> bool:
    return VOCODER_ONNX.is_file()


def models_ready(use_int8: bool | None = None) -> bool:
    return onnx_ready(use_int8) and vocoder_ready()
