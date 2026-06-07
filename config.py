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
    """Legacy env: ZIPVOICE_ONNX_INT8=1 → int8 mode."""
    return os.environ.get("ZIPVOICE_ONNX_INT8", "1").strip() in {"1", "true", "yes"}


def onnx_quant_mode() -> str:
    """
    Quant mode for inference.
    ZIPVOICE_ONNX_QUANT=fp32|fp16|int8|int4|mixed overrides ZIPVOICE_ONNX_INT8.
    Falls back to quantization.json in models/onnx/ when unset.
    """
    from onnx_quant import normalize_quant_mode, read_quant_manifest

    env_quant = os.environ.get("ZIPVOICE_ONNX_QUANT", "").strip().lower()
    if env_quant:
        return normalize_quant_mode(env_quant, use_int8=None)

    manifest = read_quant_manifest(ONNX_DIR)
    if manifest and manifest.get("mode"):
        return str(manifest["mode"])

    return normalize_quant_mode(None, use_int8=use_onnx_int8())


def onnx_files(
    quant_mode: str | None = None,
    *,
    use_int8: bool | None = None,
    mixed_config: dict | None = None,
) -> tuple[str, str]:
    from onnx_quant import DEFAULT_MIXED_CONFIG, onnx_filenames, normalize_quant_mode, read_quant_manifest

    if quant_mode is None and mixed_config is None:
        manifest = read_quant_manifest(ONNX_DIR)
        if manifest and manifest.get("mode") == "mixed":
            mixed_config = {
                "text_encoder": manifest.get("text_encoder", "int8"),
                "fm_decoder": manifest.get("fm_decoder", "fp32"),
            }
            quant_mode = "mixed"

    mode = normalize_quant_mode(
        quant_mode or onnx_quant_mode(),
        use_int8=use_int8,
    )
    mixed = mixed_config if mode == "mixed" else None
    if mode == "mixed" and mixed is None:
        mixed = DEFAULT_MIXED_CONFIG
    return onnx_filenames(mode, mixed)


def onnx_ready(
    quant_mode: str | None = None,
    *,
    use_int8: bool | None = None,
    mixed_config: dict | None = None,
) -> bool:
    from onnx_quant import DEFAULT_MIXED_CONFIG, normalize_quant_mode, onnx_ready_for_mode, read_quant_manifest

    if quant_mode is None and mixed_config is None:
        manifest = read_quant_manifest(ONNX_DIR)
        if manifest and manifest.get("mode") == "mixed":
            mixed_config = {
                "text_encoder": manifest.get("text_encoder", "int8"),
                "fm_decoder": manifest.get("fm_decoder", "fp32"),
            }
            quant_mode = "mixed"

    mode = normalize_quant_mode(
        quant_mode or onnx_quant_mode(),
        use_int8=use_int8,
    )
    mixed = mixed_config if mode == "mixed" else None
    if mode == "mixed" and mixed is None:
        mixed = DEFAULT_MIXED_CONFIG
    return onnx_ready_for_mode(ONNX_DIR, mode, mixed)


def vocoder_ready() -> bool:
    return VOCODER_ONNX.is_file()


def models_ready(
    quant_mode: str | None = None,
    *,
    use_int8: bool | None = None,
) -> bool:
    return onnx_ready(quant_mode, use_int8=use_int8) and vocoder_ready()
