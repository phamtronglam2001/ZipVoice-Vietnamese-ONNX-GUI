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

HF_VOCODER_ONNX_REPO = "wetdog/vocos-mel-24khz-onnx"  # attribution; ONNX fallback
HF_VOCODER_PYTORCH_REPO = "charactr/vocos-mel-24khz"  # default PyTorch vocoder weights
HF_VOCODER_REPO = HF_VOCODER_ONNX_REPO  # backward-compatible alias
VOCODER_ONNX = VOCODER_DIR / "mel_spec_24khz.onnx"
VOCODER_ONNX_FILENAME = "mel_spec_24khz.onnx"
VOCODER_PYTORCH_CONFIG = VOCODER_DIR / "config.yaml"
VOCODER_PYTORCH_WEIGHTS = VOCODER_DIR / "pytorch_model.bin"


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


def is_onnx_gpu_env() -> bool:
    """True when ZIPVOICE_ONNX_GPU=1 and CPU is not forced."""
    if is_force_cpu():
        return False
    return os.environ.get("ZIPVOICE_ONNX_GPU", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def apply_cpu_env() -> None:
    if is_force_cpu():
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ.setdefault("ZIPVOICE_FORCE_CPU", "1")


def set_offline_env() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def use_onnx_int8() -> bool:
    """Legacy env: ZIPVOICE_ONNX_INT8=1 → int8 mode (only when explicitly set)."""
    raw = os.environ.get("ZIPVOICE_ONNX_INT8", "").strip()
    if not raw:
        return False
    return raw.lower() in {"1", "true", "yes"}


def onnx_quant_mode() -> str:
    """
    Quant mode for inference.

    Priority: ZIPVOICE_ONNX_QUANT env → quantization.json → folder scan
    (int4 > int8 > fp16 > fp32) → legacy ZIPVOICE_ONNX_INT8 → fp32.
    """
    from onnx_quant import resolve_default_quant_mode

    env_quant = os.environ.get("ZIPVOICE_ONNX_QUANT", "").strip().lower() or None
    mode, _source = resolve_default_quant_mode(
        ONNX_DIR,
        env_quant=env_quant,
        legacy_int8_env=os.environ.get("ZIPVOICE_ONNX_INT8"),
    )
    return mode


def onnx_quant_mode_source() -> str:
    """Human-readable source for the active quant mode (env, manifest, folder, …)."""
    from onnx_quant import resolve_default_quant_mode

    env_quant = os.environ.get("ZIPVOICE_ONNX_QUANT", "").strip().lower() or None
    _mode, source = resolve_default_quant_mode(
        ONNX_DIR,
        env_quant=env_quant,
        legacy_int8_env=os.environ.get("ZIPVOICE_ONNX_INT8"),
    )
    return source


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


def _resolve_quant_context(
    quant_mode: str | None = None,
    *,
    use_int8: bool | None = None,
    mixed_config: dict | None = None,
) -> tuple[str, dict | None]:
    from onnx_quant import DEFAULT_MIXED_CONFIG, normalize_quant_mode, read_quant_manifest

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
    return mode, mixed


def onnx_ready(
    quant_mode: str | None = None,
    *,
    use_int8: bool | None = None,
    mixed_config: dict | None = None,
) -> bool:
    from onnx_quant import missing_onnx_files

    mode, mixed = _resolve_quant_context(
        quant_mode, use_int8=use_int8, mixed_config=mixed_config
    )
    return not missing_onnx_files(ONNX_DIR, mode, mixed)


def onnx_ready_report() -> str:
    """Markdown summary of quant variant readiness for GUI."""
    from onnx_quant import (
        DEFAULT_MIXED_CONFIG,
        QUANT_MODE_CHOICES,
        missing_onnx_files,
        quant_readiness_hint,
        read_quant_manifest,
    )

    active = onnx_quant_mode()
    source = onnx_quant_mode_source()
    source_labels = {
        "env": "ZIPVOICE_ONNX_QUANT",
        "manifest": "quantization.json",
        "folder": "auto-detect từ tên file",
        "legacy": "ZIPVOICE_ONNX_INT8",
        "default": "mặc định",
    }
    lines = ["### ONNX quant readiness (`models/onnx/`)"]
    lines.append(
        f"- **Active mode:** `{active}` *(via {source_labels.get(source, source)})*"
    )
    manifest = read_quant_manifest(ONNX_DIR)
    if manifest:
        lines.append(
            f"- **quantization.json:** mode=`{manifest.get('mode', '?')}`"
        )
    else:
        lines.append(
            "- **quantization.json:** *(chưa có — auto-detect int4 > int8 > fp32 từ tên file)*"
        )

    for mode in QUANT_MODE_CHOICES:
        mixed = DEFAULT_MIXED_CONFIG if mode == "mixed" else None
        missing = missing_onnx_files(ONNX_DIR, mode, mixed)
        ready = not missing
        lines.append(f"- ONNX **{mode}** ready: **{ready}**")
        if not ready:
            hint = quant_readiness_hint(mode, missing)
            if hint:
                lines.append(f"  - {hint}")
            if missing:
                lines.append(f"  - Missing: `{', '.join(missing)}`")

    return "\n".join(lines)


def vocoder_onnx_ready() -> bool:
    return VOCODER_ONNX.is_file()


def vocoder_ready() -> bool:
    """Backward-compatible alias for ONNX vocoder file."""
    return vocoder_onnx_ready()


def pytorch_vocoder_ready() -> bool:
    if not VOCODER_PYTORCH_CONFIG.is_file():
        return False
    return VOCODER_PYTORCH_WEIGHTS.is_file() or (VOCODER_DIR / "model.safetensors").is_file()


def models_ready(
    quant_mode: str | None = None,
    *,
    use_int8: bool | None = None,
    use_pytorch_vocoder: bool = True,
) -> bool:
    onnx_ok = onnx_ready(quant_mode, use_int8=use_int8)
    if use_pytorch_vocoder:
        return onnx_ok and pytorch_vocoder_ready()
    return onnx_ok and vocoder_onnx_ready()
