"""
Portable path configuration for ZipVoice Vietnamese ONNX TTS.
All inference weights are bundled under models/ (Git LFS) — no runtime HF download.
See models/THIRD_PARTY_LICENSES.md for license terms.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ASSETS_DIR = ROOT / "assets"
OUTPUT_DIR = ROOT / "output"
REF_INFO_JSON = ASSETS_DIR / "ref_info.json"
# Thư mục con trong assets/: mỗi giọng = 1 file audio + 1 file .txt cùng tên (stem).
REF_AUDIO_PAIR_DIRS = (
    ASSETS_DIR / "sample_audio",
)
REF_AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".aac")

FFMPEG_BIN = ROOT / "ffmpeg" / "bin"
SYSTEM_FFMPEG_BIN = Path(r"C:\ffmpeg\bin")

DEFAULT_MODELS_DIR = ROOT / "models"

MODELS_DIR: Path = DEFAULT_MODELS_DIR
ONNX_DIR: Path = MODELS_DIR / "onnx"
VOCODER_DIR: Path = MODELS_DIR / "vocoder"

ONNX_MODEL_JSON: Path = ONNX_DIR / "model.json"
ONNX_TOKENS: Path = ONNX_DIR / "tokens.txt"

# Vocoder ONNX: bundled local file only — export from ZipVoice-Vietnamese-GUI
# (vocos_export.py → mel_spec_24khz.onnx, 100 mel, synced with ZipVoice feat_dim).
# ISTFT via librosa at inference (vocos_istft.py).
HF_VOCODER_WEIGHTS_REPO = "charactr/vocos-mel-24khz"  # attribution only (100-mel PyTorch weights)
VOCODER_ONNX = VOCODER_DIR / "mel_spec_24khz.onnx"
VOCODER_ONNX_FILENAME = "mel_spec_24khz.onnx"
VOCODER_MEL_CHANNELS = 100
VOCODER_RUNTIME_LABEL = "Vocos ONNX (100 mel) + librosa ISTFT"
# Backward-compatible alias (attribution only — never auto-download).
HF_VOCODER_REPO = HF_VOCODER_WEIGHTS_REPO


def _sync_model_paths(models_dir: Path) -> None:
    """Update module-level model paths (call after resolving a models root)."""
    global MODELS_DIR, ONNX_DIR, VOCODER_DIR, ONNX_MODEL_JSON, ONNX_TOKENS, VOCODER_ONNX
    resolved = models_dir.resolve()
    MODELS_DIR = resolved
    ONNX_DIR = resolved / "onnx"
    VOCODER_DIR = resolved / "vocoder"
    ONNX_MODEL_JSON = ONNX_DIR / "model.json"
    ONNX_TOKENS = ONNX_DIR / "tokens.txt"
    VOCODER_ONNX = VOCODER_DIR / VOCODER_ONNX_FILENAME


def resolve_models_dir(path: str | Path | None = None) -> Path:
    """Resolve models root: relative paths are under project ROOT; empty → ./models/."""
    if path is None:
        return DEFAULT_MODELS_DIR.resolve()
    raw = str(path).strip()
    if not raw:
        return DEFAULT_MODELS_DIR.resolve()
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = (ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def set_models_dir(path: str | Path | None = None) -> Path:
    """Point runtime inference at *path* (onnx/ + vocoder/ subfolders). Returns resolved path."""
    resolved = resolve_models_dir(path)
    _sync_model_paths(resolved)
    return resolved


def models_dir_display(path: Path | None = None) -> str:
    """Human-friendly models path for GUI (relative to ROOT when possible)."""
    target = (path or MODELS_DIR).resolve()
    try:
        return str(target.relative_to(ROOT.resolve()))
    except ValueError:
        return str(target)


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
    (int4 > int8) → legacy ZIPVOICE_ONNX_INT8 → int8 (default).
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
) -> tuple[str, str]:
    from onnx_quant import normalize_quant_mode, onnx_filenames

    mode = normalize_quant_mode(
        quant_mode or onnx_quant_mode(),
        use_int8=use_int8,
    )
    return onnx_filenames(mode)


def _resolve_quant_mode(
    quant_mode: str | None = None,
    *,
    use_int8: bool | None = None,
) -> str:
    from onnx_quant import normalize_quant_mode

    return normalize_quant_mode(
        quant_mode or onnx_quant_mode(),
        use_int8=use_int8,
    )


def onnx_ready(
    quant_mode: str | None = None,
    *,
    use_int8: bool | None = None,
) -> bool:
    mode = _resolve_quant_mode(quant_mode, use_int8=use_int8)
    return not _missing_bundled_onnx_files(ONNX_DIR, mode)


def onnx_ready_report() -> str:
    """Markdown summary of quant variant readiness for GUI."""
    from onnx_quant import (
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
    lines = [f"### ONNX quant readiness (`{models_dir_display()}/onnx/`)"]
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
            "- **quantization.json:** *(chưa có — auto-detect int4 > int8 từ tên file)*"
        )

    for mode in QUANT_MODE_CHOICES:
        missing = missing_onnx_files(ONNX_DIR, mode)
        ready = not missing
        lines.append(f"- ONNX **{mode}** ready: **{ready}**")
        if not ready:
            hint = quant_readiness_hint(mode, missing)
            if hint:
                lines.append(f"  - {hint}")
            if missing:
                lines.append(f"  - Missing: `{', '.join(missing)}`")

    return "\n".join(lines)


def _env_int(name: str, default: int, *, minimum: int = 1, maximum: int | None = None) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        value = default
    else:
        try:
            value = int(raw)
        except ValueError:
            value = default
    value = max(minimum, value)
    if maximum is not None:
        value = min(value, maximum)
    return value


def cpu_max_parallel_workers() -> int:
    """Max parallel chunk workers on CPU (ProcessPool)."""
    cap = os.cpu_count() or 4
    return _env_int("ZIPVOICE_CPU_MAX_WORKERS", min(8, cap), minimum=1, maximum=cap)


def gpu_max_parallel_workers() -> int:
    """
    Max parallel chunk workers when each process loads ONNX on GPU.

    Default 1 — each ProcessPool worker creates its own CUDA sessions (VRAM × N).
    Override via ZIPVOICE_GPU_MAX_WORKERS only if you have spare VRAM (risk of OOM/crash).
    """
    cap = os.cpu_count() or 4
    return _env_int("ZIPVOICE_GPU_MAX_WORKERS", 1, minimum=1, maximum=cap)


def onnx_num_threads() -> int:
    """
    ORT intra/inter op threads per InferenceSession.

    Default: physical core count (capped at 16). Override: ZIPVOICE_ONNX_THREADS.
    """
    cap = os.cpu_count() or 4
    default = min(cap, 16)
    return _env_int("ZIPVOICE_ONNX_THREADS", default, minimum=1, maximum=cap)


def inference_batch_size() -> int:
    """
    Chunks per ONNX batch on GPU sequential path.

    Default 1. Override: ZIPVOICE_INFERENCE_BATCH (2–8 typical when VRAM allows).
    """
    return _env_int("ZIPVOICE_INFERENCE_BATCH", 1, minimum=1, maximum=32)


def ode_solver_default() -> str:
    """ODE integrator: euler | heun | midpoint. Override: ZIPVOICE_ODE_SOLVER."""
    raw = os.environ.get("ZIPVOICE_ODE_SOLVER", "euler").strip().lower()
    if raw in {"euler", "heun", "midpoint"}:
        return raw
    return "euler"


def pipeline_overlap_enabled() -> bool:
    """Pre-tokenize next chunk on CPU while GPU runs fm_decoder."""
    raw = os.environ.get("ZIPVOICE_PIPELINE_OVERLAP", "1").strip().lower()
    return raw not in {"0", "false", "no"}


def _is_lfs_pointer(path: Path) -> bool:
    """True when *path* is a Git LFS stub, not real model bytes."""
    if not path.is_file():
        return False
    if path.stat().st_size > 512:
        return False
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")[:200]
    except OSError:
        return False
    return head.startswith("version https://git-lfs.github.com")


def _model_file_ready(path: Path) -> bool:
    return path.is_file() and not _is_lfs_pointer(path)


def _missing_bundled_onnx_files(onnx_dir: Path, mode: str) -> list[str]:
    """Return missing ONNX filenames; Git LFS pointer stubs count as missing."""
    from onnx_quant import required_onnx_files

    return [
        name
        for name in required_onnx_files(mode)
        if not _model_file_ready(onnx_dir / name)
    ]


def vocoder_onnx_ready() -> bool:
    return _model_file_ready(VOCODER_ONNX)


def vocoder_deploy_instructions() -> str:
    """How to obtain mel_spec_24khz.onnx for deployment."""
    return (
        f"Đặt vocoder ONNX 100 mel tại `{models_dir_display()}/vocoder/{VOCODER_ONNX_FILENAME}`.\n"
        "  • Export từ ZipVoice-Vietnamese-GUI: Tab Export → bật **Export Vocos ONNX**\n"
        "  • Hoặc copy file đã export từ repo PyTorch sang repo ONNX-GUI\n"
        "  • Hoặc chạy `git lfs pull` nếu file bundled trong repo\n"
        f"  • Weights gốc (attribution): {HF_VOCODER_WEIGHTS_REPO}"
    )


def models_ready_report() -> list[str]:
    """Human-readable list of missing or invalid bundled model files."""
    missing: list[str] = []
    mode = onnx_quant_mode()
    for name in _missing_bundled_onnx_files(ONNX_DIR, mode):
        path = ONNX_DIR / name
        if _is_lfs_pointer(path):
            missing.append(
                f"{models_dir_display()}/onnx/{name} (Git LFS pointer — chạy: git lfs pull)"
            )
        else:
            missing.append(f"{models_dir_display()}/onnx/{name} (quant mode `{mode}`)")

    if not vocoder_onnx_ready():
        voc_path = VOCODER_ONNX
        if _is_lfs_pointer(voc_path):
            missing.append(
                f"{models_dir_display()}/vocoder/{VOCODER_ONNX_FILENAME} "
                "(Git LFS pointer — chạy: git lfs pull)"
            )
        else:
            missing.append(
                f"{models_dir_display()}/vocoder/{VOCODER_ONNX_FILENAME} "
                "(100 mel — export ZipVoice-Vietnamese-GUI hoặc git lfs pull)"
            )
    return missing


def vocoder_ready() -> bool:
    """Backward-compatible alias for ONNX vocoder file."""
    return vocoder_onnx_ready()


def models_ready(
    quant_mode: str | None = None,
    *,
    use_int8: bool | None = None,
) -> bool:
    return onnx_ready(quant_mode, use_int8=use_int8) and vocoder_onnx_ready()


_sync_model_paths(DEFAULT_MODELS_DIR)
