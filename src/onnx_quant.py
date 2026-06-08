"""
ONNX quantization helpers shared between export (PyTorch GUI) and inference (ONNX GUI).

Inference modes:
  int8 — dynamic MatMul INT8 (text_encoder_int8.onnx, fm_decoder_int8.onnx)
  int4 — block weight-only INT4 via ORT MatMulNBits (text_encoder_int4.onnx, ...)
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Literal

logger = logging.getLogger("zipvoice_onnx")

QuantMode = Literal["int8", "int4"]

QUANT_MODE_CHOICES: tuple[str, ...] = ("int8", "int4")

QUANT_MANIFEST = "quantization.json"
ONNX_COMPONENTS = ("text_encoder", "fm_decoder")

# Unquantized export basenames — build input only, not an inference mode.
BASELINE_ONNX_SUFFIX = ""

_SUFFIX: dict[QuantMode, str] = {
    "int8": "_int8",
    "int4": "_int4",
}


def normalize_quant_mode(mode: str | None, *, use_int8: bool | None = None) -> QuantMode:
    """Resolve mode from string or legacy use_int8 flag."""
    if mode is not None:
        m = str(mode).strip().lower()
        if m in QUANT_MODE_CHOICES:
            return m  # type: ignore[return-value]
    if use_int8 is True:
        return "int8"
    return "int8"


def baseline_filename(component: str) -> str:
    """Unquantized ONNX export used as quantize source (not shipped for inference)."""
    return f"{component}{BASELINE_ONNX_SUFFIX}.onnx"


def component_filename(component: str, quant: QuantMode | str) -> str:
    mode = normalize_quant_mode(str(quant))
    suffix = _SUFFIX[mode]
    return f"{component}{suffix}.onnx"


def onnx_filenames(mode: QuantMode | str) -> tuple[str, str]:
    mode = normalize_quant_mode(str(mode))
    return component_filename("text_encoder", mode), component_filename("fm_decoder", mode)


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024) if path.is_file() else 0.0


def format_sizes(onnx_dir: Path, filenames: tuple[str, str]) -> str:
    parts = []
    for name in filenames:
        p = onnx_dir / name
        if p.is_file():
            parts.append(f"`{name}` {file_size_mb(p):.1f} MB")
        else:
            parts.append(f"`{name}` missing")
    return " · ".join(parts)


def required_onnx_files(mode: QuantMode | str) -> tuple[str, str, str, str]:
    te, fm = onnx_filenames(mode)
    return te, fm, "model.json", "tokens.txt"


def missing_onnx_files(onnx_dir: Path, mode: QuantMode | str) -> list[str]:
    """Return filenames missing for the given quant mode."""
    return [
        name
        for name in required_onnx_files(mode)
        if not (onnx_dir / name).is_file()
    ]


def onnx_ready_for_mode(onnx_dir: Path, mode: QuantMode | str) -> bool:
    return not missing_onnx_files(onnx_dir, mode)


def quant_readiness_hint(mode: QuantMode | str, missing: list[str]) -> str:
    """Short user-facing hint when a quant mode is not ready."""
    mode = normalize_quant_mode(str(mode))
    if not missing:
        return ""

    onnx_missing = [n for n in missing if n.endswith(".onnx")]
    if mode == "int4" and onnx_missing:
        return (
            "Export INT4 từ PyTorch GUI (ZipVoice-Vietnamese-GUI): "
            "Tab Export → chọn **int4** → Export ONNX, rồi copy `models/onnx/` sang repo ONNX-GUI. "
            "Hoặc chạy `quantize_onnx.py --mode int4` nếu đã có bản export gốc (text_encoder.onnx)."
        )
    if mode == "int8" and onnx_missing:
        return (
            "Thiếu bản INT8 — export từ PyTorch GUI (mode **int8**) "
            "hoặc `quantize_onnx.py --mode int8` từ bản export gốc."
        )
    return f"Thiếu: {', '.join(missing)}"


def write_quant_manifest(
    onnx_dir: Path,
    mode: QuantMode,
    *,
    created: list[str] | None = None,
) -> Path:
    payload: dict = {
        "mode": mode,
        "text_encoder": mode,
        "fm_decoder": mode,
        "filenames": dict(zip(ONNX_COMPONENTS, onnx_filenames(mode))),
    }
    if created:
        payload["created"] = created
    path = onnx_dir / QUANT_MANIFEST
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_quant_manifest(onnx_dir: Path) -> dict | None:
    path = onnx_dir / QUANT_MANIFEST
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


_FOLDER_SCAN_ORDER: tuple[QuantMode, ...] = ("int4", "int8")


def detect_quant_mode_from_folder(onnx_dir: Path) -> QuantMode | None:
    """Scan models/onnx/ for the smallest ready uniform quant set (int4 > int8)."""
    for mode in _FOLDER_SCAN_ORDER:
        if onnx_ready_for_mode(onnx_dir, mode):
            return mode
    return None


def resolve_default_quant_mode(
    onnx_dir: Path,
    *,
    env_quant: str | None = None,
    legacy_int8_env: str | None = None,
) -> tuple[QuantMode, str]:
    """
    Default inference quant mode when the caller does not pass an explicit mode.

    Priority:
      1. ZIPVOICE_ONNX_QUANT env (int8 or int4 only)
      2. quantization.json ``mode``
      3. folder scan (int4 > int8)
      4. legacy ZIPVOICE_ONNX_INT8 when explicitly set
      5. int8 (default)
    """
    if env_quant:
        return normalize_quant_mode(env_quant), "env"

    manifest = read_quant_manifest(onnx_dir)
    if manifest:
        manifest_mode = str(manifest.get("mode", "")).strip().lower()
        if manifest_mode in QUANT_MODE_CHOICES:
            return manifest_mode, "manifest"  # type: ignore[return-value]

    detected = detect_quant_mode_from_folder(onnx_dir)
    if detected:
        return detected, "folder"

    if legacy_int8_env is not None and legacy_int8_env.strip():
        use_int8 = legacy_int8_env.strip().lower() in {"1", "true", "yes"}
        return normalize_quant_mode(None, use_int8=use_int8), "legacy"

    return "int8", "default"


def resolve_inference_mode(onnx_dir: Path, requested: str | None = None) -> QuantMode:
    """Pick quant mode for inference: explicit request > manifest > folder scan > default."""
    if requested:
        return normalize_quant_mode(requested)
    mode, _source = resolve_default_quant_mode(onnx_dir)
    return mode


def _quantize_int8(src: Path, dst: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
    )


def _quantize_int4(src: Path, dst: Path) -> None:
    from onnxruntime.quantization import quant_utils
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        DefaultWeightOnlyQuantConfig,
        MatMulNBitsQuantizer,
    )

    quant_config = DefaultWeightOnlyQuantConfig(
        block_size=128,
        is_symmetric=True,
        accuracy_level=4,
        quant_format=quant_utils.QuantFormat.QOperator,
        op_types_to_quantize=("MatMul",),
        bits=4,
    )
    model = quant_utils.load_model_with_shape_infer(src)
    quant = MatMulNBitsQuantizer(model, algo_config=quant_config)
    quant.process()
    quant.model.save_model_to_file(str(dst), use_external_data_format=False)


def remove_baseline_exports(onnx_dir: Path) -> list[str]:
    """Delete unquantized export files after building int4/int8 variants."""
    removed: list[str] = []
    for comp in ONNX_COMPONENTS:
        name = baseline_filename(comp)
        path = onnx_dir / name
        if path.is_file():
            mb = file_size_mb(path)
            path.unlink()
            removed.append(name)
            logger.info("Removed baseline export: %s (%.1f MB)", name, mb)
    return removed


def quantize_component(src_baseline: Path, dst: Path, quant: QuantMode) -> None:
    """Create quantized variant from unquantized ONNX export."""
    if not src_baseline.is_file():
        raise FileNotFoundError(f"Missing baseline export: {src_baseline}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file():
        dst.unlink()

    mode = normalize_quant_mode(str(quant))
    if mode == "int8":
        _quantize_int8(src_baseline, dst)
    elif mode == "int4":
        _quantize_int4(src_baseline, dst)
    else:
        raise ValueError(f"Unknown quant mode: {quant}")

    logger.info(
        "Quantized %s -> %s (%s, %.1f MB)",
        src_baseline.name,
        dst.name,
        mode,
        file_size_mb(dst),
    )


def export_quant_variants(
    onnx_dir: Path,
    mode: QuantMode,
    *,
    baseline_source_dir: Path | None = None,
    keep_baseline: bool = True,
) -> list[str]:
    """
    After unquantized base export, build requested int4/int8 artifacts.

    *baseline_source_dir* — read baseline exports here (e.g. temp dir) instead of *onnx_dir*.
    *keep_baseline* — when False, remove text_encoder.onnx / fm_decoder.onnx after quant.
    """
    src_dir = baseline_source_dir or onnx_dir
    created: list[str] = []
    base = {c: src_dir / baseline_filename(c) for c in ONNX_COMPONENTS}

    mode = normalize_quant_mode(str(mode))
    for comp in ONNX_COMPONENTS:
        out = onnx_dir / component_filename(comp, mode)
        quantize_component(base[comp], out, mode)
        created.append(out.name)

    if keep_baseline:
        if baseline_source_dir is not None:
            for comp in ONNX_COMPONENTS:
                dst = onnx_dir / baseline_filename(comp)
                shutil.copy2(base[comp], dst)
    else:
        remove_baseline_exports(onnx_dir)

    write_quant_manifest(onnx_dir, mode, created=created)
    return created
