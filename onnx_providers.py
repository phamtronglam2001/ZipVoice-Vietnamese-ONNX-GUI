"""
ONNX Runtime execution provider selection (CPU / CUDA / DirectML).
"""
from __future__ import annotations

import logging
import sys
from typing import Any

import onnxruntime as ort

from config import is_force_cpu

logger = logging.getLogger("zipvoice_onnx_gui")

ProviderEntry = str | tuple[str, dict[str, Any]]


def resolve_ort_providers(use_gpu: bool) -> tuple[list[ProviderEntry], str]:
    """
    Build ORT provider list for InferenceSession.

    Priority when use_gpu: CUDA → DirectML (Windows) → CPU.
    Returns (providers, label) where label describes the preferred accelerator.
    """
    if is_force_cpu() or not use_gpu:
        return ["CPUExecutionProvider"], "CPU"

    available = set(ort.get_available_providers())
    providers: list[ProviderEntry] = []
    label = "CPU (fallback)"

    if "CUDAExecutionProvider" in available:
        providers.append(
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kSameAsRequested",
                },
            )
        )
        label = "CUDA"
    elif sys.platform == "win32" and "DmlExecutionProvider" in available:
        providers.append("DmlExecutionProvider")
        label = "DirectML"

    providers.append("CPUExecutionProvider")
    if label.startswith("CPU"):
        logger.warning(
            "GPU requested but no CUDA/DirectML EP in onnxruntime (%s). "
            "Install onnxruntime-gpu (NVIDIA) or use CPU package + DirectML build. "
            "Available: %s",
            ort.__version__,
            ", ".join(sorted(available)),
        )
    return providers, label


def session_active_provider(session: ort.InferenceSession) -> str:
    used = session.get_providers()
    return used[0] if used else "unknown"


def create_inference_session(
    model_path: str,
    *,
    sess_options: ort.SessionOptions,
    use_gpu: bool,
    quant_mode: str | None = None,
    component: str = "model",
) -> ort.InferenceSession:
    """
    Create InferenceSession with GPU preference and int4 CPU fallback when needed.
    """
    providers, label = resolve_ort_providers(use_gpu)
    quant = (quant_mode or "").lower()
    int4_like = "int4" in quant

    try:
        session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers,
        )
    except Exception as exc:
        if use_gpu and not is_force_cpu():
            logger.warning(
                "%s: GPU session failed (%s) — falling back to CPU.",
                component,
                exc,
            )
            session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
        else:
            raise

    active = session_active_provider(session)
    if use_gpu and active == "CPUExecutionProvider" and label != "CPU":
        if int4_like:
            logger.warning(
                "%s: INT4/MatMulNBits may be CPU-only on this ORT build — "
                "using CPUExecutionProvider (quant_mode=%s).",
                component,
                quant_mode,
            )
        else:
            logger.warning(
                "%s: requested %s but session uses CPUExecutionProvider.",
                component,
                label,
            )
    elif use_gpu:
        logger.info("%s: ORT provider=%s (requested %s)", component, active, label)

    return session


def provider_status_message(use_gpu: bool) -> str:
    """Short status for GUI / logs."""
    if is_force_cpu():
        return "CPU (ZIPVOICE_FORCE_CPU=1)"
    providers, label = resolve_ort_providers(use_gpu)
    if not use_gpu:
        return "CPU"
    available = ort.get_available_providers()
    if label == "CPU (fallback)":
        return (
            "CPU — GPU không khả dụng. Cài `onnxruntime-gpu` "
            f"(ORT {ort.__version__}, EPs: {', '.join(available)})"
        )
    return label
