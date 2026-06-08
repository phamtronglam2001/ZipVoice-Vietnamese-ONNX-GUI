"""
GPU / ONNX Runtime diagnostics for ZipVoice ONNX GUI.

Run from repo root:
  .venv\\Scripts\\python.exe scripts\\diagnose_gpu.py

Prints ORT execution providers, CUDA probe, NVIDIA GPU list (nvidia-smi),
and recommendations for hybrid Intel+NVIDIA laptops (Task Manager GPU 0 vs 1).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import onnxruntime as ort

from onnx_providers import (
    describe_accelerator,
    ensure_cuda_runtime_on_path,
    is_cuda_execution_provider_loadable,
    provider_status_message,
    query_nvidia_gpus,
    resolve_ort_providers,
)


def _section(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def _run_nvidia_smi() -> str | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        proc = subprocess.run(
            [
                exe,
                "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if proc.returncode != 0:
            return proc.stderr.strip() or f"nvidia-smi exit {proc.returncode}"
        return proc.stdout.strip()
    except OSError as exc:
        return str(exc)


def _env_flags() -> None:
    keys = (
        "ZIPVOICE_FORCE_CPU",
        "ZIPVOICE_ONNX_GPU",
        "CUDA_VISIBLE_DEVICES",
        "ZIPVOICE_GPU_MAX_WORKERS",
        "ZIPVOICE_CPU_MAX_WORKERS",
    )
    for key in keys:
        val = os.environ.get(key)
        print(f"  {key}={val!r}" if val is not None else f"  {key}=(not set)")


def _recommendations(
    *,
    cuda_loadable: bool,
    providers_label: str,
    has_dml_only: bool,
) -> None:
    print()
    print("Recommendations:")
    if has_dml_only:
        print(
            "  - Only DirectML EP available: ONNX may run on Intel iGPU (GPU 0 in Task Manager),"
        )
        print("    not NVIDIA. Install onnxruntime-gpu + CUDA DLLs (install_gpu.bat).")
    if not cuda_loadable and "CUDA" in ort.get_available_providers():
        print(
            "  - CUDA EP listed but DLL probe failed: run install_gpu.bat or install"
        )
        print("    nvidia-cublas-cu12, nvidia-cudnn-cu12, nvidia-cuda-runtime-cu12.")
    if cuda_loadable:
        print(
            "  - CUDA device_id=0 maps to the first NVIDIA GPU in nvidia-smi (not Intel iGPU)."
        )
        print(
            "  - In Task Manager > Performance, watch **GPU 1 (NVIDIA)** for 3D/CUDA usage;"
        )
        print("    Intel GPU 0 may show activity from desktop compositor / DirectML, not ORT CUDA.")
    print(
        "  - Crash 0xc0000005 with free VRAM is often multi-process GPU (parallel workers),"
    )
    print("    not VRAM exhaustion. Set parallel workers = 1 on GPU (default).")
    print(
        "  - Force NVIDIA for graphics: Windows Settings > System > Display > Graphics"
    )
    print("    > add python.exe > High performance (NVIDIA).")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() == "":
        print("  - run_gpu.bat sets CUDA_VISIBLE_DEVICES=0 (first NVIDIA GPU).")
    print(f"  - Active ORT preference (use_gpu=True): {providers_label}")


def main() -> int:
    ensure_cuda_runtime_on_path()

    _section("Environment")
    _env_flags()

    _section("ONNX Runtime")
    print(f"  Version: {ort.__version__}")
    eps = ort.get_available_providers()
    print(f"  Available EPs: {', '.join(eps)}")

    cuda_loadable = is_cuda_execution_provider_loadable()
    print(f"  CUDA loadable (DLL probe): {cuda_loadable}")

    providers, label = resolve_ort_providers(True)
    print(f"  resolve_ort_providers(use_gpu=True): label={label!r}")
    for entry in providers:
        print(f"    - {entry!r}")

    print(f"  provider_status_message(True): {provider_status_message(True)}")

    _section("NVIDIA GPUs (nvidia-smi)")
    smi = _run_nvidia_smi()
    if smi is None:
        print("  nvidia-smi not found — no NVIDIA driver in PATH or no discrete GPU.")
    else:
        for line in smi.splitlines():
            print(f"  {line}")

    gpus = query_nvidia_gpus()
    if gpus:
        for g in gpus:
            print(
                f"  CUDA device_id={g['index']}: {g['name']} "
                f"({g.get('memory_total_mib', '?')} MiB total)"
            )
    elif smi:
        print("  (Could not parse GPU list — see nvidia-smi output above.)")

    _section("Accelerator summary")
    print(f"  {describe_accelerator(use_gpu=True)}")

    has_dml = "DmlExecutionProvider" in eps
    has_cuda_ep = "CUDAExecutionProvider" in eps
    has_dml_only = has_dml and not (has_cuda_ep and cuda_loadable)

    _section("Notes: Task Manager vs CUDA")
    print(
        "  - Intel iGPU cannot run CUDAExecutionProvider. If CUDA fails, ORT falls back"
    )
    print("    to CPU (not Intel GPU for CUDA EP).")
    print(
        "  - DirectML (DmlExecutionProvider) CAN use Intel iGPU — only if CUDA is unavailable."
    )
    print(
        "  - Hybrid laptops: Task Manager often labels Intel as GPU 0, NVIDIA as GPU 1."
    )
    print(
        "  - CUDA device_id=0 always means first NVIDIA GPU in CUDA's device list."
    )
    print(
        "  - Prompt mel (librosa via vocos_fbank.py) runs on CPU; not moved to GPU."
    )

    _recommendations(
        cuda_loadable=cuda_loadable,
        providers_label=label,
        has_dml_only=has_dml_only,
    )

    print()
    return 0 if cuda_loadable or label == "DirectML" else 1


if __name__ == "__main__":
    raise SystemExit(main())
