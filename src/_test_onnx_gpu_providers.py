"""Smoke test: ORT provider resolution (CPU-only env OK)."""
from __future__ import annotations

import onnxruntime as ort

from onnx_providers import (
    ensure_cuda_runtime_on_path,
    is_cuda_execution_provider_loadable,
    provider_status_message,
    resolve_ort_providers,
)


def main() -> None:
    ensure_cuda_runtime_on_path()
    print("onnxruntime", ort.__version__)
    print("available EPs:", ort.get_available_providers())
    print("CUDA loadable:", is_cuda_execution_provider_loadable())
    for flag in (False, True):
        providers, label = resolve_ort_providers(flag)
        print(f"use_gpu={flag} -> label={label!r}, providers={providers!r}")
        print(f"  status: {provider_status_message(flag)}")
    print("OK")


if __name__ == "__main__":
    main()
