"""
ONNX Runtime execution provider selection (CPU / CUDA / DirectML).
"""
from __future__ import annotations

import logging
import os
import shutil
import site
import subprocess
import sys
from pathlib import Path
from typing import Any

import onnxruntime as ort

from config import VOCODER_RUNTIME_LABEL, is_force_cpu

logger = logging.getLogger("zipvoice_onnx_gui")

ProviderEntry = str | tuple[str, dict[str, Any]]

# ORT 1.26.x CUDA 12 EP — quick pre-check before loading provider DLL (Windows).
CUDA_REQUIRED_DLLS: tuple[str, ...] = (
    "cublasLt64_12.dll",
    "cudnn64_9.dll",
)

_cuda_loadable: bool | None = None
_cuda_warned: bool = False
_cuda_path_prepared: bool = False
_dll_directories_added: list[str] = []


def _ort_capi_dir() -> Path:
    return Path(ort.__file__).resolve().parent / "capi"


def _probe_cuda_provider_library() -> bool:
    """
    Try loading onnxruntime_providers_cuda.dll after PATH fix.

    More reliable than checking individual CUDA DLL names (e.g. cufft64_11.dll).
    """
    if sys.platform != "win32":
        return all(_dll_findable(name) for name in CUDA_REQUIRED_DLLS)

    dll_path = _ort_capi_dir() / "onnxruntime_providers_cuda.dll"
    if not dll_path.is_file():
        return False

    import ctypes

    prev_mode = ctypes.windll.kernel32.SetErrorMode(0x8003)  # SEM_FAIL | SEM_NOOPEN
    try:
        ctypes.WinDLL(str(dll_path))
        return True
    except OSError:
        return False
    finally:
        ctypes.windll.kernel32.SetErrorMode(prev_mode)


def _site_package_roots() -> list[Path]:
    roots: list[Path] = []
    for entry in site.getsitepackages():
        roots.append(Path(entry))
    user = site.getusersitepackages()
    if user:
        roots.append(Path(user))
    return roots


def _cuda_dll_search_dirs() -> list[Path]:
    """Candidate directories that may contain CUDA/cuDNN runtime DLLs."""
    dirs: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        key = str(path).lower()
        if key in seen:
            return
        if path.is_dir():
            seen.add(key)
            dirs.append(path)

    for root in _site_package_roots():
        nvidia_root = root / "nvidia"
        if nvidia_root.is_dir():
            for pkg_dir in sorted(nvidia_root.iterdir()):
                add(pkg_dir / "bin")

    toolkit = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if toolkit.is_dir():
        for ver_dir in sorted(toolkit.glob("v12.*"), reverse=True):
            add(ver_dir / "bin")

    cuda_path = os.environ.get("CUDA_PATH", "").strip()
    if cuda_path:
        add(Path(cuda_path) / "bin")

    cudnn_path = os.environ.get("CUDNN_PATH", "").strip()
    if cudnn_path:
        add(Path(cudnn_path) / "bin")

    return dirs


def ensure_cuda_runtime_on_path() -> list[str]:
    """
    Prepend discovered CUDA/cuDNN bin dirs to PATH (Windows-focused, safe on Linux).

    On Windows Python 3.8+, also registers DLL directories via os.add_dll_directory
    so onnxruntime_providers_cuda.dll can resolve NVIDIA pip wheel dependencies.

    Returns list of directories newly prepended to PATH.
    """
    global _cuda_path_prepared

    added: list[str] = []
    if not _cuda_path_prepared:
        path = os.environ.get("PATH", "")
        parts = path.split(os.pathsep) if path else []

        for bin_dir in reversed(_cuda_dll_search_dirs()):
            bin_str = str(bin_dir)
            if bin_str not in parts:
                parts.insert(0, bin_str)
                added.append(bin_str)

        if added:
            os.environ["PATH"] = os.pathsep.join(parts)

        if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
            capi = _ort_capi_dir()
            register_dirs = [capi, *_cuda_dll_search_dirs()]
            for directory in register_dirs:
                dir_str = str(directory)
                if not directory.is_dir() or dir_str in _dll_directories_added:
                    continue
                os.add_dll_directory(dir_str)
                _dll_directories_added.append(dir_str)

        _cuda_path_prepared = True

    return added


def _dll_findable(name: str) -> bool:
    if os.path.isfile(name):
        return True
    for directory in _cuda_dll_search_dirs():
        if (directory / name).is_file():
            return True
    found = os.environ.get("PATH", "")
    for directory in found.split(os.pathsep):
        if directory and (Path(directory) / name).is_file():
            return True
    return False


def is_cuda_execution_provider_loadable(*, warn: bool = True) -> bool:
    """
    True when CUDAExecutionProvider is listed and required CUDA 12 DLLs are loadable.

    Result is cached for the process. Logs at most one warning when CUDA is unavailable.
    """
    global _cuda_loadable, _cuda_warned

    if _cuda_loadable is not None:
        return _cuda_loadable

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        _cuda_loadable = False
        return False

    ensure_cuda_runtime_on_path()

    if not all(_dll_findable(name) for name in CUDA_REQUIRED_DLLS):
        missing = [dll for dll in CUDA_REQUIRED_DLLS if not _dll_findable(dll)]
        _cuda_loadable = False
        if warn and not _cuda_warned:
            _cuda_warned = True
            logger.warning(
                "CUDA không khả dụng — thiếu DLL: %s. "
                "Chạy install_gpu.bat (cài nvidia-cublas-cu12, nvidia-cudnn-cu12, "
                "nvidia-cuda-runtime-cu12, nvidia-cufft-cu12) hoặc cài CUDA Toolkit 12 + cuDNN 9. "
                "App sẽ dùng CPU khi bật GPU.",
                ", ".join(missing),
            )
        return False

    if not _probe_cuda_provider_library():
        _cuda_loadable = False
        if warn and not _cuda_warned:
            _cuda_warned = True
            logger.warning(
                "CUDA không khả dụng — không load được onnxruntime_providers_cuda.dll "
                "(thiếu thư viện CUDA/cuDNN hoặc driver NVIDIA). "
                "Chạy install_gpu.bat hoặc cài CUDA Toolkit 12 + cuDNN 9. "
                "App sẽ dùng CPU khi bật GPU.",
            )
        return False

    _cuda_loadable = True
    return True


def query_nvidia_gpus() -> list[dict[str, str | int]]:
    """Parse nvidia-smi CSV (index, name, memory.total) when available."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        proc = subprocess.run(
            [
                exe,
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if proc.returncode != 0:
            return []
        gpus: list[dict[str, str | int]] = []
        for line in proc.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                idx = int(parts[0])
            except ValueError:
                continue
            name = parts[1]
            mem = parts[2] if len(parts) > 2 else ""
            entry: dict[str, str | int] = {"index": idx, "name": name}
            if mem.isdigit():
                entry["memory_total_mib"] = int(mem)
            gpus.append(entry)
        return gpus
    except OSError:
        return []


def cuda_device_name(device_id: int = 0) -> str | None:
    """Human-readable NVIDIA GPU name for CUDA device_id, or None."""
    for gpu in query_nvidia_gpus():
        if gpu.get("index") == device_id:
            return str(gpu.get("name", ""))
    return None


def describe_accelerator(*, use_gpu: bool, device_id: int = 0) -> str:
    """Short accelerator description for logs / diagnostics."""
    if is_force_cpu():
        return "CPU (ZIPVOICE_FORCE_CPU=1)"
    if not use_gpu:
        return "CPU (GPU disabled)"

    _, label = resolve_ort_providers(use_gpu=True)
    if label == "CUDA":
        name = cuda_device_name(device_id)
        if name:
            return f"CUDA device_id={device_id} — {name}"
        return f"CUDA device_id={device_id} (NVIDIA — run scripts/diagnose_gpu.py)"
    if label == "DirectML":
        return (
            "DirectML (may use Intel iGPU on hybrid laptops — "
            "check Task Manager GPU 0, not NVIDIA)"
        )
    if label == "CPU (CUDA DLL thiếu)":
        return "CPU fallback (CUDA DLL missing)"
    return "CPU fallback (no GPU EP)"


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

    if "CUDAExecutionProvider" in available and is_cuda_execution_provider_loadable():
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
        logger.warning(
            "CUDA không khả dụng — dùng DirectML. Trên laptop Intel+NVIDIA, DirectML "
            "có thể chạy trên GPU tích hợp Intel (Task Manager GPU 0), không phải RTX. "
            "Chạy install_gpu.bat hoặc scripts/diagnose_gpu.py để kiểm tra."
        )

    providers.append("CPUExecutionProvider")
    if label.startswith("CPU"):
        if "CUDAExecutionProvider" in available and not is_cuda_execution_provider_loadable(
            warn=False
        ):
            label = "CPU (CUDA DLL thiếu)"
        else:
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
    if use_gpu and active == "CPUExecutionProvider" and label not in {
        "CPU",
        "CPU (fallback)",
        "CPU (CUDA DLL thiếu)",
    }:
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
    elif use_gpu and active != "CPUExecutionProvider":
        extra = ""
        if active == "CUDAExecutionProvider":
            name = cuda_device_name(0)
            extra = f", device_id=0" + (f" ({name})" if name else "")
        elif active == "DmlExecutionProvider":
            extra = " (có thể là Intel iGPU — xem Task Manager GPU 0)"
        logger.info(
            "%s: ORT provider=%s (requested %s%s)",
            component,
            active,
            label,
            extra,
        )

    return session


def provider_status_message(use_gpu: bool) -> str:
    """Short status for GUI / logs."""
    if is_force_cpu():
        return "CPU (ZIPVOICE_FORCE_CPU=1)"
    if not use_gpu:
        return "CPU"

    _, label = resolve_ort_providers(use_gpu=True)
    if label == "CPU (CUDA DLL thiếu)":
        return (
            "CPU — thiếu CUDA 12/cuDNN 9 DLL. "
            "Chạy install_gpu.bat hoặc tắt 'Dùng GPU' trong GUI."
        )
    if label == "CPU (fallback)":
        available = ort.get_available_providers()
        return (
            "CPU — GPU không khả dụng. Cài `onnxruntime-gpu` "
            f"(ORT {ort.__version__}, EPs: {', '.join(available)})"
        )
    return label


_PROVIDER_LABELS: dict[str, str] = {
    "CUDAExecutionProvider": "CUDA (GPU)",
    "CPUExecutionProvider": "CPU",
    "DmlExecutionProvider": "DirectML (GPU)",
}


def friendly_provider_name(ep: str) -> str:
    return _PROVIDER_LABELS.get(ep, ep)


def _cpu_fallback_suffix(use_gpu: bool, active_ep: str) -> str:
    """Vietnamese note when GPU was requested but ORT uses CPU."""
    if not use_gpu or active_ep != "CPUExecutionProvider":
        return ""
    if is_force_cpu():
        return " (ZIPVOICE_FORCE_CPU — fallback CPU)"
    _, label = resolve_ort_providers(use_gpu=True)
    if label == "CPU (CUDA DLL thiếu)":
        return " (CUDA không khả dụng — fallback CPU)"
    if label == "CPU (fallback)":
        return " (GPU không khả dụng — fallback CPU)"
    return " (fallback CPU)"


def predict_runtime_device_summary(use_gpu: bool) -> str:
    """Predicted ORT device before engine load (checkbox + CUDA probe)."""
    if is_force_cpu():
        base = "Dự kiến: CPU (ZIPVOICE_FORCE_CPU=1)"
    elif not use_gpu:
        base = "Dự kiến: CPU (tắt GPU trong cài đặt)"
    else:
        _, label = resolve_ort_providers(use_gpu=True)
        if label == "CUDA":
            name = cuda_device_name(0)
            gpu_note = f" ({name})" if name else ""
            base = (
                f"Dự kiến: CUDA device_id=0{gpu_note} — "
                "xem GPU NVIDIA trong Task Manager (thường GPU 1, không phải Intel GPU 0)"
            )
        elif label == "DirectML":
            base = (
                "Dự kiến: DirectML — có thể dùng Intel iGPU (Task Manager GPU 0); "
                "cài CUDA EP để dùng NVIDIA"
            )
        elif label == "CPU (CUDA DLL thiếu)":
            base = "Dự kiến: CPU (CUDA không khả dụng — thiếu DLL, fallback)"
        else:
            base = "Dự kiến: CPU (GPU không khả dụng — fallback)"
    return f"{base} · Vocoder: {VOCODER_RUNTIME_LABEL}"


def runtime_provider_log_lines(engine) -> list[str]:
    """Per-component active ORT provider lines for StatusLog."""
    use_gpu = bool(getattr(engine, "use_gpu", False))
    te = session_active_provider(engine.model.text_encoder)
    fm = session_active_provider(engine.model.fm_decoder)
    voc = session_active_provider(engine.vocoder)
    return [
        f"ORT text_encoder: {te}{_cpu_fallback_suffix(use_gpu, te)}",
        f"ORT fm_decoder: {fm}{_cpu_fallback_suffix(use_gpu, fm)}",
        f"ORT vocoder: {voc}{_cpu_fallback_suffix(use_gpu, voc)}",
    ]


def get_runtime_device_summary(engine) -> str:
    """Actual runtime device after OnnxTTSEngine load."""
    use_gpu = bool(getattr(engine, "use_gpu", False))
    te = session_active_provider(engine.model.text_encoder)
    fm = session_active_provider(engine.model.fm_decoder)
    voc = session_active_provider(engine.vocoder)
    onnx_eps = {"text_encoder": te, "fm_decoder": fm, "vocoder": voc}

    unique = set(onnx_eps.values())
    if len(unique) == 1:
        ep = next(iter(unique))
        label = friendly_provider_name(ep)
        suffix = _cpu_fallback_suffix(use_gpu, ep)
        gpu_detail = ""
        if ep == "CUDAExecutionProvider":
            name = cuda_device_name(0)
            gpu_detail = f" device_id=0" + (f" ({name})" if name else "")
        elif ep == "DmlExecutionProvider":
            gpu_detail = " (có thể Intel iGPU — Task Manager GPU 0)"
        components = ", ".join(onnx_eps.keys())
        return f"Thực tế: {label}{gpu_detail}{suffix} — {components}"
    parts = [
        f"{name}={friendly_provider_name(ep)}"
        for name, ep in onnx_eps.items()
    ]
    return f"Thực tế: {', '.join(parts)}"
