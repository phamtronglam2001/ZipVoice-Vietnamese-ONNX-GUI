"""
ONNX Runtime SessionOptions builder — graph optimization, threading, profiling.
"""
from __future__ import annotations

import os

import onnxruntime as ort

from config import onnx_num_threads


def build_session_options(
    *,
    num_threads: int | None = None,
    enable_profiling: bool = False,
) -> ort.SessionOptions:
    """Build ORT SessionOptions with performance-oriented defaults."""
    threads = num_threads if num_threads is not None else onnx_num_threads()
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = threads
    opts.intra_op_num_threads = threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_mem_pattern = True
    opts.enable_cpu_mem_arena = True

    if enable_profiling or _env_flag("ZIPVOICE_ORT_PROFILE"):
        opts.enable_profiling = True

    return opts


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}
