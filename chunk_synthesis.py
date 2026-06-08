"""
Parallel TTS chunk synthesis — one ONNX Runtime session per worker process.
ORT InferenceSession is not thread-safe; ProcessPoolExecutor avoids sharing sessions.
"""
from __future__ import annotations

import gc
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

import numpy as np

from config import cpu_max_parallel_workers, gpu_max_parallel_workers, is_force_cpu
from status_log import StatusLog
from audio.ref_audio import preprocess_ref_audio_text
from text.chunking import TtsChunk
from text.pipeline import prepare_for_tts, prepare_tts_text

logger = logging.getLogger("zipvoice_onnx_gui")

_worker_engine = None


def max_parallel_workers(use_gpu: bool = False) -> int:
    if use_gpu:
        # Each worker loads full ONNX on GPU — VRAM scales with worker count.
        return gpu_max_parallel_workers()
    return cpu_max_parallel_workers()


def ui_parallel_workers_max(use_gpu: bool = False) -> int:
    """Slider/spinbox maximum; Gradio requires minimum strictly less than maximum."""
    return max(2, max_parallel_workers(use_gpu=use_gpu))


def clamp_parallel_workers(workers: int, use_gpu: bool = False) -> int:
    return min(max(1, int(workers)), max_parallel_workers(use_gpu))


def parallel_workers_clamp_message(
    requested: int, effective: int, use_gpu: bool = False
) -> str:
    """User-facing warning when parallel workers are reduced for safety."""
    cap = max_parallel_workers(use_gpu=use_gpu)
    if use_gpu:
        return (
            f"GPU: giảm workers {requested} → {effective} "
            f"(tối đa {cap} — mỗi worker load ONNX riêng trên GPU, dễ hết VRAM/crash). "
            "Khuyến nghị 1 worker trên GPU."
        )
    return (
        f"Giảm workers {requested} → {effective} "
        f"(CPU tối đa {cap})."
    )


def _worker_init(quant_mode: str, use_gpu: bool) -> None:
    global _worker_engine
    os.environ["ZIPVOICE_ONNX_GPU"] = "1" if use_gpu else "0"
    if use_gpu and not is_force_cpu():
        os.environ.pop("ZIPVOICE_FORCE_CPU", None)
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    from onnx_engine import OnnxTTSEngine

    OnnxTTSEngine._instance = None
    _worker_engine = OnnxTTSEngine(
        quant_mode=quant_mode,
        use_gpu=use_gpu,
    )


def _worker_synthesize_chunk(task: tuple) -> tuple[int, np.ndarray, str | None]:
    (
        index,
        normalized,
        prompt_normalized,
        ref_audio,
        speed,
        num_step,
        guidance_scale,
        t_shift,
        ode_seed,
        use_fixed_seed,
    ) = task
    try:
        wav = _worker_engine.generate(
            prompt_text=prompt_normalized,
            prompt_wav=ref_audio,
            text=normalized,
            speed=speed,
            num_step=int(num_step),
            guidance_scale=float(guidance_scale),
            t_shift=float(t_shift),
            ode_seed=int(ode_seed),
            use_fixed_seed=bool(use_fixed_seed),
            chunk_index=index,
        )
        return index, wav, None
    except Exception as exc:
        return index, np.array([], dtype=np.float32), str(exc)


@dataclass
class ChunkSynthResult:
    wave_parts: list[np.ndarray]
    normalized_preview: str
    elapsed_s: float
    effective_workers: int
    parallel_used: bool
    chunks_synthesized: int


ProgressFn = Callable[[float, str], None]


def _noop_progress(
    _frac: float | tuple[int, int | None] | None,
    desc: str | None = None,
    **_: object,
) -> None:
    pass


def synthesize_tts_chunks(
    *,
    engine,
    tts_chunks: list[TtsChunk],
    wave_parts: list[np.ndarray],
    norm_pipeline: list[str],
    tts_input_mode: str,
    ref_audio_path: str,
    ref_text: str,
    speed: float,
    num_step: int,
    guidance_scale: float,
    t_shift: float,
    onnx_quant_mode: str,
    parallel_workers: int,
    use_onnx_gpu: bool = False,
    ode_seed: int = 42,
    use_fixed_seed: bool = True,
    progress: ProgressFn | None = None,
    status_log: StatusLog | None = None,
) -> ChunkSynthResult:
    """
    Synthesize all chunks from scratch with current parameters.
    Falls back to sequential when workers==1 or only one pending chunk.
    """
    if progress is None:
        progress = _noop_progress
    use_gpu = bool(use_onnx_gpu) and not is_force_cpu()
    requested_workers = int(parallel_workers)
    workers = clamp_parallel_workers(requested_workers, use_gpu=use_gpu)
    if requested_workers != workers:
        clamp_msg = parallel_workers_clamp_message(
            requested_workers, workers, use_gpu=use_gpu
        )
        logger.warning(clamp_msg)
        if status_log is not None:
            status_log.warn(clamp_msg)

    def _ref_show_info(msg: str) -> None:
        logger.info(msg)
        if status_log is not None:
            status_log.info(f"Ref audio: {msg}")

    if status_log is not None:
        status_log.stage_begin("Tiền xử lý giọng mẫu")
        status_log.info(f"ref_audio_path={ref_audio_path}")
        status_log.info(f"ref_text_len={len(ref_text.strip())} ký tự")

    ref_audio, resolved_ref_text = preprocess_ref_audio_text(
        ref_audio_path,
        ref_text,
        show_info=_ref_show_info,
    )
    prompt_normalized = prepare_tts_text(resolved_ref_text, norm_pipeline)

    if status_log is not None:
        status_log.info(f"prompt_normalized_len={len(prompt_normalized)} ký tự")
        status_log.stage_end("Tiền xử lý giọng mẫu", f"temp_wav={ref_audio}")

    pending: list[tuple[int, str]] = []
    normalized_preview = ""
    total = len(tts_chunks)

    for i, tts_chunk in enumerate(tts_chunks):
        normalized = prepare_for_tts(
            tts_chunk.text, norm_pipeline, tts_input_mode, already_normalized=True
        )
        if not normalized.strip():
            msg = f"Bỏ qua chunk {i + 1}/{total}: rỗng sau chuẩn hóa"
            logger.warning("skip chunk %d/%d: empty after normalize", i + 1, total)
            if status_log is not None:
                status_log.warn(msg)
            wave_parts[i] = np.array([], dtype=np.float32)
            continue
        if not normalized_preview:
            normalized_preview = normalized[:500] + (
                "…" if len(normalized) > 500 else ""
            )
        pending.append((i, normalized))

    if not pending:
        return ChunkSynthResult(
            wave_parts=wave_parts,
            normalized_preview=normalized_preview,
            elapsed_s=0.0,
            effective_workers=1,
            parallel_used=False,
            chunks_synthesized=0,
        )

    if status_log is not None:
        status_log.stage_begin("Tổng hợp chunk")
    t0 = time.perf_counter()
    use_parallel = workers > 1 and len(pending) > 1

    if use_parallel:
        logger.info(
            "parallel synthesis | workers=%d | pending=%d/%d chunks",
            workers,
            len(pending),
            total,
        )
        if status_log is not None:
            status_log.info(
                f"Chế độ song song: {workers} workers, "
                f"{len(pending)}/{total} chunk cần tổng hợp"
            )
        _run_parallel(
            pending=pending,
            wave_parts=wave_parts,
            prompt_normalized=prompt_normalized,
            ref_audio=ref_audio,
            speed=speed,
            num_step=num_step,
            guidance_scale=guidance_scale,
            t_shift=t_shift,
            onnx_quant_mode=onnx_quant_mode,
            use_onnx_gpu=use_gpu,
            ode_seed=ode_seed,
            use_fixed_seed=use_fixed_seed,
            workers=workers,
            total=total,
            progress=progress,
            status_log=status_log,
        )
    else:
        logger.info(
            "sequential synthesis | pending=%d/%d chunks",
            len(pending),
            total,
        )
        if status_log is not None:
            status_log.info(
                f"Chế độ tuần tự: {len(pending)}/{total} chunk cần tổng hợp"
            )
        _run_sequential(
            pending=pending,
            wave_parts=wave_parts,
            prompt_normalized=prompt_normalized,
            ref_audio=ref_audio,
            speed=speed,
            num_step=num_step,
            guidance_scale=guidance_scale,
            t_shift=t_shift,
            engine=engine,
            ode_seed=ode_seed,
            use_fixed_seed=use_fixed_seed,
            total=total,
            progress=progress,
            status_log=status_log,
        )

    elapsed = time.perf_counter() - t0
    mode = "parallel" if use_parallel else "sequential"
    logger.info(
        "chunk synthesis done | mode=%s | workers=%d | elapsed=%.1fs | chunks=%d",
        mode,
        workers if use_parallel else 1,
        elapsed,
        len(pending),
    )
    if status_log is not None:
        status_log.stage_end(
            "Tổng hợp chunk",
            f"mode={mode}, workers={workers if use_parallel else 1}, "
            f"chunks={len(pending)}",
        )
    return ChunkSynthResult(
        wave_parts=wave_parts,
        normalized_preview=normalized_preview,
        elapsed_s=elapsed,
        effective_workers=workers if use_parallel else 1,
        parallel_used=use_parallel,
        chunks_synthesized=len(pending),
    )


def _run_sequential(
    *,
    pending: list[tuple[int, str]],
    wave_parts: list[np.ndarray],
    prompt_normalized: str,
    ref_audio: str,
    speed: float,
    num_step: int,
    guidance_scale: float,
    t_shift: float,
    engine,
    ode_seed: int,
    use_fixed_seed: bool,
    total: int,
    progress: ProgressFn,
    status_log: StatusLog | None = None,
) -> None:
    done = total - len(pending)
    for i, normalized in pending:
        chunk_label = f"Chunk {i + 1}/{total} ({len(normalized)} ký tự)"
        if status_log is not None:
            status_log.info(f"Bắt đầu {chunk_label}")
        progress(
            (done + 1) / total,
            desc=f"Đang tổng hợp đoạn {i + 1}/{total} (ONNX)...",
        )
        t_chunk = time.perf_counter()
        wav = engine.generate(
            prompt_text=prompt_normalized,
            prompt_wav=ref_audio,
            text=normalized,
            speed=speed,
            num_step=int(num_step),
            guidance_scale=float(guidance_scale),
            t_shift=float(t_shift),
            ode_seed=ode_seed,
            use_fixed_seed=use_fixed_seed,
            chunk_index=i,
        )
        if wav.size == 0:
            msg = (
                f"Bỏ qua {chunk_label}: wav rỗng (0 mel frames)"
            )
            logger.warning(
                "skip chunk %d/%d (%d chars): empty wav (0 mel frames)",
                i + 1,
                total,
                len(normalized),
            )
            if status_log is not None:
                status_log.warn(msg)
        elif status_log is not None:
            status_log.info(
                f"Xong {chunk_label} — {wav.size} samples "
                f"({time.perf_counter() - t_chunk:.2f}s)"
            )
        wave_parts[i] = wav
        del wav
        done += 1
        if is_force_cpu():
            gc.collect()


def _run_parallel(
    *,
    pending: list[tuple[int, str]],
    wave_parts: list[np.ndarray],
    prompt_normalized: str,
    ref_audio: str,
    speed: float,
    num_step: int,
    guidance_scale: float,
    t_shift: float,
    onnx_quant_mode: str,
    use_onnx_gpu: bool,
    ode_seed: int,
    use_fixed_seed: bool,
    workers: int,
    total: int,
    progress: ProgressFn,
    status_log: StatusLog | None = None,
) -> None:
    norm_by_index = {i: normalized for i, normalized in pending}
    tasks = [
        (
            i,
            normalized,
            prompt_normalized,
            ref_audio,
            speed,
            num_step,
            guidance_scale,
            t_shift,
            ode_seed,
            use_fixed_seed,
        )
        for i, normalized in pending
    ]
    pool_workers = min(workers, len(tasks))
    completed = 0
    base_done = total - len(pending)

    with ProcessPoolExecutor(
        max_workers=pool_workers,
        initializer=_worker_init,
        initargs=(onnx_quant_mode, use_onnx_gpu),
    ) as pool:
        futures = [pool.submit(_worker_synthesize_chunk, task) for task in tasks]
        for fut in as_completed(futures):
            index, wav, err = fut.result()
            if err is not None:
                err_msg = f"Lỗi tổng hợp chunk {index + 1}/{total}: {err}"
                if status_log is not None:
                    status_log.error(err_msg)
                raise RuntimeError(err_msg) from None
            normalized_len = len(norm_by_index.get(index, ""))
            chunk_label = f"Chunk {index + 1}/{total} ({normalized_len} ký tự)"
            if wav.size == 0:
                logger.warning(
                    "skip chunk %d/%d (%d chars): empty wav (0 mel frames)",
                    index + 1,
                    total,
                    normalized_len,
                )
                if status_log is not None:
                    status_log.warn(
                        f"Bỏ qua {chunk_label}: wav rỗng (0 mel frames)"
                    )
            elif status_log is not None:
                status_log.info(f"Xong {chunk_label} — {wav.size} samples")
            wave_parts[index] = wav
            del wav
            completed += 1
            progress(
                (base_done + completed) / total,
                desc=(
                    f"Song song: {completed}/{len(pending)} chunk "
                    f"({pool_workers} workers)..."
                ),
            )
            if is_force_cpu():
                gc.collect()
