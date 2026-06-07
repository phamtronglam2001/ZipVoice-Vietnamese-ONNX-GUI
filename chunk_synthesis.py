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

from config import is_force_cpu
from utils import (
    TtsChunk,
    prepare_for_tts,
    prepare_tts_text,
    preprocess_ref_audio_text,
    save_tts_checkpoint_chunk,
)

logger = logging.getLogger("zipvoice_onnx_gui")

_worker_engine = None


def max_parallel_workers(use_gpu: bool = False) -> int:
    if use_gpu:
        # Each worker loads full ONNX on GPU — limit VRAM use.
        return min(2, os.cpu_count() or 2)
    return min(8, os.cpu_count() or 4)


def clamp_parallel_workers(
    workers: int,
    use_gpu: bool = False,
    *,
    use_pytorch_vocoder: bool = True,
) -> int:
    if use_pytorch_vocoder:
        return 1
    return min(max(1, int(workers)), max_parallel_workers(use_gpu))


def _worker_init(quant_mode: str, use_gpu: bool) -> None:
    global _worker_engine
    os.environ["ZIPVOICE_ONNX_GPU"] = "1" if use_gpu else "0"
    if use_gpu and not is_force_cpu():
        os.environ.pop("ZIPVOICE_FORCE_CPU", None)
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    from onnx_engine import OnnxTTSEngine

    OnnxTTSEngine._instance = None
    _worker_engine = OnnxTTSEngine(quant_mode=quant_mode, use_gpu=use_gpu)


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


def _noop_progress(_frac: float, _desc: str) -> None:
    pass


def synthesize_tts_chunks(
    *,
    engine,
    tts_chunks: list[TtsChunk],
    wave_parts: list[np.ndarray],
    resume_from: int,
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
    use_pytorch_vocoder: bool = True,
    manifest: dict,
    output_dir,
    progress: ProgressFn | None = None,
) -> ChunkSynthResult:
    """
    Synthesize all pending chunks (respecting checkpoint resume).
    Falls back to sequential when workers==1, resume active, or only one pending chunk.
    """
    progress = progress or _noop_progress
    use_gpu = bool(use_onnx_gpu) and not is_force_cpu()
    workers = clamp_parallel_workers(
        parallel_workers,
        use_gpu=use_gpu,
        use_pytorch_vocoder=use_pytorch_vocoder,
    )
    if use_pytorch_vocoder and parallel_workers > 1:
        logger.info(
            "parallel_workers=%d → 1 (PyTorch vocoder — không hỗ trợ song song)",
            parallel_workers,
        )
    if resume_from > 0 and workers > 1:
        logger.info(
            "parallel_workers=%d → 1 (checkpoint resume từ chunk %d)",
            workers,
            resume_from + 1,
        )
        workers = 1

    ref_audio, resolved_ref_text = preprocess_ref_audio_text(
        ref_audio_path,
        ref_text,
        show_info=logger.info,
    )
    prompt_normalized = prepare_tts_text(resolved_ref_text, norm_pipeline)

    pending: list[tuple[int, str]] = []
    normalized_preview = ""
    total = len(tts_chunks)

    for i, tts_chunk in enumerate(tts_chunks):
        if i < resume_from and wave_parts[i].size > 0:
            continue
        normalized = prepare_for_tts(
            tts_chunk.text, norm_pipeline, tts_input_mode, already_normalized=True
        )
        if not normalized.strip():
            logger.warning(
                "skip chunk %d/%d: empty after normalize",
                i + 1,
                total,
            )
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

    t0 = time.perf_counter()
    use_parallel = workers > 1 and len(pending) > 1

    if use_parallel:
        logger.info(
            "parallel synthesis | workers=%d | pending=%d/%d chunks",
            workers,
            len(pending),
            total,
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
            workers=workers,
            total=total,
            engine=engine,
            manifest=manifest,
            output_dir=output_dir,
            progress=progress,
        )
    else:
        logger.info(
            "sequential synthesis | pending=%d/%d chunks",
            len(pending),
            total,
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
            total=total,
            manifest=manifest,
            output_dir=output_dir,
            progress=progress,
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
    return ChunkSynthResult(
        wave_parts=wave_parts,
        normalized_preview=normalized_preview,
        elapsed_s=elapsed,
        effective_workers=workers if use_parallel else 1,
        parallel_used=use_parallel,
        chunks_synthesized=len(pending),
    )


def _save_chunk(
    engine,
    index: int,
    wav: np.ndarray,
    manifest: dict,
    output_dir,
) -> None:
    save_tts_checkpoint_chunk(
        output_dir, index, wav, engine.sampling_rate, manifest
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
    total: int,
    manifest: dict,
    output_dir,
    progress: ProgressFn,
) -> None:
    done = total - len(pending)
    for i, normalized in pending:
        progress(
            (done + 1) / total,
            desc=f"Đang tổng hợp đoạn {i + 1}/{total} (ONNX)...",
        )
        wav = engine.generate(
            prompt_text=prompt_normalized,
            prompt_wav=ref_audio,
            text=normalized,
            speed=speed,
            num_step=int(num_step),
            guidance_scale=float(guidance_scale),
            t_shift=float(t_shift),
        )
        if wav.size == 0:
            logger.warning(
                "skip chunk %d/%d (%d chars): empty wav (0 mel frames)",
                i + 1,
                total,
                len(normalized),
            )
        wave_parts[i] = wav
        _save_chunk(engine, i, wav, manifest, output_dir)
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
    workers: int,
    total: int,
    engine,
    manifest: dict,
    output_dir,
    progress: ProgressFn,
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
                raise RuntimeError(
                    f"Lỗi tổng hợp chunk {index + 1}/{total}: {err}"
                ) from None
            if wav.size == 0:
                normalized_len = len(norm_by_index.get(index, ""))
                logger.warning(
                    "skip chunk %d/%d (%d chars): empty wav (0 mel frames)",
                    index + 1,
                    total,
                    normalized_len,
                )
            wave_parts[index] = wav
            _save_chunk(engine, index, wav, manifest, output_dir)
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
