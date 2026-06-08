"""

Shared TTS inference pipeline — callable from Gradio, Slint, or CLI without Gradio deps.

Extracted from app.infer_tts so GUI layers only handle presentation and I/O.

"""

from __future__ import annotations
import logging
import traceback
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from assets_loader import get_voice_by_id
from chunk_synthesis import (
    ChunkSynthResult,
    clamp_parallel_workers,
    iter_synthesize_tts_chunks,
    parallel_workers_clamp_message,
    synthesize_tts_chunks,
)
from config import (
    ONNX_DIR,
    VOCODER_ONNX,
    VOCODER_RUNTIME_LABEL,
    is_force_cpu,
    onnx_files,
    onnx_ready,
    vocoder_deploy_instructions,
    vocoder_onnx_ready,
)

from export_audio import save_output
from onnx_engine import OnnxTTSEngine, format_ode_seed_log
from onnx_providers import (

    get_runtime_device_summary,

    predict_runtime_device_summary,

    provider_status_message,

    runtime_provider_log_lines,
)

from onnx_quant import file_size_mb, missing_onnx_files, quant_readiness_hint
from status_log import StatusLog
from audio.post_process import join_tts_audio_chunks
from text.chunking import split_text_for_tts
from text.io import read_text_file
from text.normalizers import build_normalize_pipeline, format_normalize_pipeline
from text.pipeline import (
    INPUT_MODE_CHOICES,
    format_tts_timing_line,
    normalize_full_document,
    parse_input_mode,
    preview_normalize_output,
)

logger = logging.getLogger("zipvoice_onnx_gui")

MAX_GEN_TEXT_CHARS = 500_000

ProgressFn = Callable[[float, str | None], None]

LogNotifyFn = Callable[[], None]

@dataclass

class TTSRequest:

    asset_voice_id: str

    ref_audio_path: str | None

    ref_text: str

    gen_text: str

    speed: float = 1.0

    export_format: str = "WAV 24kHz"

    norm_pipeline: list[str] | None = None

    chunk_max_chars: int = 135

    chunk_min_chars: int = 70

    pause_sentence: float = 0.35

    pause_paragraph: float = 0.65

    pause_chapter: float = 2.0

    pause_enum_item: float = 0.45

    pause_forced: float = 0.28

    onnx_quant_mode: str = "int8"

    synth_num_step: int = 16

    synth_guidance_scale: float = 1.0

    synth_t_shift: float = 0.5

    input_mode: str = "raw"

    gen_txt_source_path: str | None = None

    parallel_workers: int = 1

    use_onnx_gpu: bool = False

    ode_seed: int = 42

    use_fixed_seed: bool = True

@dataclass

class TTSProgress:

    """Intermediate snapshot for streaming GUIs (Gradio generator yields)."""

    norm_preview: str = ""

    runtime_device: str = ""

    status_log_text: str = ""

@dataclass

class TTSResult:

    saved_path: Path

    sample_rate: int

    wave: np.ndarray

    status_summary: str

    norm_preview: str

    status_log_text: str = ""

    runtime_device: str = ""

@dataclass

class TTSError:

    message: str

    status_log_text: str = ""

    exc: Exception | None = field(default=None, repr=False)

def resolve_gen_text_for_pipeline(

    gen_text: str,

    gen_txt_source_path: str | None,
) -> str:

    if gen_txt_source_path:

        return read_text_file(gen_txt_source_path)

    return gen_text

def resolve_ref_path(

    ref_audio_path: str | None,

    asset_voice_id: str,

    voice_cache: list,
) -> str | None:

    if ref_audio_path:

        return ref_audio_path

    voice = get_voice_by_id(asset_voice_id, voice_cache)

    if voice:

        return voice.audio_path

    return None

def _noop_progress(_frac: float, _desc: str | None = None) -> None:

    pass

def _log_engine_details(
    log: StatusLog,
    engine: OnnxTTSEngine,
    quant_mode: str,
    *,
    num_step: int,
    guidance_scale: float,
    t_shift: float,
    speed: float,
) -> None:

    te_name, fm_name = onnx_files(quant_mode)

    te_path = ONNX_DIR / te_name

    fm_path = ONNX_DIR / fm_name

    log.info(f"quant_mode={quant_mode}")

    log.info(f"text_encoder: {te_path} ({file_size_mb(te_path):.1f} MB)")

    log.info(f"fm_decoder: {fm_path} ({file_size_mb(fm_path):.1f} MB)")

    log.info(f"ORT device (cài đặt): {provider_status_message(engine.use_gpu)}")

    for line in runtime_provider_log_lines(engine):

        log.info(line)

    log.info(f"Thiết bị thực tế: {get_runtime_device_summary(engine)}")

    log.info(
        f"Vocoder: {VOCODER_RUNTIME_LABEL} | {VOCODER_ONNX} "
        f"({file_size_mb(VOCODER_ONNX):.1f} MB)"
    )

    log.info(f"Sampling rate: {engine.sampling_rate} Hz")

    log.info(

        f"Synth: speed={speed}, num_step={num_step}, "

        f"guidance_scale={guidance_scale}, t_shift={t_shift}"

    )

def _format_chunk_sizes(chunks) -> str:

    sizes = [len(c.text) for c in chunks]

    if not sizes:

        return "(trống)"

    preview = ", ".join(str(s) for s in sizes[:15])

    if len(sizes) > 15:

        preview += f", … (+{len(sizes) - 15} chunk)"

    return (

        f"[{preview}] — {len(sizes)} chunk, "

        f"min={min(sizes)}, max={max(sizes)} ký tự"

    )

def iter_tts_pipeline(

    req: TTSRequest,

    *,

    voice_cache: list,

    progress: ProgressFn | None = None,

    status_log: StatusLog | None = None,

    log_engine: bool = True,

    on_log: LogNotifyFn | None = None,
) -> Iterator[TTSProgress | TTSResult | TTSError]:

    """Run full ONNX TTS pipeline, yielding progress snapshots then result or error."""

    log = status_log or StatusLog()

    prog = progress or _noop_progress

    norm_preview_short = ""

    runtime_device = ""

    def snap(

        *,

        norm_preview: str | None = None,

        runtime_device_override: str | None = None,
    ) -> TTSProgress:

        nonlocal norm_preview_short, runtime_device

        if norm_preview is not None:

            norm_preview_short = norm_preview[:2000] if norm_preview else ""

        if runtime_device_override is not None:

            runtime_device = runtime_device_override

        if on_log:

            on_log()

        return TTSProgress(

            norm_preview=norm_preview_short,

            runtime_device=runtime_device,

            status_log_text=log.text(),
        )

    try:

        use_gpu = bool(req.use_onnx_gpu) and not is_force_cpu()

        runtime_device = predict_runtime_device_summary(use_gpu)

        log.heading("Bắt đầu tổng hợp ONNX")

        yield snap()

        ref_audio_path = resolve_ref_path(

            req.ref_audio_path, req.asset_voice_id, voice_cache

        )

        if not ref_audio_path:

            log.error("Chưa chọn giọng mẫu (assets/ hoặc upload)")

            err = TTSError(

                "Chọn giọng từ menu assets/ hoặc upload file giọng mẫu.",

                log.text(),
            )

            yield err

            return

        gen_text = resolve_gen_text_for_pipeline(req.gen_text, req.gen_txt_source_path)

        if not gen_text.strip():

            log.error("Văn bản đọc trống")

            yield TTSError(

                "Vui lòng nhập văn bản cần đọc (ô số 3).",

                log.text(),
            )

            return

        ref_text = req.ref_text

        if not ref_text.strip():

            voice = get_voice_by_id(req.asset_voice_id, voice_cache)

            if voice and voice.transcript.strip():

                ref_text = voice.transcript

        if not ref_text.strip():

            log.error("Thiếu transcript giọng mẫu")

            yield TTSError(

                "Bắt buộc nhập transcript giọng mẫu (ô số 2). "

                "App không tự nhận dạng — bạn phải xác nhận lời nói trong file audio.",

                log.text(),
            )

            return

        if len(gen_text) > MAX_GEN_TEXT_CHARS:

            log.error(f"Văn bản quá dài: {len(gen_text):,} ký tự")

            yield TTSError(

                f"Văn bản quá dài ({len(gen_text):,} ký tự, tối đa {MAX_GEN_TEXT_CHARS:,}). "

                "Chia thành nhiều file .txt.",

                log.text(),
            )

            return

        log.heading("Đầu vào")

        log.info(f"asset_voice_id={req.asset_voice_id or '(upload thủ công)'}")

        log.info(f"ref_audio={ref_audio_path}")

        log.info(f"ref_text_len={len(ref_text.strip())} ký tự")

        log.info(f"gen_text_len={len(gen_text)} ký tự")

        if req.gen_txt_source_path:

            log.info(f"gen_txt_source={req.gen_txt_source_path}")

        log.info(f"export_format={req.export_format}, speed={req.speed}")

        yield snap()

        try:

            norm_pipeline = build_normalize_pipeline(req.norm_pipeline)

        except ValueError as exc:

            log.error(str(exc))

            yield TTSError(str(exc), log.text(), exc)

            return

        norm_label = format_normalize_pipeline(norm_pipeline)

        tts_input_mode = parse_input_mode(req.input_mode)

        if tts_input_mode == "prepared" and norm_pipeline:

            warn_msg = f"input_mode=prepared — bỏ qua pipeline: {norm_label}"

            logger.warning("input_mode=prepared — bỏ qua pipeline chuẩn hóa: %s", norm_label)

            log.warn(warn_msg)

            norm_label = f"{INPUT_MODE_CHOICES['prepared']} (pipeline bỏ qua)"

        requested_workers = int(req.parallel_workers)
        workers = clamp_parallel_workers(requested_workers, use_gpu=use_gpu)
        if requested_workers != workers:
            log.warn(
                parallel_workers_clamp_message(
                    requested_workers, workers, use_gpu=use_gpu
                )
            )
        vocoder_label = "onnx" if vocoder_onnx_ready() else "missing"

        log.heading("Chuẩn hóa & chia chunk")

        log.info(f"input_mode={tts_input_mode}")

        log.info(f"normalize_pipeline={norm_label or '(trống)'}")

        log.info(

            f"chunk_min_chars={req.chunk_min_chars}, chunk_max_chars={req.chunk_max_chars}, pauses: "

            f"câu={req.pause_sentence}s, đoạn={req.pause_paragraph}s, "

            f"chương={req.pause_chapter}s, enum={req.pause_enum_item}s, "

            f"cắt={req.pause_forced}s"

        )

        log.info(

            f"onnx_quant={req.onnx_quant_mode}, workers={workers}, gpu={use_gpu}"

        )

        log.info(
            format_ode_seed_log(
                ode_seed=req.ode_seed,
                use_fixed_seed=req.use_fixed_seed,
            )
        )

        yield snap()

        logger.info(

            "run_tts_pipeline | asset=%s | gen_len=%d | ref_len=%d | export=%s | norm=%s | mode=%s | chunk=%d | quant=%s | workers=%d | gpu=%s | vocoder=%s",

            req.asset_voice_id,

            len(gen_text),

            len(ref_text.strip()),

            req.export_format,

            norm_label,

            tts_input_mode,

            req.chunk_max_chars,

            req.onnx_quant_mode,

            workers,

            use_gpu,

            vocoder_label,
        )

        if not onnx_ready(req.onnx_quant_mode):

            missing = missing_onnx_files(ONNX_DIR, req.onnx_quant_mode)

            hint = quant_readiness_hint(req.onnx_quant_mode, missing)

            log.error(f"Thiếu ONNX: {', '.join(missing)}")

            yield TTSError(

                f"Chưa có ONNX cho mode **{req.onnx_quant_mode}**.\n"

                f"Thiếu: `{', '.join(missing)}`\n\n{hint}",

                log.text(),
            )

            return

        if not vocoder_onnx_ready():

            log.error("Thiếu ONNX vocoder weights")

            yield TTSError(

                "Thiếu ONNX vocoder trong `models/vocoder/`.\n\n"
                f"{vocoder_deploy_instructions()}",

                log.text(),
            )

            return

        log.stage_begin("Chuẩn hóa toàn văn")

        normalized_doc = normalize_full_document(

            gen_text, norm_pipeline, tts_input_mode

        )

        log.stage_end("Chuẩn hóa toàn văn", f"{len(normalized_doc)} ký tự")

        yield snap()

        log.stage_begin("Chia chunk")

        chunk_merge_log: list[str] = []
        tts_chunks = split_text_for_tts(

            normalized_doc,

            max_chars=int(req.chunk_max_chars),

            min_chars=int(req.chunk_min_chars),

            pause_sentence=float(req.pause_sentence),

            pause_paragraph=float(req.pause_paragraph),

            pause_chapter=float(req.pause_chapter),

            pause_enum_item=float(req.pause_enum_item),

            pause_forced_split=float(req.pause_forced),
            merge_log=chunk_merge_log,
        )

        for merge_msg in chunk_merge_log:
            log.info(merge_msg)

        log.info(f"Số chunk: {len(tts_chunks)}")

        log.info(f"Kích thước: {_format_chunk_sizes(tts_chunks)}")

        log.stage_end("Chia chunk")

        yield snap()

        log.heading("Load ONNX engine")

        if req.use_onnx_gpu and not use_gpu:

            log.warn("Yêu cầu GPU nhưng bị tắt (ZIPVOICE_FORCE_CPU hoặc không có CUDA EP)")

            logger.warning("GPU requested but disabled (ZIPVOICE_FORCE_CPU or no CUDA EP)")

        log.stage_begin("Load ONNX engine")

        engine = OnnxTTSEngine.get(

            quant_mode=req.onnx_quant_mode,

            use_gpu=use_gpu,
        )

        runtime_device = get_runtime_device_summary(engine)

        device_note = runtime_device.removeprefix("Thực tế: ")

        if log_engine:

            _log_engine_details(

                log,

                engine,

                req.onnx_quant_mode,

                num_step=req.synth_num_step,

                guidance_scale=req.synth_guidance_scale,

                t_shift=req.synth_t_shift,

                speed=req.speed,
            )

        log.stage_end("Load ONNX engine")

        yield snap(runtime_device_override=runtime_device)

        wave_parts: list[np.ndarray] = [

            np.array([], dtype=np.float32) for _ in range(len(tts_chunks))

        ]

        synth_holder = ChunkSynthResult(
            wave_parts=wave_parts,
            normalized_preview="",
            elapsed_s=0.0,
            effective_workers=1,
            parallel_used=False,
            chunks_synthesized=0,
        )

        try:

            for _ in iter_synthesize_tts_chunks(

                engine=engine,

                tts_chunks=tts_chunks,

                wave_parts=wave_parts,

                norm_pipeline=norm_pipeline,

                tts_input_mode=tts_input_mode,

                ref_audio_path=ref_audio_path,

                ref_text=ref_text,

                speed=float(req.speed),

                num_step=int(req.synth_num_step),

                guidance_scale=float(req.synth_guidance_scale),

                t_shift=float(req.synth_t_shift),
                onnx_quant_mode=req.onnx_quant_mode,

                parallel_workers=workers,

                use_onnx_gpu=use_gpu,

                ode_seed=int(req.ode_seed),

                use_fixed_seed=bool(req.use_fixed_seed),

                progress=prog,

                status_log=log,

                result_holder=synth_holder,
            ):

                yield snap()

            synth_result = synth_holder

        except RuntimeError as exc:

            log.error(str(exc))

            yield TTSError(str(exc), log.text(), exc)

            return

        normalized_preview = synth_result.normalized_preview

        yield snap(norm_preview=normalized_preview or "")

        log.stage_begin("Ghép audio")

        final_wave = join_tts_audio_chunks(wave_parts, tts_chunks, engine.sampling_rate)

        if final_wave is None or final_wave.size == 0:

            msg = (

                "Không tạo được âm thanh — một số đoạn quá ngắn hoặc không có mel frame. "

                "Thử giảm độ dài prompt giọng mẫu hoặc gộp đoạn văn ngắn hơn."

            )

            log.error(msg)

            yield TTSError(msg, log.text())

            return

        duration_s = final_wave.size / engine.sampling_rate

        log.stage_end(

            "Ghép audio",

            f"{final_wave.size} samples, ~{duration_s:.1f}s @ {engine.sampling_rate}Hz",
        )

        yield snap(norm_preview=normalized_preview or "")

        log.stage_begin("Lưu file output")

        saved = save_output(

            final_wave,

            engine.sampling_rate,

            req.export_format,

            voice_label=req.asset_voice_id or "upload",

            text_preview=gen_text,
        )

        log.info(f"output_path={saved}")

        log.stage_end("Lưu file output")

        yield snap(norm_preview=normalized_preview or "")

        est_min = max(1, int(len(tts_chunks) * 5 / 60))

        onnx_mode = str(req.onnx_quant_mode).upper()

        timing_line = format_tts_timing_line(

            synth_result.elapsed_s,

            len(tts_chunks),

            req.onnx_quant_mode,

            chunks_synthesized=synth_result.chunks_synthesized,

            parallel_workers=(

                synth_result.effective_workers if synth_result.parallel_used else None

            ),
        )

        parallel_note = ""

        if synth_result.parallel_used:

            parallel_note = f" · **Song song:** {synth_result.effective_workers} workers"

        vocoder_note = VOCODER_RUNTIME_LABEL

        norm_full_preview = preview_normalize_output(

            gen_text,

            norm_pipeline,

            chunk_max_chars=int(req.chunk_max_chars),

            chunk_min_chars=int(req.chunk_min_chars),

            mode=tts_input_mode,
        )

        status = (

            f"Đã lưu: `{saved}`\n\n"

            f"**ONNX:** {onnx_mode} · **Vocoder:** {vocoder_note} · **Thiết bị:** {device_note} · "

            f"**Chuẩn hóa:** {norm_label} · "

            f"**Chunks:** {len(tts_chunks)} (min {req.chunk_min_chars}, max {req.chunk_max_chars} ký tự/chunk, "

            f"~{est_min} phút ước tính){parallel_note}\n\n"

            f"**Thời gian:** {timing_line}\n\n"

            f"**Text sau chuẩn hóa (đoạn 1):** {normalized_preview or '(trống)'}"

        )

        log.heading("Hoàn tất")

        log.info(timing_line)

        log.info(f"File: {saved}")

        logger.info("run_tts_pipeline done | file=%s | %s", saved, timing_line)

        yield TTSResult(

            saved_path=saved,

            sample_rate=engine.sampling_rate,

            wave=final_wave,

            status_summary=status,

            norm_preview=norm_full_preview,

            status_log_text=log.text(),

            runtime_device=runtime_device,
        )

    except ImportError as exc:

        logger.warning("Normalize dependency: %s", exc)

        log.error(str(exc))

        yield TTSError(str(exc), log.text(), exc)

    except ValueError as exc:

        logger.warning("Validation: %s", exc)

        log.error(str(exc))

        yield TTSError(str(exc), log.text(), exc)

    except Exception as exc:

        logger.error("run_tts_pipeline failed:\n%s", traceback.format_exc())

        log.error(f"Lỗi: {exc}")

        yield TTSError(f"Lỗi: {exc}\nChi tiết: logs/app.log", log.text(), exc)

def run_tts_pipeline(

    req: TTSRequest,

    *,

    voice_cache: list,

    progress: ProgressFn | None = None,

    status_log: StatusLog | None = None,

    log_engine: bool = True,

    on_log: LogNotifyFn | None = None,
) -> TTSResult | TTSError:

    """Run full ONNX TTS pipeline. Returns TTSResult or TTSError (never raises gr.Error)."""

    result: TTSResult | TTSError | None = None

    for event in iter_tts_pipeline(

        req,

        voice_cache=voice_cache,

        progress=progress,

        status_log=status_log,

        log_engine=log_engine,

        on_log=on_log,
    ):

        if isinstance(event, (TTSResult, TTSError)):

            result = event

    assert result is not None

    return result

def preview_normalize_text(

    gen_text: str,

    gen_txt_source_path: str | None,

    norm_pipeline: list[str] | None,

    chunk_max_chars: int,

    chunk_min_chars: int,

    input_mode: str,
) -> str:

    source_text = resolve_gen_text_for_pipeline(gen_text, gen_txt_source_path)

    return preview_normalize_output(

        source_text,

        norm_pipeline,

        chunk_max_chars=int(chunk_max_chars),

        chunk_min_chars=int(chunk_min_chars),

        mode=parse_input_mode(input_mode),
    )

