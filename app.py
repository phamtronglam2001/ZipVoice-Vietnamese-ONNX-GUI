"""
ZipVoice Vietnamese ONNX TTS — local Gradio GUI.
All model weights bundled under models/ (Git LFS); no Hugging Face download at setup.
"""
from __future__ import annotations

import os
import sys
import traceback

import gradio as gr
import numpy as np

from assets_loader import MANUAL_CHOICE, dropdown_choices, get_voice_by_id, scan_ref_voices
from config import (
    ASSETS_DIR,
    ONNX_DIR,
    OUTPUT_DIR,
    ROOT,
    VOCODER_DIR,
    ensure_ffmpeg_on_path,
    is_force_cpu,
    is_onnx_gpu_env,
    models_ready,
    onnx_quant_mode as get_onnx_quant_mode,
    onnx_ready,
    pytorch_vocoder_ready,
)
from export_audio import EXPORT_CHOICES, save_output
from preset_io import (
    collect_gui_state,
    load_preset,
    preset_dropdown_choices,
    save_preset,
    apply_preset_to_gui,
)
from runtime_log import setup_runtime_logging

ensure_ffmpeg_on_path()
logger = setup_runtime_logging(name="zipvoice_onnx_gui")

if not models_ready():
    print(
        "\n[ERROR] Runtime not ready. Run setup first:\n"
        "  Windows:  install_cpu.bat\n"
    )
    if not onnx_ready():
        print("[ERROR] Missing ONNX files in models/onnx/")
    if not pytorch_vocoder_ready():
        print(
            "[ERROR] Missing PyTorch Vocos weights in models/vocoder/\n"
            "  Need config.yaml + pytorch_model.bin (charactr/vocos-mel-24khz).\n"
            "  Run: python download_models.py --pytorch-vocoder\n"
            "  Or copy from ZipVoice-Vietnamese-GUI repo."
        )
    sys.exit(1)

from chunk_synthesis import clamp_parallel_workers, max_parallel_workers, synthesize_tts_chunks  # noqa: E402
from onnx_engine import OnnxTTSEngine  # noqa: E402
from onnx_providers import provider_status_message  # noqa: E402
from utils import (  # noqa: E402
    AUDIOBOOK_PRESET_PIPELINE,
    DEFAULT_NORMALIZE_PIPELINE,
    NORMALIZE_ADD_CHOICES,
    PAUSE_CHAPTER_DEFAULT,
    PAUSE_ENUM_DEFAULT,
    PAUSE_FORCED_SPLIT_DEFAULT,
    PAUSE_PARAGRAPH_DEFAULT,
    PAUSE_SENTENCE_DEFAULT,
    build_normalize_pipeline,
    compute_tts_checkpoint_key,
    format_normalize_pipeline,
    format_normalize_pipeline_list,
    format_tts_timing_line,
    join_tts_audio_chunks,
    load_tts_checkpoint_chunks,
    pipeline_add_step,
    pipeline_move,
    pipeline_remove_at,
    pipeline_selector_choices,
    _parse_pipeline_index,
    INPUT_MODE_CHOICES,
    export_normalized_text_file,
    parse_input_mode,
    normalize_full_document,
    preview_normalize_output,
    read_text_file,
    split_text_for_tts,
)

MAX_GEN_TEXT_CHARS = 500_000

# Gradio textarea may corrupt Unicode/line breaks — CSS fixes display only.
GRADIO_UNICODE_TEXTBOX_CSS = """
.zv-unicode-text textarea,
.zv-unicode-text input {
    font-family: "Segoe UI", "Noto Sans", "DejaVu Sans", system-ui, sans-serif !important;
    white-space: pre-wrap !important;
    overflow-y: auto !important;
    resize: vertical !important;
}
.zv-scroll-text textarea {
    max-height: 60vh !important;
    overflow-y: scroll !important;
}
"""
TEXTBOX_UNICODE_CLASS = "zv-unicode-text zv-scroll-text"
_voice_cache: list = []


def _refresh_voices() -> tuple[gr.Dropdown, str]:
    global _voice_cache
    _voice_cache = scan_ref_voices()
    choices = dropdown_choices(_voice_cache)
    info = f"Đã load **{len(_voice_cache)}** giọng từ `assets/ref_info.json`"
    if not _voice_cache:
        info += (
            "\n\nChưa có giọng hợp lệ. Kiểm tra `assets/ref_info.json` "
            "và đường dẫn `audio_path`."
        )
    return gr.Dropdown(choices=choices, value=MANUAL_CHOICE), info


def _on_voice_pick(voice_id: str) -> tuple[str | None, str, str]:
    voice = get_voice_by_id(voice_id, _voice_cache)
    if voice is None:
        return None, "", "Chế độ: upload thủ công"
    note = f"Đã chọn: `{voice.id}`"
    if not voice.transcript:
        note += " — **chưa có `text` trong ref_info.json**, bắt buộc điền ô số 2."
    return voice.audio_path, voice.transcript, note


def _on_gen_txt_upload(file_path: str | None):
    """Populate textbox for display; keep path in State for pipeline source-of-truth."""
    if not file_path:
        return gr.update(), None
    try:
        return read_text_file(file_path), file_path
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc


def _resolve_gen_text_for_pipeline(
    gen_text: str,
    gen_txt_source_path: str | None,
) -> str:
    """Pipeline/TTS must not trust Gradio textbox when input came from .txt upload.

    Gradio may alter Unicode normalization, line breaks, or special characters
    in textarea values. Re-read the original file (UTF-8 / UTF-8-BOM) instead.
    """
    if gen_txt_source_path:
        try:
            return read_text_file(gen_txt_source_path)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
    return gen_text


def _refresh_pipeline_ui(
    steps: list[str] | None,
    selected_index: str | None = None,
) -> tuple[list[str], str, gr.Dropdown]:
    validated = build_normalize_pipeline(steps)
    choices = pipeline_selector_choices(validated)
    valid_values = {c[1] for c in choices}
    if not choices:
        selector_value = None
    elif selected_index is not None and selected_index in valid_values:
        selector_value = selected_index
    else:
        selector_value = choices[0][1]
    return (
        validated,
        format_normalize_pipeline_list(validated),
        gr.Dropdown(choices=choices, value=selector_value),
    )


def _on_pipeline_add(steps: list[str] | None, new_step: str):
    try:
        old_len = len(steps or [])
        updated = pipeline_add_step(steps, new_step)
        if len(updated) > old_len:
            select = str(len(updated) - 1)
        elif updated:
            select = "0"
        else:
            select = None
        return _refresh_pipeline_ui(updated, select)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc


def _on_pipeline_remove(steps: list[str] | None, index: str | None):
    idx = _parse_pipeline_index(index)
    updated = pipeline_remove_at(steps, index)
    if not updated:
        select = None
    elif idx is not None:
        select = str(min(idx, len(updated) - 1))
    else:
        select = "0"
    return _refresh_pipeline_ui(updated, select)


def _on_pipeline_move_up(steps: list[str] | None, index: str | None):
    idx = _parse_pipeline_index(index)
    updated = pipeline_move(steps, index, -1)
    select = str(idx - 1) if idx is not None and idx > 0 else index
    return _refresh_pipeline_ui(updated, select)


def _on_pipeline_move_down(steps: list[str] | None, index: str | None):
    idx = _parse_pipeline_index(index)
    updated = pipeline_move(steps, index, 1)
    select = str(idx + 1) if idx is not None else index
    return _refresh_pipeline_ui(updated, select)


def _on_pipeline_reset():
    return _refresh_pipeline_ui(list(DEFAULT_NORMALIZE_PIPELINE))


def _on_pipeline_audiobook_preset():
    steps = list(AUDIOBOOK_PRESET_PIPELINE)
    return _refresh_pipeline_ui(steps)


def _on_load_preset(preset_name: str | None):
    if not preset_name:
        raise gr.Error("Chọn preset trong danh sách profiles/.")
    try:
        preset = load_preset(preset_name)
    except (FileNotFoundError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc
    updates = apply_preset_to_gui(preset)
    preset_status_msg = updates.pop("preset_status", f"Đã tải preset **{preset.name}**")
    asset_info = preset_status_msg
    if preset.voice_mode == "bundled" and preset.voice_id:
        voice = get_voice_by_id(preset.voice_id, _voice_cache)
        if voice:
            updates["ref_audio"] = gr.update(value=voice.audio_path)
            updates["ref_text"] = gr.update(value=voice.transcript)
            asset_info = f"Preset **{preset.name}** · giọng `{voice.id}`"
        else:
            asset_info = (
                f"Preset **{preset.name}** — giọng `{preset.voice_id}` không tìm thấy"
            )
    return (
        updates["voice_dropdown"],
        updates["ref_audio"],
        updates["ref_text"],
        asset_info,
        updates["norm_pipeline_state"],
        updates["norm_pipeline_display"],
        updates["norm_sel_step"],
        updates["chunk_max_chars"],
        updates["pause_sentence"],
        updates["pause_paragraph"],
        updates["pause_chapter"],
        updates["pause_enum_item"],
        updates["pause_forced"],
        updates["speed"],
        updates["export_format"],
        updates["onnx_quant_mode"],
        updates["synth_num_step"],
        updates["synth_guidance_scale"],
        updates["synth_t_shift"],
        updates["input_mode"],
        updates.get("parallel_workers", gr.update(value=1)),
        updates.get("use_onnx_gpu", gr.update(value=is_onnx_gpu_env())),
        updates.get("use_pytorch_vocoder", gr.update(value=True)),
        gr.update(visible=not preset.use_pytorch_vocoder),
        preset_status_msg,
    )


def _on_save_preset(
    save_name: str,
    voice_dropdown: str,
    ref_audio_path: str | None,
    ref_text: str,
    norm_pipeline: list[str] | None,
    chunk_max_chars: int,
    pause_sentence: float,
    pause_paragraph: float,
    pause_chapter: float,
    pause_enum_item: float,
    pause_forced: float,
    speed: float,
    export_format: str,
    onnx_quant_mode: str,
    synth_num_step: int,
    synth_guidance_scale: float,
    synth_t_shift: float,
    input_mode: str,
    parallel_workers: int,
    use_onnx_gpu: bool,
    use_pytorch_vocoder: bool,
):
    if not save_name or not save_name.strip():
        raise gr.Error("Nhập tên file preset (vd: sach_ai_vy).")
    preset = collect_gui_state(
        voice_dropdown=voice_dropdown,
        ref_audio_path=ref_audio_path,
        ref_text=ref_text,
        norm_pipeline=norm_pipeline,
        chunk_max_chars=chunk_max_chars,
        pause_sentence=pause_sentence,
        pause_paragraph=pause_paragraph,
        pause_chapter=pause_chapter,
        pause_enum_item=pause_enum_item,
        pause_forced=pause_forced,
        speed=speed,
        export_format=export_format,
        onnx_quant_mode=onnx_quant_mode,
        num_step=synth_num_step,
        guidance_scale=synth_guidance_scale,
        t_shift=synth_t_shift,
        preset_name=save_name.strip(),
        input_mode=input_mode,
        parallel_workers=parallel_workers,
        use_onnx_gpu=use_onnx_gpu,
        use_pytorch_vocoder=use_pytorch_vocoder,
    )
    try:
        path = save_preset(save_name.strip(), preset)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    choices = preset_dropdown_choices()
    return (
        gr.update(choices=choices, value=path.stem),
        f"Đã lưu preset → `{path}`",
    )


def preview_normalize(
    gen_text: str,
    gen_txt_source_path: str | None,
    norm_pipeline: list[str] | None,
    chunk_max_chars: int,
    input_mode: str,
) -> str:
    try:
        source_text = _resolve_gen_text_for_pipeline(gen_text, gen_txt_source_path)
        return preview_normalize_output(
            source_text,
            norm_pipeline,
            chunk_max_chars=int(chunk_max_chars),
            mode=parse_input_mode(input_mode),
        )
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    except ImportError as exc:
        raise gr.Error(str(exc)) from exc


def export_normalized_text(
    gen_text: str,
    gen_txt_source_path: str | None,
    norm_pipeline: list[str] | None,
    input_mode: str,
    export_path: str,
    gen_txt_file: str | None,
) -> tuple[str, str]:
    source_text = _resolve_gen_text_for_pipeline(gen_text, gen_txt_source_path)
    if not source_text.strip():
        raise gr.Error("Vui lòng nhập văn bản (ô 3) trước khi xuất.")
    try:
        norm_pipeline = build_normalize_pipeline(norm_pipeline)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    mode = parse_input_mode(input_mode)
    hint = gen_txt_source_path or gen_txt_file or "text"
    path_arg = export_path.strip() if export_path and export_path.strip() else None
    try:
        saved = export_normalized_text_file(
            source_text,
            norm_pipeline,
            mode=mode,
            output_path=path_arg,
            source_hint=hint,
        )
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    preview = preview_normalize_output(
        source_text,
        norm_pipeline,
        chunk_max_chars=135,
        mode=mode,
    )
    return preview, f"Đã lưu **{saved}** ({saved.stat().st_size:,} bytes)"


def _resolve_ref_path(ref_audio_path: str | None, asset_voice_id: str) -> str | None:
    if ref_audio_path:
        return ref_audio_path
    voice = get_voice_by_id(asset_voice_id, _voice_cache)
    if voice:
        return voice.audio_path
    return None


def _toggle_onnx_vocoder_warning(use_pytorch_vocoder: bool):
    return gr.update(
        visible=not use_pytorch_vocoder,
    )


def infer_tts(
    asset_voice_id: str,
    ref_audio_path: str | None,
    ref_text: str,
    gen_text: str,
    speed: float,
    export_format: str,
    norm_pipeline: list[str] | None,
    chunk_max_chars: int,
    pause_sentence: float,
    pause_paragraph: float,
    pause_chapter: float,
    pause_enum_item: float,
    pause_forced: float,
    onnx_quant_mode: str,
    synth_num_step: int,
    synth_guidance_scale: float,
    synth_t_shift: float,
    input_mode: str,
    gen_txt_source_path: str | None,
    parallel_workers: int = 1,
    use_onnx_gpu: bool = False,
    use_pytorch_vocoder: bool = True,
    progress=gr.Progress(),
) -> tuple[str | None, tuple[int, np.ndarray] | None, str, str]:
    ref_audio_path = _resolve_ref_path(ref_audio_path, asset_voice_id)
    if not ref_audio_path:
        raise gr.Error("Chọn giọng từ menu assets/ hoặc upload file giọng mẫu.")
    gen_text = _resolve_gen_text_for_pipeline(gen_text, gen_txt_source_path)
    if not gen_text.strip():
        raise gr.Error("Vui lòng nhập văn bản cần đọc (ô số 3).")
    if not ref_text.strip():
        voice = get_voice_by_id(asset_voice_id, _voice_cache)
        if voice and voice.transcript.strip():
            ref_text = voice.transcript
    if not ref_text.strip():
        raise gr.Error(
            "Bắt buộc nhập transcript giọng mẫu (ô số 2). "
            "App không tự nhận dạng — bạn phải xác nhận lời nói trong file audio."
        )
    if len(gen_text) > MAX_GEN_TEXT_CHARS:
        raise gr.Error(
            f"Văn bản quá dài ({len(gen_text):,} ký tự, tối đa {MAX_GEN_TEXT_CHARS:,}). "
            "Chia thành nhiều file .txt."
        )

    try:
        try:
            norm_pipeline = build_normalize_pipeline(norm_pipeline)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc

        voice = get_voice_by_id(asset_voice_id, _voice_cache)
        norm_label = format_normalize_pipeline(norm_pipeline)
        tts_input_mode = parse_input_mode(input_mode)
        if tts_input_mode == "prepared" and norm_pipeline:
            logger.warning(
                "input_mode=prepared — bỏ qua pipeline chuẩn hóa: %s",
                norm_label,
            )
            norm_label = f"{INPUT_MODE_CHOICES['prepared']} (pipeline bỏ qua)"
        use_gpu = bool(use_onnx_gpu) and not is_force_cpu()
        use_pt_vocoder = bool(use_pytorch_vocoder)
        workers = clamp_parallel_workers(
            parallel_workers,
            use_gpu=use_gpu,
            use_pytorch_vocoder=use_pt_vocoder,
        )
        logger.info(
            "infer_tts | asset=%s | gen_len=%d | ref_len=%d | export=%s | norm=%s | mode=%s | chunk=%d | quant=%s | workers=%d | gpu=%s | vocoder=%s",
            asset_voice_id,
            len(gen_text),
            len(ref_text.strip()),
            export_format,
            norm_label,
            tts_input_mode,
            chunk_max_chars,
            onnx_quant_mode,
            workers,
            use_gpu,
            "pytorch" if use_pt_vocoder else "onnx",
        )

        if not onnx_ready(onnx_quant_mode):
            from onnx_quant import missing_onnx_files, quant_readiness_hint

            missing = missing_onnx_files(ONNX_DIR, onnx_quant_mode)
            hint = quant_readiness_hint(onnx_quant_mode, missing)
            raise gr.Error(
                f"Chưa có ONNX cho mode **{onnx_quant_mode}**.\n"
                f"Thiếu: `{', '.join(missing)}`\n\n{hint}"
            )

        if use_pt_vocoder and not pytorch_vocoder_ready():
            raise gr.Error(
                "Thiếu PyTorch Vocos weights trong `models/vocoder/`.\n"
                "Cần `config.yaml` + `pytorch_model.bin` "
                "([charactr/vocos-mel-24khz](https://huggingface.co/charactr/vocos-mel-24khz)).\n"
                "Chạy: `python download_models.py --pytorch-vocoder` "
                "hoặc copy từ repo PyTorch ZipVoice-Vietnamese-GUI."
            )

        engine = OnnxTTSEngine.get(
            quant_mode=onnx_quant_mode,
            use_gpu=use_gpu,
            use_pytorch_vocoder=use_pt_vocoder,
        )
        device_note = provider_status_message(use_gpu)
        if use_onnx_gpu and not use_gpu:
            logger.warning("GPU requested but disabled (ZIPVOICE_FORCE_CPU or no CUDA EP)")
        normalized_doc = normalize_full_document(
            gen_text, norm_pipeline, tts_input_mode
        )
        tts_chunks = split_text_for_tts(
            normalized_doc,
            max_chars=int(chunk_max_chars),
            pause_sentence=float(pause_sentence),
            pause_paragraph=float(pause_paragraph),
            pause_chapter=float(pause_chapter),
            pause_enum_item=float(pause_enum_item),
            pause_forced_split=float(pause_forced),
        )
        checkpoint_key = compute_tts_checkpoint_key(
            gen_text,
            norm_pipeline,
            int(chunk_max_chars),
            speed=float(speed),
            pause_sentence=float(pause_sentence),
            pause_paragraph=float(pause_paragraph),
            pause_chapter=float(pause_chapter),
            pause_forced=float(pause_forced),
            asset_voice_id=asset_voice_id or "",
            input_mode=tts_input_mode,
        )
        manifest = {
            "checkpoint_key": checkpoint_key,
            "num_chunks": len(tts_chunks),
            "pipeline": norm_pipeline,
            "input_mode": tts_input_mode,
            "chunk_max_chars": int(chunk_max_chars),
            "speed": float(speed),
            "pause_sentence": float(pause_sentence),
            "pause_paragraph": float(pause_paragraph),
            "pause_chapter": float(pause_chapter),
            "pause_forced": float(pause_forced),
            "asset_voice_id": asset_voice_id or "",
        }
        cached_chunks, resume_from = load_tts_checkpoint_chunks(
            OUTPUT_DIR, checkpoint_key, len(tts_chunks)
        )
        if resume_from > 0:
            logger.info("resume từ chunk %d/%d", resume_from + 1, len(tts_chunks))
        wave_parts: list[np.ndarray] = [
            cached_chunks.get(i, np.array([], dtype=np.float32))
            for i in range(len(tts_chunks))
        ]
        try:
            synth_result = synthesize_tts_chunks(
                engine=engine,
                tts_chunks=tts_chunks,
                wave_parts=wave_parts,
                resume_from=resume_from,
                norm_pipeline=norm_pipeline,
                tts_input_mode=tts_input_mode,
                ref_audio_path=ref_audio_path,
                ref_text=ref_text,
                speed=float(speed),
                num_step=int(synth_num_step),
                guidance_scale=float(synth_guidance_scale),
                t_shift=float(synth_t_shift),
                onnx_quant_mode=onnx_quant_mode,
                parallel_workers=workers,
                use_onnx_gpu=use_gpu,
                use_pytorch_vocoder=use_pt_vocoder,
                manifest=manifest,
                output_dir=OUTPUT_DIR,
                progress=progress,
            )
        except RuntimeError as exc:
            raise gr.Error(str(exc)) from exc
        normalized_preview = synth_result.normalized_preview

        final_wave = join_tts_audio_chunks(wave_parts, tts_chunks, engine.sampling_rate)
        if final_wave is None or final_wave.size == 0:
            raise gr.Error(
                "Không tạo được âm thanh — một số đoạn quá ngắn hoặc không có mel frame. "
                "Thử giảm độ dài prompt giọng mẫu hoặc gộp đoạn văn ngắn hơn."
            )

        saved = save_output(
            final_wave,
            engine.sampling_rate,
            export_format,
            voice_label=asset_voice_id or "upload",
            text_preview=gen_text,
        )

        est_min = max(1, int(len(tts_chunks) * 5 / 60))
        onnx_mode = str(onnx_quant_mode).upper()
        resume_note = (
            f" · **Resume:** chunk {resume_from + 1}"
            if resume_from > 0
            else ""
        )
        timing_line = format_tts_timing_line(
            synth_result.elapsed_s,
            len(tts_chunks),
            onnx_quant_mode,
            chunks_synthesized=synth_result.chunks_synthesized,
            parallel_workers=(
                synth_result.effective_workers if synth_result.parallel_used else None
            ),
        )
        parallel_note = ""
        if synth_result.parallel_used:
            parallel_note = f" · **Song song:** {synth_result.effective_workers} workers"
        elif workers > 1 and resume_from > 0:
            parallel_note = " · **Song song:** tắt (resume checkpoint)"
        elif use_pt_vocoder and parallel_workers > 1:
            parallel_note = " · **Song song:** tắt (PyTorch vocoder)"
        vocoder_note = (
            "PyTorch Vocos (khuyên dùng)"
            if use_pt_vocoder
            else "ONNX wetdog + librosa ISTFT"
        )
        norm_full_preview = preview_normalize_output(
            gen_text,
            norm_pipeline,
            chunk_max_chars=int(chunk_max_chars),
            mode=tts_input_mode,
        )
        status = (
            f"Đã lưu: `{saved}`\n\n"
            f"**ONNX:** {onnx_mode} · **Vocoder:** {vocoder_note} · **Thiết bị:** {device_note} · **Chuẩn hóa:** {norm_label} · "
            f"**Chunks:** {len(tts_chunks)} (max {chunk_max_chars} ký tự/chunk, "
            f"~{est_min} phút ước tính){resume_note}{parallel_note}\n\n"
            f"**Thời gian:** {timing_line}\n\n"
            f"**Text sau chuẩn hóa (đoạn 1):** {normalized_preview or '(trống)'}"
        )
        logger.info("infer_tts done | file=%s | %s", saved, timing_line)
        return str(saved), (engine.sampling_rate, final_wave), status, norm_full_preview

    except gr.Error:
        raise
    except ImportError as exc:
        logger.warning("Normalize dependency: %s", exc)
        raise gr.Error(str(exc)) from exc
    except ValueError as exc:
        logger.warning("Validation: %s", exc)
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        logger.error("infer_tts failed:\n%s", traceback.format_exc())
        raise gr.Error(f"Lỗi: {exc}\nChi tiết: logs/app.log") from exc


def build_ui() -> gr.Blocks:
    global _voice_cache
    _voice_cache = scan_ref_voices()
    initial_choices = dropdown_choices(_voice_cache)

    with gr.Blocks(title="ZipVoice Vietnamese ONNX TTS") as demo:
        gen_txt_source_path = gr.State(value=None)
        gr.Markdown(
            """
# ZipVoice Vietnamese ONNX TTS

Inference qua **ONNX Runtime** (ZipVoice) + **Vocos PyTorch** (mặc định, chất lượng tốt nhất).
Giọng mẫu từ `assets/` · File xuất lưu vào `output/`
            """
        )
        if not pytorch_vocoder_ready():
            gr.Markdown(
                f"⚠️ **Thiếu PyTorch Vocos weights** trong `{VOCODER_DIR}` — "
                "cần `config.yaml` + `pytorch_model.bin`. "
                "Chạy `python download_models.py --pytorch-vocoder` hoặc copy từ repo PyTorch."
            )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Giọng mẫu (assets/)")
                voice_dropdown = gr.Dropdown(
                    label="Chọn giọng có sẵn",
                    choices=initial_choices,
                    value=MANUAL_CHOICE,
                    interactive=True,
                )
                refresh_btn = gr.Button("Làm mới danh sách", size="sm")
                asset_info = gr.Markdown(
                    f"Đã load **{len(_voice_cache)}** giọng từ `ref_info.json`"
                )
                ref_audio = gr.Audio(
                    label="Hoặc upload giọng mẫu (3–15 giây)",
                    type="filepath",
                )
            with gr.Column(scale=2):
                ref_text = gr.Textbox(
                    label="2) Nội dung TRONG file giọng mẫu (bắt buộc)",
                    placeholder="Tự điền từ ref_info.json khi chọn giọng",
                    lines=3,
                    max_lines=12,
                    elem_classes=[TEXTBOX_UNICODE_CLASS],
                )
                gen_text = gr.Textbox(
                    label="3) Văn bản CẦN ĐỌC (output TTS)",
                    placeholder="Văn bản bạn muốn tổng hợp bằng giọng mẫu...",
                    lines=12,
                    max_lines=40,
                    elem_classes=[TEXTBOX_UNICODE_CLASS],
                )
                gen_txt_file = gr.File(
                    label="Hoặc upload file .txt / .md (điền vào ô 3)",
                    file_types=[".txt", ".text", ".md"],
                    type="filepath",
                )
                input_mode = gr.Radio(
                    label="Chế độ văn bản đầu vào",
                    choices=[(label, key) for key, label in INPUT_MODE_CHOICES.items()],
                    value="raw",
                    info="Chọn «Đã chuẩn hóa» khi dùng file .txt đã xuất và chỉnh sửa — không chạy pipeline lại.",
                )

        with gr.Accordion("Preset — toàn bộ cấu hình đọc sách", open=False):
            gr.Markdown(
                "Preset lưu **giọng, pipeline chuẩn hóa, chunk, nghỉ, tổng hợp, xuất file**. "
                "Tải preset để áp dụng mọi thiết lập cho audiobook nhất quán."
            )
            _preset_choices = preset_dropdown_choices()
            preset_dropdown = gr.Dropdown(
                label="Chọn preset (profiles/*.json)",
                choices=_preset_choices,
                value=_preset_choices[0][1] if _preset_choices else None,
                interactive=True,
            )
            with gr.Row():
                preset_load_btn = gr.Button("Tải preset", variant="secondary")
                preset_save_name = gr.Textbox(
                    label="Tên lưu preset",
                    placeholder="vd: sach_ai_vy",
                    scale=2,
                )
                preset_save_btn = gr.Button("Lưu preset", scale=1)
            preset_status = gr.Markdown("")
            synth_num_step = gr.State(value=16)
            synth_guidance_scale = gr.State(value=1.0)
            synth_t_shift = gr.State(value=0.5)

        with gr.Row():
            speed = gr.Slider(0.3, 2.0, value=1.0, step=0.1, label="Tốc độ")
            export_format = gr.Dropdown(
                label="Định dạng xuất",
                choices=list(EXPORT_CHOICES.keys()),
                value="WAV 24kHz",
            )
            from onnx_quant import QUANT_MODE_CHOICES

            _default_quant = get_onnx_quant_mode()
            onnx_quant_mode = gr.Dropdown(
                label="ONNX quant mode",
                choices=list(QUANT_MODE_CHOICES),
                value=_default_quant,
                info="Tự khớp quantization.json hoặc auto-detect int4/int8/fp32 từ models/onnx/",
            )

        _norm_add_choices = [(label, key) for key, label in NORMALIZE_ADD_CHOICES.items()]
        _default_sel = pipeline_selector_choices(DEFAULT_NORMALIZE_PIPELINE)
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown(
                    "### Pipeline chuẩn hóa\n"
                    "1. **Chọn loại chuẩn hóa** → **Thêm vào pipeline** · "
                    "2. Chọn bước trong **Pipeline hiện tại** → **Lên** / **Xuống** / **Xóa** · "
                    "3. **Xem trước chuẩn hóa** và **TTS** dùng cùng pipeline.\n\n"
                    "Mặc định: **pipeline trống** (chỉ post-process). Preset sách: toàn bộ chuẩn hóa (sea-g2p → … → VieNeu)."
                )
                norm_pipeline_state = gr.State(value=[])
                with gr.Row():
                    norm_add_step = gr.Dropdown(
                        label="Chọn loại chuẩn hóa",
                        choices=_norm_add_choices,
                        value="vieneu",
                        scale=3,
                    )
                    norm_add_btn = gr.Button("Thêm vào pipeline", size="sm", scale=1)
                norm_sel_step = gr.Dropdown(
                    label="Pipeline hiện tại — chọn bước để sửa",
                    choices=_default_sel,
                    value=_default_sel[0][1] if _default_sel else None,
                )
                with gr.Row():
                    norm_up_btn = gr.Button("Lên", size="sm", min_width=60)
                    norm_down_btn = gr.Button("Xuống", size="sm", min_width=60)
                    norm_remove_btn = gr.Button("Xóa", size="sm", min_width=60)
                    norm_reset_btn = gr.Button("Mặc định", size="sm", min_width=80)
                    norm_audiobook_btn = gr.Button(
                        "Preset: Sách/Audiobook", size="sm", min_width=140
                    )
                    pipeline_quick_save_name = gr.Textbox(
                        label="Tên lưu preset",
                        placeholder="vd: sach_ai_vy",
                        scale=2,
                        min_width=120,
                    )
                    pipeline_quick_save_btn = gr.Button(
                        "Lưu preset", size="sm", min_width=90
                    )
                norm_pipeline_display = gr.Markdown(
                    value=format_normalize_pipeline_list([])
                )
            chunk_max_chars = gr.Slider(
                80,
                220,
                value=135,
                step=5,
                label="Max ký tự / chunk",
                info="ZipVoice ~100 token/chunk. Giảm nếu OOM; tăng nhẹ cho đoạn ngắn.",
            )
        with gr.Accordion("Tinh chỉnh nghỉ (audiobook)", open=False):
            with gr.Row():
                pause_sentence = gr.Slider(
                    0.1, 1.5, value=PAUSE_SENTENCE_DEFAULT, step=0.05,
                    label="Nghỉ câu (s)",
                )
                pause_paragraph = gr.Slider(
                    0.2, 2.0, value=PAUSE_PARAGRAPH_DEFAULT, step=0.05,
                    label="Nghỉ đoạn (s)",
                )
                pause_chapter = gr.Slider(
                    0.5, 4.0, value=PAUSE_CHAPTER_DEFAULT, step=0.1,
                    label="Nghỉ chương (s)",
                )
                pause_enum_item = gr.Slider(
                    0.2, 2.5, value=PAUSE_ENUM_DEFAULT, step=0.05,
                    label="Nghỉ mục liệt kê (s)",
                )
                pause_forced = gr.Slider(
                    0.05, 0.8, value=PAUSE_FORCED_SPLIT_DEFAULT, step=0.01,
                    label="Nghỉ cắt phẩy (s)",
                )

        _max_cpu_workers = max_parallel_workers(use_gpu=False)
        _default_gpu = is_onnx_gpu_env()
        with gr.Accordion("Hiệu năng", open=False):
            use_pytorch_vocoder = gr.Checkbox(
                label="Dùng Vocos PyTorch (khuyên dùng)",
                value=True,
                info=(
                    "Chất lượng tốt nhất. Weights: models/vocoder/config.yaml + "
                    "pytorch_model.bin (charactr/vocos-mel-24khz)."
                ),
            )
            onnx_vocoder_warning = gr.Markdown(
                value=(
                    "⚠️ **Vocoder ONNX + librosa ISTFT** có thể kém chất lượng hơn PyTorch Vocos, "
                    "đặc biệt với **INT4**. Chỉ dùng khi cần inference thuần ONNX (không torch)."
                ),
                visible=False,
            )
            use_onnx_gpu = gr.Checkbox(
                label="Dùng GPU (CUDA / DirectML)",
                value=_default_gpu,
                info=(
                    "Cần cài onnxruntime-gpu (NVIDIA CUDA) hoặc ORT có DirectML. "
                    "Env: ZIPVOICE_ONNX_GPU=1. INT4 có thể chạy CPU fallback."
                ),
            )
            parallel_workers = gr.Slider(
                1,
                _max_cpu_workers,
                value=1,
                step=1,
                label="Số luồng chunk song song (workers)",
                info=(
                    f"Mặc định 1 (tuần tự). CPU tối đa {_max_cpu_workers}; "
                    "GPU tối đa 2 (mỗi worker load ONNX riêng — tốn VRAM). "
                    "PyTorch vocoder → luôn tuần tự. Resume checkpoint → tự về 1."
                ),
            )
        use_pytorch_vocoder.change(
            _toggle_onnx_vocoder_warning,
            inputs=[use_pytorch_vocoder],
            outputs=[onnx_vocoder_warning],
        )

        with gr.Row():
            preview_norm_btn = gr.Button(
                "Xem trước chuẩn hóa (ô 3)",
                size="sm",
                variant="secondary",
            )
            export_norm_btn = gr.Button(
                "Xuất text đã chuẩn hóa (.txt)",
                size="sm",
                variant="secondary",
            )
        export_norm_path = gr.Textbox(
            label="Đường dẫn xuất .txt (tùy chọn)",
            placeholder="Để trống → output/<tên_file>_normalized.txt",
            lines=1,
        )
        export_norm_status = gr.Markdown("")
        norm_preview = gr.Textbox(
            label="Text đã chuẩn hóa (đầy đủ) — xem trước / sau TTS",
            lines=16,
            max_lines=50,
            interactive=False,
            elem_classes=[TEXTBOX_UNICODE_CLASS],
            placeholder="Cấu hình pipeline → nhập văn bản ô 3 → Xem trước hoặc Xuất .txt",
        )

        btn = gr.Button("Tổng hợp giọng nói (ONNX)", variant="primary")
        save_status = gr.Markdown("")

        with gr.Row():
            output_file = gr.File(label="Tải file output/", type="filepath")
            output_audio = gr.Audio(label="Nghe thử", type="filepath")

        gr.Markdown(
            """
**ONNX quant:** INT4 cần `text_encoder_int4.onnx` + `fm_decoder_int4.onnx` — export từ PyTorch GUI hoặc `quantize_onnx.py --mode int4`.

**Mẹo soạn manuscript (audiobook)**

- Xuống dòng = đoạn mới; dòng `Chương 1` / `Lời nói đầu` / `Phụ lục` → nghỉ chương dài hơn.
- Pipeline mặc định **trống** — giữ nguyên văn bản, chỉ dọn dấu câu. Dùng **Preset: Sách/Audiobook** khi cần pipeline đầy đủ (sea-g2p → … → VieNeu).
- **Xem trước chuẩn hóa** để QC ranh giới chunk; checkpoint tại `output/.checkpoints/latest/` để resume.

**Cấu trúc:** `models/onnx/` + `models/vocoder/` · Nghỉ mặc định: 0.35s/câu, 0.65s/đoạn, 2.0s/chương, 0.28s/cắt phẩy.
            """
        )

        gen_txt_file.change(
            _on_gen_txt_upload,
            inputs=[gen_txt_file],
            outputs=[gen_text, gen_txt_source_path],
        )
        refresh_btn.click(_refresh_voices, outputs=[voice_dropdown, asset_info])
        voice_dropdown.change(
            _on_voice_pick,
            inputs=[voice_dropdown],
            outputs=[ref_audio, ref_text, asset_info],
        )
        _pipeline_outputs = [
            norm_pipeline_state,
            norm_pipeline_display,
            norm_sel_step,
        ]
        norm_add_btn.click(
            _on_pipeline_add,
            inputs=[norm_pipeline_state, norm_add_step],
            outputs=_pipeline_outputs,
        )
        norm_remove_btn.click(
            _on_pipeline_remove,
            inputs=[norm_pipeline_state, norm_sel_step],
            outputs=_pipeline_outputs,
        )
        norm_up_btn.click(
            _on_pipeline_move_up,
            inputs=[norm_pipeline_state, norm_sel_step],
            outputs=_pipeline_outputs,
        )
        norm_down_btn.click(
            _on_pipeline_move_down,
            inputs=[norm_pipeline_state, norm_sel_step],
            outputs=_pipeline_outputs,
        )
        norm_reset_btn.click(_on_pipeline_reset, outputs=_pipeline_outputs)
        norm_audiobook_btn.click(_on_pipeline_audiobook_preset, outputs=_pipeline_outputs)
        pipeline_quick_save_btn.click(
            _on_save_preset,
            inputs=[
                pipeline_quick_save_name,
                voice_dropdown,
                ref_audio,
                ref_text,
                norm_pipeline_state,
                chunk_max_chars,
                pause_sentence,
                pause_paragraph,
                pause_chapter,
                pause_enum_item,
                pause_forced,
                speed,
                export_format,
                onnx_quant_mode,
                synth_num_step,
                synth_guidance_scale,
                synth_t_shift,
                input_mode,
                parallel_workers,
                use_onnx_gpu,
                use_pytorch_vocoder,
            ],
            outputs=[preset_dropdown, preset_status],
        )
        preset_load_btn.click(
            _on_load_preset,
            inputs=[preset_dropdown],
            outputs=[
                voice_dropdown,
                ref_audio,
                ref_text,
                asset_info,
                norm_pipeline_state,
                norm_pipeline_display,
                norm_sel_step,
                chunk_max_chars,
                pause_sentence,
                pause_paragraph,
                pause_chapter,
                pause_enum_item,
                pause_forced,
                speed,
                export_format,
                onnx_quant_mode,
                synth_num_step,
                synth_guidance_scale,
                synth_t_shift,
                input_mode,
                parallel_workers,
                use_onnx_gpu,
                use_pytorch_vocoder,
                onnx_vocoder_warning,
                preset_status,
            ],
        )
        preset_save_btn.click(
            _on_save_preset,
            inputs=[
                preset_save_name,
                voice_dropdown,
                ref_audio,
                ref_text,
                norm_pipeline_state,
                chunk_max_chars,
                pause_sentence,
                pause_paragraph,
                pause_chapter,
                pause_enum_item,
                pause_forced,
                speed,
                export_format,
                onnx_quant_mode,
                synth_num_step,
                synth_guidance_scale,
                synth_t_shift,
                input_mode,
                parallel_workers,
                use_onnx_gpu,
                use_pytorch_vocoder,
            ],
            outputs=[preset_dropdown, preset_status],
        )
        preview_norm_btn.click(
            preview_normalize,
            inputs=[
                gen_text,
                gen_txt_source_path,
                norm_pipeline_state,
                chunk_max_chars,
                input_mode,
            ],
            outputs=[norm_preview],
        )
        export_norm_btn.click(
            export_normalized_text,
            inputs=[
                gen_text,
                gen_txt_source_path,
                norm_pipeline_state,
                input_mode,
                export_norm_path,
                gen_txt_file,
            ],
            outputs=[norm_preview, export_norm_status],
        )
        btn.click(
            infer_tts,
            inputs=[
                voice_dropdown,
                ref_audio,
                ref_text,
                gen_text,
                speed,
                export_format,
                norm_pipeline_state,
                chunk_max_chars,
                pause_sentence,
                pause_paragraph,
                pause_chapter,
                pause_enum_item,
                pause_forced,
                onnx_quant_mode,
                synth_num_step,
                synth_guidance_scale,
                synth_t_shift,
                input_mode,
                gen_txt_source_path,
                parallel_workers,
                use_onnx_gpu,
                use_pytorch_vocoder,
            ],
            outputs=[output_file, output_audio, save_status, norm_preview],
            concurrency_limit=1,
        )

    return demo


if __name__ == "__main__":
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    host = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7862"))
    share = os.environ.get("GRADIO_SHARE", "0") == "1"

    logger.info(
        "Starting Gradio ONNX on http://%s:%s (force_cpu=%s, onnx_gpu_env=%s)",
        host,
        port,
        is_force_cpu(),
        is_onnx_gpu_env(),
    )
    demo = build_ui()
    demo.queue(default_concurrency_limit=1).launch(
        server_name=host,
        server_port=port,
        share=share,
        show_error=True,
        allowed_paths=[str(ROOT), str(ASSETS_DIR), str(OUTPUT_DIR)],
        theme=gr.themes.Soft(),
        css=GRADIO_UNICODE_TEXTBOX_CSS,
    )
