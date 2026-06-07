"""
ZipVoice Vietnamese ONNX TTS — local Gradio GUI.
ONNX weights bundled; vocoder + tokenizer runtime via setup once.
"""
from __future__ import annotations

import gc
import os
import sys
import tempfile
import traceback

import gradio as gr
import numpy as np

from assets_loader import MANUAL_CHOICE, dropdown_choices, get_voice_by_id, scan_ref_voices
from config import (
    ASSETS_DIR,
    OUTPUT_DIR,
    ROOT,
    ensure_ffmpeg_on_path,
    is_force_cpu,
    models_ready,
    onnx_ready,
)
from export_audio import EXPORT_CHOICES, save_output
from runtime_log import setup_runtime_logging

ensure_ffmpeg_on_path()
logger = setup_runtime_logging(name="zipvoice_onnx_gui")

if not models_ready():
    print(
        "\n[ERROR] Runtime not ready. Run setup first:\n"
        "  Windows:  install_cpu.bat\n"
        "  PowerShell: .\\setup_cpu.ps1\n"
    )
    if not onnx_ready():
        print("[ERROR] Missing ONNX files in models/onnx/")
    sys.exit(1)

from onnx_engine import OnnxTTSEngine  # noqa: E402
from utils import (  # noqa: E402
    NORMALIZE_STEP_CHOICES,
    build_normalize_pipeline,
    format_normalize_pipeline,
    join_tts_audio_chunks,
    prepare_tts_text,
    preprocess_ref_audio_text,
    read_text_file,
    save_spectrogram,
    preview_normalize_output,
    split_text_for_tts,
)

MAX_GEN_TEXT_CHARS = 500_000
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


def _on_gen_txt_upload(file_path: str | None) -> str:
    if not file_path:
        return gr.update()
    try:
        return read_text_file(file_path)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc


def preview_normalize(
    gen_text: str,
    norm_step1: str,
    norm_step2: str,
    norm_step3: str,
    chunk_max_chars: int,
) -> str:
    try:
        return preview_normalize_output(
            gen_text,
            norm_step1,
            norm_step2,
            norm_step3,
            chunk_max_chars=int(chunk_max_chars),
        )
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    except ImportError as exc:
        raise gr.Error(str(exc)) from exc


def _resolve_ref_path(ref_audio_path: str | None, asset_voice_id: str) -> str | None:
    if ref_audio_path:
        return ref_audio_path
    voice = get_voice_by_id(asset_voice_id, _voice_cache)
    if voice:
        return voice.audio_path
    return None


def infer_tts(
    asset_voice_id: str,
    ref_audio_path: str | None,
    ref_text: str,
    gen_text: str,
    speed: float,
    export_format: str,
    norm_step1: str,
    norm_step2: str,
    norm_step3: str,
    chunk_max_chars: int,
    use_int8: bool,
    progress=gr.Progress(),
) -> tuple[str | None, tuple[int, np.ndarray] | None, str | None, str]:
    ref_audio_path = _resolve_ref_path(ref_audio_path, asset_voice_id)
    if not ref_audio_path:
        raise gr.Error("Chọn giọng từ menu assets/ hoặc upload file giọng mẫu.")
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
            norm_pipeline = build_normalize_pipeline(norm_step1, norm_step2, norm_step3)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc

        voice = get_voice_by_id(asset_voice_id, _voice_cache)
        norm_label = format_normalize_pipeline(norm_pipeline)
        logger.info(
            "infer_tts | asset=%s | gen_len=%d | ref_len=%d | export=%s | norm=%s | chunk=%d | int8=%s",
            asset_voice_id,
            len(gen_text),
            len(ref_text.strip()),
            export_format,
            norm_label,
            chunk_max_chars,
            use_int8,
        )

        engine = OnnxTTSEngine.get(use_int8=use_int8)
        tts_chunks = split_text_for_tts(gen_text, max_chars=int(chunk_max_chars))
        wave_parts: list[np.ndarray] = []
        ref_audio = ""
        resolved_ref_text = ""
        normalized_preview = ""

        for i, tts_chunk in enumerate(tts_chunks):
            progress(
                (i + 1) / len(tts_chunks),
                desc=f"Đang tổng hợp đoạn {i + 1}/{len(tts_chunks)} (ONNX)...",
            )

            if i == 0:
                ref_audio, resolved_ref_text = preprocess_ref_audio_text(
                    ref_audio_path,
                    ref_text,
                    show_info=logger.info,
                )
            else:
                resolved_ref_text = ref_text.strip() or resolved_ref_text

            prompt_normalized = prepare_tts_text(resolved_ref_text, norm_pipeline)
            normalized = prepare_tts_text(tts_chunk.text, norm_pipeline)
            if not normalized_preview:
                normalized_preview = normalized[:500] + (
                    "…" if len(normalized) > 500 else ""
                )

            wav = engine.generate(
                prompt_text=prompt_normalized,
                prompt_wav=ref_audio,
                text=normalized,
                speed=speed,
            ).detach().numpy()[0]
            wave_parts.append(wav)
            del wav
            if is_force_cpu():
                gc.collect()

        final_wave = join_tts_audio_chunks(wave_parts, tts_chunks, engine.sampling_rate)
        assert final_wave is not None and final_wave.size > 0

        saved = save_output(
            final_wave,
            engine.sampling_rate,
            export_format,
            voice_label=asset_voice_id or "upload",
            text_preview=gen_text,
        )

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            spec_path = tmp.name
        save_spectrogram(final_wave, spec_path)

        est_min = max(1, int(len(tts_chunks) * 5 / 60))
        onnx_mode = "INT8" if use_int8 else "FP32"
        status = (
            f"Đã lưu: `{saved}`\n\n"
            f"**ONNX:** {onnx_mode} · **Chuẩn hóa:** {norm_label} · "
            f"**Chunks:** {len(tts_chunks)} (max {chunk_max_chars} ký tự/chunk, "
            f"~{est_min} phút CPU ước tính)\n\n"
            f"**Text sau chuẩn hóa (đoạn 1):** {normalized_preview or '(trống)'}"
        )
        logger.info("infer_tts done | file=%s", saved)
        return str(saved), (engine.sampling_rate, final_wave), spec_path, status

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
        gr.Markdown(
            """
# ZipVoice Vietnamese ONNX TTS

Inference qua **ONNX Runtime** (không cần checkpoint PyTorch ~470 MB).
Giọng mẫu từ `assets/` · File xuất lưu vào `output/`
            """
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
                )
                gen_text = gr.Textbox(
                    label="3) Văn bản CẦN ĐỌC (output TTS)",
                    placeholder="Văn bản bạn muốn tổng hợp bằng giọng mẫu...",
                    lines=8,
                )
                gen_txt_file = gr.File(
                    label="Hoặc upload file .txt / .md (điền vào ô 3)",
                    file_types=[".txt", ".text", ".md"],
                    type="filepath",
                )

        with gr.Row():
            speed = gr.Slider(0.3, 2.0, value=1.0, step=0.1, label="Tốc độ")
            export_format = gr.Dropdown(
                label="Định dạng xuất",
                choices=list(EXPORT_CHOICES.keys()),
                value="WAV 24kHz",
            )
            use_int8 = gr.Checkbox(
                label="Dùng ONNX INT8 (nhanh hơn, nhẹ hơn RAM)",
                value=True,
            )

        _norm_choices = [(label, key) for key, label in NORMALIZE_STEP_CHOICES.items()]
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown(
                    "### Chuẩn hóa text (pipeline, tối đa 3 bước)\n"
                    "Áp dụng lần lượt A → B → C. Cả 3 thư viện chỉ trả về **text** "
                    "(không phoneme) nên có thể xếp chuỗi."
                )
                with gr.Row():
                    norm_step1 = gr.Dropdown(
                        label="Bước 1",
                        choices=_norm_choices,
                        value="vinorm",
                    )
                    norm_step2 = gr.Dropdown(
                        label="Bước 2 (tuỳ chọn)",
                        choices=_norm_choices,
                        value="none",
                    )
                    norm_step3 = gr.Dropdown(
                        label="Bước 3 (tuỳ chọn)",
                        choices=_norm_choices,
                        value="none",
                    )
            chunk_max_chars = gr.Slider(
                80,
                220,
                value=135,
                step=5,
                label="Max ký tự / chunk",
                info="ZipVoice ~100 token/chunk. Giảm nếu OOM; tăng nhẹ cho đoạn ngắn.",
            )

        with gr.Row():
            preview_norm_btn = gr.Button(
                "Xem trước chuẩn hóa (ô 3)",
                size="sm",
                variant="secondary",
            )
        norm_preview = gr.Textbox(
            label="Kết quả pipeline chuẩn hóa — chạy trước khi TTS",
            lines=12,
            max_lines=24,
            interactive=False,
            placeholder="Chọn pipeline → nhập văn bản ô 3 → bấm nút trên",
        )

        btn = gr.Button("Tổng hợp giọng nói (ONNX)", variant="primary")
        save_status = gr.Markdown("")

        with gr.Row():
            output_file = gr.File(label="Tải file output/", type="filepath")
            output_audio = gr.Audio(label="Nghe thử", type="filepath")
        output_spec = gr.Image(label="Spectrogram")

        gr.Markdown(
            """
**Văn bản dài / sách:** upload `.txt` hoặc `.md` → chia theo đoạn (`\\n`) → câu → gộp đến max chunk;
nghỉ **0.35s/câu**, **0.65s/đoạn**, **1.2s/tiêu đề chương**.

**Cấu trúc:** `models/onnx/` (đã có sẵn) · `models/vocoder/` (setup tải) · `vendor/ZipVoice` (tokenizer)
            """
        )

        gen_txt_file.change(_on_gen_txt_upload, inputs=[gen_txt_file], outputs=[gen_text])
        refresh_btn.click(_refresh_voices, outputs=[voice_dropdown, asset_info])
        voice_dropdown.change(
            _on_voice_pick,
            inputs=[voice_dropdown],
            outputs=[ref_audio, ref_text, asset_info],
        )
        preview_norm_btn.click(
            preview_normalize,
            inputs=[gen_text, norm_step1, norm_step2, norm_step3, chunk_max_chars],
            outputs=[norm_preview],
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
                norm_step1,
                norm_step2,
                norm_step3,
                chunk_max_chars,
                use_int8,
            ],
            outputs=[output_file, output_audio, output_spec, save_status],
            concurrency_limit=1,
        )

    return demo


if __name__ == "__main__":
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    host = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7862"))
    share = os.environ.get("GRADIO_SHARE", "0") == "1"

    logger.info("Starting Gradio ONNX on http://%s:%s (cpu=%s)", host, port, is_force_cpu())
    demo = build_ui()
    demo.queue(default_concurrency_limit=1).launch(
        server_name=host,
        server_port=port,
        share=share,
        show_error=True,
        allowed_paths=[str(ROOT), str(ASSETS_DIR), str(OUTPUT_DIR)],
        theme=gr.themes.Soft(),
    )
