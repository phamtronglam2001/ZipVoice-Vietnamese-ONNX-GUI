"""
TTS controller for Slint GUI — wraps tts_pipeline without Gradio.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from assets_loader import (
    MANUAL_CHOICE,
    RefVoice,
    dropdown_choices,
    format_voice_load_summary,
    get_voice_by_id,
    scan_ref_voices,
)
from config import (
    is_force_cpu,
    is_onnx_gpu_env,
    onnx_quant_mode as get_onnx_quant_mode,
    onnx_quant_mode_source,
)
from export_audio import EXPORT_CHOICES
from onnx_providers import predict_runtime_device_summary
from onnx_quant import QUANT_MODE_CHOICES
from status_log import StatusLog
from runtime_log import LOG_FILE
from text.normalizers import (
    AUDIOBOOK_PRESET_PIPELINE,
    DEFAULT_NORMALIZE_PIPELINE,
    NORMALIZE_ADD_CHOICES,
    build_normalize_pipeline,
    format_normalize_pipeline,
    format_normalize_pipeline_list,
    pipeline_add_step,
    pipeline_move,
    pipeline_remove_at,
)
from text.chunking import (
    PAUSE_CHAPTER_DEFAULT,
    PAUSE_ENUM_DEFAULT,
    PAUSE_FORCED_SPLIT_DEFAULT,
    PAUSE_PARAGRAPH_DEFAULT,
    PAUSE_SENTENCE_DEFAULT,
)
from text.pipeline import INPUT_MODE_CHOICES, parse_input_mode, preview_chunks_output, preview_normalize_output
from tts_pipeline import (
    TTSProgress,
    TTSRequest,
    TTSResult,
    TTSError,
    iter_tts_pipeline,
    resolve_gen_text_for_pipeline,
)

logger = logging.getLogger("zipvoice_slint_gui")


@dataclass
class SlintGuiState:
    voice_id: str = MANUAL_CHOICE
    ref_audio_path: str = ""
    ref_text: str = ""
    gen_text: str = ""
    gen_txt_source_path: str = ""
    input_mode: str = "raw"
    speed: float = 1.0
    export_format: str = "WAV 24kHz"
    onnx_quant_mode: str = field(default_factory=get_onnx_quant_mode)
    norm_pipeline: list[str] = field(default_factory=list)
    chunk_max_chars: int = 135
    chunk_min_chars: int = 70
    pause_sentence: float = PAUSE_SENTENCE_DEFAULT
    pause_paragraph: float = PAUSE_PARAGRAPH_DEFAULT
    pause_chapter: float = PAUSE_CHAPTER_DEFAULT
    pause_enum_item: float = PAUSE_ENUM_DEFAULT
    pause_forced: float = PAUSE_FORCED_SPLIT_DEFAULT
    synth_num_step: int = 16
    synth_guidance_scale: float = 1.0
    synth_t_shift: float = 0.5
    parallel_workers: int = 1
    use_onnx_gpu: bool = field(default_factory=is_onnx_gpu_env)
    inference_batch_size: int = 1
    ode_solver: str = "euler"
    pipeline_overlap: bool = True


class TTSController:
    """State + inference for Slint; thread-safe synthesis with callbacks."""

    def __init__(self) -> None:
        self.voices: list[RefVoice] = scan_ref_voices()
        self.state = SlintGuiState()
        self.status_log = StatusLog()
        self._busy = False
        self._lock = threading.Lock()

    def refresh_voices(self) -> tuple[list[tuple[str, str]], str]:
        self.voices = scan_ref_voices()
        choices = dropdown_choices(self.voices)
        info = format_voice_load_summary(self.voices)
        return choices, info

    def voice_labels(self) -> list[str]:
        return [label for label, _vid in dropdown_choices(self.voices)]

    def voice_ids(self) -> list[str]:
        return [vid for _label, vid in dropdown_choices(self.voices)]

    def pick_voice(self, voice_id: str) -> tuple[str, str, str]:
        """Return (ref_audio_path, ref_text, note)."""
        voice = get_voice_by_id(voice_id, self.voices)
        if voice is None:
            return "", "", "Chế độ: upload thủ công"
        note = f"Đã chọn: {voice.id}"
        if not voice.transcript:
            note += " — chưa có transcript, bắt buộc điền ô số 2."
        return voice.audio_path, voice.transcript, note

    def pipeline_display(self) -> str:
        try:
            steps = build_normalize_pipeline(self.state.norm_pipeline)
            self.state.norm_pipeline = steps
        except ValueError:
            steps = []
        return format_normalize_pipeline_list(steps)

    def pipeline_add(self, step_key: str) -> str:
        try:
            self.state.norm_pipeline = pipeline_add_step(self.state.norm_pipeline, step_key)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return self.pipeline_display()

    def pipeline_remove(self, index: int) -> str:
        self.state.norm_pipeline = pipeline_remove_at(
            self.state.norm_pipeline, str(index) if index >= 0 else None
        )
        return self.pipeline_display()

    def pipeline_move(self, index: int, direction: int) -> str:
        self.state.norm_pipeline = pipeline_move(
            self.state.norm_pipeline, str(index), direction
        )
        return self.pipeline_display()

    def pipeline_reset(self) -> str:
        self.state.norm_pipeline = list(DEFAULT_NORMALIZE_PIPELINE)
        return self.pipeline_display()

    def pipeline_audiobook(self) -> str:
        self.state.norm_pipeline = list(AUDIOBOOK_PRESET_PIPELINE)
        return self.pipeline_display()

    def predicted_runtime_device(self) -> str:
        use_gpu = bool(self.state.use_onnx_gpu) and not is_force_cpu()
        return predict_runtime_device_summary(use_gpu)

    def build_request(self) -> TTSRequest:
        s = self.state
        ref_path = s.ref_audio_path.strip() or None
        gen_src = s.gen_txt_source_path.strip() or None
        return TTSRequest(
            asset_voice_id=s.voice_id,
            ref_audio_path=ref_path,
            ref_text=s.ref_text,
            gen_text=s.gen_text,
            speed=s.speed,
            export_format=s.export_format,
            norm_pipeline=list(s.norm_pipeline),
            chunk_max_chars=s.chunk_max_chars,
            chunk_min_chars=s.chunk_min_chars,
            pause_sentence=s.pause_sentence,
            pause_paragraph=s.pause_paragraph,
            pause_chapter=s.pause_chapter,
            pause_enum_item=s.pause_enum_item,
            pause_forced=s.pause_forced,
            onnx_quant_mode=s.onnx_quant_mode,
            synth_num_step=s.synth_num_step,
            synth_guidance_scale=s.synth_guidance_scale,
            synth_t_shift=s.synth_t_shift,
            input_mode=s.input_mode,
            gen_txt_source_path=gen_src,
            parallel_workers=s.parallel_workers,
            use_onnx_gpu=s.use_onnx_gpu,
            ode_seed=42,
            use_fixed_seed=True,
            ode_solver=s.ode_solver,
            inference_batch_size=s.inference_batch_size,
            pipeline_overlap=s.pipeline_overlap,
        )

    @property
    def is_busy(self) -> bool:
        with self._lock:
            return self._busy

    def synthesize_async(
        self,
        on_progress: Callable[[float, str], None],
        on_log: Callable[[str], None],
        on_done: Callable[[TTSResult | TTSError], None],
        on_snapshot: Callable[[str, str, str], None] | None = None,
    ) -> bool:
        """Start synthesis on a worker thread.

        on_snapshot(norm_preview, status_log_text, runtime_device) fires on each
        pipeline progress tick (same streaming model as Gradio).
        """
        with self._lock:
            if self._busy:
                return False
            self._busy = True

        def worker() -> None:
            self.status_log.clear()
            on_log(self.status_log.text())

            def progress(frac: float, desc: str | None = None) -> None:
                on_progress(float(frac), desc or "")

            def notify_log() -> None:
                on_log(self.status_log.text())

            def notify_snapshot(
                *,
                norm_preview: str = "",
                runtime_device: str = "",
            ) -> None:
                if on_snapshot is None:
                    return
                on_snapshot(
                    norm_preview,
                    self.status_log.text(),
                    runtime_device,
                )

            req = self.build_request()
            result: TTSResult | TTSError | None = None
            runtime_device = self.predicted_runtime_device()
            notify_snapshot(runtime_device=runtime_device)

            for event in iter_tts_pipeline(
                req,
                voice_cache=self.voices,
                progress=progress,
                status_log=self.status_log,
                on_log=notify_log,
            ):
                if isinstance(event, TTSProgress):
                    notify_log()
                    notify_snapshot(
                        norm_preview=event.norm_preview,
                        runtime_device=event.runtime_device or runtime_device,
                    )
                    if event.runtime_device:
                        runtime_device = event.runtime_device
                elif isinstance(event, (TTSResult, TTSError)):
                    result = event

            on_log(self.status_log.text())
            if result is not None:
                on_done(result)
            with self._lock:
                self._busy = False

        threading.Thread(target=worker, daemon=True).start()
        return True

    def startup_log_text(self) -> str:
        """Environment banner shown in the log tab at startup."""
        s = self.state
        log = StatusLog(mirror_console=False)
        log.heading("Slint GUI — sẵn sàng")
        log.info(f"File log: {LOG_FILE}")
        log.info(f"Giọng assets: {len(self.voices)}")
        quant = get_onnx_quant_mode()
        log.info(f"ONNX quant mặc định: {quant} (nguồn: {onnx_quant_mode_source()})")
        gpu_on = s.use_onnx_gpu and not is_force_cpu()
        log.info(f"GPU checkbox/env: {gpu_on}")
        log.info(f"Thiết bị dự kiến: {self.predicted_runtime_device()}")
        if gpu_on and quant == "int4":
            log.warn(
                "int4 (MatMulNBits): fm_decoder thường chạy CPU trên ORT — "
                "Dedicated VRAM thấp, Task Manager có thể chỉ thấy «Copy». "
                "Thử int8 trong tab «Hiệu năng & synth» để dùng ~1GB VRAM như Gradio."
            )
        elif gpu_on and quant == "int8":
            log.info(
                "int8 + CUDA: kỳ vọng Dedicated GPU memory ~1GB và 3D/CUDA khi tổng hợp "
                "(không chỉ Copy)."
            )
        log.blank()
        log.info(
            "Sau «Tổng hợp», xem dòng ORT text_encoder / fm_decoder / vocoder trong log — "
            "CPUExecutionProvider = chạy CPU dù bật GPU."
        )
        log.info("Bấm «Xem trước chuẩn hóa» để xem chunk; «Tổng hợp» để chạy ONNX.")
        return log.text()

    def append_ui_log(self, msg: str, *, level: str = "info") -> str:
        """Append a line to the shared status log (UI actions, pipeline edits)."""
        if level == "warn":
            self.status_log.warn(msg)
        elif level == "error":
            self.status_log.error(msg)
        else:
            self.status_log.info(msg)
        return self.status_log.text()

    def preview_normalize(self) -> tuple[str, str, str]:
        """Return (norm_preview, chunk_preview, status_log_text)."""
        s = self.state
        gen_src = s.gen_txt_source_path.strip() or None
        source_text = resolve_gen_text_for_pipeline(s.gen_text, gen_src)
        input_mode = s.input_mode
        pipeline = build_normalize_pipeline(s.norm_pipeline)
        pipeline_label = format_normalize_pipeline(pipeline)

        log = StatusLog(mirror_console=False)
        log.heading("Xem trước chuẩn hóa & chunk")
        log.info(f"Nguồn: {gen_src or '(ô nhập trực tiếp)'}")
        log.info(f"input_mode={input_mode}")
        log.info(f"pipeline={pipeline_label or '(trống)'}")
        log.info(
            f"chunk min={s.chunk_min_chars}, max={s.chunk_max_chars}; "
            f"nghỉ câu={s.pause_sentence}s, đoạn={s.pause_paragraph}s, "
            f"chương={s.pause_chapter}s, enum={s.pause_enum_item}s, "
            f"cắt={s.pause_forced}s"
        )
        log.info(f"gen_text_len={len(source_text)} ký tự")

        norm_preview = preview_normalize_output(
            source_text,
            pipeline,
            chunk_max_chars=int(s.chunk_max_chars),
            chunk_min_chars=int(s.chunk_min_chars),
            mode=parse_input_mode(input_mode),
            pause_sentence=float(s.pause_sentence),
            pause_paragraph=float(s.pause_paragraph),
            pause_chapter=float(s.pause_chapter),
            pause_enum_item=float(s.pause_enum_item),
            pause_forced_split=float(s.pause_forced),
            include_chunk_preview=True,
            max_preview_chars=4000,
        )
        chunk_preview = preview_chunks_output(
            source_text,
            pipeline,
            chunk_max_chars=int(s.chunk_max_chars),
            chunk_min_chars=int(s.chunk_min_chars),
            mode=parse_input_mode(input_mode),
            pause_sentence=float(s.pause_sentence),
            pause_paragraph=float(s.pause_paragraph),
            pause_chapter=float(s.pause_chapter),
            pause_enum_item=float(s.pause_enum_item),
            pause_forced_split=float(s.pause_forced),
            show_micro_merge=True,
        )
        log.blank()
        log.info(f"── Chi tiết chunk: {len(chunk_preview.splitlines())} dòng (bên dưới) ──")
        return norm_preview, chunk_preview, log.text() + "\n\n" + chunk_preview

    @staticmethod
    def export_format_labels() -> list[str]:
        return list(EXPORT_CHOICES.keys())

    @staticmethod
    def quant_mode_labels() -> list[str]:
        return list(QUANT_MODE_CHOICES)

    @staticmethod
    def input_mode_labels() -> list[str]:
        return list(INPUT_MODE_CHOICES.values())

    @staticmethod
    def input_mode_keys() -> list[str]:
        return list(INPUT_MODE_CHOICES.keys())

    @staticmethod
    def normalize_step_labels() -> list[str]:
        return list(NORMALIZE_ADD_CHOICES.values())

    @staticmethod
    def normalize_step_keys() -> list[str]:
        return list(NORMALIZE_ADD_CHOICES.keys())

    def load_gen_text_file(self, path: str) -> tuple[str, str]:
        from text.io import read_text_file

        text = read_text_file(path)
        self.state.gen_text = text
        self.state.gen_txt_source_path = path
        return text, path

    def set_ref_audio_file(self, path: str) -> None:
        self.state.ref_audio_path = path
        self.state.voice_id = MANUAL_CHOICE
