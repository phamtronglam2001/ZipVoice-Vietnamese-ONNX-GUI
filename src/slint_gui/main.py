"""
ZipVoice Vietnamese ONNX TTS — Slint GUI entry point.

Run from project root:
    run_slint_gui.bat
Or with PYTHONPATH:
    set PYTHONPATH=src
    python src/slint_gui/main.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import (  # noqa: E402
    ASSETS_DIR,
    OUTPUT_DIR,
    ensure_ffmpeg_on_path,
    is_force_cpu,
    models_ready,
    onnx_ready,
    vocoder_deploy_instructions,
    vocoder_onnx_ready,
)
from chunk_synthesis import (  # noqa: E402
    clamp_parallel_workers,
    max_parallel_workers,
    ui_parallel_workers_max,
)
from runtime_log import setup_runtime_logging  # noqa: E402

ensure_ffmpeg_on_path()
logger = setup_runtime_logging(name="zipvoice_slint_gui")

if not models_ready():
    print(
        "\n[ERROR] Runtime not ready. Run setup first:\n"
        "  Windows: install_cpu.bat\n"
    )
    if not onnx_ready():
        print("[ERROR] Missing ONNX files in models/onnx/")
    if not vocoder_onnx_ready():
        print("[ERROR] Missing ONNX vocoder in models/vocoder/ (mel_spec_24khz.onnx)")
        print(vocoder_deploy_instructions())
    sys.exit(1)

try:
    import slint  # noqa: E402
except ImportError:
    print(
        "[ERROR] Slint chưa cài. Chạy:\n"
        "  .venv\\Scripts\\python.exe -m pip install -r requirements-slint.txt\n"
        "  hoặc: run_slint_gui.bat\n"
    )
    sys.exit(1)

from assets_loader import MANUAL_CHOICE  # noqa: E402
from slint_gui.backend.tts_controller import TTSController  # noqa: E402
from slint_gui.slint_utils import bind_string_list_model, slint_int  # noqa: E402
from slint_gui.theme_prefs import load_dark_mode, save_dark_mode  # noqa: E402
from tts_pipeline import TTSResult, TTSError  # noqa: E402

UI_PATH = Path(__file__).resolve().parent / "ui" / "app.slint"
components = slint.load_file(str(UI_PATH))


def _invoke_ui(fn) -> None:
    """Schedule UI update on Slint event loop (safe from worker threads)."""
    try:
        slint.invoke_from_event_loop(fn)
    except AttributeError:
        fn()


def _pick_file(title: str, patterns: list[tuple[str, str]]) -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(title=title, filetypes=patterns)
        root.destroy()
        return path or None
    except Exception as exc:
        logger.warning("File dialog failed: %s", exc)
        return None


class MainWindow(components.MainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.controller = TTSController()
        self._sync_static_options()
        self._sync_voices()
        self.pipeline_display = self.controller.pipeline_display()
        self.runtime_device = self.controller.predicted_runtime_device()
        self.dark_mode = load_dark_mode(default=False)
        self.apply_theme()

    @slint.callback
    def theme_toggle_clicked(self) -> None:
        self.dark_mode = not bool(self.dark_mode)
        self.apply_theme()
        save_dark_mode(bool(self.dark_mode))

    def _sync_static_options(self) -> None:
        c = self.controller
        bind_string_list_model(self, "export_format_options", c.export_format_labels())
        bind_string_list_model(self, "quant_mode_options", c.quant_mode_labels())
        bind_string_list_model(self, "norm_step_options", c.normalize_step_labels())
        bind_string_list_model(self, "input_mode_options", c.input_mode_labels())
        try:
            self.quant_mode_index = c.quant_mode_labels().index(c.state.onnx_quant_mode)
        except ValueError:
            self.quant_mode_index = 0
        step_keys = c.normalize_step_keys()
        try:
            self.norm_step_index = step_keys.index("vieneu")
        except ValueError:
            self.norm_step_index = 0
        self.use_onnx_gpu = c.state.use_onnx_gpu
        self._sync_parallel_workers_limits()

    def _worker_hint(self, use_gpu: bool) -> str:
        max_w = max_parallel_workers(use_gpu=use_gpu)
        if use_gpu:
            return (
                f"GPU: khuyến nghị 1 worker (tuần tự). Tối đa {max_w} — "
                "mỗi worker load ONNX riêng, tốn VRAM. Env: ZIPVOICE_GPU_MAX_WORKERS."
            )
        return (
            f"Mặc định 1 (tuần tự). CPU tối đa {max_w} workers. "
            "Env: ZIPVOICE_CPU_MAX_WORKERS."
        )

    def _sync_parallel_workers_limits(self) -> None:
        use_gpu = bool(self.use_onnx_gpu) and not is_force_cpu()
        self.parallel_workers_max = ui_parallel_workers_max(use_gpu=use_gpu)
        self.parallel_workers = clamp_parallel_workers(
            int(self.parallel_workers), use_gpu=use_gpu
        )
        self.workers_hint = self._worker_hint(use_gpu)

    def _sync_voices(self) -> None:
        _choices, info = self.controller.refresh_voices()
        bind_string_list_model(self, "voice_options", self.controller.voice_labels())
        self.asset_info = info
        self.voice_index = 0
        self.controller.state.voice_id = MANUAL_CHOICE

    def _pull_state_from_ui(self) -> None:
        s = self.controller.state
        ids = self.controller.voice_ids()
        idx = min(max(0, slint_int(self.voice_index)), max(0, len(ids) - 1))
        s.voice_id = ids[idx] if ids else MANUAL_CHOICE
        s.ref_audio_path = self.ref_audio_path
        s.ref_text = self.ref_text
        s.gen_text = self.gen_text
        s.gen_txt_source_path = self.gen_txt_source_path
        mode_keys = self.controller.input_mode_keys()
        mode_i = min(slint_int(self.input_mode_index), max(0, len(mode_keys) - 1))
        s.input_mode = mode_keys[mode_i]
        s.speed = float(self.speed_value)
        exp = self.controller.export_format_labels()
        exp_i = min(slint_int(self.export_format_index), max(0, len(exp) - 1))
        s.export_format = exp[exp_i]
        quants = self.controller.quant_mode_labels()
        quant_i = min(slint_int(self.quant_mode_index), max(0, len(quants) - 1))
        s.onnx_quant_mode = quants[quant_i]
        s.chunk_max_chars = slint_int(self.chunk_max_chars, 135)
        s.pause_sentence = float(self.pause_sentence)
        s.pause_paragraph = float(self.pause_paragraph)
        s.pause_chapter = float(self.pause_chapter)
        s.pause_enum_item = float(self.pause_enum)
        s.pause_forced = float(self.pause_forced)
        s.use_onnx_gpu = bool(self.use_onnx_gpu)
        use_gpu = s.use_onnx_gpu and not is_force_cpu()
        s.parallel_workers = clamp_parallel_workers(
            int(self.parallel_workers), use_gpu=use_gpu
        )
        self.parallel_workers = s.parallel_workers
        s.synth_num_step = int(self.synth_num_step)
        s.synth_guidance_scale = float(self.synth_guidance_scale)
        s.synth_t_shift = float(self.synth_t_shift)

    @slint.callback
    def gpu_setting_changed(self) -> None:
        self._pull_state_from_ui()
        self._sync_parallel_workers_limits()
        self.runtime_device = self.controller.predicted_runtime_device()

    @slint.callback
    def refresh_voices(self) -> None:
        self._sync_voices()

    @slint.callback
    def voice_changed(self, index: int) -> None:
        index = slint_int(index)
        self.voice_index = index
        ids = self.controller.voice_ids()
        if not ids or index >= len(ids):
            return
        voice_id = ids[index]
        self.controller.state.voice_id = voice_id
        ref_path, ref_text, note = self.controller.pick_voice(voice_id)
        self.ref_audio_path = ref_path
        self.ref_text = ref_text
        self.asset_info = note

    @slint.callback
    def browse_ref_audio(self) -> None:
        path = _pick_file(
            "Chọn file giọng mẫu",
            [("Audio", "*.wav *.mp3 *.flac *.ogg"), ("All", "*.*")],
        )
        if path:
            self.ref_audio_path = path
            self.controller.set_ref_audio_file(path)

    @slint.callback
    def browse_gen_txt_file(self) -> None:
        path = _pick_file(
            "Chọn file văn bản",
            [("Text", "*.txt *.md *.text"), ("All", "*.*")],
        )
        if path:
            text, src = self.controller.load_gen_text_file(path)
            self.gen_text = text
            self.gen_txt_source_path = src

    @slint.callback
    def pipeline_add_clicked(self) -> None:
        keys = self.controller.normalize_step_keys()
        if not keys:
            return
        idx = min(slint_int(self.norm_step_index), len(keys) - 1)
        try:
            self.pipeline_display = self.controller.pipeline_add(keys[idx])
        except ValueError as exc:
            self.status_log = str(exc)

    @slint.callback
    def pipeline_remove_clicked(self) -> None:
        self.pipeline_display = self.controller.pipeline_remove(
            slint_int(self.pipeline_selected_index)
        )

    @slint.callback
    def pipeline_up_clicked(self) -> None:
        self.pipeline_display = self.controller.pipeline_move(
            slint_int(self.pipeline_selected_index), -1
        )

    @slint.callback
    def pipeline_down_clicked(self) -> None:
        self.pipeline_display = self.controller.pipeline_move(
            slint_int(self.pipeline_selected_index), 1
        )

    @slint.callback
    def pipeline_reset_clicked(self) -> None:
        self.pipeline_display = self.controller.pipeline_reset()

    @slint.callback
    def pipeline_audiobook_clicked(self) -> None:
        self.pipeline_display = self.controller.pipeline_audiobook()

    @slint.callback
    def preview_normalize_clicked(self) -> None:
        self._pull_state_from_ui()
        try:
            self.norm_preview = self.controller.preview_normalize()
        except (ValueError, ImportError) as exc:
            self.status_summary = str(exc)

    @slint.callback
    def synthesize_clicked(self) -> None:
        if self.busy:
            return
        self._pull_state_from_ui()
        self.busy = True
        self.progress_value = 0.0
        self.progress_label = "Đang tổng hợp..."
        self.status_summary = "Đang xử lý..."
        self.runtime_device = self.controller.predicted_runtime_device()

        def on_progress(frac: float, desc: str) -> None:
            def update() -> None:
                self.progress_value = min(1.0, max(0.0, frac))
                if desc:
                    self.progress_label = desc

            _invoke_ui(update)

        def on_log(text: str) -> None:
            def update() -> None:
                self.status_log = text

            _invoke_ui(update)

        def on_done(result: TTSResult | TTSError) -> None:
            def finish() -> None:
                self.busy = False
                if isinstance(result, TTSError):
                    self.status_summary = result.message
                    self.status_log = result.status_log_text
                    self.progress_label = "Lỗi"
                    self.progress_value = 0.0
                else:
                    self.status_summary = result.status_summary
                    self.status_log = result.status_log_text
                    self.norm_preview = result.norm_preview
                    self.output_file_path = str(result.saved_path)
                    if result.runtime_device:
                        self.runtime_device = result.runtime_device
                    self.progress_label = "Hoàn tất"
                    self.progress_value = 1.0

            _invoke_ui(finish)

        started = self.controller.synthesize_async(on_progress, on_log, on_done)
        if not started:
            self.busy = False
            self.status_summary = "Đang chạy tổng hợp khác — vui lòng đợi."


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Starting Slint GUI")
    window = MainWindow()
    window.show()
    window.run()


if __name__ == "__main__":
    main()
