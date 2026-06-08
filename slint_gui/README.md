# ZipVoice Vietnamese ONNX — Slint GUI

Native desktop GUI synced with the Gradio app. Uses shared `tts_pipeline.py` at project root (same stages, defaults, and inference params).

## Prerequisites

- Completed project setup (`install_cpu.bat` or GPU install)
- Models in `models/onnx/` and `models/vocoder/`
- Python venv at `.venv/`

## Install Slint

```bat
.venv\Scripts\pip install -r requirements-slint.txt
```

Or double-click `run_slint_gui.bat` (auto-installs Slint if missing).

## Run

```bat
run_slint_gui.bat
```

Uses the same CPU/GPU env as Gradio launchers (reads `.install_mode`; or use `run_cpu.bat` / `run_gpu.bat` env pattern).

Or:

```bat
.venv\Scripts\python.exe slint_gui\main.py
```

Verify import / UI load:

```bat
.venv\Scripts\python.exe -c "import slint_gui; import slint; slint.load_file('slint_gui/ui/app.slint')"
```

## Layout

```
slint_gui/
  main.py                 # Entry point, Slint window + callbacks
  ui/app.slint            # Main window UI (tabs)
  backend/tts_controller.py  # State + async synthesis wrapper
```

Shared inference: `tts_pipeline.py` (project root). Gradio `app.py` also calls `iter_tts_pipeline` from the same module.

## Synced with Gradio

| Feature | Notes |
|---------|--------|
| Voice dropdown + manual ref audio (file picker) | Same `assets/ref_info.json` |
| Ref text, gen text, .txt/.md upload | Re-reads source file for pipeline (UTF-8) |
| Input mode (raw / prepared) | Same labels & behavior |
| Normalization pipeline | Add / move / remove / reset / audiobook preset |
| Chunk size & pause sliders | Same defaults (135 chars, pause 0.35/0.65/2.0/0.45/0.28 s) |
| Speed, export format, ONNX quant | `QUANT_MODE_CHOICES` + auto-detect default |
| GPU toggle | `use_onnx_gpu` default from env (`is_onnx_gpu_env`) |
| **Thiết bị thực tế** | Predicted before run; updated after ONNX engine load |
| Synth params | num_step=16, guidance=1.0, t_shift=0.5 |
| Parallel workers | Same clamp rules as Gradio |
| **Xem trước chuẩn hóa** | `preview_normalize_text()` |
| **Tổng hợp ONNX** | `run_tts_pipeline()` — same status log stages as Gradio |
| Status log (debug) | Stages: Đầu vào → Chuẩn hóa → Chia chunk → Load ONNX → synth → Ghép → Lưu → Hoàn tất |
| Output file path | Read-only path (open in OS file manager / player) |
| Progress indicator | Chunk synthesis progress |

## Intentionally omitted

| Gradio feature | Reason |
|----------------|--------|
| In-app WAV playback | No convenient Slint audio widget; use OS player on output path |
| Drag-drop / mic ref audio | File picker is enough on desktop |
| Export normalized `.txt` | Low value vs Gradio; use Gradio or CLI if needed |
| Preset load/save (`profiles/`) | JSON preset UI not worth duplicating yet; use Gradio presets |
| Markdown in status text | Plain text in Slint text areas |
| Gradio file download widget | Output path + Explorer is simpler on desktop |

## Notes

- Does **not** replace `app.py`; both GUIs share `tts_pipeline.py`.
- Slint deps live in `requirements-slint.txt` at project root.
