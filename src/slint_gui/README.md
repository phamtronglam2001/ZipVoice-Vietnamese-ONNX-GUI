# ZipVoice Vietnamese ONNX — Slint GUI (production)

Native desktop GUI for **production TTS**. Uses shared `src/tts_pipeline.py` (same stages, defaults, and inference params as Gradio).

**Gradio** (`src/app.py`) is the **debug** surface: status-log tooling, export normalized `.txt`, export per-chunk WAVs, presets, in-app playback. Those features stay in Gradio only — not ported to Slint.

## Prerequisites

- Completed project setup (`install_cpu.bat` or `install_gpu.bat`)
- Models in `models/onnx/` and `models/vocoder/`
- Python venv at `.venv/` (created by `uv venv` — may not include `pip.exe`)

## Install Slint

```bat
.venv\Scripts\python.exe -m pip install -r requirements-slint.txt
```

Or double-click `run_slint_gui.bat` (auto-installs Slint if missing; uses `python -m pip` or `uv pip` when pip.exe is absent).

## Run

```bat
run_slint_gui.bat
```

Uses the same CPU/GPU env as Gradio launchers: `run_slint_gui.bat` reads `.install_mode` and sets `ZIPVOICE_ONNX_GPU=1` when GPU (same as `run_gpu.bat`). Force CPU: `ZIPVOICE_FORCE_CPU=1`.

Or from repo root with `PYTHONPATH=src`:

```bat
set PYTHONPATH=src
.venv\Scripts\python.exe src\slint_gui\main.py
```

Verify import / UI load:

```bat
set PYTHONPATH=src
.venv\Scripts\python.exe -c "import slint; slint.load_file('src/slint_gui/ui/app.slint')"
```

## Layout

```
src/slint_gui/
  main.py                    # Entry point, Slint window + callbacks
  ui/app.slint               # Main window UI (tabs)
  backend/tts_controller.py  # State + async synthesis wrapper
```

Shared inference: `src/tts_pipeline.py`. Gradio `src/app.py` also calls `iter_tts_pipeline` from the same module.

## Production features (Slint)

| Feature | Notes |
|---------|--------|
| Voice dropdown + manual ref audio (file picker) | Same `assets/ref_info.json` |
| Ref text, gen text, .txt/.md upload | Re-reads source file for pipeline (UTF-8) |
| Input mode (raw / prepared) | Same labels & behavior |
| Normalization pipeline | Add / move / remove / reset / audiobook preset |
| Chunk size & pause sliders | Same defaults (135 chars, pause 0.35/0.65/2.0/0.45/0.28 s) |
| Speed, export format, ONNX quant | `QUANT_MODE_CHOICES` + auto-detect default |
| GPU toggle | `use_onnx_gpu` default from env (`is_onnx_gpu_env`) |
| **Dark / light mode** | Header button — syncs `Palette.color-scheme`; saved in `profiles/slint_ui.json` |
| **Thiết bị thực tế** | Predicted before run; updated after ONNX engine load |
| Synth params | num_step=16, guidance=1.0, t_shift=0.5 |
| Parallel workers | Same clamp rules as Gradio |
| **Xem trước chuẩn hóa** | `preview_normalize_text()` |
| **Tổng hợp ONNX** | `run_tts_pipeline()` — same status log stages as Gradio |
| Status log | Stages: Đầu vào → Chuẩn hóa → Chia chunk → Load ONNX → synth → Ghép → Lưu → Hoàn tất |
| Output file path | Read-only path (open in OS file manager / player) |
| Progress indicator | Chunk synthesis progress |

## Debug-only (Gradio — not in Slint)

| Feature | Where |
|---------|--------|
| Export từng chunk WAV | Gradio accordion **Debug** → `output/chunk_test/` |
| Export normalized `.txt` | Gradio button |
| Preset load/save (`profiles/`) | Gradio preset accordion |
| In-app WAV playback | Gradio audio widget |
| CLI script | `scripts/export_chunk_wavs.py` |

## Intentionally omitted from Slint

| Gradio feature | Reason |
|----------------|--------|
| In-app WAV playback | No convenient Slint audio widget; use OS player on output path |
| Drag-drop / mic ref audio | File picker is enough on desktop |
| Export normalized `.txt` | Debug / QC — use Gradio |
| Preset load/save | Debug / workflow — use Gradio presets |
| Export per-chunk WAVs | Debug quality review — Gradio or `scripts/export_chunk_wavs.py` |
| Markdown in status text | Plain text in Slint text areas |
| Gradio file download widget | Output path + Explorer is simpler on desktop |

## Notes

- Does **not** replace `src/app.py`; both GUIs share `src/tts_pipeline.py`.
- Slint deps live in `requirements-slint.txt` at project root.
- **Theme:** std-widgets follow `Palette.color-scheme`. Window background/text use `Palette.*` so widgets and chrome match. If colors looked inverted before, Windows dark mode + hardcoded light `#f5f6f8` background caused the mismatch — use **Chế độ tối / Chế độ sáng** in the header.
