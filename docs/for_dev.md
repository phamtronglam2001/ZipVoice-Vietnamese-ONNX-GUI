# For developers

Python code lives under `src/`. Launchers (`.bat`) stay at repo root and set `PYTHONPATH=%~dp0src`.

## Repo layout

```
<root>/
  *.bat, README*.md, LICENSE, requirements*.txt
  assets/    models/    profiles/    docs/    scripts/    src/
```

| Path | Role |
|------|------|
| `assets/` | Bundled reference voices + `ref_info.json` |
| `models/` | ZipVoice ONNX weights, vocoder, licenses |
| `profiles/` | JSON presets |
| `scripts/` | Diagnostics & debug utilities (not imported as app code) |
| `src/` | All application Python |

## `src/` layout

```
src/
  app.py                 # Gradio (debug)
  cli_tts.py             # CLI entry
  tts_pipeline.py        # Full TTS orchestration
  chunk_synthesis.py     # Parallel chunk workers
  chunk_export.py        # Debug: export each chunk WAV + manifest
  onnx_engine.py         # ZipVoice ONNX + vocoder decode
  espeak_tokenizer.py    # piper_phonemize → tokens.txt
  config.py              # ROOT → repo root (parent of src/)
  preset_io.py           # JSON presets under profiles/
  text/
    chunking.py          # TtsChunk, split_text_for_tts, min/max merge, micro-chunk \n join
    pipeline.py          # normalize_full_document, prepare_tts_text, preview/export
    io.py                # read_text_file
    normalizers/
      __init__.py        # NORMALIZERS registry, build_normalize_pipeline, GUI labels
      dot_newline.py     # optional step: period+space → newline
      period_linebreak.py
      vieneu_text.py
  audio/
    post_process.py      # join_tts_audio_chunks, leading_pause, forced-split crossfade
    ref_audio.py         # preprocess_ref_audio_text, resample
  slint_gui/             # Slint production GUI
    main.py
    ui/app.slint
    backend/tts_controller.py
```

## Launchers

| Script | Entry |
|--------|-------|
| `install_cpu.bat` | `uv` venv + CPU deps; writes `.install_mode=cpu` |
| `install_gpu.bat` | GPU ORT + CUDA DLLs; writes `.install_mode=gpu` |
| `run_slint_gui.bat` | `src/slint_gui/main.py` |
| `run_cli.bat` | `src/cli_tts.py` (passes CLI args) |
| `run_gui.bat` | Delegates to `run_cpu.bat` or `run_gpu.bat` via `.install_mode` |
| `run_cpu.bat` / `run_gpu.bat` | `src/app.py` (Gradio) |

All set `PYTHONPATH=%~dp0src`. Scripts under `scripts/` prepend `ROOT / "src"` to `sys.path` when run directly.

## Text pipeline

1. **Normalize** — `text.pipeline.normalize_full_document()` runs user-selected steps from `text.normalizers.NORMALIZERS` in GUI order via `normalize_text_pipeline()`. Optional `dot_newline` (`(?<![0-9])\.\s+` → `\n`) converts period+space to newlines when added to the pipeline (typically before `period_break`).
2. **Chunk** — `text.chunking.split_text_for_tts()` respects max/min chars; micro-chunks below `min_chars` merge with `\n` between parts (one synthesis call).
3. **Per chunk** — `prepare_tts_text(..., already_normalized=True)` light cleanup only.
4. **G2P** — `EspeakTokenizer` → ONNX `generate()` per chunk.
5. **Join audio** — `audio.post_process.join_tts_audio_chunks()` inserts `pause_after` / `leading_pause` between chunks (not `\n`).

## Adding a normalizer

1. Implement `def my_step(text: str) -> str` in `src/text/normalizers/my_step.py`.
2. Register in `src/text/normalizers/__init__.py`:

```python
from text.normalizers.my_step import my_step

NORMALIZERS["my_step"] = my_step
NORMALIZE_BACKENDS["my_step"] = "My step label"
NORMALIZE_STEP_CHOICES["my_step"] = NORMALIZE_BACKENDS["my_step"]
NORMALIZE_ADD_CHOICES["my_step"] = NORMALIZE_BACKENDS["my_step"]
```

3. Add tests in `src/test_normalize_pipeline.py`.

## Chunk preview (GUI debug)

`text.chunking.format_chunks_preview()` formats final post-merge `TtsChunk` list for inspection before synthesis:

```
Chunk 1/7 (92 chars, pause_after=0.65s, paragraph)
  một.[NL]top các mẫu viết đoạn văn...
```

- `[NL]` — literal newline inside chunk text (often from micro-chunk merge via `\n`).
- `pause_after` / `leading_pause` — silence inserted between chunks in `join_tts_audio_chunks`.
- `[micro-merged]` — chunk text contains `\n` (short lines were joined into one synthesis unit).
- Gradio: **Debug → Xem trước chunk (ô 3)** uses `text.pipeline.preview_chunks_output()` with current pipeline, min/max chars, and pause sliders.

## Pause model (current)

| Level | Mechanism |
|-------|-----------|
| Text | `dot_newline`: `. ` → newline; `period_break`: `()[]{}` → newline; enum `một.` → line break |
| Micro-merge | `_join_merged_chunk_parts`: `\n` between merged short pieces |
| Between chunks | `TtsChunk.pause_after`, `leading_pause` → silence in `join_tts_audio_chunks` |

No mid-wave char-ratio cut or VAD.

## Imports (migration from old `utils.py`)

| Old | New |
|-----|-----|
| `from utils import split_text_for_tts, TtsChunk` | `from text.chunking import ...` |
| `from utils import normalize_full_document` | `from text.pipeline import ...` |
| `from utils import build_normalize_pipeline` | `from text.normalizers import ...` |
| `from utils import join_tts_audio_chunks` | `from audio.post_process import ...` |
| `from utils import preprocess_ref_audio_text` | `from audio.ref_audio import ...` |

## Tests

From repo root (venv active or `.venv\Scripts\python.exe`):

```bat
set PYTHONPATH=src
python -m unittest test_normalize_pipeline -v
```

PowerShell:

```powershell
$env:PYTHONPATH="src"; python -m unittest test_normalize_pipeline -v
```

Both `test_normalize_pipeline` and `src.test_normalize_pipeline` work with `PYTHONPATH=src`.
