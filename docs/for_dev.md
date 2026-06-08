# For developers

Internal layout after the `src/` reorganization. Launchers (`.bat`) stay at repo root; Python code lives under `src/`.

## Layout

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
      period_linebreak.py
      vieneu_text.py
  audio/
    post_process.py      # join_tts_audio_chunks, leading_pause, forced-split crossfade
    ref_audio.py         # preprocess_ref_audio_text, resample
  slint_gui/             # Slint (production)

scripts/                 # Root utilities (set sys.path to src/ or use PYTHONPATH)
```

Repo root keeps: `run_*.bat`, `install_*.bat`, `assets/`, `models/`, `profiles/`, `output/`, `logs/`, `docs/`.

## PYTHONPATH

Batch launchers set `PYTHONPATH=%~dp0src` before invoking Python. Scripts under `scripts/` prepend `ROOT / "src"` to `sys.path`.

## Text pipeline

1. **Normalize** — `text.pipeline.normalize_full_document()` runs steps from `text.normalizers.NORMALIZERS` (user-ordered in GUI).
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

## Pause model (current)

| Level | Mechanism |
|-------|-----------|
| Text | `period_break`: `()[]{}` → newline; enum `một.` → line break |
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

From repo root (with venv active or `.venv\Scripts\python.exe`):

```bat
set PYTHONPATH=src
python -m unittest test_normalize_pipeline -v
```

Or PowerShell:

```powershell
$env:PYTHONPATH="src"; python -m unittest test_normalize_pipeline -v
```
