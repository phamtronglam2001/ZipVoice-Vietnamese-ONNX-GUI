# ZipVoice Vietnamese ONNX GUI

Offline Gradio application for Vietnamese zero-shot text-to-speech using **ONNX Runtime**. No full PyTorch ZipVoice checkpoint (~470 MB) is required at inference time.

ONNX weights are exported from [hynt/ZipVoice-Vietnamese-2500h](https://huggingface.co/hynt/ZipVoice-Vietnamese-2500h). The Vocos vocoder remains PyTorch-based, consistent with the upstream ZipVoice ONNX design.

**Author:** Pham Trong Lam · **License:** Non-Commercial (see `LICENSE`)

English | [Tiếng Việt](README.md)

**Full PyTorch GUI:** [ZipVoice-Vietnamese-GUI](https://github.com/phamtronglam2001/ZipVoice-Vietnamese-GUI)

---

## Comparison with the PyTorch GUI

| | [ZipVoice-Vietnamese-GUI](https://github.com/phamtronglam2001/ZipVoice-Vietnamese-GUI) | This repository |
|---|---|---|
| Inference | PyTorch checkpoint | ONNX (INT8 default) |
| First-time download | ~2 GB | ~50 MB (vocoder only) |
| ONNX in repo | No | Yes (`models/onnx/`, Git LFS) |
| `vendor/ZipVoice` clone | Yes | **No** |
| Default port | 7860 | 7862 |

---

## Quick start (Windows, CPU)

```bat
install_cpu.bat
run_cpu.bat
```

Open http://127.0.0.1:7862

---

## Features

- Nine bundled reference voices (`assets/ref_audio/`, `ref_info.json`)
- Manual reference upload; **transcript required** (no ASR)
- `.txt` / `.md` upload for long-form text
- **Three-step normalization pipeline** (see table below)
- **Preview normalization** before synthesis (no model load)
- Smart chunking with pauses for sentences, paragraphs, chapters, and numbered list items
- WAV / MP3 export to `output/`
- ONNX **INT8** (default) or FP32

---

## Normalization pipeline (Steps 1 → 2 → 3)

Up to **three steps** in order. Duplicate backends are rejected.

| Backend | pip required? | Role |
|---------|---------------|------|
| **VieNeu** | No (built-in) | Punctuation / noise cleanup from VieNeu-TTS `core_utils` |
| **TTS structure** | No (built-in) | Parentheses → commas; numbered items → line break + ~1 s pause |
| **vinorm** | `pip install vinorm` | NSW normalization — **not bundled in ZipVoice** |
| **vietnormalizer** | pip | Broader Vietnamese cleanup |
| **sea-g2p Normalizer** | pip | Rich NSW rules (**Normalizer only**, no G2P) |
| **None** | — | Skip this step |

### TTS structure (`period_linebreak.py`)

| Rule | Input example | After processing |
|------|---------------|------------------|
| Brackets → comma (breath pause) | `mẫu (mẹ)` | `mẫu, mẹ` |
| Number + period → line break | `một. next paragraph` | `một.` + newline + `next paragraph` |
| Pause after list item | standalone `một.` block | ~**1.0 s** before next block |

Supports `()`, `[]`, and `{}`.

### Suggested pipelines

| Content type | Step 1 | Step 2 | Step 3 |
|--------------|--------|--------|--------|
| GUI default | VieNeu | TTS structure | (none) |
| Numbers / symbols | VieNeu | TTS structure | vinorm or sea-g2p |
| Noisy OCR | VieNeu | vietnormalizer | TTS structure |

All backends output **plain text**. ZipVoice phonemizes via Espeak separately, so chaining is safe.

Use **Preview normalization** on box 3 to verify output (including embedded `\n` line breaks) before running TTS.

---

## This is not browser-only ONNX

Gradio uses the browser for the **UI only**. Inference runs in a **local Python process** (ONNX Runtime + PyTorch vocoder). This is not [onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/).

---

## Why PyTorch is still required

| Component | Runtime |
|-----------|---------|
| Text encoder + flow-matching decoder | **ONNX Runtime** |
| Vocos vocoder | **PyTorch** |
| Audio tensors / I/O glue | `torch` / `torchaudio` |

The full ZipVoice PyTorch checkpoint (~470 MB) is **not** downloaded.

---

## Project layout

```
models/onnx/          # Bundled ONNX weights (Git LFS)
models/vocoder/       # Downloaded once (~50 MB)
espeak_tokenizer.py   # Espeak via piper_phonemize
vocos_fbank.py        # Prompt mel features
vieneu_text.py        # VieNeu punctuation cleanup
period_linebreak.py   # Brackets→commas, list breaks
app.py                # Gradio GUI
onnx_engine.py        # ONNX + Vocos inference
utils.py              # Pipeline + long-text chunking
```

---

## Dependencies

**Runtime** (`requirements-cpu.txt` + `setup_cpu.ps1`):

`onnxruntime`, `torch`, `torchaudio`, `vocos`, `piper_phonemize`, `gradio`, `pydub`, `scipy`, `soundfile`, and optionally `vinorm`, `vietnormalizer`, `sea-g2p`.

**One-time setup** (`requirements-setup.txt`): `huggingface_hub` for vocoder download.

**Removed:** `vendor/ZipVoice` clone, `lhotse`, `jieba`, `librosa`, `matplotlib`, and other unused packages from the full PyTorch project.

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ZIPVOICE_FORCE_CPU` | `1` | Force CPU |
| `ZIPVOICE_ONNX_INT8` | `1` | Use INT8 ONNX weights |
| `GRADIO_SERVER_PORT` | `7862` | GUI port |

---

## Troubleshooting

| Symptom | Action |
|---------|--------|
| `Models not found` | Run `install_cpu.bat` |
| `vinorm` not installed | `pip install vinorm`, or remove vinorm from pipeline; use VieNeu / TTS structure |
| Parenthetical text runs together | Enable **TTS structure** in Step 2 |
| List item `một.` glued to next line | TTS structure + **Preview normalization** |
| OOM on long books | Lower **Max chars / chunk** (100–110) |
| Slow synthesis | Keep INT8 enabled |

Logs: `logs/app.log`

---

## License

- Base model [hynt/ZipVoice-Vietnamese-2500h](https://huggingface.co/hynt/ZipVoice-Vietnamese-2500h): CC-BY-NC-SA-4.0
- This GUI and bundled voices: Non-Commercial (`LICENSE`)
