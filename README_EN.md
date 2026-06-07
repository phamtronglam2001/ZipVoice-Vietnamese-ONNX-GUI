# ZipVoice Vietnamese ONNX GUI

Offline Gradio application for Vietnamese zero-shot text-to-speech using **ONNX Runtime**. No full PyTorch ZipVoice checkpoint (~470 MB) is required at inference time.

ONNX weights are exported from [hynt/ZipVoice-Vietnamese-2500h](https://huggingface.co/hynt/ZipVoice-Vietnamese-2500h). The Vocos vocoder remains PyTorch-based, consistent with the upstream ZipVoice ONNX design.

**Author:** Pham Trong Lam · **License:** Non-Commercial (see `LICENSE`)

English | [Tiếng Việt](README.md)

---

## Comparison with the full PyTorch GUI

| | [ZipVoice-Vietnamese-GUI](https://github.com/phamtronglam2001/ZipVoice-Vietnamese-GUI) | This repository |
|---|---|---|
| Inference backend | PyTorch checkpoint | ONNX (INT8 by default) |
| First-time download | ~2 GB (model + vocoder) | ~50 MB (vocoder only) |
| ONNX weights in repo | No | Yes (`models/onnx/`) |
| Default port | 7860 | 7862 |

---

## Quick start (Windows, CPU)

```bat
install_cpu.bat
run_cpu.bat
```

Open http://127.0.0.1:7862

The installer creates a virtual environment, installs CPU PyTorch and ONNX Runtime, and downloads the Vocos vocoder plus the ZipVoice tokenizer runtime. ONNX model files are already included in the repository.

---

## Features

### Voice cloning (zero-shot)

- Nine bundled reference voices in `assets/ref_audio/` with transcripts in `ref_info.json`
- Manual reference upload (3–15 seconds of clean speech)
- **Transcript is required** — the app does not run automatic speech recognition

### Long-form text and audiobooks

- Upload `.txt` or `.md` files for book-length input
- Intelligent chunking: paragraph → sentence → clause, respecting `max chars / chunk`
- Natural pauses between chunks: **0.35 s** (sentence), **0.65 s** (paragraph), **1.2 s** (chapter heading)
- Export to WAV or MP3 (32 kbps / 128 kbps, 24 kHz) under `output/`

### Text normalization pipeline

Vietnamese TTS benefits from normalizing non-standard words (numbers, dates, symbols, abbreviations) before phonemization. This project supports **chaining up to three normalizers** in a user-defined order:

| Step | Library | Role |
|------|---------|------|
| 1–3 (optional chain) | **vinorm** (TTSnorm) | Fast, lightweight NSW normalization |
| | **vietnormalizer** | Broader Vietnamese text cleanup |
| | **sea-g2p Normalizer** | Rich NSW rules (G2P module is **not** used) |

**Why chaining is safe here:** all three libraries expose a *text-in, text-out* normalizer. ZipVoice uses Espeak phonemization separately. We never pipe phoneme strings through a second normalizer, so a pipeline such as `vinorm → vietnormalizer → sea-g2p` is valid.

Recommended starting points:

- General prose: `vinorm` only
- Mixed numerals and symbols: `vinorm → sea-g2p`
- Noisy OCR or legacy typography: `vietnormalizer → vinorm`

Duplicate steps in the pipeline are rejected by the UI.

Use **Preview normalization** to run the pipeline on box 3 text before synthesis. This step does not load ONNX or vocoder weights.

### ONNX runtime options

- **INT8** (default): smaller memory footprint, faster CPU inference
- **FP32**: reference quality when INT8 quantization is too aggressive for a passage

---

## This is not browser-only ONNX

Gradio opens a URL in your browser for the **user interface only**. All inference runs in a **local Python process** on your machine:

- ONNX Runtime executes the text encoder and flow-matching decoder
- PyTorch runs the Vocos vocoder and lightweight tensor glue code
- `piper_phonemize` / espeak phonemizes Vietnamese text

This is **not** [onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/). You cannot skip installing runtimes and “just open a page” to synthesize speech.

For a future **standalone `.exe`**, the same binaries are bundled inside the executable (PyInstaller/Nuitka, etc.) — the dependency stack below still applies, only the packaging changes.

---

## Minimal dependencies

| Package | Purpose |
|---------|---------|
| `onnxruntime` | ONNX inference (encoder + decoder) |
| `torch`, `torchaudio` | Vocos vocoder; audio tensors |
| `vocos` | Mel-spectrogram → waveform |
| `piper_phonemize` | Espeak Vietnamese phonemization |
| `gradio` | Desktop-style web GUI |
| `pydub`, `scipy`, `soundfile` | Reference audio prep; WAV/MP3 export |
| `vinorm`, `vietnormalizer`, `sea-g2p` | Optional normalization pipeline |

**Removed** (were copied from the full PyTorch project but unused here): `lhotse`, `jieba`, `pypinyin`, `cn2an`, `inflect`, `librosa`, `matplotlib`, `safetensors`, `onnx` (export tool), entire `vendor/ZipVoice` clone.

Built-in modules replace the vendor tree: `espeak_tokenizer.py`, `vocos_fbank.py`.

---

## Project layout

```
models/onnx/          # Bundled ONNX weights
models/vocoder/       # Downloaded once by setup
espeak_tokenizer.py   # Minimal Espeak tokenizer
vocos_fbank.py        # Prompt mel features (no lhotse)
app.py                # Gradio GUI
onnx_engine.py        # ONNX + Vocos inference
utils.py              # Normalization + long-text chunking
```

---

## Why is PyTorch still required?

This is intentional, not a packaging mistake.

| Component | Runtime |
|-----------|---------|
| Text encoder + flow-matching decoder | **ONNX Runtime** (bundled weights) |
| Vocos vocoder (mel → waveform) | **PyTorch** (upstream ZipVoice ONNX design) |
| Audio I/O, tensors between ONNX steps | `torch` / `torchaudio` |

You do **not** download the full ZipVoice PyTorch checkpoint (~470 MB). Setup only pulls the Vocos vocoder (~50 MB) plus the ZipVoice source tree for the Espeak tokenizer.

---

## Requirements

- Python 3.10 or newer
- Internet access **once** during `install_cpu.bat` (vocoder + vendor clone)
- `piper_phonemize` / espeak (installed automatically by the setup script)

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ZIPVOICE_FORCE_CPU` | `1` in `run_cpu.bat` | Force CPU execution |
| `ZIPVOICE_ONNX_INT8` | `1` | Prefer INT8 ONNX weights at engine load |
| `GRADIO_SERVER_PORT` | `7862` | Gradio listen port |

---

## Model and license notice

- Base model: [hynt/ZipVoice-Vietnamese-2500h](https://huggingface.co/hynt/ZipVoice-Vietnamese-2500h) (CC-BY-NC-SA-4.0)
- This GUI and bundled reference audio: Non-Commercial license in `LICENSE`
- Users are responsible for lawful use of cloned voices and generated speech

---

## Troubleshooting

| Symptom | Action |
|---------|--------|
| `Models not found` on launch | Run `install_cpu.bat` |
| Import error for `vinorm` / `sea-g2p` | Re-run setup or `pip install -r requirements-cpu.txt` |
| Out-of-memory on long books | Lower **Max chars / chunk** (e.g. 100–110) |
| Slow synthesis | Enable INT8; close other heavy applications |

Logs are written to `logs/app.log`.
