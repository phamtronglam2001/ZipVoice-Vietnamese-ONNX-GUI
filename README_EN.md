# ZipVoice Vietnamese ONNX GUI

Offline Vietnamese zero-shot TTS: ZipVoice ONNX (int4/int8) + 100-mel Vocos vocoder + **Slint** (production) and **Gradio** (debug) GUIs.

**Author:** [Pham Trong Lam](https://github.com/phamtronglam2001) · **License:** Non-Commercial (`LICENSE`) · [Tiếng Việt](README.md)

Weights exported from [hynt/ZipVoice-Vietnamese-2500h](https://huggingface.co/hynt/ZipVoice-Vietnamese-2500h); bundled vocoder `models/vocoder/mel_spec_24khz.onnx`. Third-party terms: `models/THIRD_PARTY_LICENSES.md`.

---

## Features

- Zero-shot voice: bundled voices in `assets/` or upload WAV + transcript
- **int4** / **int8** quant (CPU or CUDA/DirectML via ONNX Runtime)
- Configurable text normalization pipeline (extensible registry in `src/text/normalizers/`)
- Chunk **min / max characters**; short micro-segments merged with `\n` before one synthesis call
- Audiobook pauses: sentence / paragraph / chapter / enum / comma split
- JSON presets (`profiles/`), CLI (`cli_tts.py`)
- Gradio: per-chunk WAV export, ODE seed, runtime device log

---

## Quick install (Windows)

```bat
install_cpu.bat
rem or install_gpu.bat  (NVIDIA CUDA)
```

Requires **Git LFS** for ONNX weights. Espeak via `piper_phonemize` wheel (install scripts handle it).

---

## Run

| Purpose | Command |
|---------|---------|
| **Production GUI (Slint)** | `run_slint_gui.bat` |
| **Debug GUI (Gradio)** | `run_gui.bat` / `run_gpu.bat` / `run_cpu.bat` |
| **CLI** | `run_cli.bat` or `python src\cli_tts.py synthesize --help` |

---

## TTS flow (summary)

```
Text → normalize (`src/text/normalizers`) → split chunks (`src/text/chunking`)
  → per chunk: Espeak G2P → ZipVoice ONNX → vocoder
  → join WAV + pauses (`src/audio/post_process`)
```

Over-short **micro-chunks** are merged into **one** synthesis (joined with `\n`). Each full TTS chunk still gets its own `generate()`; gaps between chunks use `pause_after`, not newline merge.

---

## Chunk settings

| Parameter | Default | Meaning |
|-----------|---------|---------|
| **Min chars / chunk** | 70 | Merge tiny segments (avoid weak mel / voice drift) |
| **Max chars / chunk** | 135 | Upper bound; lower if OOM |

Available in Gradio, Slint, and presets.

---

## Acknowledgments

| Component | Source | License |
|-----------|--------|---------|
| ZipVoice / ONNX | [k2-fsa/ZipVoice](https://github.com/k2-fsa/ZipVoice) | Apache-2.0 |
| VI checkpoint | [hynt/ZipVoice-Vietnamese-2500h](https://huggingface.co/hynt/ZipVoice-Vietnamese-2500h) | CC-BY-NC-SA-4.0 |
| Vocos | [gemelo-ai/vocos](https://github.com/gemelo-ai/vocos) · [charactr/vocos-mel-24khz](https://huggingface.co/charactr/vocos-mel-24khz) | MIT |
| VieNeu text hygiene | [pnnbao97/VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS) | per upstream |
| sea-g2p | [pnnbao97/sea-g2p](https://github.com/pnnbao97/sea-g2p) | per upstream |
| Espeak / piper_phonemize | [espeak-ng](https://github.com/espeak-ng/espeak-ng) · [k2-fsa/icefall](https://github.com/k2-fsa/icefall) | per upstream |
| ONNX Runtime | [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) | MIT |

GUI, Slint, chunk/audio pipeline, presets: **Pham Trong Lam** — Non-Commercial (`LICENSE`).

Output from `hynt` models must comply with **CC-BY-NC-SA-4.0** and be labeled AI-generated.

---

## Development

Folder layout, adding normalizers, imports: **[docs/for_dev.md](docs/for_dev.md)**

```bat
python -m unittest test_normalize_pipeline -v
```

---

## GPU / workers

- GPU: **workers = 1** (multiple CUDA processes often crash)
- Diagnostics: `python scripts/diagnose_gpu.py`
- Env: `ZIPVOICE_ONNX_GPU=1`, `ZIPVOICE_GPU_MAX_WORKERS=1`
