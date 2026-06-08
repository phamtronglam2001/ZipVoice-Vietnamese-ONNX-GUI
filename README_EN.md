# ZipVoice Vietnamese ONNX GUI

Offline Vietnamese zero-shot TTS: ZipVoice ONNX (int4/int8) + 100-mel Vocos vocoder + **Gradio** GUI (recommended). Desktop **Slint** GUI is **incomplete** — see [TODO](#todo).

**Author:** [Pham Trong Lam](https://github.com/phamtronglam2001) · **License:** Non-Commercial (`LICENSE`) · [Tiếng Việt](README.md)

Weights exported from [hynt/ZipVoice-Vietnamese-2500h](https://huggingface.co/hynt/ZipVoice-Vietnamese-2500h); bundled vocoder `models/vocoder/mel_spec_24khz.onnx`. Third-party terms: `models/THIRD_PARTY_LICENSES.md`.

---

## Features

- Zero-shot voice: bundled voices in `assets/` (`ref_info.json` or `sample_audio/` pairs) or upload WAV/MP3 + transcript
- **int4** / **int8** quant (CPU or CUDA/DirectML via ONNX Runtime)
- Configurable text normalization pipeline (optional dot_newline step, extensible registry in `src/text/normalizers/`)
- Chunk **min / max characters**; short micro-segments merged with `\n` before one synthesis call
- Audiobook pauses: sentence / paragraph / chapter / enum / comma split
- JSON presets (`profiles/`), CLI (`run_cli.bat`)
- Gradio: chunk preview, per-chunk WAV export, ODE seed, runtime device log

---

## Install (Windows)

| Script | Purpose |
|--------|---------|
| `install_cpu.bat` | Create `.venv`, install CPU deps, write `.install_mode=cpu` |
| `install_gpu.bat` | Install `onnxruntime-gpu` + CUDA DLLs, write `.install_mode=gpu` |

Requires **Git LFS** for ONNX weights. Espeak via `piper_phonemize` wheel (install scripts handle it).

---

## Run

All `.bat` launchers set `PYTHONPATH=%~dp0src` then invoke modules under `src/`.

| Purpose | Command |
|---------|---------|
| **GUI (Gradio — recommended)** | `run_gui.bat` (auto CPU/GPU) or `run_cpu.bat` / `run_gpu.bat` |
| **CLI** | `run_cli.bat` → `src/cli_tts.py` |
| ~~Slint desktop~~ | `run_slint_gui.bat` — **not ready**, see [TODO](#todo) |

### Gradio tabs

| Tab | Contents |
|-----|----------|
| **Giọng & văn bản** | Reference voice, input text, output file, preview audio, status log |
| **Chuẩn hoá text** | Input mode, normalize preview / export `.txt` |
| **Chuẩn hóa & chunk** | Normalization pipeline, min/max chunk, audiobook pauses |
| **Hiệu năng & preset** | Speed, ONNX quant, GPU, workers, JSON presets |
| **Debug** | Chunk preview, ODE seed, per-chunk WAV export |

---

## Reference voices (`assets/`)

The app loads voices from **two formats** (merged in the voice dropdown; click **Refresh** after adding files):

1. **`ref_info.json`** — each entry: `name`, `audio_path`, `text` (transcript required).
2. **`sample_audio/`** — each voice = **one audio file + one `.txt` with the same stem** (e.g. `Bá-Vinh.mp3` + `Bá-Vinh.txt`).

Supported audio: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, … See [`assets/README.txt`](assets/README.txt).

Bundled: **9** voices from `ref_info.json` + **30** from `sample_audio/`.

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

Available in Gradio and presets.

---

## TODO

Items **not finished** — do not treat as production-ready:

- [ ] **Slint GUI** (`src/slint_gui/`, `run_slint_gui.bat`) — desktop scaffold (UI + shared `tts_pipeline`) but **unstable**: silent crash on ONNX synthesis, Slint Python 1.9.x binding issues, missing preset/chunk export vs Gradio. **Use Gradio** (`run_gui.bat`) for now. Notes: [`src/slint_gui/README.md`](src/slint_gui/README.md).

---

## Acknowledgments

| Component | Source | License |
|-----------|--------|---------|
| ZipVoice / ONNX | [k2-fsa/ZipVoice](https://github.com/k2-fsa/ZipVoice) | Apache-2.0 |
| VI checkpoint | [hynt/ZipVoice-Vietnamese-2500h](https://huggingface.co/hynt/ZipVoice-Vietnamese-2500h) | CC-BY-NC-SA-4.0 |
| `sample_audio/` reference voices (30) | [contextboxai/ViZipvoice](https://huggingface.co/contextboxai/ViZipvoice) (`audio/` — mp3 + transcript) | Apache-2.0 |
| Vocos | [gemelo-ai/vocos](https://github.com/gemelo-ai/vocos) · [charactr/vocos-mel-24khz](https://huggingface.co/charactr/vocos-mel-24khz) | MIT |
| VieNeu text hygiene | [pnnbao97/VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS) | per upstream |
| sea-g2p | [pnnbao97/sea-g2p](https://github.com/pnnbao97/sea-g2p) | per upstream |
| Espeak / piper_phonemize | [espeak-ng](https://github.com/espeak-ng/espeak-ng) · [k2-fsa/icefall](https://github.com/k2-fsa/icefall) | per upstream |
| ONNX Runtime | [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) | MIT |

Gradio GUI, chunk/audio pipeline, presets: **Pham Trong Lam** — Non-Commercial (`LICENSE`).

Output from `hynt` models must comply with **CC-BY-NC-SA-4.0** and be labeled AI-generated.

---

## Development

Folder layout, adding normalizers, imports: **[docs/for_dev.md](docs/for_dev.md)**

```bat
set PYTHONPATH=src
python -m unittest test_normalize_pipeline -v
```

---

## GPU / workers

- GPU: **workers = 1** (multiple CUDA processes often crash)
- Diagnostics: `python scripts/diagnose_gpu.py`
- Env: `ZIPVOICE_ONNX_GPU=1`, `ZIPVOICE_GPU_MAX_WORKERS=1`
