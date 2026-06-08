<div align="center">

# 🎙️ ZipVoice Vietnamese ONNX GUI

**Offline, zero-shot Vietnamese Text-to-Speech — clone any voice from a few seconds of audio, fully on-device.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-int4%2Fint8-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-F97316)](https://www.gradio.app/)
[![Release](https://img.shields.io/badge/release-v1.0.0-success)](https://github.com/phamtronglam2001/ZipVoice-Vietnamese-ONNX-GUI/releases/tag/v1.0.0)
[![License](https://img.shields.io/badge/license-Non--Commercial-red)](LICENSE)

**[Pham Trong Lam](https://github.com/phamtronglam2001)** · [Tiếng Việt](README_VI.md)

</div>

---

## 📖 Overview

High-quality Vietnamese TTS usually means sending text to a paid cloud API — a problem for privacy-sensitive content, audiobooks, and offline/air-gapped use.

**ZipVoice Vietnamese ONNX GUI** runs the entire pipeline **locally**: it quantizes the [ZipVoice](https://github.com/k2-fsa/ZipVoice) flow-matching model to **ONNX int4/int8**, pairs it with a 100-mel Vocos vocoder, and wraps everything in a desktop-friendly **Gradio** app. Give it ~3–15 seconds of reference audio plus a transcript and it clones that voice for arbitrary Vietnamese text — no GPU required, no data leaves your machine.

> **In one line:** a production-style, end-to-end neural TTS system — model quantization, a configurable text-normalization pipeline, parallel chunked inference, and a polished GUI — built for **long-form Vietnamese audiobooks**.

---

## ✨ Key Features

- 🗣️ **Zero-shot voice cloning** — clone from a reference clip + transcript, or pick from **39 bundled voices**.
- ⚡ **ONNX int4 / int8 quantization** — runs on CPU, or CUDA / DirectML via ONNX Runtime.
- 🧩 **Configurable normalization pipeline** — composable steps (VieNeu, sentence-structure, sea-g2p, …) via an extensible registry.
- 📚 **Audiobook-grade chunking** — smart min/max splitting, micro-chunk merging, and sentence / paragraph / chapter / enumeration pauses.
- 🚀 **Performance tuning** — parallel chunk workers, prompt caching, GPU batching, and selectable ODE solvers.
- 🎛️ **Polished Gradio GUI** — voice picker, live status log, normalization preview, per-chunk WAV export, and presets.
- 💾 **JSON presets + CLI** — reproducible configs and batch-friendly automation.

---

## 🛠️ Engineering Highlights

What this project demonstrates, beyond "calling a model":

| Area | What was built |
|------|----------------|
| **Model optimization** | Quantized ZipVoice to ONNX **int4/int8**; bundled vocoder export (100-mel, aligned with `feat_dim`) + librosa ISTFT |
| **Inference performance** | Parallel chunk workers, **prompt caching**, GPU batching, ODE-solver selection (`euler`/`heun`/`midpoint`), CPU/GPU pipeline overlap |
| **Text processing** | Modular **normalizer registry** (plug-in steps), Vietnamese G2P via espeak/`piper_phonemize`, audiobook pause modeling |
| **Cross-hardware support** | Auto CPU/GPU launch, CUDA + DirectML providers, runtime device introspection & diagnostics |
| **Software design** | Shared `tts_pipeline` core reused across Gradio GUI, CLI, and an experimental Slint desktop UI; clean `src/` layout with unit tests |
| **Productization** | One-click Windows installers, presets, status logging, licensing/attribution matrix |

---

## 🏗️ Architecture

```
            ┌────────────┐    ┌──────────────┐    ┌──────────────────┐
  Text ───► │ Normalize  │──► │   Chunk      │──► │  Per chunk:       │
            │ pipeline   │    │ (min/max +   │    │  Espeak G2P →     │
            │ (registry) │    │  micro-merge)│    │  ZipVoice ONNX →  │
            └────────────┘    └──────────────┘    │  Vocos + ISTFT    │
                                                   └────────┬─────────┘
                                                            ▼
                                          ┌─────────────────────────────┐
                                          │ Join WAV + pauses            │
                                          │ (sentence/paragraph/chapter) │
                                          └─────────────────────────────┘
```

Each TTS chunk is its own `generate()` call; pauses between chunks use `pause_after` (not newline merge). Over-short micro-chunks are joined with `\n` into a single synthesis call to avoid weak mel / voice drift.

<details>
<summary><strong>Repository layout</strong></summary>

```
src/
  app.py                 # Gradio GUI (recommended entry)
  cli_tts.py             # CLI entry
  tts_pipeline.py        # Full TTS orchestration (shared core)
  chunk_synthesis.py     # Parallel chunk workers
  onnx_engine.py         # ZipVoice ONNX + vocoder decode
  espeak_tokenizer.py    # piper_phonemize → tokens.txt
  text/                  # chunking + normalization pipeline
    normalizers/         # plug-in registry (vieneu, period_linebreak, dot_newline, …)
  audio/                 # post-processing (join, pauses, ref-audio prep)
  slint_gui/             # experimental desktop UI (WIP)
assets/   models/   profiles/   scripts/   docs/
```

Full developer notes: **[docs/for_dev.md](docs/for_dev.md)**.

</details>

---

## 🚀 Quickstart (Windows)

```bat
REM 1. Install (pick one)
install_cpu.bat        REM CPU-only
install_gpu.bat        REM onnxruntime-gpu + CUDA

REM 2. Launch the GUI (auto CPU/GPU)
run_gui.bat
```

> Requires **Git LFS** to pull the ONNX weights. Espeak is installed automatically via the `piper_phonemize` wheel.

| Entry point | Command |
|-------------|---------|
| **GUI (recommended)** | `run_gui.bat` (or `run_cpu.bat` / `run_gpu.bat`) |
| **CLI** | `run_cli.bat` → `src/cli_tts.py` |
| Slint desktop (experimental) | `run_slint_gui.bat` — see [Roadmap](#-roadmap) |

---

## 🖥️ Usage

### Demo

![Gradio GUI](docs/screenshot.png)

Reference voice, input text, and synthesized output (click **play** to open the audio file on GitHub):

<table>
<tr>
<td colspan="2"><h3>Đinh Quyết &nbsp;<code>Đinh-Quyết</code></h3></td>
</tr>
<tr>
<td width="50%">
<b>Reference</b>&ensp;<a href="docs/demo/ref_voice.mp3">play</a><br>
<i>Trong trang, việc ứng dụng công nghệ và chuyển đổi số đang là yếu tố chi phối các mô hình này.</i>
</td>
<td width="50%">
<b>Text to synthesize</b><br>
<i>Tiếng Việt là tiếng nói thiêng liêng và giàu đẹp của dân tộc Việt Nam. Vẻ đẹp của tiếng Việt trước hết thể hiện ở âm thanh giàu nhạc tính với những thanh điệu trầm bổng, khiến lời nói nghe như một khúc hát dịu dàng. Không chỉ đẹp, tiếng Việt còn rất giàu với vốn từ phong phú, có thể diễn đạt mọi cung bậc cảm xúc của con người. Những từ láy, từ tượng thanh, tượng hình làm cho lời văn trở nên sinh động và giàu hình ảnh. Qua thời gian, tiếng Việt ngày càng phát triển và khẳng định sức sống mạnh mẽ. Vì vậy, mỗi chúng ta cần yêu quý và giữ gìn sự trong sáng của tiếng mẹ đẻ.</i>
</td>
</tr>
<tr>
<td colspan="2">
<b>Output (TTS)</b>&ensp;<a href="docs/demo/output.wav">play</a>
</td>
</tr>
</table>

> Regenerate demo assets: [`docs/demo/README.md`](docs/demo/README.md)

### Gradio tabs

| Tab | Contents |
|-----|----------|
| **Giọng & văn bản** | Reference voice, input text, output file, audio preview, status log |
| **Chuẩn hoá text** | Input mode, normalization preview / export `.txt` |
| **Chuẩn hóa & chunk** | Normalization pipeline, min/max chunk, audiobook pauses |
| **Hiệu năng & preset** | Speed, ONNX quant, GPU, workers, JSON presets |
| **Debug** | Chunk preview, ODE seed, per-chunk WAV export |

### Reference voices (`assets/`)

The voice picker merges **two formats** (click **Refresh** after adding files):

1. **`ref_info.json`** — each entry has `name`, `audio_path`, `text` (transcript required).
2. **`sample_audio/`** — one audio file + a same-stem `.txt` transcript (e.g. `Bá-Vinh.mp3` + `Bá-Vinh.txt`).

Supported audio: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, … Bundled: **9** voices from `ref_info.json` + **30** from `sample_audio/`. Details in [`assets/README.txt`](assets/README.txt).

### Chunk settings

| Parameter | Default | Meaning |
|-----------|---------|---------|
| **Min chars / chunk** | 70 | Merge tiny segments (avoid weak mel / voice drift) |
| **Max chars / chunk** | 135 | Upper bound; lower if you hit OOM |

---

## ⚙️ Performance

Per chunk: `text_encoder` ×1 → `fm_decoder` × `num_step` (ODE) → `vocoder` ×1 → librosa ISTFT.

| Lever | Where | Knob |
|-------|-------|------|
| ORT graph opt + threads | `onnx_session_opts.py` | `ZIPVOICE_ONNX_THREADS`, GUI **Hiệu năng** |
| Prompt cache | `OnnxTTSEngine.prepare_prompt()` | automatic |
| GPU batching | `OnnxTTSEngine.generate_batch()` | `ZIPVOICE_INFERENCE_BATCH`, GUI **Batch size** |
| ODE solver | `euler` / `heun` / `midpoint` | `ZIPVOICE_ODE_SOLVER`, GUI **ODE solver** |
| CPU overlap | pre-tokenize next chunk | `ZIPVOICE_PIPELINE_OVERLAP=1` |

```bat
set PYTHONPATH=src
python scripts/profile_inference.py --gpu --quant int4 --batch 4
python scripts/diagnose_gpu.py
```

> **GPU note:** use **workers = 1** on GPU (multiple CUDA processes often crash). 

---

## 🗺️ Roadmap

- [ ] **Slint desktop GUI** (`src/slint_gui/`) — a native desktop front-end sharing the same `tts_pipeline` core. Currently experimental: silent crashes on ONNX synthesis, Slint Python 1.9.x binding issues, and missing preset/chunk-export parity with Gradio. **Use the Gradio GUI for now.** Technical notes: [`src/slint_gui/README.md`](src/slint_gui/README.md).

---

## 🧪 Development

```bat
set PYTHONPATH=src
python -m unittest test_normalize_pipeline test_inference_perf -v
```

Folder layout, adding a normalizer, import paths, and the pause model: **[docs/for_dev.md](docs/for_dev.md)**.

---

## 🙏 Acknowledgments & License

Weights are exported from [hynt/ZipVoice-Vietnamese-2500h](https://huggingface.co/hynt/ZipVoice-Vietnamese-2500h); the vocoder `models/vocoder/mel_spec_24khz.onnx` is bundled. Full third-party terms: [`models/THIRD_PARTY_LICENSES.md`](models/THIRD_PARTY_LICENSES.md).

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

The Gradio GUI, chunk/audio pipeline, and presets are by **Pham Trong Lam** — Non-Commercial ([`LICENSE`](LICENSE)). Audio generated from `hynt` models must comply with **CC-BY-NC-SA-4.0** and be labeled AI-generated.

---

## 👤 Author

**Pham Trong Lam** — [github.com/phamtronglam2001](https://github.com/phamtronglam2001)
End-to-end ML engineering: model quantization, inference optimization, and shipping usable tools around them.
