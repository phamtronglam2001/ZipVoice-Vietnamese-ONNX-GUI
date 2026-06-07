# ZipVoice Vietnamese ONNX GUI

Offline Gradio application for Vietnamese zero-shot text-to-speech using **ONNX Runtime**. No full PyTorch ZipVoice checkpoint (~470 MB) is required at inference time.

ONNX weights are exported from [hynt/ZipVoice-Vietnamese-2500h](https://huggingface.co/hynt/ZipVoice-Vietnamese-2500h). The Vocos vocoder runs as **ONNX** ([wetdog/vocos-mel-24khz-onnx](https://huggingface.co/wetdog/vocos-mel-24khz-onnx)) with **librosa ISTFT** — all weights are **bundled in `models/`** (Git LFS); no Hugging Face download at install.

**Author:** [Pham Trong Lam](https://github.com/phamtronglam2001) · **Repository:** https://github.com/phamtronglam2001/ZipVoice-Vietnamese-ONNX-GUI · **License:** Non-Commercial (see `LICENSE`)

English | [Tiếng Việt](README.md)

**Full PyTorch GUI:** [ZipVoice-Vietnamese-GUI](https://github.com/phamtronglam2001/ZipVoice-Vietnamese-GUI)

---

## Provenance & attributions

| Component | GitHub / source | License |
|-----------|-----------------|---------|
| **GUI code + sample `.wav`** | [phamtronglam2001/ZipVoice-Vietnamese-ONNX-GUI](https://github.com/phamtronglam2001/ZipVoice-Vietnamese-ONNX-GUI) | Non-Commercial |
| **ZipVoice ONNX** | Exported from [hynt/ZipVoice-Vietnamese-2500h](https://huggingface.co/hynt/ZipVoice-Vietnamese-2500h) · upstream [k2-fsa/ZipVoice](https://github.com/k2-fsa/ZipVoice) | CC-BY-NC-SA-4.0 · Apache-2.0 |
| **Vocos vocoder** | Architecture [gemelo-ai/vocos](https://github.com/gemelo-ai/vocos) · ONNX [wetdog/vocos-mel-24khz-onnx](https://huggingface.co/wetdog/vocos-mel-24khz-onnx) | MIT · model card |
| **ONNX Runtime** | [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) | MIT |
| **Text normalizers** | See [detailed table](#third-party-source-attributions) | Per repository |

---

## Acknowledgments

### Vietnamese checkpoint — Hugging Face

| | |
|---|---|
| **Author / publisher** | [**Nguyen Thien Hy**](https://huggingface.co/hynt) (`hynt`) |
| **Base model** | [hynt/ZipVoice-Vietnamese-2500h](https://huggingface.co/hynt/ZipVoice-Vietnamese-2500h) — bundled ONNX weights are exported from this checkpoint |
| **Demo Space** | [hynt/ZipVoice-Vietnamese-100h](https://huggingface.co/spaces/hynt/ZipVoice-Vietnamese-100h) |
| **Training data** | PhoAudioBook, ViVoice, UEH (model card); demucs for music removal |

### Base ZipVoice — k2-fsa

[k2-fsa/ZipVoice](https://github.com/k2-fsa/ZipVoice) · Zhu, Han et al., [arXiv:2506.13053](https://arxiv.org/abs/2506.13053) · Apache-2.0

### ONNX vocoder

Architecture [gemelo-ai/vocos](https://github.com/gemelo-ai/vocos) (Siuzdak et al.) · original weights [charactr/vocos-mel-24khz](https://huggingface.co/charactr/vocos-mel-24khz) · ONNX export [wetdog/vocos-mel-24khz-onnx](https://huggingface.co/wetdog/vocos-mel-24khz-onnx)

### Referenced datasets

[thivux/phoaudiobook](https://huggingface.co/datasets/thivux/phoaudiobook) (Vu et al., ACL 2025) · ViVoice · UEH — per `hynt` model card.

> This GUI (Pham Trong Lam) does not replace `hynt` / `k2-fsa`. Follow CC-BY-NC-SA-4.0 and disclose AI-generated audio.

---

## Third-party source attributions

### ONNX inference stack

| Component | GitHub / source | File in this repo |
|-----------|-----------------|-------------------|
| **ZipVoice ONNX** | [k2-fsa/ZipVoice](https://github.com/k2-fsa/ZipVoice) | `models/onnx/*.onnx`, `onnx_engine.py` |
| **Base checkpoint** | [hynt/ZipVoice-Vietnamese-2500h](https://huggingface.co/hynt/ZipVoice-Vietnamese-2500h) | exported weights |
| **Espeak tokenizer** | [k2-fsa/ZipVoice](https://github.com/k2-fsa/ZipVoice) (vendored) | `espeak_tokenizer.py` |
| **piper_phonemize** | [k2-fsa/icefall](https://github.com/k2-fsa/icefall) | installed via `install_cpu.bat` |
| **Espeak** | [espeak-ng/espeak-ng](https://github.com/espeak-ng/espeak-ng) | phonemization backend |
| **VocosFbank** | [k2-fsa/ZipVoice](https://github.com/k2-fsa/ZipVoice) `feature.py` | `vocos_fbank.py` (ported) |
| **Vocos decode** | [gemelo-ai/vocos](https://github.com/gemelo-ai/vocos) · ONNX [wetdog/vocos-mel-24khz-onnx](https://huggingface.co/wetdog/vocos-mel-24khz-onnx) | `onnx_engine.py`, `vocos_istft.py` |

### Text normalization pipeline (GUI)

| Backend | GitHub / source | File / install |
|---------|-----------------|----------------|
| **VieNeu** | [pnnbao97/VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS) — `vieneu_utils/core_utils.py` | `vieneu_text.py` (built-in) |
| **TTS structure** | Original to this repo (Pham Trong Lam) | `period_linebreak.py` (built-in) |
| **vinorm** | [NoahDrisort/vinorm](https://github.com/NoahDrisort/vinorm) | `pip install vinorm` |
| **vietnormalizer** | [nghimestudio/vietnormalizer](https://github.com/nghimestudio/vietnormalizer) | `pip install vietnormalizer` |
| **sea-g2p Normalizer** | [pnnbao97/sea-g2p](https://github.com/pnnbao97/sea-g2p) | `pip install sea-g2p` — Normalizer only |

**Long-text chunking** (`utils.py`): inspired by [VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS), reimplemented for ZipVoice.

### UI

| Component | GitHub |
|-----------|--------|
| **Gradio** | [gradio-app/gradio](https://github.com/gradio-app/gradio) |

---

## Comparison with the PyTorch GUI

| | [ZipVoice-Vietnamese-GUI](https://github.com/phamtronglam2001/ZipVoice-Vietnamese-GUI) | This repository |
|---|---|---|
| Inference | PyTorch checkpoint | ONNX (INT8 default) |
| First-time download | ~2 GB | **None** (clone + `git lfs pull`) |
| ONNX in repo | No | Yes (`models/onnx/` + `models/vocoder/`, Git LFS) |
| `vendor/ZipVoice` clone | Yes | **No** |
| Default port | 7860 | 7862 |

---

## Quick start (Windows, CPU)

Requires [uv](https://docs.astral.sh/uv/) on PATH. **Batch only** — no PowerShell setup script.

```bat
git clone https://github.com/phamtronglam2001/ZipVoice-Vietnamese-ONNX-GUI.git
cd ZipVoice-Vietnamese-ONNX-GUI
git lfs install
git lfs pull
install_cpu.bat
run_cpu.bat
```

Open http://127.0.0.1:7862

If `models/vocoder/mel_spec_24khz.onnx` is missing after clone, run `python download_models.py` (optional HTTP fallback, no `huggingface_hub`).

---

## Features

- Nine bundled reference voices (`assets/ref_audio/`, `ref_info.json`)
- Manual reference upload; **transcript required** (no ASR)
- `.txt` / `.md` upload for long-form text
- **Configurable normalization pipeline** (default empty; audiobook preset)
- **Preview normalization** before synthesis (no model load)
- Smart chunking with pauses for sentences, paragraphs, chapters, and numbered list items
- WAV / MP3 export to `output/`
- ONNX **INT8** (default) or FP32
- **Presets** (`profiles/*.json`) — save/load full audiobook config (GUI + CLI)

---

## Presets / profiles (`profiles/`)

JSON **schema v1** stores voice, normalization pipeline, chunk size, pauses, ONNX synthesis params, and export format.

| Default file | Description |
|--------------|-------------|
| `profiles/none.json` | Empty pipeline, manual voice upload |
| `profiles/sach.json` | VieNeu → TTS structure → vinorm, voice **Ái Vy** |

### GUI

**Preset** accordion:

- Pick a file under `profiles/`
- **Load preset** — applies voice, pipeline, chunk, pauses, speed, INT8, export format
- **Save preset** — writes `profiles/<name>.json` from current UI state

### CLI (load preset only — no pipeline editing on the command line)

```bat
run_cli.bat list-voices
run_cli.bat profile list
run_cli.bat profile show sach
run_cli.bat preview -p sach -t "Chapter 1. Hello."
run_cli.bat synthesize -p sach -f book.txt -o output/book.wav
```

- `--profile` / `-p` (default `none`) loads the full `PresetConfig`
- Only `-o` / `--output` overrides the output path

See `preset_io.py`, `cli_tts.py`, `run_cli.bat`.

---

## Normalization pipeline (ordered steps)

Add, remove, and reorder steps on the GUI. **Each step receives the previous step's output** (`text₀ → step₁ → text₁ → …`). No step-count limit; duplicate backends are rejected.

| Backend | pip required? | Role |
|---------|---------------|------|
| **VieNeu** | No (built-in) | Port of [pnnbao97/VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS) `core_utils` |
| **Join PDF line breaks** | No (built-in) | `join_soft_breaks` — merge short lowercase lines (OCR/PDF wraps) |
| **Newline → sentence** | No (built-in) | `newline_sentence` — `Chương 1\nNội dung` → `Chương 1.\nNội dung` |
| **TTS structure** | No (built-in) | `period_break` — brackets→commas; `một. next` → `một.\nnext` |
| **vinorm** | `pip install vinorm` | [NoahDrisort/vinorm](https://github.com/NoahDrisort/vinorm) |
| **vietnormalizer** | pip | [nghimestudio/vietnormalizer](https://github.com/nghimestudio/vietnormalizer) |
| **sea-g2p Normalizer** | pip | [pnnbao97/sea-g2p](https://github.com/pnnbao97/sea-g2p) — Normalizer only |
| **None** | — | Skip this step |

### TTS structure (`period_linebreak.py`)

| Rule | Input example | After processing |
|------|---------------|------------------|
| Brackets → comma (breath pause) | `mẫu (mẹ)` | `mẫu, mẹ` |
| Number + period → line break | `một. next paragraph` | `một.` + newline + `next paragraph` |
| Pause after list item | standalone `một.` block | ~**1.0 s** before next block |

Supports `()`, `[]`, and `{}`.

**TTS / preview flow:** normalize the **full document** once (`normalize_full_document`) → **split** (`split_text_for_tts`) → per-chunk light cleanup only. **Preview normalization** shows chained pipeline output with preserved `\n` and added periods.

### Suggested pipelines

| Content type | Suggested chain |
|--------------|-----------------|
| GUI default | *(empty pipeline)* |
| Audiobook | VieNeu → TTS structure → vinorm (or sea-g2p) |
| OCR / PDF line wraps | Join PDF → VieNeu → TTS structure |
| Chapter headings | add **Newline → sentence** before or after TTS structure |

All backends output **plain text**. ZipVoice phonemizes via Espeak separately, so chaining is safe.

Use **Preview normalization** on box 3 to verify output (including embedded `\n` line breaks) before running TTS.

---

## This is not browser-only ONNX

Gradio uses the browser for the **UI only**. Inference runs in a **local Python process** (ONNX Runtime + librosa ISTFT). This is not [onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/).

---

## Inference stack (no PyTorch)

| Component | Runtime |
|-----------|---------|
| Text encoder + flow-matching decoder | **ONNX Runtime + numpy** |
| Prompt mel + flow steps | **numpy / scipy / librosa** |
| Vocos vocoder | **ONNX wetdog + librosa ISTFT** (`mel_spec_24khz.onnx`, ~54 MB) |

The full ZipVoice PyTorch checkpoint (~470 MB) is **not** downloaded.

---

## Project layout

```
models/onnx/          # Bundled ZipVoice ONNX (Git LFS)
models/vocoder/       # Bundled Vocos ONNX mel_spec_24khz.onnx (Git LFS)
models/THIRD_PARTY_LICENSES.md
espeak_tokenizer.py   # Espeak via piper_phonemize
vocos_fbank.py        # Prompt mel features
vocos_istft.py        # librosa ISTFT from mag/x/y
vieneu_text.py        # VieNeu punctuation cleanup
period_linebreak.py   # Brackets→commas, list breaks
app.py                # Gradio GUI
onnx_engine.py        # ONNX + Vocos inference
utils.py              # Pipeline + long-text chunking
```

---

## Dependencies

| File | Contents |
|------|----------|
| `install_cpu.bat` | uv venv + pip deps + verify bundled models |
| `requirements-cpu.txt` | onnxruntime, gradio, librosa, scipy, … |
| `requirements-normalize.txt` | optional NSW packages |
| `download_models.py` | optional vocoder fallback if Git LFS not pulled |

**Removed:** PyTorch/torchaudio, `vocos` pip package, `vendor/ZipVoice` clone, `lhotse`, `jieba`, `matplotlib`, and other unused packages from the full PyTorch project.

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
| `Models not found` | Run `git lfs pull`, then `install_cpu.bat`; or `python download_models.py` for vocoder only |
| `vinorm` not installed | `pip install vinorm`, or remove vinorm from pipeline; use VieNeu / TTS structure |
| Parenthetical text runs together | Enable **TTS structure** in Step 2 |
| List item `một.` glued to next line | TTS structure + **Preview normalization** |
| OOM on long books | Lower **Max chars / chunk** (100–110) |
| Slow synthesis | Keep INT8 enabled |

Logs: `logs/app.log`

---

## License

- Base model [hynt/ZipVoice-Vietnamese-2500h](https://huggingface.co/hynt/ZipVoice-Vietnamese-2500h): CC-BY-NC-SA-4.0
- Bundled weights: see [`models/THIRD_PARTY_LICENSES.md`](models/THIRD_PARTY_LICENSES.md)
- This GUI and bundled voices: Non-Commercial (`LICENSE`) — Pham Trong Lam
- ZipVoice upstream: [k2-fsa/ZipVoice](https://github.com/k2-fsa/ZipVoice) — Apache-2.0
- Third-party libraries: see [Third-party source attributions](#third-party-source-attributions)
