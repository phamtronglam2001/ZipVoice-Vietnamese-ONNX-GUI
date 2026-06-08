# ZipVoice Vietnamese ONNX GUI

Offline Gradio application for Vietnamese zero-shot text-to-speech. **ZipVoice** (text_encoder + fm_decoder) runs via **ONNX Runtime** — no full PyTorch ZipVoice checkpoint (~470 MB) at inference time. **Vocos ONNX vocoder 100 mel** (local export from ZipVoice-Vietnamese-GUI) + **librosa ISTFT** (`vocos_istft.py`).

ONNX weights are exported from [hynt/ZipVoice-Vietnamese-2500h](https://huggingface.co/hynt/ZipVoice-Vietnamese-2500h). ZipVoice (int4/int8 in `models/onnx/`) and the **100-mel** vocoder (`models/vocoder/mel_spec_24khz.onnx`, export from ZipVoice-Vietnamese-GUI) are **bundled** in this repo via **Git LFS** — no network model download at install time.

**Author:** [Pham Trong Lam](https://github.com/phamtronglam2001) · **Repository:** https://github.com/phamtronglam2001/ZipVoice-Vietnamese-ONNX-GUI · **License:** Non-Commercial (see `LICENSE`)

English | [Tiếng Việt](README.md)

**Full PyTorch GUI:** [ZipVoice-Vietnamese-GUI](https://github.com/phamtronglam2001/ZipVoice-Vietnamese-GUI)

---

## Provenance & attributions

| Component | GitHub / source | License |
|-----------|-----------------|---------|
| **GUI code + sample `.wav`** | [phamtronglam2001/ZipVoice-Vietnamese-ONNX-GUI](https://github.com/phamtronglam2001/ZipVoice-Vietnamese-ONNX-GUI) | Non-Commercial |
| **ZipVoice ONNX** | Exported from [hynt/ZipVoice-Vietnamese-2500h](https://huggingface.co/hynt/ZipVoice-Vietnamese-2500h) · upstream [k2-fsa/ZipVoice](https://github.com/k2-fsa/ZipVoice) | CC-BY-NC-SA-4.0 · Apache-2.0 |
| **Vocos vocoder** | Architecture [gemelo-ai/vocos](https://github.com/gemelo-ai/vocos) · PyTorch [charactr/vocos-mel-24khz](https://huggingface.co/charactr/vocos-mel-24khz) (100 mel) · local ONNX `mel_spec_24khz.onnx` | MIT · model card |
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

### Vocoder

Architecture [gemelo-ai/vocos](https://github.com/gemelo-ai/vocos) (Siuzdak et al.) · original PyTorch weights [charactr/vocos-mel-24khz](https://huggingface.co/charactr/vocos-mel-24khz) (100 mel) · bundled ONNX `models/vocoder/mel_spec_24khz.onnx` (export from ZipVoice-Vietnamese-GUI)

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
| **VocosFbank** | [k2-fsa/ZipVoice](https://github.com/k2-fsa/ZipVoice) `feature.py` | `vocos_fbank.py` — librosa `htk=True`, `norm=None` (matches torchaudio upstream, RMSE ~0.017) |
| **Vocos decode** | [gemelo-ai/vocos](https://github.com/gemelo-ai/vocos) · ONNX 100 mel + librosa ISTFT | `onnx_engine.py`, `vocos_istft.py` |
| **ONNX EP (CPU/CUDA)** | [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) | `onnx_providers.py` |
| **Shared TTS pipeline** | Original to this repo | `tts_pipeline.py`, `chunk_synthesis.py` |
| **Status log** | Original to this repo | `status_log.py` |

### Text normalization pipeline (GUI)

| Backend | GitHub / source | File / install |
|---------|-----------------|----------------|
| **VieNeu** | [pnnbao97/VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS) — `vieneu_utils/core_utils.py` | `vieneu_text.py` (built-in) |
| **TTS structure** | Original to this repo (Pham Trong Lam) | `period_linebreak.py` (built-in) |
| **Newline → sentence** | Original to this repo | `period_linebreak.py` (`newline_sentence`) |
| **Join PDF line breaks** | Original to this repo | `period_linebreak.py` (`join_soft_breaks`) |
| **sea-g2p Normalizer** | [pnnbao97/sea-g2p](https://github.com/pnnbao97/sea-g2p) | `pip install sea-g2p` — Normalizer only |

**Long-text chunking** (`utils.py`): inspired by [VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS), reimplemented for ZipVoice.

> **Note:** [vinorm](https://github.com/NoahDrisort/vinorm) and [vietnormalizer](https://github.com/nghimestudio/vietnormalizer) were previously supported as optional NSW backends; they are **not bundled**. Use `sea-g2p` instead.

### UI

| Component | GitHub |
|-----------|--------|
| **Gradio** | [gradio-app/gradio](https://github.com/gradio-app/gradio) |
| **Slint (scaffold)** | [slint-ui/slint](https://github.com/slint-ui/slint) — `slint_gui/` |

---

## Comparison with the PyTorch GUI

| | [ZipVoice-Vietnamese-GUI](https://github.com/phamtronglam2001/ZipVoice-Vietnamese-GUI) | This repository |
|---|---|---|
| ZipVoice inference | PyTorch checkpoint ~470 MB | ONNX (int4 / int8) |
| First-time download | ~2 GB | **None** (clone + `git lfs pull`) |
| ONNX in repo | No | Yes (`models/onnx/` + `models/vocoder/`, Git LFS) |
| Default vocoder | PyTorch Vocos | ONNX 100 mel (local) + librosa ISTFT |
| `vendor/ZipVoice` clone | Yes | **No** |
| Default port | 7860 | 7862 |
| Native desktop GUI | No | Slint scaffold (`run_slint_gui.bat`) |

---

## Quick start (Windows)

Requires [uv](https://docs.astral.sh/uv/) on PATH. **Batch files only** — no PowerShell setup script.

### CPU (default)

```bat
git clone https://github.com/phamtronglam2001/ZipVoice-Vietnamese-ONNX-GUI.git
cd ZipVoice-Vietnamese-ONNX-GUI
git lfs install
git lfs pull
install_cpu.bat
run_gui.bat
```

`install_cpu.bat` asks **Co GPU NVIDIA CUDA?** — answer `N` for CPU-only, or `Y` to install `onnxruntime-gpu` + CUDA DLLs via pip (same as `install_gpu.bat`).

### GPU (NVIDIA CUDA)

After clone + `git lfs pull`:

```bat
install_gpu.bat
run_gpu.bat
```

- Removes the CPU `onnxruntime` package before installing `onnxruntime-gpu` (they cannot coexist).
- `requirements-gpu.txt` adds **`nvidia-cublas-cu12`**, **`nvidia-cudnn-cu12`**, **`nvidia-cuda-runtime-cu12`**, **`nvidia-cufft-cu12`** — a full CUDA Toolkit 12 install is **not required** when using pip wheels on Windows.
- `install_gpu.bat` verifies bundled `models/` weights, ONNX vocoder import, and CUDA EP (`onnx_providers.is_cuda_execution_provider_loadable`).
- Alternatively install [CUDA Toolkit 12](https://developer.nvidia.com/cuda-downloads) + cuDNN 9 — `onnx_providers.py` adds `bin` directories to PATH / `add_dll_directory`.
- After GPU install, `.install_mode` = `gpu` → `run_gpu.bat` / `run_gui.bat` auto-enable `ZIPVOICE_ONNX_GPU=1`.

Open http://127.0.0.1:7862

If weights are missing after clone:

- ZipVoice ONNX: `git lfs pull`
- Vocoder 100 mel: place at `models/vocoder/mel_spec_24khz.onnx` (export from ZipVoice-Vietnamese-GUI)

---

## Running the app

| Script | Description |
|--------|-------------|
| `run_gui.bat` | **Recommended** — Gradio GUI; reads `.install_mode` → CPU or GPU |
| `run_cpu.bat` | Gradio GUI **CPU-only** (`ZIPVOICE_FORCE_CPU=1`) |
| `run_gpu.bat` | Gradio GUI **GPU** (`ZIPVOICE_ONNX_GPU=1`, CUDA DLL check) |
| `run_cli.bat` | CLI TTS (preset/profile) |
| `run_slint_gui.bat` | Slint desktop GUI (scaffold) — also installs `requirements-slint.txt` |

---

## Features

- Nine bundled reference voices (`assets/ref_audio/`, `ref_info.json`)
- Manual reference upload; **transcript required** (no ASR)
- `.txt` / `.md` upload for long-form text
- **Configurable normalization pipeline** (default empty; audiobook preset)
- **Preview normalization** before synthesis (no model load)
- Smart chunking with pauses for sentences, paragraphs, chapters, and numbered list items
- WAV / MP3 export to `output/`
- **ONNX quant mode** — int8 / int4 (dropdown; default from `quantization.json` or auto-detect)
- **GPU** — `run_gpu.bat` auto-enables; or «Use GPU» checkbox; or `set ZIPVOICE_ONNX_GPU=1`
- **Status log (debug)** — textbox tracking synthesis stages (`StatusLog`)
- **Preview / download** — returns WAV **filepath** (`type="filepath"`), avoiding float32 preview conversion issues
- **Every synthesis run is fresh** — no chunk checkpoint resume (`output/.checkpoints/` removed)
- **Presets** (`profiles/*.json`) — save/load full audiobook config (GUI + CLI)

---

## Inference architecture

```
Text → normalization pipeline → chunks → Espeak phonemize
  → VocosFbank (mel prompt from ref audio)
  → ONNX ZipVoice (text_encoder + fm_decoder, numpy flow steps)
  → Vocos decode (ONNX 100 mel + librosa ISTFT)
  → join chunks + pauses → WAV/MP3
```

| Component | Runtime | Notes |
|-----------|---------|-------|
| text_encoder + fm_decoder | **ONNX Runtime** (+ numpy flow) | No PyTorch ZipVoice checkpoint ~470 MB |
| Prompt mel (ref audio) | **VocosFbank** — librosa | Matches [k2-fsa/ZipVoice `feature.py`](https://github.com/k2-fsa/ZipVoice) (`htk=True`, `norm=None`) |
| **Vocos vocoder** | **ONNX 100 mel** + `vocos_istft.py` | `mel_spec_24khz.onnx` (Git LFS or GUI export) |
| Execution provider | CPU / CUDA / DirectML | `onnx_providers.py` |

Shared TTS logic: `tts_pipeline.py` (Gradio, Slint, CLI).

---

## ONNX quant modes (`models/onnx/`)

| Mode | ONNX files | Notes |
|------|------------|-------|
| **int8** | `*_int8.onnx` | **Recommended** — balance of speed + quality (default when no manifest) |
| **int4** | `*_int4.onnx` | Fast / small; **experimental** — may be lower quality than int8 |

**Automatic mode selection** (priority):

1. Environment variable `ZIPVOICE_ONNX_QUANT` (`int8` or `int4`)
2. `models/onnx/quantization.json` (`mode`, `filenames`, …)
3. Folder scan: int4 > int8
4. Legacy `ZIPVOICE_ONNX_INT8=1` → int8
5. Default: **int8**

This repo ships a sample `quantization.json` with `"mode": "int4"`. Export int8 variants from [ZipVoice-Vietnamese-GUI](https://github.com/phamtronglam2001/ZipVoice-Vietnamese-GUI) or run `quantize_onnx.py`.

**Quality recommendation:** **int8** with ONNX vocoder. INT4 is faster but may be lower quality than int8.

---

## Vocoder (ONNX)

| | ONNX 100 mel + librosa ISTFT |
|---|---|
| Weights | `models/vocoder/mel_spec_24khz.onnx` — export ZipVoice-Vietnamese-GUI (100 mel, matches ZipVoice) |
| Decode | ONNX Runtime + `vocos_istft.py` |
| Prompt mel | `vocos_fbank.py` — librosa (`htk=True`, `norm=None`) |

Verify bundled files: `python download_models.py` (local check only, no download). `install_cpu.bat` / `install_gpu.bat` run the same check at the end.

---

## GPU (CUDA / DirectML)

- **Use GPU (CUDA / DirectML)** checkbox in the GUI (**Performance** accordion), or run `run_gpu.bat` / `set ZIPVOICE_ONNX_GPU=1`.
- Requires `install_gpu.bat` (or `Y` during `install_cpu.bat`).
- `onnx_providers.py`:
  - `ensure_cuda_runtime_on_path()` — adds DLLs from pip `nvidia-*` or CUDA Toolkit to PATH
  - `is_cuda_execution_provider_loadable()` — probes `onnxruntime_providers_cuda.dll` + `cublasLt64_12.dll`, `cudnn64_9.dll`
  - If CUDA cannot load → **CPU fallback** (warning in log, no crash)
- INT4 on GPU may CPU-fallback depending on ORT build.
- Parallel chunks: max N workers on CPU (`ZIPVOICE_CPU_MAX_WORKERS`); **GPU defaults to 1 worker** (`ZIPVOICE_GPU_MAX_WORKERS`) — each worker is a separate process loading ONNX on GPU.

---

## Presets / profiles (`profiles/`)

JSON **schema v1** stores voice, normalization pipeline, chunk size, pauses, ONNX synthesis params, and export format.

| Default file | Description |
|--------------|-------------|
| `profiles/none.json` | Empty pipeline, manual voice upload |
| `profiles/sach.json` | Full pipeline (sea-g2p → … → VieNeu), voice **Ái Vy** |

### GUI

**Preset** accordion:

- Pick a file under `profiles/`
- **Load preset** — applies voice, pipeline, chunk, pauses, speed, quant mode, export format
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
- `--skip-normalize` / `--input-prepared` — skip normalization pipeline
- `--output-normalized PATH` — save normalized text to `.txt`
- `--normalize-only` — normalize only, no TTS
- `--gpu` — ONNX Runtime CUDA/DirectML

See `preset_io.py`, `cli_tts.py`, `run_cli.bat`.

> Legacy preset field `synthesis.use_int8` still maps to int8; the GUI uses the **ONNX quant mode** dropdown.

---

## Normalization pipeline (ordered steps)

Add, remove, and reorder steps on the GUI. **Each step receives the previous step's output** (`text₀ → step₁ → text₁ → …`). No step-count limit; duplicate backends are rejected.

| Backend | pip required? | Role |
|---------|---------------|------|
| **VieNeu** | No (built-in) | Port of [pnnbao97/VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS) `core_utils` |
| **Join PDF line breaks** | No (built-in) | `join_soft_breaks` — merge short lowercase lines (OCR/PDF wraps) |
| **Newline → sentence** | No (built-in) | `newline_sentence` — `Chương 1\nNội dung` → `Chương 1.\nNội dung` |
| **TTS structure** | No (built-in) | `period_break` — brackets→commas; `một. next` → `một.\nnext` |
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
| Audiobook | sea-g2p → TTS structure → … → VieNeu |
| OCR / PDF line wraps | Join PDF → VieNeu → TTS structure |
| Chapter headings | add **Newline → sentence** before or after TTS structure |

All backends output **plain text**. ZipVoice phonemizes via Espeak separately, so chaining is safe.

Use **Preview normalization** on box 3 to verify output (including embedded `\n` line breaks) before running TTS.

---

## Slint GUI (scaffold — future Gradio replacement)

Experimental native desktop GUI sharing `tts_pipeline.py`:

```bat
run_slint_gui.bat
```

Or:

```bat
.venv\Scripts\pip install -r requirements-slint.txt
.venv\Scripts\python.exe slint_gui\main.py
```

| Implemented (MVP) | Not yet (TODO) |
|-------------------|----------------|
| Voice, ref text, pipeline, chunk, pauses, quant, GPU, vocoder | Full preset load/save |
| ONNX synthesis + status log + progress | In-app audio playback |
| Normalize preview | Export normalized `.txt`, drag-drop ref audio |

Details: [`slint_gui/README.md`](slint_gui/README.md)

---

## Debug & logging

| Channel | Description |
|---------|-------------|
| **Status log (debug)** | Gradio textbox — `StatusLog` class (`status_log.py`): timestamps, normalize / chunk / ONNX / vocoder stages |
| **`logs/app.log`** | Python log file (`zipvoice_onnx_gui`) — CUDA probe, inference errors, DLL warnings |
| Slint | Similar status text area via `tts_controller.py` |

Each **Synthesize** click starts a fresh log; **no** chunk checkpoint resume.

---

## This is not browser-only ONNX

Gradio uses the browser for the **UI only**. Inference runs in a **local Python process** (ONNX Runtime + ONNX vocoder + librosa ISTFT). This is not [onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/).

---

## Project layout

```
models/onnx/              # ZipVoice ONNX + quantization.json (Git LFS)
models/vocoder/           # mel_spec_24khz.onnx (Git LFS)
models/THIRD_PARTY_LICENSES.md
assets/ref_audio/         # 9 reference voices + ref_info.json
profiles/                 # Preset JSON (none, sach, …)
output/                   # Exported WAV/MP3 (fresh each run, no .checkpoints/)
logs/app.log              # Application log
app.py                    # Gradio GUI
tts_pipeline.py           # Shared TTS pipeline (Gradio / Slint / CLI)
chunk_synthesis.py        # Multi-chunk synthesis / workers
onnx_engine.py            # ONNX ZipVoice + vocoder decode
onnx_providers.py         # CPU / CUDA / DirectML, ensure_cuda_runtime_on_path
onnx_quant.py             # Quant modes, quantization.json, quantize helpers
status_log.py             # StatusLog for GUI debug textbox
vocos_fbank.py            # Mel prompt — matches ZipVoice upstream
vocos_istft.py            # librosa ISTFT (ONNX vocoder)
espeak_tokenizer.py       # Espeak G2P (piper_phonemize)
vieneu_text.py            # VieNeu punctuation cleanup
period_linebreak.py       # Brackets→commas, list breaks
preset_io.py              # Load/save preset, GUI ↔ JSON
cli_tts.py                # CLI profile-driven
utils.py                  # Pipeline + long-text chunking
slint_gui/                # Slint desktop scaffold (main.py, ui/, backend/)
install_cpu.bat           # CPU install or optional GPU prompt
install_gpu.bat           # onnxruntime-gpu + nvidia DLL pip packages
run_gui.bat               # Gradio auto (port 7862)
run_cpu.bat               # Gradio CPU-only
run_gpu.bat               # Gradio GPU (ZIPVOICE_ONNX_GPU=1)
run_slint_gui.bat         # Slint GUI
run_cli.bat               # CLI TTS
requirements-cpu.txt      # onnxruntime + librosa (fbank)
requirements-gpu.txt      # onnxruntime-gpu + nvidia-cublas/cudnn/cuda-runtime/cufft
requirements-slint.txt    # slint (separate)
requirements-normalize.txt
download_models.py        # Verify bundled ONNX + vocoder (local only)
```

---

## Dependencies

| File | Contents |
|------|----------|
| `install_cpu.bat` | uv venv + pip deps + verify bundled models; optional GPU (`Y`) |
| `install_gpu.bat` | `onnxruntime-gpu` + `requirements-gpu.txt` + model/vocoder verify + CUDA probe |
| `run_gui.bat` | Start Gradio GUI — auto CPU/GPU from `.install_mode` |
| `run_cpu.bat` | Gradio GUI CPU-only |
| `run_gpu.bat` | Gradio GUI GPU (`ZIPVOICE_ONNX_GPU=1`) |
| `run_slint_gui.bat` | Slint GUI (auto-installs `requirements-slint.txt`) |
| `run_cli.bat` | CLI TTS (preset/profile) |
| `requirements-cpu.txt` | onnxruntime, gradio, librosa, … |
| `requirements-gpu.txt` | onnxruntime-gpu, nvidia-cublas/cudnn/cuda-runtime/cufft-cu12, librosa, … |
| `requirements-slint.txt` | `slint>=1.9.0` (separate desktop GUI) |
| `requirements-normalize.txt` | optional NSW packages |
| `download_models.py` | verify bundled local files (ZipVoice ONNX + 100-mel vocoder) |

**Removed:** `vendor/ZipVoice` clone, chunk checkpoint resume `output/.checkpoints/`, `lhotse`, `jieba`, `matplotlib`, and other unused packages from the full PyTorch project.

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ZIPVOICE_FORCE_CPU` | `1` (`run_cpu.bat` / CPU install) | Force CPU; hide GPU |
| `ZIPVOICE_ONNX_GPU` | `1` (`run_gpu.bat` / GPU install) | Use CUDA/DirectML when available |
| `ZIPVOICE_ONNX_QUANT` | *(from manifest)* | Override quant mode: `int8` / `int4` |
| `ZIPVOICE_ONNX_INT8` | *(legacy)* | `1` → int8 when `ZIPVOICE_ONNX_QUANT` is unset |
| `GRADIO_SERVER_PORT` | `7862` | GUI port |

---

## Troubleshooting

| Symptom | Action |
|---------|--------|
| `Models not found` | Run `git lfs pull`, then `install_cpu.bat` |
| Missing ONNX vocoder | Place `models/vocoder/mel_spec_24khz.onnx` (100 mel, export ZipVoice-Vietnamese-GUI) |
| Missing sea-g2p | `pip install -r requirements-normalize.txt` |
| **Broadband noise / hiss** | Ensure `vocos_fbank.py` uses librosa `htk=True`, `norm=None`; try **int8** instead of int4 |
| Broken / distorted audio preview | Fixed: output returns **WAV filepath**, not float32 array — update to latest repo |
| GPU enabled but still on CPU | Run `install_gpu.bat`; check `logs/app.log` (missing `cublasLt64_12.dll` / `cudnn64_9.dll`); install pip `nvidia-*` or CUDA Toolkit 12 |
| `onnxruntime` vs `onnxruntime-gpu` conflict | Install only one package — use `install_gpu.bat` |
| Stale / partial TTS output | **No checkpoint resume** — each synthesis runs from scratch; delete old files in `output/` if needed |
| Parenthetical text runs together | Enable **TTS structure** in Step 2 |
| List item `một.` glued to next line | TTS structure + **Preview normalization** |
| OOM / crash `0xc0000005` on GPU | Often **not** RTX VRAM exhaustion — usually **parallel workers > 1** (each ProcessPool worker loads its own CUDA sessions → driver crash). Set **workers = 1**; try **int4**; lower **Max chars / chunk** (100–110) |
| OOM on long books | Lower **Max chars / chunk** (100–110); GPU workers = 1; CPU can use more workers |
| Task Manager: free VRAM but still errors | You may be watching the **wrong GPU**: on Intel+NVIDIA laptops **GPU 0** is often Intel iGPU, **GPU 1** is NVIDIA. ORT **CUDA** runs only on NVIDIA (`device_id=0` = first GPU in `nvidia-smi`, not Intel). Run `scripts/diagnose_gpu.py` |
| GPU enabled but NVIDIA usage stays flat | Missing CUDA DLLs → **CPU** fallback (CUDA EP does not use Intel). Check **Status log** / `logs/app.log` for `Thực tế: CUDA device_id=0 (RTX …)` |
| DirectML only, no CUDA | CPU `onnxruntime` + DirectML may run on **Intel iGPU** (Task Manager GPU 0). Run `install_gpu.bat` for `CUDAExecutionProvider` on the RTX |

**GPU diagnostics:** `.venv\Scripts\python.exe scripts\diagnose_gpu.py` — EP list, CUDA probe, `nvidia-smi`, Task Manager / `CUDA_VISIBLE_DEVICES` hints.

Logs: `logs/app.log` · Debug: **Status log** textbox on the GUI

---

## License

- Base model [hynt/ZipVoice-Vietnamese-2500h](https://huggingface.co/hynt/ZipVoice-Vietnamese-2500h): CC-BY-NC-SA-4.0
- Bundled weights: see [`models/THIRD_PARTY_LICENSES.md`](models/THIRD_PARTY_LICENSES.md)
- This GUI and bundled voices: Non-Commercial (`LICENSE`) — Pham Trong Lam
- ZipVoice upstream: [k2-fsa/ZipVoice](https://github.com/k2-fsa/ZipVoice) — Apache-2.0
- Third-party libraries: see [Third-party source attributions](#third-party-source-attributions)
