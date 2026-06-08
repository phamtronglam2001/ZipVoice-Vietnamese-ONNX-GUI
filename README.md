# ZipVoice Vietnamese ONNX GUI

TTS tiếng Việt zero-shot **offline**: ZipVoice ONNX (int4/int8) + vocoder Vocos 100 mel + GUI **Gradio** (khuyến nghị). GUI desktop **Slint** đang dở — xem [TODO](#todo).

**Tác giả:** [Pham Trong Lam](https://github.com/phamtronglam2001) · **License:** Non-Commercial (`LICENSE`) · [English](README_EN.md)

Model weights export từ [hynt/ZipVoice-Vietnamese-2500h](https://huggingface.co/hynt/ZipVoice-Vietnamese-2500h); vocoder bundled `models/vocoder/mel_spec_24khz.onnx`. Chi tiết license bên thứ ba: `models/THIRD_PARTY_LICENSES.md`.

---

## Tính năng

- Giọng zero-shot: chọn giọng trong `assets/` hoặc upload WAV + transcript
- Quant **int4** / **int8** (CPU hoặc CUDA/DirectML qua ONNX Runtime)
- Pipeline chuẩn hóa text tùy chỉnh (VieNeu, dot_newline step, sea-g2p, cấu trúc TTS, …) — registry trong `src/text/normalizers/`
- Chia chunk **min / max ký tự**; gộp micro-chunk ngắn bằng `\n` trước khi synth
- Nghỉ audiobook: câu / đoạn / chương / enum / cắt phẩy
- Preset JSON (`profiles/`), CLI (`run_cli.bat`)
- Gradio: chunk preview + export từng chunk WAV, ODE seed, log thiết bị ONNX

---

## Cài đặt (Windows)

| Script | Mục đích |
|--------|----------|
| `install_cpu.bat` | Tạo `.venv`, cài CPU deps, ghi `.install_mode=cpu` |
| `install_gpu.bat` | Cài `onnxruntime-gpu` + CUDA DLL, ghi `.install_mode=gpu` |

Cần **Git LFS** để pull weights ONNX. Espeak qua wheel `piper_phonemize` (script cài tự xử lý).

---

## Chạy ứng dụng

Mọi launcher `.bat` đặt `PYTHONPATH=%~dp0src` rồi gọi module trong `src/`.

| Mục đích | Lệnh |
|----------|------|
| **GUI (Gradio — khuyến nghị)** | `run_gui.bat` (auto CPU/GPU) hoặc `run_cpu.bat` / `run_gpu.bat` |
| **CLI** | `run_cli.bat` → `src/cli_tts.py` |
| ~~Slint desktop~~ | `run_slint_gui.bat` — **chưa dùng được**, xem [TODO](#todo) |

Gradio: synth, preset, chunk preview, export debug, log ONNX. Slint tạm **không khuyến nghị** cho đến khi TODO xong.

---

## Luồng TTS (tóm tắt)

```
Văn bản → chuẩn hóa (`src/text/normalizers`) → chia chunk (`src/text/chunking`)
  → mỗi chunk: Espeak G2P → ZipVoice ONNX → vocoder
  → nối WAV + nghỉ (`src/audio/post_process`)
```

Micro-chunk quá ngắn được gộp **trong một lần synth** (nối bằng `\n`). Mỗi chunk TTS chính thức vẫn là một lần `generate()` riêng; nghỉ giữa chunk bằng `pause_after`, không gộp chunk bằng newline.

---

## Cấu hình chunk

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| **Min ký tự / chunk** | 70 | Gộp phần quá ngắn (tránh mel yếu / lạc giọng) |
| **Max ký tự / chunk** | 135 | Trần độ dài; giảm nếu OOM |

Có trong Gradio và preset.

---

## TODO

Các hạng mục **chưa hoàn thành** — đừng coi là production-ready:

- [ ] **Slint GUI** (`src/slint_gui/`, `run_slint_gui.bat`) — scaffold desktop (UI + `tts_pipeline` chung) nhưng **chưa ổn định**: crash/tự thoát im lặng khi tổng hợp ONNX, binding Slint Python 1.9.x còn lỗi, thiếu preset/export chunk so với Gradio. **Hiện dùng Gradio** (`run_gui.bat`). Ghi chú kỹ thuật: [`src/slint_gui/README.md`](src/slint_gui/README.md).

## Lời cảm ơn

| Thành phần | Nguồn | License |
|------------|-------|---------|
| ZipVoice / ONNX stack | [k2-fsa/ZipVoice](https://github.com/k2-fsa/ZipVoice) | Apache-2.0 |
| Checkpoint VI | [hynt/ZipVoice-Vietnamese-2500h](https://huggingface.co/hynt/ZipVoice-Vietnamese-2500h) | CC-BY-NC-SA-4.0 |
| Vocos | [gemelo-ai/vocos](https://github.com/gemelo-ai/vocos) · [charactr/vocos-mel-24khz](https://huggingface.co/charactr/vocos-mel-24khz) | MIT |
| VieNeu text hygiene | [pnnbao97/VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS) | theo repo gốc |
| sea-g2p | [pnnbao97/sea-g2p](https://github.com/pnnbao97/sea-g2p) | theo repo gốc |
| Espeak / piper_phonemize | [espeak-ng](https://github.com/espeak-ng/espeak-ng) · [k2-fsa/icefall](https://github.com/k2-fsa/icefall) | theo repo gốc |
| ONNX Runtime | [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) | MIT |

GUI Gradio, pipeline chunk/audio, preset: **Pham Trong Lam** — Non-Commercial (`LICENSE`).

Audio sinh ra từ model `hynt` phải tuân thủ **CC-BY-NC-SA-4.0** và ghi rõ AI-generated.

---

## Phát triển

Cấu trúc thư mục, thêm normalizer, import paths: **[docs/for_dev.md](docs/for_dev.md)**

```bat
set PYTHONPATH=src
python -m unittest test_normalize_pipeline -v
```

---

## GPU / workers

- GPU: **workers = 1** (mỗi process load CUDA riêng → crash nếu >1)
- Chẩn đoán: `python scripts/diagnose_gpu.py`
- Env: `ZIPVOICE_ONNX_GPU=1`, `ZIPVOICE_GPU_MAX_WORKERS=1`
