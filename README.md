# ZipVoice Vietnamese ONNX GUI

TTS tiếng Việt zero-shot **offline**: ZipVoice ONNX (int4/int8) + vocoder Vocos 100 mel + GUI **Slint** (production) và **Gradio** (debug).

**Tác giả:** [Pham Trong Lam](https://github.com/phamtronglam2001) · **License:** Non-Commercial (`LICENSE`) · [English](README_EN.md)

Model weights export từ [hynt/ZipVoice-Vietnamese-2500h](https://huggingface.co/hynt/ZipVoice-Vietnamese-2500h); vocoder bundled `models/vocoder/mel_spec_24khz.onnx`. Chi tiết license bên thứ ba: `models/THIRD_PARTY_LICENSES.md`.

---

## Tính năng

- Giọng zero-shot: chọn giọng trong `assets/` hoặc upload WAV + transcript
- Quant **int4** / **int8** (CPU hoặc CUDA/DirectML qua ONNX Runtime)
- Pipeline chuẩn hóa text tùy chỉnh (VieNeu, sea-g2p, cấu trúc TTS, …) — registry mở rộng trong `text/normalizers/`
- Chia chunk **min / max ký tự**; gộp micro-chunk ngắn bằng `\n` trước khi synth
- Nghỉ audiobook: câu / đoạn / chương / enum / cắt phẩy
- Preset JSON (`profiles/`), CLI (`cli_tts.py`)
- Gradio: export từng chunk WAV, ODE seed, log thiết bị ONNX

---

## Cài đặt nhanh (Windows)

```bat
install_cpu.bat
rem hoặc install_gpu.bat  (NVIDIA CUDA)
```

Cần **Git LFS** để pull weights ONNX. Espeak qua wheel `piper_phonemize` (script cài tự xử lý).

---

## Chạy ứng dụng

| Mục đích | Lệnh |
|----------|------|
| **GUI production (Slint)** | `run_slint_gui.bat` |
| **GUI debug (Gradio)** | `run_gui.bat` / `run_gpu.bat` / `run_cpu.bat` |
| **CLI** | `python cli_tts.py synthesize --help` |

Slint: giọng, preset, synth sách dài. Gradio: debug chunk, seed, xem pipeline chuẩn hóa.

---

## Luồng TTS (tóm tắt)

```
Văn bản → chuẩn hóa (text/normalizers) → chia chunk (text/chunking)
  → mỗi chunk: Espeak G2P → ZipVoice ONNX → vocoder
  → nối WAV + nghỉ (audio/post_process)
```

Micro-chunk quá ngắn được gộp **trong một lần synth** (nối bằng `\n`). Mỗi chunk TTS chính thức vẫn là một lần `generate()` riêng; nghỉ giữa chunk bằng `pause_after`, không gộp chunk bằng newline.

---

## Cấu hình chunk

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| **Min ký tự / chunk** | 70 | Gộp phần quá ngắn (tránh mel yếu / lạc giọng) |
| **Max ký tự / chunk** | 135 | Trần độ dài; giảm nếu OOM |

Có trong Gradio, Slint, preset.

---

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

GUI, Slint, pipeline chunk/audio, preset: **Pham Trong Lam** — Non-Commercial (`LICENSE`).

Audio sinh ra từ model `hynt` phải tuân thủ **CC-BY-NC-SA-4.0** và ghi rõ AI-generated.

---

## Phát triển

Cấu trúc thư mục, thêm normalizer, import paths: **[docs/for_dev.md](docs/for_dev.md)**

```bat
python -m unittest test_normalize_pipeline -v
```

---

## GPU / workers

- GPU: **workers = 1** (mỗi process load CUDA riêng → crash nếu >1)
- Chẩn đoán: `python scripts/diagnose_gpu.py`
- Env: `ZIPVOICE_ONNX_GPU=1`, `ZIPVOICE_GPU_MAX_WORKERS=1`
