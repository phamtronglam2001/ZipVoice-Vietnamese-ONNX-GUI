# ZipVoice Vietnamese ONNX GUI

GUI Gradio offline cho TTS tiếng Việt zero-shot, chạy inference qua **ONNX Runtime** — không cần checkpoint PyTorch ZipVoice (~470 MB).

Model ONNX export từ [hynt/ZipVoice-Vietnamese-2500h](https://huggingface.co/hynt/ZipVoice-Vietnamese-2500h). Vocoder Vocos vẫn dùng PyTorch (theo thiết kế upstream).

**Tác giả:** Pham Trong Lam · **License:** Non-Commercial (xem `LICENSE`)

## So với repo PyTorch đầy đủ

| | [ZipVoice-Vietnamese-GUI](https://github.com/phamtronglam2001/ZipVoice-Vietnamese-GUI) | Repo này |
|---|---|---|
| Inference | PyTorch checkpoint | ONNX (INT8 mặc định) |
| Download setup | ~2 GB (model + vocoder) | ~100 MB (vocoder + tokenizer) |
| ONNX trong repo | Không | Có sẵn `models/onnx/` |
| Port mặc định | 7860 | 7862 |

## Cài đặt nhanh (Windows CPU)

```bat
install_cpu.bat
run_cpu.bat
```

Mở http://127.0.0.1:7862

## Tính năng GUI

- 9 giọng mẫu trong `assets/ref_audio/` + `ref_info.json`
- Upload giọng mẫu + transcript bắt buộc (không ASR)
- Upload `.txt` / `.md` cho **sách dài**
- **Chuẩn hóa text:** không / vinorm / vietnormalizer / sea-g2p
- **Chia chunk thông minh:** đoạn → câu → gộp đến max ký tự; nghỉ 0.35s/câu, 0.65s/đoạn, 1.2s/chương
- Xuất WAV / MP3 → `output/`
- ONNX INT8 (mặc định, nhanh hơn) hoặc FP32

## Cấu trúc

```
models/onnx/          # text_encoder*.onnx, fm_decoder*.onnx, model.json, tokens.txt (có sẵn)
models/vocoder/       # tải bởi setup
vendor/ZipVoice/      # clone bởi setup (EspeakTokenizer)
assets/ref_audio/     # giọng mẫu
app.py                # Gradio GUI
onnx_engine.py        # ONNX inference
utils.py              # chuẩn hóa + chia văn bản dài
```

## Yêu cầu

- Python 3.10+
- Internet **một lần** khi chạy `install_cpu.bat` (vocoder + vendor)
- espeak qua `piper_phonemize` (cài tự động trong setup)

## Biến môi trường

| Biến | Mặc định | Ý nghĩa |
|------|----------|---------|
| `ZIPVOICE_FORCE_CPU` | `1` (run_cpu.bat) | Chạy CPU |
| `ZIPVOICE_ONNX_INT8` | `1` | Dùng ONNX INT8 khi khởi động engine |
| `GRADIO_SERVER_PORT` | `7862` | Port GUI |
