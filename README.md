# ZipVoice Vietnamese ONNX GUI

GUI Gradio offline cho TTS tiếng Việt zero-shot, chạy inference qua **ONNX Runtime** — không cần checkpoint PyTorch ZipVoice (~470 MB).

Model ONNX export từ [hynt/ZipVoice-Vietnamese-2500h](https://huggingface.co/hynt/ZipVoice-Vietnamese-2500h). Vocoder Vocos vẫn dùng PyTorch (theo thiết kế upstream).

**Tác giả:** Pham Trong Lam · **License:** Non-Commercial (xem `LICENSE`)

Tiếng Việt | [English](README_EN.md)

## So với repo PyTorch đầy đủ

| | [ZipVoice-Vietnamese-GUI](https://github.com/phamtronglam2001/ZipVoice-Vietnamese-GUI) | Repo này |
|---|---|---|
| Inference | PyTorch checkpoint | ONNX (INT8 mặc định) |
| Download setup | ~2 GB (model + vocoder) | ~50 MB (vocoder only) |
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
- **Chuẩn hóa text (pipeline):** chọn tối đa 3 bước lần lượt — vinorm → vietnormalizer → sea-g2p (hoặc bất kỳ thứ tự nào)
- **Xem trước chuẩn hóa:** nút riêng + ô kết quả trước khi TTS (không load model)
- **Chia chunk thông minh:** đoạn → câu → gộp đến max ký tự; nghỉ 0.35s/câu, 0.65s/đoạn, 1.2s/chương
- Xuất WAV / MP3 → `output/`
- ONNX INT8 (mặc định, nhanh hơn) hoặc FP32

### Pipeline chuẩn hóa

Chọn **Bước 1 → 2 → 3** (mỗi bước: không / vinorm / vietnormalizer / sea-g2p). Cả 3 thư viện chỉ dùng **Normalizer** (đầu ra là text, không phoneme) nên có thể xếp chuỗi — ví dụ `vinorm → sea-g2p` cho sách nhiều số. Không cho phép trùng thư viện.

## Cấu trúc

```
models/onnx/          # text_encoder*.onnx, fm_decoder*.onnx, model.json, tokens.txt (có sẵn)
models/vocoder/       # tải bởi setup (~50 MB)
espeak_tokenizer.py   # Espeak G2P (piper_phonemize) — không clone ZipVoice
vocos_fbank.py        # mel features cho prompt audio
app.py                # Gradio GUI (mở browser = chỉ UI, inference chạy Python local)
onnx_engine.py        # ONNX Runtime + Vocos PyTorch
utils.py              # chuẩn hóa + chia văn bản dài
```

## Không phải ONNX trên browser

Gradio mở **http://127.0.0.1:7862** trong trình duyệt nhưng toàn bộ inference chạy **process Python trên máy** (ONNX Runtime + PyTorch vocoder). Đây không phải `onnxruntime-web` — không thể “chỉ mở browser là chạy” mà không cài runtime.

## Dependencies tối thiểu (`requirements-cpu.txt`)

| Gói | Vì sao cần |
|-----|------------|
| `onnxruntime` | text_encoder + fm_decoder ONNX |
| `torch` + `torchaudio` | Vocos vocoder + tensor giữa các bước ONNX |
| `vocos` | mel → waveform |
| `piper_phonemize` | Espeak phonemize tiếng Việt |
| `gradio` | GUI |
| `pydub` / `scipy` / `soundfile` | xử lý giọng mẫu + xuất WAV/MP3 |
| `vinorm` / `vietnormalizer` / `sea-g2p` | pipeline chuẩn hóa (tuỳ chọn) |

**Đã bỏ:** `lhotse`, `jieba`, `pypinyin`, `cn2an`, `inflect`, `librosa`, `matplotlib`, `safetensors`, `onnx` (export), clone `vendor/ZipVoice`.

## Tại sao vẫn cần PyTorch?

ONNX chỉ thay **text encoder + flow-matching decoder** (~600 MB checkpoint PyTorch).
**Vocos vocoder** (mel → waveform) vẫn chạy PyTorch — đúng thiết kế upstream ZipVoice ONNX.
Ngoài ra `torch`/`torchaudio` dùng cho tensor giữa các bước ONNX và load file giọng mẫu.
**Không** cần tải checkpoint ZipVoice PyTorch ~470 MB.

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
