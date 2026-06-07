# ZipVoice Vietnamese ONNX GUI

GUI Gradio offline cho TTS tiếng Việt zero-shot, chạy inference qua **ONNX Runtime** — không cần checkpoint PyTorch ZipVoice (~470 MB).

Model ONNX export từ [hynt/ZipVoice-Vietnamese-2500h](https://huggingface.co/hynt/ZipVoice-Vietnamese-2500h). Vocoder Vocos vẫn dùng PyTorch (theo thiết kế upstream).

**Tác giả:** Pham Trong Lam · **License:** Non-Commercial (xem `LICENSE`)

Tiếng Việt | [English](README_EN.md)

**Repo PyTorch đầy đủ:** [ZipVoice-Vietnamese-GUI](https://github.com/phamtronglam2001/ZipVoice-Vietnamese-GUI)

---

## So với repo PyTorch

| | [ZipVoice-Vietnamese-GUI](https://github.com/phamtronglam2001/ZipVoice-Vietnamese-GUI) | Repo này |
|---|---|---|
| Inference | PyTorch checkpoint ~470 MB | ONNX (INT8 mặc định) |
| Download setup | ~2 GB | ~50 MB (vocoder only) |
| ONNX trong repo | Không | Có sẵn `models/onnx/` (Git LFS) |
| Clone `vendor/ZipVoice` | Có | **Không** |
| Port mặc định | 7860 | 7862 |

---

## Cài đặt nhanh (Windows CPU)

```bat
install_cpu.bat
run_cpu.bat
```

Mở http://127.0.0.1:7862

---

## Tính năng GUI

- **9 giọng mẫu** trong `assets/ref_audio/` + `ref_info.json`
- Upload giọng mẫu + **transcript bắt buộc** (không ASR)
- Upload `.txt` / `.md` cho **sách dài**
- **Pipeline chuẩn hóa** tối đa 3 bước (xem bảng dưới)
- **Xem trước chuẩn hóa** — nút riêng, không load model TTS
- **Chia chunk** đoạn → câu → max ký tự; nghỉ theo câu / đoạn / chương / mục đánh số
- Xuất WAV / MP3 → `output/`
- ONNX **INT8** (mặc định) hoặc FP32

---

## Pipeline chuẩn hóa (Bước 1 → 2 → 3)

Chọn tối đa **3 bước** theo thứ tự. Không cho trùng thư viện.

| Backend | Cài pip? | Vai trò |
|---------|----------|---------|
| **VieNeu** | Không (built-in) | Dọn punctuation/noise — port từ VieNeu-TTS `core_utils` |
| **Cấu trúc TTS** | Không (built-in) | Ngoặc `()` `[]` `{}` → dấu phẩy; `một.` / `2.` → xuống dòng + nghỉ ~1s |
| **vinorm** | `pip install vinorm` | NSW (số, ngày…) — **không có trong ZipVoice**, package PyPI riêng |
| **vietnormalizer** | pip | Chuẩn hóa tiếng Việt rộng hơn |
| **sea-g2p Normalizer** | pip | NSW mạnh (chỉ Normalizer, **không** G2P) |
| **Không** | — | Bỏ qua bước đó |

### Cấu trúc TTS (`period_linebreak.py`) — chi tiết

| Quy tắc | Ví dụ vào | Sau xử lý |
|----------|-----------|-----------|
| Ngoặc → phẩy (ngắt hơi) | `mẫu (mẹ)` | `mẫu, mẹ` |
| Số + chấm → xuống dòng | `một. đọc đoạn` | `một.` + xuống dòng + `đọc đoạn` |
| Nghỉ sau mục đánh số | `một.` (đoạn riêng) | ~**1.0 s** trước đoạn tiếp |

### Gợi ý pipeline

| Loại văn bản | Bước 1 | Bước 2 | Bước 3 |
|--------------|--------|--------|--------|
| Mặc định GUI | VieNeu | Cấu trúc TTS | (none) |
| Sách nhiều số | VieNeu | Cấu trúc TTS | vinorm hoặc sea-g2p |
| OCR / typography lỗi | VieNeu | vietnormalizer | Cấu trúc TTS |

> **Lưu ý:** ZipVoice chỉ dùng **Espeak** để phonemize. Các bước trên đều trả về **text** — có thể xếp chuỗi an toàn.

---

## Không phải ONNX trên browser

Gradio mở trình duyệt chỉ là **giao diện**. Inference chạy **process Python local** (ONNX Runtime + PyTorch vocoder). Không phải `onnxruntime-web`.

---

## Tại sao vẫn cần PyTorch?

| Thành phần | Runtime |
|------------|---------|
| text_encoder + fm_decoder | **ONNX Runtime** |
| Vocos vocoder | **PyTorch** |
| Tensor / load audio | `torch` / `torchaudio` |

Không tải checkpoint ZipVoice PyTorch ~470 MB.

---

## Cấu trúc thư mục

```
models/onnx/          # ONNX weights (có sẵn, Git LFS)
models/vocoder/       # Tải khi setup (~50 MB)
espeak_tokenizer.py   # Espeak G2P (piper_phonemize)
vocos_fbank.py        # Mel features prompt audio
vieneu_text.py        # VieNeu punctuation cleanup
period_linebreak.py   # Ngoặc→phẩy, số+chấm→xuống dòng
app.py                # Gradio GUI
onnx_engine.py        # ONNX + Vocos inference
utils.py              # Pipeline normalize + chunk sách dài
```

---

## Dependencies

**Runtime** (`requirements-cpu.txt` + setup script):

| Gói | Vì sao |
|-----|--------|
| `onnxruntime` | Inference ONNX |
| `torch`, `torchaudio` | Vocoder + tensor |
| `vocos` | mel → waveform |
| `piper_phonemize` | Espeak tiếng Việt |
| `gradio` | GUI |
| `pydub`, `scipy`, `soundfile` | Audio I/O |
| `vinorm`, `vietnormalizer`, `sea-g2p` | Pipeline NSW (tùy chọn) |

**Setup một lần** (`requirements-setup.txt`): `huggingface_hub` — tải vocoder.

**Đã bỏ:** clone `vendor/ZipVoice`, `lhotse`, `jieba`, `librosa`, `matplotlib`, …

---

## Biến môi trường

| Biến | Mặc định | Ý nghĩa |
|------|----------|---------|
| `ZIPVOICE_FORCE_CPU` | `1` | Chạy CPU |
| `ZIPVOICE_ONNX_INT8` | `1` | ONNX INT8 |
| `GRADIO_SERVER_PORT` | `7862` | Port GUI |

---

## Troubleshooting

| Vấn đề | Cách xử lý |
|--------|------------|
| `Models not found` | `install_cpu.bat` |
| `Chưa cài vinorm` | `pip install vinorm` hoặc bỏ vinorm khỏi pipeline; dùng VieNeu / Cấu trúc TTS |
| Đọc liền trong ngoặc | Bật **Cấu trúc TTS** ở Bước 2 |
| Mục `một.` dính đoạn sau | Cấu trúc TTS + xem **Xem trước chuẩn hóa** |
| OOM sách dài | Giảm **Max ký tự/chunk** (100–110) |

Log: `logs/app.log`
