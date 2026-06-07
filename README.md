# ZipVoice Vietnamese ONNX GUI

GUI Gradio offline cho TTS tiếng Việt zero-shot, chạy inference qua **ONNX Runtime** — không cần checkpoint PyTorch ZipVoice (~470 MB).

Model ONNX export từ [hynt/ZipVoice-Vietnamese-2500h](https://huggingface.co/hynt/ZipVoice-Vietnamese-2500h). Vocoder Vocos chạy **ONNX** ([wetdog/vocos-mel-24khz-onnx](https://huggingface.co/wetdog/vocos-mel-24khz-onnx)) + **librosa ISTFT** — **toàn bộ weights có sẵn trong `models/`** (Git LFS), không tải Hugging Face khi cài đặt.

**Tác giả:** [Pham Trong Lam](https://github.com/phamtronglam2001) · **Repository:** https://github.com/phamtronglam2001/ZipVoice-Vietnamese-ONNX-GUI · **License:** Non-Commercial (xem `LICENSE`)

Tiếng Việt | [English](README_EN.md)

**Repo PyTorch đầy đủ:** [ZipVoice-Vietnamese-GUI](https://github.com/phamtronglam2001/ZipVoice-Vietnamese-GUI)

---

## Nguồn gốc & trích dẫn

| Thành phần | GitHub / nguồn | Giấy phép |
|------------|----------------|-----------|
| **Mã GUI + giọng mẫu `.wav`** | [phamtronglam2001/ZipVoice-Vietnamese-ONNX-GUI](https://github.com/phamtronglam2001/ZipVoice-Vietnamese-ONNX-GUI) | Non-Commercial |
| **ZipVoice ONNX** | Export từ [hynt/ZipVoice-Vietnamese-2500h](https://huggingface.co/hynt/ZipVoice-Vietnamese-2500h) · upstream [k2-fsa/ZipVoice](https://github.com/k2-fsa/ZipVoice) | CC-BY-NC-SA-4.0 · Apache-2.0 |
| **Vocos vocoder** | Kiến trúc [gemelo-ai/vocos](https://github.com/gemelo-ai/vocos) · ONNX [wetdog/vocos-mel-24khz-onnx](https://huggingface.co/wetdog/vocos-mel-24khz-onnx) | MIT · model card |
| **ONNX Runtime** | [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) | MIT |
| **Chuẩn hóa text** | Xem [bảng chi tiết](#trích-dẫn-mã-nguồn--thư-viện-bên-thứ-ba) | Theo từng repo |

---

## Lời cảm ơn (Acknowledgments)

### Checkpoint tiếng Việt — Hugging Face

| | |
|---|---|
| **Tác giả / publisher** | [**Nguyen Thien Hy**](https://huggingface.co/hynt) (`hynt`) |
| **Model gốc** | [hynt/ZipVoice-Vietnamese-2500h](https://huggingface.co/hynt/ZipVoice-Vietnamese-2500h) — ONNX trong repo export từ checkpoint này |
| **Demo Space** | [hynt/ZipVoice-Vietnamese-100h](https://huggingface.co/spaces/hynt/ZipVoice-Vietnamese-100h) |
| **Dữ liệu train** | PhoAudioBook, ViVoice, UEH (model card); demucs tách nhạc nền |

### ZipVoice gốc — k2-fsa

[k2-fsa/ZipVoice](https://github.com/k2-fsa/ZipVoice) · Zhu, Han et al., [arXiv:2506.13053](https://arxiv.org/abs/2506.13053) · Apache-2.0

### Vocoder ONNX

Kiến trúc [gemelo-ai/vocos](https://github.com/gemelo-ai/vocos) (Siuzdak et al.) · weights gốc [charactr/vocos-mel-24khz](https://huggingface.co/charactr/vocos-mel-24khz) · export ONNX [wetdog/vocos-mel-24khz-onnx](https://huggingface.co/wetdog/vocos-mel-24khz-onnx)

### Dataset tham chiếu

[thivux/phoaudiobook](https://huggingface.co/datasets/thivux/phoaudiobook) (Vu et al., ACL 2025) · ViVoice · UEH — theo model card `hynt`.

> Repo GUI (Pham Trong Lam) không thay thế `hynt` / `k2-fsa`. Xin tuân thủ CC-BY-NC-SA-4.0 và ghi rõ audio là AI-generated.

---

## Trích dẫn mã nguồn & thư viện bên thứ ba

### Inference (ONNX stack)

| Thành phần | GitHub / nguồn | File trong repo |
|------------|----------------|-----------------|
| **ZipVoice ONNX** | [k2-fsa/ZipVoice](https://github.com/k2-fsa/ZipVoice) (`infer_zipvoice_onnx.py`) | `models/onnx/*.onnx`, `onnx_engine.py` |
| **Checkpoint gốc** | [hynt/ZipVoice-Vietnamese-2500h](https://huggingface.co/hynt/ZipVoice-Vietnamese-2500h) | weights đã export |
| **Espeak tokenizer** | [k2-fsa/ZipVoice](https://github.com/k2-fsa/ZipVoice) (vendored) | `espeak_tokenizer.py` |
| **piper_phonemize** | [k2-fsa/icefall](https://github.com/k2-fsa/icefall) | wheel cài qua `install_cpu.bat` |
| **Espeak** | [espeak-ng/espeak-ng](https://github.com/espeak-ng/espeak-ng) | backend phonemize |
| **VocosFbank (mel prompt)** | [k2-fsa/ZipVoice](https://github.com/k2-fsa/ZipVoice) `zipvoice/utils/feature.py` | `vocos_fbank.py` (port) |
| **Vocos decode** | [gemelo-ai/vocos](https://github.com/gemelo-ai/vocos) · ONNX [wetdog/vocos-mel-24khz-onnx](https://huggingface.co/wetdog/vocos-mel-24khz-onnx) | `onnx_engine.py`, `vocos_istft.py` |

### Pipeline chuẩn hóa text (GUI)

| Backend | GitHub / nguồn | File / cài đặt |
|---------|----------------|----------------|
| **VieNeu** | [pnnbao97/VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS) — `vieneu_utils/core_utils.py` | `vieneu_text.py` (built-in) |
| **Cấu trúc TTS** | Viết trong repo này (Pham Trong Lam) | `period_linebreak.py` (built-in) |
| **Xuống dòng → câu** | Viết trong repo này | `period_linebreak.py` (`newline_sentence`) |
| **Gộp xuống dòng PDF** | Viết trong repo này | `period_linebreak.py` (`join_soft_breaks`) |
| **vinorm** | [NoahDrisort/vinorm](https://github.com/NoahDrisort/vinorm) | `pip install vinorm` |
| **vietnormalizer** | [nghimestudio/vietnormalizer](https://github.com/nghimestudio/vietnormalizer) | `pip install vietnormalizer` |
| **sea-g2p Normalizer** | [pnnbao97/sea-g2p](https://github.com/pnnbao97/sea-g2p) | `pip install sea-g2p` — chỉ Normalizer |

**Chunk sách dài** (`utils.py`): lấy cảm hứng từ [VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS), triển khai riêng.

### UI

| Thành phần | GitHub |
|------------|--------|
| **Gradio** | [gradio-app/gradio](https://github.com/gradio-app/gradio) |

---

## So với repo PyTorch

| | [ZipVoice-Vietnamese-GUI](https://github.com/phamtronglam2001/ZipVoice-Vietnamese-GUI) | Repo này |
|---|---|---|
| Inference | PyTorch checkpoint ~470 MB | ONNX (INT8 mặc định) |
| Download setup | ~2 GB | **Không** (clone + `git lfs pull`) |
| ONNX trong repo | Không | Có (`models/onnx/` + `models/vocoder/`, Git LFS) |
| Clone `vendor/ZipVoice` | Có | **Không** |
| Port mặc định | 7860 | 7862 |

---

## Cài đặt nhanh (Windows CPU)

Cần [uv](https://docs.astral.sh/uv/) trên PATH.

```bat
git clone https://github.com/phamtronglam2001/ZipVoice-Vietnamese-ONNX-GUI.git
cd ZipVoice-Vietnamese-ONNX-GUI
git lfs install
git lfs pull
install_cpu.bat
run_cpu.bat
```

Mở http://127.0.0.1:7862 — **không gọi PowerShell**.

Nếu thiếu `models/vocoder/mel_spec_24khz.onnx` sau clone, chạy `python download_models.py` (fallback HTTP, không cần `huggingface_hub`).

---

## Tính năng GUI

- **9 giọng mẫu** trong `assets/ref_audio/` + `ref_info.json`
- Upload giọng mẫu + **transcript bắt buộc** (không ASR)
- Upload `.txt` / `.md` cho **sách dài**
- **Pipeline chuẩn hóa** danh sách tùy chỉnh (mặc định trống; preset Sách gợi ý VieNeu → Cấu trúc → vinorm)
- **Xem trước chuẩn hóa** — nút riêng, không load model TTS
- **Chia chunk** đoạn → câu → max ký tự; nghỉ theo câu / đoạn / chương / mục đánh số
- Xuất WAV / MP3 → `output/`
- ONNX **INT8** (mặc định) hoặc FP32
- **Preset/profile** (`profiles/*.json`) — lưu/tải toàn bộ cấu hình đọc sách (GUI + CLI)

---

## Preset / profile (`profiles/`)

Preset JSON **schema v1** gom mọi thiết lập audiobook: giọng, pipeline chuẩn hóa, chunk, nghỉ, tham số tổng hợp ONNX, định dạng xuất.

| File mặc định | Mô tả |
|---------------|--------|
| `profiles/none.json` | Pipeline trống, giọng upload thủ công |
| `profiles/sach.json` | VieNeu → Cấu trúc TTS → vinorm, giọng **Ái Vy** |

### GUI

Accordion **Preset** trên GUI:

- **Chọn preset** từ `profiles/*.json`
- **Tải preset** — áp dụng giọng, pipeline, chunk, nghỉ, tốc độ, INT8, xuất file
- **Lưu preset** — nhập tên → ghi `profiles/<tên>.json` từ trạng thái hiện tại

### CLI (chỉ load preset, không sửa pipeline trên dòng lệnh)

```bat
run_cli.bat list-voices
run_cli.bat profile list
run_cli.bat profile show sach
run_cli.bat preview -p sach -t "Chương 1. Xin chào."
run_cli.bat synthesize -p sach -f book.txt -o output/book.wav
```

- `--profile` / `-p` (mặc định `none`) nạp **toàn bộ** `PresetConfig`
- Chỉ `-o` / `--output` ghi đè đường dẫn file ra; mọi tham số khác lấy từ preset
- `--skip-normalize` / `--input-prepared` — input đã chuẩn hóa (bỏ qua pipeline)
- `--output-normalized PATH` — ghi full text sau pipeline ra `.txt`
- `--normalize-only` (lệnh `synthesize`) — chỉ chuẩn hóa, không TTS

Ví dụ schema rút gọn:

```json
{
  "schema_version": 1,
  "name": "Sách — Ái Vy",
  "voice": { "mode": "bundled", "voice_id": "ai_vy" },
  "voice_manual": { "mode": "manual", "ref_wav": "", "ref_text": "" },
  "pipeline": ["vieneu", "period_break", "vinorm"],
  "chunk_max_chars": 135,
  "pause": { "sentence": 0.35, "paragraph": 0.65, "chapter": 2.0, "enum_item": 1.0, "forced_split": 0.28 },
  "synthesis": { "use_int8": true, "num_step": 16, "speed": 1.0, "guidance_scale": 1.0, "t_shift": 0.5 },
  "export": { "format": "WAV 24kHz" },
  "input_mode": "raw"
}
```

Module: `preset_io.py` · CLI: `cli_tts.py` · `run_cli.bat`

### Xuất / nhập lại text đã chuẩn hóa (workflow sách dài)

Luồng khuyến nghị khi cần **chỉnh sửa thủ công** trước TTS:

```
book.txt → pipeline → book_normalized.txt → (sửa tay) → TTS (bỏ qua pipeline) → audio
```

**GUI**

1. Chế độ **Văn bản gốc** — nhập/upload `book.txt`, cấu hình pipeline, bấm **Xem trước** hoặc **Xuất text đã chuẩn hóa (.txt)**.
2. Mở file `output/<tên>_normalized.txt`, sửa lỗi NSW / dấu câu / xuống dòng.
3. Upload file đã sửa vào ô 3, chọn **Đã chuẩn hóa (bỏ qua pipeline)** → **Tổng hợp**.

Ô **Text đã chuẩn hóa** hiển thị **toàn bộ** văn bản (không cắt preview). Preset có thể lưu `"input_mode": "prepared"`.

**CLI**

```bat
run_cli.bat synthesize -p sach -f book.txt --normalize-only --output-normalized output/book_normalized.txt
run_cli.bat synthesize -p sach -f output/book_normalized.txt --skip-normalize -o output/book.wav
```

| Flag | Ý nghĩa |
|------|---------|
| `--normalize-only` | Chỉ chạy pipeline, in/lưu `.txt`, không TTS |
| `--output-normalized path.txt` | Lưu text sau chuẩn hóa |
| `--skip-normalize` / `--input-prepared` | Input đã chuẩn hóa — bỏ qua pipeline (chỉ dọn dấu câu nhẹ, **không** lowercase) |

---

## Pipeline chuẩn hóa (chuỗi bước tuần tự)

Thêm/xóa/sắp xếp bước trên GUI — **mỗi bước nhận output của bước trước** (`text₀ → bước₁ → text₁ → …`). Không giới hạn số bước; không cho trùng cùng backend.

| Backend | Cài pip? | Vai trò |
|---------|----------|---------|
| **VieNeu** | Không (built-in) | Port [pnnbao97/VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS) `core_utils` |
| **Gộp xuống dòng PDF** | Không (built-in) | `join_soft_breaks` — gộp dòng ngắn viết thường (OCR/PDF) |
| **Xuống dòng → câu** | Không (built-in) | `newline_sentence` — `Chương 1\nNội dung` → `Chương 1.\nNội dung` |
| **Cấu trúc TTS** | Không (built-in) | `period_break` — ngoặc→phẩy; `một. đoạn` → `một.\nđoạn` |
| **vinorm** | `pip install vinorm` | [NoahDrisort/vinorm](https://github.com/NoahDrisort/vinorm) |
| **vietnormalizer** | pip | [nghimestudio/vietnormalizer](https://github.com/nghimestudio/vietnormalizer) |
| **sea-g2p Normalizer** | pip | [pnnbao97/sea-g2p](https://github.com/pnnbao97/sea-g2p) — chỉ Normalizer |
| **Không** | — | Bỏ qua bước đó |

### Cấu trúc TTS (`period_linebreak.py`) — chi tiết

| Quy tắc | Ví dụ vào | Sau xử lý |
|----------|-----------|-----------|
| Ngoặc → phẩy (ngắt hơi) | `mẫu (mẹ)` | `mẫu, mẹ` |
| Số + chấm → xuống dòng | `một. đọc đoạn` | `một.` + xuống dòng + `đọc đoạn` |
| Nghỉ sau mục đánh số | `một.` (đoạn riêng) | ~**1.0 s** trước đoạn tiếp |

**Luồng TTS / preview:** chuẩn hóa **toàn bộ** văn bản (`normalize_full_document`) → **chia chunk** (`split_text_for_tts`) → mỗi chunk chỉ dọn dấu câu nhẹ. Ô **Xem trước** hiển thị output chuỗi pipeline, **giữ `\n`** và dấu `.` đã thêm.

### Gợi ý pipeline

| Loại văn bản | Gợi ý |
|--------------|-------|
| Mặc định GUI | *(pipeline trống)* |
| Sách / audiobook | VieNeu → Cấu trúc TTS → vinorm (hoặc sea-g2p) |
| OCR / PDF ngắt dòng | Gộp PDF → VieNeu → Cấu trúc TTS |
| Tiêu đề chương | thêm **Xuống dòng → câu** trước hoặc sau Cấu trúc TTS |

> **Lưu ý:** ZipVoice chỉ dùng **Espeak** để phonemize. Các bước trên đều trả về **text** — có thể xếp chuỗi an toàn.

---

## Không phải ONNX trên browser

Gradio mở trình duyệt chỉ là **giao diện**. Inference chạy **process Python local** (ONNX Runtime + librosa ISTFT). Không phải `onnxruntime-web`.

---

## Stack inference (không PyTorch)

| Thành phần | Runtime | Ghi chú |
|------------|---------|---------|
| text_encoder + fm_decoder | **ONNX Runtime + numpy** | Không checkpoint PyTorch ~470 MB |
| Prompt mel + flow steps | **numpy / scipy / librosa** | Không torchaudio |
| **Vocos vocoder** | **ONNX wetdog + librosa ISTFT** | `mel_spec_24khz.onnx` (~54 MB) |

Cài đặt: **`install_cpu.bat`** (uv).

---

## Cấu trúc thư mục

```
models/onnx/          # ZipVoice ONNX (Git LFS)
models/vocoder/       # Vocos mel_spec_24khz.onnx (Git LFS)
models/THIRD_PARTY_LICENSES.md
espeak_tokenizer.py   # Espeak G2P (piper_phonemize)
vocos_fbank.py        # Mel features prompt audio
vocos_istft.py        # librosa ISTFT từ mag/x/y
vieneu_text.py        # VieNeu punctuation cleanup
period_linebreak.py   # Ngoặc→phẩy, số+chấm→xuống dòng
profiles/             # Preset JSON (none, sach, …)
preset_io.py          # Load/save preset, GUI ↔ JSON
cli_tts.py            # CLI profile-driven
run_cli.bat           # Chạy CLI
app.py                # Gradio GUI
onnx_engine.py        # ONNX + Vocos inference
utils.py              # Pipeline normalize + chunk sách dài
```

---

## Dependencies

| File | Nội dung |
|------|----------|
| `install_cpu.bat` | uv venv + pip deps + kiểm tra models có sẵn |
| `requirements-cpu.txt` | onnxruntime, gradio, librosa, scipy, … |
| `requirements-normalize.txt` | vinorm, sea-g2p (tùy chọn) |
| `download_models.py` | fallback vocoder nếu chưa `git lfs pull` |

**Đã bỏ:** PyTorch/torchaudio, `vocos` pip package, clone `vendor/ZipVoice`, `lhotse`, `jieba`, `matplotlib`, …

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
| `Models not found` | `git lfs pull` rồi `install_cpu.bat`; hoặc `python download_models.py` (vocoder) |
| `Chưa cài vinorm` | `pip install vinorm` hoặc bỏ vinorm khỏi pipeline; dùng VieNeu / Cấu trúc TTS |
| Đọc liền trong ngoặc | Bật **Cấu trúc TTS** ở Bước 2 |
| Mục `một.` dính đoạn sau | Cấu trúc TTS + xem **Xem trước chuẩn hóa** |
| OOM sách dài | Giảm **Max ký tự/chunk** (100–110) |

Log: `logs/app.log`

---

## License

- **Repo này (mã + giọng mẫu):** [Non-Commercial](LICENSE) — Pham Trong Lam
- **Weights bundled:** xem [`models/THIRD_PARTY_LICENSES.md`](models/THIRD_PARTY_LICENSES.md)
- **Checkpoint gốc:** [hynt/ZipVoice-Vietnamese-2500h](https://huggingface.co/hynt/ZipVoice-Vietnamese-2500h) — CC-BY-NC-SA-4.0
- **ZipVoice upstream:** [k2-fsa/ZipVoice](https://github.com/k2-fsa/ZipVoice) — Apache-2.0
- **Thư viện bên thứ ba:** xem [Trích dẫn](#trích-dẫn-mã-nguồn--thư-viện-bên-thứ-ba)
