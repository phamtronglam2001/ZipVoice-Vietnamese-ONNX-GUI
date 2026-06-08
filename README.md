# ZipVoice Vietnamese ONNX GUI

GUI Gradio offline cho TTS tiếng Việt zero-shot. **ZipVoice** (text_encoder + fm_decoder) chạy qua **ONNX Runtime** — không cần checkpoint PyTorch ZipVoice (~470 MB). **Vocoder Vocos ONNX 100 mel** (export local từ ZipVoice-Vietnamese-GUI) + **librosa ISTFT** (`vocos_istft.py`).

Model ONNX export từ [hynt/ZipVoice-Vietnamese-2500h](https://huggingface.co/hynt/ZipVoice-Vietnamese-2500h). Weights ZipVoice (int4/int8 trong `models/onnx/`) và vocoder **100 mel** (`models/vocoder/mel_spec_24khz.onnx`, export ZipVoice-Vietnamese-GUI) được **bundle sẵn** trong repo qua **Git LFS** — không tải model từ mạng lúc cài đặt.

**Tác giả:** [Pham Trong Lam](https://github.com/phamtronglam2001) · **Repository:** https://github.com/phamtronglam2001/ZipVoice-Vietnamese-ONNX-GUI · **License:** Non-Commercial (xem `LICENSE`)

Tiếng Việt | [English](README_EN.md)

**Repo PyTorch đầy đủ:** [ZipVoice-Vietnamese-GUI](https://github.com/phamtronglam2001/ZipVoice-Vietnamese-GUI)

---

## Nguồn gốc & trích dẫn

| Thành phần | GitHub / nguồn | Giấy phép |
|------------|----------------|-----------|
| **Mã GUI + giọng mẫu `.wav`** | [phamtronglam2001/ZipVoice-Vietnamese-ONNX-GUI](https://github.com/phamtronglam2001/ZipVoice-Vietnamese-ONNX-GUI) | Non-Commercial |
| **ZipVoice ONNX** | Export từ [hynt/ZipVoice-Vietnamese-2500h](https://huggingface.co/hynt/ZipVoice-Vietnamese-2500h) · upstream [k2-fsa/ZipVoice](https://github.com/k2-fsa/ZipVoice) | CC-BY-NC-SA-4.0 · Apache-2.0 |
| **Vocos vocoder** | Kiến trúc [gemelo-ai/vocos](https://github.com/gemelo-ai/vocos) · PyTorch [charactr/vocos-mel-24khz](https://huggingface.co/charactr/vocos-mel-24khz) (100 mel) · ONNX export local `mel_spec_24khz.onnx` | MIT · model card |
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

### Vocoder

Kiến trúc [gemelo-ai/vocos](https://github.com/gemelo-ai/vocos) (Siuzdak et al.) · weights gốc PyTorch [charactr/vocos-mel-24khz](https://huggingface.co/charactr/vocos-mel-24khz) (100 mel) · ONNX bundled `models/vocoder/mel_spec_24khz.onnx` (export ZipVoice-Vietnamese-GUI)

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
| **VocosFbank (mel prompt)** | [k2-fsa/ZipVoice](https://github.com/k2-fsa/ZipVoice) `zipvoice/utils/feature.py` | `vocos_fbank.py` — librosa `htk=True`, `norm=None` (khớp torchaudio upstream, RMSE ~0.017) |
| **Vocos decode** | [gemelo-ai/vocos](https://github.com/gemelo-ai/vocos) · ONNX 100 mel + librosa ISTFT | `onnx_engine.py`, `vocos_istft.py` |
| **ONNX EP (CPU/CUDA)** | [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) | `onnx_providers.py` |
| **Pipeline TTS dùng chung** | Viết trong repo này | `tts_pipeline.py`, `chunk_synthesis.py` |
| **Nhật ký trạng thái** | Viết trong repo này | `status_log.py` |

### Pipeline chuẩn hóa text (GUI)

| Backend | GitHub / nguồn | File / cài đặt |
|---------|----------------|----------------|
| **VieNeu** | [pnnbao97/VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS) — `vieneu_utils/core_utils.py` | `vieneu_text.py` (built-in) |
| **Cấu trúc TTS** | Viết trong repo này (Pham Trong Lam) | `period_linebreak.py` (built-in) |
| **Xuống dòng → câu** | Viết trong repo này | `period_linebreak.py` (`newline_sentence`) |
| **Gộp xuống dòng PDF** | Viết trong repo này | `period_linebreak.py` (`join_soft_breaks`) |
| **sea-g2p Normalizer** | [pnnbao97/sea-g2p](https://github.com/pnnbao97/sea-g2p) | `pip install sea-g2p` — chỉ Normalizer |

**Chunk sách dài** (`utils.py`): lấy cảm hứng từ [VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS), triển khai riêng.

> **Lưu ý:** [vinorm](https://github.com/NoahDrisort/vinorm) và [vietnormalizer](https://github.com/nghimestudio/vietnormalizer) từng được hỗ trợ như backend NSW tùy chọn; hiện **không cài kèm**. Dùng `sea-g2p` thay thế.

### UI

| Thành phần | GitHub |
|------------|--------|
| **Gradio** | [gradio-app/gradio](https://github.com/gradio-app/gradio) |
| **Slint (scaffold)** | [slint-ui/slint](https://github.com/slint-ui/slint) — `slint_gui/` |

---

## So với repo PyTorch

| | [ZipVoice-Vietnamese-GUI](https://github.com/phamtronglam2001/ZipVoice-Vietnamese-GUI) | Repo này |
|---|---|---|
| Inference ZipVoice | PyTorch checkpoint ~470 MB | ONNX (int4 / int8) |
| Download setup | ~2 GB | **Không** (clone + `git lfs pull`) |
| ONNX trong repo | Không | Có (`models/onnx/` + `models/vocoder/`, Git LFS) |
| Vocoder mặc định | PyTorch Vocos | ONNX 100 mel (local) + librosa ISTFT |
| Clone `vendor/ZipVoice` | Có | **Không** |
| Port mặc định | 7860 | 7862 |
| GUI desktop native | Không | Slint scaffold (`run_slint_gui.bat`) |

---

## Cài đặt nhanh (Windows)

Cần [uv](https://docs.astral.sh/uv/) trên PATH. **Chỉ dùng file `.bat`** — không gọi PowerShell.

### CPU (mặc định)

```bat
git clone https://github.com/phamtronglam2001/ZipVoice-Vietnamese-ONNX-GUI.git
cd ZipVoice-Vietnamese-ONNX-GUI
git lfs install
git lfs pull
install_cpu.bat
run_gui.bat
```

`install_cpu.bat` hỏi **Co GPU NVIDIA CUDA?** — trả lời `N` cho CPU thuần, hoặc `Y` để cài `onnxruntime-gpu` + DLL CUDA qua pip (tương tự `install_gpu.bat`).

### GPU (NVIDIA CUDA)

Sau clone + `git lfs pull`:

```bat
install_gpu.bat
run_gpu.bat
```

- Gỡ `onnxruntime` CPU trước khi cài `onnxruntime-gpu` (hai gói không cùng tồn tại).
- `requirements-gpu.txt` cài thêm **`nvidia-cublas-cu12`**, **`nvidia-cudnn-cu12`**, **`nvidia-cuda-runtime-cu12`**, **`nvidia-cufft-cu12`** — **không bắt buộc** cài full CUDA Toolkit 12 nếu dùng wheel pip trên.
- `install_gpu.bat` kiểm tra weights trong `models/`, vocoder ONNX, và CUDA EP (`onnx_providers.is_cuda_execution_provider_loadable`).
- Hoặc cài [CUDA Toolkit 12](https://developer.nvidia.com/cuda-downloads) + cuDNN 9 — `onnx_providers.py` tự thêm thư mục `bin` vào PATH / `add_dll_directory`.
- Sau cài GPU, `.install_mode` = `gpu` → `run_gpu.bat` / `run_gui.bat` tự bật `ZIPVOICE_ONNX_GPU=1`.

Mở http://127.0.0.1:7862

Nếu thiếu weights sau clone:

- ZipVoice ONNX: `git lfs pull`
- Vocoder 100 mel: đặt tại `models/vocoder/mel_spec_24khz.onnx` (export ZipVoice-Vietnamese-GUI)

---

## Chạy ứng dụng

| Script | Mô tả |
|--------|--------|
| `run_gui.bat` | **Khuyến nghị** — Gradio GUI; đọc `.install_mode` → CPU hoặc GPU |
| `run_cpu.bat` | Gradio GUI **CPU-only** (`ZIPVOICE_FORCE_CPU=1`) |
| `run_gpu.bat` | Gradio GUI **GPU** (`ZIPVOICE_ONNX_GPU=1`, kiểm tra CUDA DLL) |
| `run_cli.bat` | CLI TTS (preset/profile) |
| `run_slint_gui.bat` | Slint desktop GUI (scaffold) — cài thêm `requirements-slint.txt` |

---

## Tính năng GUI

- **9 giọng mẫu** trong `assets/ref_audio/` + `ref_info.json`
- Upload giọng mẫu + **transcript bắt buộc** (không ASR)
- Upload `.txt` / `.md` cho **sách dài**
- **Pipeline chuẩn hóa** danh sách tùy chỉnh (mặc định trống; preset Sách gợi ý pipeline đầy đủ sea-g2p → … → VieNeu)
- **Xem trước chuẩn hóa** — nút riêng, không load model TTS
- **Chia chunk** đoạn → câu → max ký tự; nghỉ theo câu / đoạn / chương / mục đánh số
- Xuất WAV / MP3 → `output/`
- **ONNX quant mode** — int8 / int4 (dropdown; mặc định từ `quantization.json` hoặc auto-detect)
- **GPU** — `run_gpu.bat` tự bật; hoặc checkbox «Dùng GPU» trên GUI; hoặc `set ZIPVOICE_ONNX_GPU=1`
- **Nhật ký trạng thái (debug)** — textbox theo dõi từng bước tổng hợp (`StatusLog`)
- **Nghe thử / tải file** — trả về đường dẫn WAV (`type="filepath"`), tránh lỗi chuyển đổi float32 preview
- **Mỗi lần tổng hợp là run mới** — không resume checkpoint chunk cũ (`output/.checkpoints/` đã bỏ)
- **Preset/profile** (`profiles/*.json`) — lưu/tải toàn bộ cấu hình đọc sách (GUI + CLI)

---

## Kiến trúc inference

```
Text → pipeline chuẩn hóa → chunk → Espeak phonemize
  → VocosFbank (mel prompt từ ref audio)
  → ONNX ZipVoice (text_encoder + fm_decoder, flow steps numpy)
  → Vocos decode (ONNX 100 mel + librosa ISTFT)
  → ghép chunk + nghỉ → WAV/MP3
```

| Thành phần | Runtime | Ghi chú |
|------------|---------|---------|
| text_encoder + fm_decoder | **ONNX Runtime** (+ numpy flow) | Không checkpoint PyTorch ~470 MB |
| Prompt mel (ref audio) | **VocosFbank** — librosa | Khớp [k2-fsa/ZipVoice `feature.py`](https://github.com/k2-fsa/ZipVoice) (`htk=True`, `norm=None`) |
| **Vocos vocoder** | **ONNX 100 mel** + `vocos_istft.py` | `mel_spec_24khz.onnx` (Git LFS hoặc export GUI) |
| Execution provider | CPU / CUDA / DirectML | `onnx_providers.py` |

Logic TTS dùng chung: `tts_pipeline.py` (Gradio, Slint, CLI).

---

## ONNX quant mode (`models/onnx/`)

| Mode | File ONNX | Ghi chú |
|------|-----------|---------|
| **int8** | `*_int8.onnx` | **Khuyên dùng** — cân bằng tốc độ + chất lượng (mặc định khi không có manifest) |
| **int4** | `*_int4.onnx` | Nhanh / nhỏ; **thử nghiệm** — có thể kém hơn int8 |

**Chọn mode tự động** (ưu tiên):

1. Biến môi trường `ZIPVOICE_ONNX_QUANT` (`int8` hoặc `int4`)
2. `models/onnx/quantization.json` (`mode`, `filenames`, …)
3. Quét thư mục: int4 > int8
4. Legacy `ZIPVOICE_ONNX_INT8=1` → int8
5. Mặc định: **int8**

Repo hiện bundle mẫu `quantization.json` với `"mode": "int4"`. Export thêm bản int8 từ [ZipVoice-Vietnamese-GUI](https://github.com/phamtronglam2001/ZipVoice-Vietnamese-GUI) hoặc `quantize_onnx.py`.

**Khuyến nghị chất lượng:** **int8** + vocoder ONNX. INT4 nhanh hơn nhưng có thể kém hơn int8.

---

## Vocoder (ONNX)

| | ONNX 100 mel + librosa ISTFT |
|---|---|
| Weights | `models/vocoder/mel_spec_24khz.onnx` — export ZipVoice-Vietnamese-GUI (100 mel, khớp ZipVoice) |
| Decode | ONNX Runtime + `vocos_istft.py` |
| Prompt mel | `vocos_fbank.py` — librosa (`htk=True`, `norm=None`) |

Kiểm tra file bundled: `python download_models.py` (chỉ verify local, không download). Sau `install_cpu.bat` / `install_gpu.bat` cũng chạy bước tương tự.

---

## GPU (CUDA / DirectML)

- Checkbox **Dùng GPU (CUDA / DirectML)** trên GUI (accordion **Hiệu năng**), hoặc chạy `run_gpu.bat` / `set ZIPVOICE_ONNX_GPU=1`.
- Cần cài `install_gpu.bat` (hoặc `Y` khi `install_cpu.bat`).
- `onnx_providers.py`:
  - `ensure_cuda_runtime_on_path()` — thêm DLL từ pip `nvidia-*` hoặc CUDA Toolkit vào PATH
  - `is_cuda_execution_provider_loadable()` — probe `onnxruntime_providers_cuda.dll` + `cublasLt64_12.dll`, `cudnn64_9.dll`
  - Nếu CUDA không load được → **fallback CPU** (cảnh báo trong log, không crash)
- INT4 trên GPU có thể fallback CPU tùy ORT build.
- Song song chunk: CPU tối đa N workers (`ZIPVOICE_CPU_MAX_WORKERS`); **GPU mặc định 1 worker** (`ZIPVOICE_GPU_MAX_WORKERS`) — mỗi worker là process riêng load ONNX trên GPU.

---

## Preset / profile (`profiles/`)

Preset JSON **schema v1** gom mọi thiết lập audiobook: giọng, pipeline chuẩn hóa, chunk, nghỉ, tham số tổng hợp ONNX, định dạng xuất.

| File mặc định | Mô tả |
|---------------|--------|
| `profiles/none.json` | Pipeline trống, giọng upload thủ công |
| `profiles/sach.json` | Pipeline đầy đủ (sea-g2p → … → VieNeu), giọng **Ái Vy** |

### GUI

Accordion **Preset** trên GUI:

- **Chọn preset** từ `profiles/*.json`
- **Tải preset** — áp dụng giọng, pipeline, chunk, nghỉ, tốc độ, quant, xuất file
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
- `--gpu` — ONNX Runtime CUDA/DirectML

Ví dụ schema rút gọn:

```json
{
  "schema_version": 1,
  "name": "Sách — Ái Vy",
  "voice": { "mode": "bundled", "voice_id": "ai_vy" },
  "pipeline": ["sea_g2p", "period_break", "newline_sentence", "join_soft_breaks", "vieneu"],
  "chunk_max_chars": 135,
  "pause": { "sentence": 0.35, "paragraph": 0.65, "chapter": 2.0, "enum_item": 1.0, "forced_split": 0.28 },
  "synthesis": { "use_int8": true, "num_step": 16, "speed": 1.0, "guidance_scale": 1.0, "t_shift": 0.5 },
  "export": { "format": "WAV 24kHz" },
  "input_mode": "raw"
}
```

> Preset legacy `synthesis.use_int8` vẫn map sang int8; GUI mới dùng dropdown **ONNX quant mode**.

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
| Sách / audiobook | sea-g2p → Cấu trúc TTS → … → VieNeu |
| OCR / PDF ngắt dòng | Gộp PDF → VieNeu → Cấu trúc TTS |
| Tiêu đề chương | thêm **Xuống dòng → câu** trước hoặc sau Cấu trúc TTS |

> **Lưu ý:** ZipVoice chỉ dùng **Espeak** để phonemize. Các bước trên đều trả về **text** — có thể xếp chuỗi an toàn.

---

## Slint GUI (desktop — đồng bộ Gradio qua `tts_pipeline.py`)

Chạy: `run_slint_gui.bat` · Chi tiết synced/omitted: [`slint_gui/README.md`](slint_gui/README.md)

---

## Debug & nhật ký

| Kênh | Mô tả |
|------|--------|
| **Nhật ký trạng thái (debug)** | Textbox Gradio — class `StatusLog` (`status_log.py`): timestamp, giai đoạn chuẩn hóa / chunk / ONNX / vocoder |
| **`logs/app.log`** | Log file Python (`zipvoice_onnx_gui`) — CUDA probe, lỗi inference, cảnh báo DLL |
| Slint | Text area status tương tự qua `tts_controller.py` |

Mỗi lần bấm **Tổng hợp** bắt đầu log mới; **không** resume chunk cũ từ thư mục checkpoint.

---

## Không phải ONNX trên browser

Gradio mở trình duyệt chỉ là **giao diện**. Inference chạy **process Python local** (ONNX Runtime + vocoder ONNX + librosa ISTFT). Không phải `onnxruntime-web`.

---

## Cấu trúc thư mục

```
models/onnx/              # ZipVoice ONNX + quantization.json (Git LFS)
models/vocoder/           # mel_spec_24khz.onnx (Git LFS)
models/THIRD_PARTY_LICENSES.md
assets/ref_audio/         # 9 giọng mẫu + ref_info.json
profiles/                 # Preset JSON (none, sach, …)
output/                   # WAV/MP3 xuất (mỗi run mới, không .checkpoints/)
logs/app.log              # Log ứng dụng
app.py                    # Gradio GUI
tts_pipeline.py           # Pipeline TTS dùng chung (Gradio / Slint / CLI)
chunk_synthesis.py        # Tổng hợp đa chunk / workers
onnx_engine.py            # ONNX ZipVoice + vocoder decode
onnx_providers.py         # CPU / CUDA / DirectML, ensure_cuda_runtime_on_path
onnx_quant.py             # Quant mode, quantization.json, quantize helpers
status_log.py             # StatusLog cho GUI debug textbox
vocos_fbank.py            # Mel prompt — khớp ZipVoice upstream
vocos_istft.py            # librosa ISTFT (vocoder ONNX)
espeak_tokenizer.py       # Espeak G2P (piper_phonemize)
vieneu_text.py            # VieNeu punctuation cleanup
period_linebreak.py       # Ngoặc→phẩy, số+chấm→xuống dòng
preset_io.py              # Load/save preset, GUI ↔ JSON
cli_tts.py                # CLI profile-driven
utils.py                  # Pipeline normalize + chunk sách dài
slint_gui/                # Slint desktop scaffold (main.py, ui/, backend/)
install_cpu.bat           # Cài CPU hoặc hỏi GPU
install_gpu.bat           # Cài onnxruntime-gpu + nvidia DLL pip
run_gui.bat               # Gradio auto (port 7862)
run_cpu.bat               # Gradio CPU-only
run_gpu.bat               # Gradio GPU (ZIPVOICE_ONNX_GPU=1)
run_slint_gui.bat         # Slint GUI
run_cli.bat               # CLI TTS
requirements-cpu.txt      # onnxruntime + librosa (fbank)
requirements-gpu.txt      # onnxruntime-gpu + nvidia-cublas/cudnn/cuda-runtime/cufft
requirements-slint.txt    # slint (tách riêng)
requirements-normalize.txt
download_models.py        # Verify bundled ONNX + vocoder (local only)
```

---

## Dependencies

| File | Nội dung |
|------|----------|
| `install_cpu.bat` | uv venv + pip deps + kiểm tra models; tùy chọn GPU (`Y`) |
| `install_gpu.bat` | `onnxruntime-gpu` + `requirements-gpu.txt` + kiểm tra models/vocoder + probe CUDA |
| `run_gui.bat` | Khởi động Gradio GUI — tự chọn CPU/GPU theo `.install_mode` |
| `run_cpu.bat` | Gradio GUI CPU-only |
| `run_gpu.bat` | Gradio GUI GPU (`ZIPVOICE_ONNX_GPU=1`) |
| `run_slint_gui.bat` | Slint GUI (auto cài `requirements-slint.txt`) |
| `run_cli.bat` | CLI TTS (preset/profile) |
| `requirements-cpu.txt` | onnxruntime, gradio, librosa, … |
| `requirements-gpu.txt` | onnxruntime-gpu, nvidia-cublas/cudnn/cuda-runtime/cufft-cu12, librosa, … |
| `requirements-slint.txt` | `slint>=1.9.0` (GUI desktop riêng) |
| `requirements-normalize.txt` | sea-g2p (tùy chọn) |
| `download_models.py` | kiểm tra file bundled local (ZipVoice ONNX + vocoder 100 mel) |

**Đã bỏ:** clone `vendor/ZipVoice`, resume checkpoint `output/.checkpoints/`, `lhotse`, `jieba`, `matplotlib`, …

---

## Biến môi trường

| Biến | Mặc định | Ý nghĩa |
|------|----------|---------|
| `ZIPVOICE_FORCE_CPU` | `1` (`run_cpu.bat` / CPU install) | Ép CPU; ẩn GPU |
| `ZIPVOICE_ONNX_GPU` | `1` (`run_gpu.bat` / GPU install) | Dùng CUDA/DirectML khi có |
| `ZIPVOICE_ONNX_QUANT` | *(từ manifest)* | Ghi đè quant mode: `int8` / `int4` |
| `ZIPVOICE_ONNX_INT8` | *(legacy)* | `1` → int8 nếu không set `ZIPVOICE_ONNX_QUANT` |
| `GRADIO_SERVER_PORT` | `7862` | Port GUI |

---

## Troubleshooting

| Vấn đề | Cách xử lý |
|--------|------------|
| `Models not found` | `git lfs pull` rồi `install_cpu.bat` |
| Thiếu ONNX vocoder | Đặt `models/vocoder/mel_spec_24khz.onnx` (100 mel, export ZipVoice-Vietnamese-GUI) |
| Thiếu sea-g2p | `pip install -r requirements-normalize.txt` |
| **Nhiễu / tiếng rè rộng băng tần** | Đảm bảo `vocos_fbank.py` dùng librosa `htk=True`, `norm=None`; thử **int8** thay int4 |
| Audio preview lỗi / méo | Đã sửa: output trả **filepath WAV**, không array float32 — cập nhật repo mới nhất |
| Bật GPU nhưng vẫn CPU | Chạy `install_gpu.bat`; kiểm tra `logs/app.log` (thiếu `cublasLt64_12.dll` / `cudnn64_9.dll`); cài pip `nvidia-*` hoặc CUDA Toolkit 12 |
| `onnxruntime` vs `onnxruntime-gpu` xung đột | Chỉ cài một gói — dùng `install_gpu.bat` |
| Kết quả TTS cũ / chunk lẻ | **Không còn resume checkpoint** — mỗi lần Tổng hợp chạy từ đầu; xóa file cũ trong `output/` nếu cần |
| Đọc liền trong ngoặc | Bật **Cấu trúc TTS** ở Bước 2 |
| Mục `một.` dính đoạn sau | Cấu trúc TTS + xem **Xem trước chuẩn hóa** |
| OOM / crash `0xc0000005` khi GPU | Thường **không phải** hết VRAM RTX — thường do **workers song song > 1** (mỗi process load ONNX riêng trên CUDA → crash driver). Đặt **Workers = 1**; thử **int4**; giảm **Max ký tự/chunk** (100–110) |
| OOM sách dài | Giảm **Max ký tự/chunk** (100–110); workers GPU = 1; CPU có thể tăng workers |
| Task Manager: VRAM trống nhưng vẫn lỗi | Đang xem **sai GPU**: laptop Intel+NVIDIA — **GPU 0** thường là Intel iGPU, **GPU 1** là NVIDIA. ORT **CUDA** chỉ chạy trên NVIDIA (`device_id=0` = GPU đầu trong `nvidia-smi`, không phải Intel). Chạy `scripts/diagnose_gpu.py` |
| Bật GPU nhưng Task Manager không thấy NVIDIA tăng | CUDA DLL thiếu → fallback **CPU** (không dùng Intel cho CUDA EP). Xem **Nhật ký trạng thái** / `logs/app.log` dòng `Thực tế: CUDA device_id=0 (RTX …)` |
| Chỉ có DirectML, không CUDA | `onnxruntime` CPU + DirectML có thể chạy trên **Intel iGPU** (GPU 0). Cài `install_gpu.bat` để dùng `CUDAExecutionProvider` trên RTX |

**Chẩn đoán GPU:** `.venv\Scripts\python.exe scripts\diagnose_gpu.py` — liệt kê EP, CUDA probe, `nvidia-smi`, và gợi ý Task Manager / `CUDA_VISIBLE_DEVICES`.

Log: `logs/app.log` · Nhật ký debug: textbox **Nhật ký trạng thái** trên GUI

---

## License

- **Repo này (mã + giọng mẫu):** [Non-Commercial](LICENSE) — Pham Trong Lam
- **Weights bundled:** xem [`models/THIRD_PARTY_LICENSES.md`](models/THIRD_PARTY_LICENSES.md)
- **Checkpoint gốc:** [hynt/ZipVoice-Vietnamese-2500h](https://huggingface.co/hynt/ZipVoice-Vietnamese-2500h) — CC-BY-NC-SA-4.0
- **ZipVoice upstream:** [k2-fsa/ZipVoice](https://github.com/k2-fsa/ZipVoice) — Apache-2.0
- **Thư viện bên thứ ba:** xem [Trích dẫn](#trích-dẫn-mã-nguồn--thư-viện-bên-thứ-ba)
