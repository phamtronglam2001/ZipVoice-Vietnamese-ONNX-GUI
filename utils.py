"""
Audio preprocessing, text chunking, and spectrogram helpers.
Transcript giọng mẫu bắt buộc nhập thủ công — không auto ASR.
"""
from __future__ import annotations

import logging
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from pydub import AudioSegment, silence
from scipy.io import wavfile
from scipy.signal import resample_poly

from config import apply_cpu_env, ensure_ffmpeg_on_path, set_offline_env

ensure_ffmpeg_on_path()
apply_cpu_env()
set_offline_env()

logger = logging.getLogger("zipvoice_gui")


def resample_to_24khz(input_path: str, output_path: str) -> None:
    orig_sr, audio = wavfile.read(input_path)
    if len(audio.shape) == 2:
        audio = audio.mean(axis=1)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    resampled = resample_poly(audio, 24000, orig_sr)
    resampled_int16 = (resampled * 32767).astype(np.int16)
    wavfile.write(output_path, 24000, resampled_int16)


# ─── Long-text chunking (paragraph → sentence → clause, inspired by VieNeu-TTS) ───

RE_NEWLINE = re.compile(r"[\r\n]+")
RE_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
RE_MINOR_PUNCT = re.compile(r"(?<=[,;:\-–—])\s+")
RE_CHAPTER_HEADING = re.compile(
    r"^(?:chương|phần|mục|chapter|part)\s+[\dIVXLCivxlc]+",
    re.IGNORECASE,
)
RE_MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+")
RE_HR_LINE = re.compile(r"^[\-*_]{3,}\s*$")


@dataclass
class TtsChunk:
    text: str
    pause_after: float = 0.35
    is_sentence_end: bool = True
    is_paragraph_end: bool = False
    is_chapter_break: bool = False


def _is_chapter_heading(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    return bool(
        RE_CHAPTER_HEADING.match(s)
        or RE_MARKDOWN_HEADING.match(s)
        or RE_HR_LINE.match(s)
    )


def _split_long_sentence(sentence: str, max_chars: int) -> list[tuple[str, bool]]:
    """Return (fragment, is_sentence_end). Forced splits use is_sentence_end=False."""
    sentence = sentence.strip()
    if not sentence:
        return []
    if len(sentence) <= max_chars:
        return [(sentence, True)]

    parts: list[tuple[str, bool]] = []
    sub_parts = RE_MINOR_PUNCT.split(sentence)
    buffer = ""
    for part in sub_parts:
        part = part.strip()
        if not part:
            continue
        candidate = f"{buffer} {part}".strip() if buffer else part
        if len(candidate) <= max_chars:
            buffer = candidate
        else:
            if buffer:
                parts.append((buffer, False))
            if len(part) <= max_chars:
                buffer = part
            else:
                words, current = part.split(), ""
                for word in words:
                    cand = f"{current} {word}".strip() if current else word
                    if current and len(cand) > max_chars:
                        parts.append((current, False))
                        current = word
                    else:
                        current = cand
                buffer = current
    if buffer:
        parts.append((buffer, True))
    if parts:
        parts[-1] = (parts[-1][0], True)
    return parts


def split_text_for_tts(
    text: str,
    max_chars: int = 135,
    pause_sentence: float = 0.35,
    pause_paragraph: float = 0.65,
    pause_chapter: float = 1.2,
    pause_forced_split: float = 0.12,
) -> list[TtsChunk]:
    """
    Chia văn bản dài cho ZipVoice:
    1) tôn trọng xuống dòng / tiêu đề chương
    2) gộp câu trong cùng đoạn đến max_chars
    3) cắt câu dài tại dấu phẩy / khoảng trắng
    """
    if not text or not text.strip():
        return []

    raw_blocks: list[tuple[str, bool]] = []
    for block in RE_NEWLINE.split(text.strip()):
        block = block.strip()
        if not block:
            continue
        is_chapter = _is_chapter_heading(block)
        raw_blocks.append((block, is_chapter))

    if not raw_blocks:
        raw_blocks = [(text.strip(), False)]

    merged_blocks: list[tuple[str, bool]] = []
    for block, is_chapter in raw_blocks:
        if is_chapter and merged_blocks and not merged_blocks[-1][1]:
            merged_blocks.append((block, True))
        elif is_chapter:
            merged_blocks.append((block, True))
        else:
            merged_blocks.append((block, False))

    chunks: list[TtsChunk] = []
    total_blocks = len(merged_blocks)

    for bi, (block, is_chapter_block) in enumerate(merged_blocks):
        sentences = RE_SENTENCE_END.split(block)
        buffer = ""
        block_chunks: list[TtsChunk] = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(sentence) > max_chars:
                if buffer:
                    block_chunks.append(
                        TtsChunk(text=buffer, is_sentence_end=True, pause_after=pause_sentence)
                    )
                    buffer = ""
                for frag, sent_end in _split_long_sentence(sentence, max_chars):
                    block_chunks.append(
                        TtsChunk(
                            text=frag,
                            is_sentence_end=sent_end,
                            pause_after=pause_forced_split if not sent_end else pause_sentence,
                        )
                    )
            elif buffer and len(buffer) + 1 + len(sentence) > max_chars:
                block_chunks.append(
                    TtsChunk(text=buffer, is_sentence_end=True, pause_after=pause_sentence)
                )
                buffer = sentence
            else:
                buffer = f"{buffer} {sentence}".strip() if buffer else sentence

        if buffer:
            block_chunks.append(
                TtsChunk(text=buffer, is_sentence_end=True, pause_after=pause_sentence)
            )

        if not block_chunks:
            continue

        is_last_block = bi == total_blocks - 1
        last = block_chunks[-1]
        if is_chapter_block:
            last.is_chapter_break = True
            last.pause_after = pause_chapter
        elif not is_last_block:
            last.is_paragraph_end = True
            last.pause_after = pause_paragraph
        elif last.is_sentence_end:
            last.pause_after = pause_sentence

        chunks.extend(block_chunks)

    if not chunks:
        return [TtsChunk(text=text.strip(), pause_after=0.0)]

    chunks[-1].pause_after = 0.0
    return chunks


def join_tts_audio_chunks(
    wave_chunks: list[np.ndarray],
    tts_chunks: list[TtsChunk],
    sample_rate: int,
) -> np.ndarray:
    if not wave_chunks:
        return np.array([], dtype=np.float32)
    if len(wave_chunks) == 1:
        return wave_chunks[0]

    final = wave_chunks[0]
    for i in range(1, len(wave_chunks)):
        pause_s = 0.0
        if i - 1 < len(tts_chunks):
            pause_s = tts_chunks[i - 1].pause_after
        gap = int(sample_rate * max(0.0, pause_s))
        if gap > 0:
            final = np.concatenate([final, np.zeros(gap, dtype=final.dtype), wave_chunks[i]])
        else:
            final = np.concatenate([final, wave_chunks[i]])
    return final


def chunk_text(text: str, max_chars: int = 135) -> list[str]:
    """API cũ — trả list str (không metadata pause)."""
    return [c.text for c in split_text_for_tts(text, max_chars=max_chars)]


def read_text_file(path: str | None, max_chars: int = 500_000) -> str:
    """Đọc .txt upload; hỗ trợ UTF-8 / UTF-8-BOM / CP1258."""
    if not path:
        return ""
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"Không tìm thấy file: {path}")
    if p.suffix.lower() not in {".txt", ".text", ".md"}:
        raise ValueError("Chỉ hỗ trợ file .txt / .md")

    raw = p.read_bytes()
    if len(raw) > max_chars * 4:
        raise ValueError(f"File quá lớn (>{max_chars:,} ký tự ước tính). Chia nhỏ file trước.")

    for enc in ("utf-8-sig", "utf-8", "cp1258", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            text = ""
    else:
        text = raw.decode("utf-8", errors="replace")

    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(text) > max_chars:
        raise ValueError(
            f"File có {len(text):,} ký tự — vượt giới hạn {max_chars:,}. "
            "Chia thành nhiều file hoặc tăng giới hạn trong config."
        )
    return text


def _remove_silence_edges(audio: AudioSegment, silence_threshold: int = -42) -> AudioSegment:
    non_silent_start = silence.detect_leading_silence(
        audio, silence_threshold=silence_threshold
    )
    audio = audio[non_silent_start:]
    end_ms = audio.duration_seconds
    for ms in reversed(audio):
        if ms.dBFS > silence_threshold:
            break
        end_ms -= 0.001
    return audio[: int(end_ms * 1000)]


def preprocess_ref_audio_text(
    ref_audio_orig: str,
    ref_text: str = "",
    clip_short: bool = True,
    show_info=print,
) -> tuple[str, str]:
    ref_text = ref_text.strip()
    if not ref_text:
        raise ValueError(
            "Bắt buộc nhập transcript giọng mẫu (ô số 2 hoặc trường `text` "
            "trong ref_info.json). App không tự nhận dạng giọng nói."
        )

    show_info("Đang xử lý file giọng mẫu...")
    logger.info("Using manual transcript | len=%d", len(ref_text))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        aseg = AudioSegment.from_file(ref_audio_orig)

        if clip_short:
            non_silent_segs = silence.split_on_silence(
                aseg,
                min_silence_len=1000,
                silence_thresh=-50,
                keep_silence=1000,
                seek_step=10,
            )
            non_silent_wave = AudioSegment.silent(duration=0)
            for seg in non_silent_segs:
                if len(non_silent_wave) > 6000 and len(non_silent_wave + seg) > 15000:
                    show_info("Audio over 15s — clipping (pass 1).")
                    break
                non_silent_wave += seg

            if len(non_silent_wave) > 15000:
                non_silent_segs = silence.split_on_silence(
                    aseg,
                    min_silence_len=100,
                    silence_thresh=-40,
                    keep_silence=1000,
                    seek_step=10,
                )
                non_silent_wave = AudioSegment.silent(duration=0)
                for seg in non_silent_segs:
                    if len(non_silent_wave) > 6000 and len(non_silent_wave + seg) > 15000:
                        show_info("Audio over 15s — clipping (pass 2).")
                        break
                    non_silent_wave += seg
                aseg = non_silent_wave

            if len(aseg) > 15000:
                aseg = aseg[:15000]
                show_info("Audio over 15s — hard clip.")

        aseg = _remove_silence_edges(aseg) + AudioSegment.silent(duration=50)
        aseg.export(f.name, format="wav")
        ref_audio = f.name

    if not ref_text.endswith(". ") and not ref_text.endswith("。"):
        ref_text = ref_text + (". " if not ref_text.endswith(".") else " ")

    return ref_audio, ref_text


def post_process_text(text: str) -> str:
    text = " " + text + " "
    for bad, good in [
        (" . . ", " . "),
        (" .. ", " . "),
        (" , , ", " , "),
        (" ,, ", " , "),
    ]:
        text = text.replace(bad, good)
    text = text.replace('"', "")
    return " ".join(text.split())


NORMALIZE_BACKENDS: dict[str, str] = {
    "none": "Không (chỉ dọn dấu câu)",
    "vinorm": "vinorm (TTSnorm)",
    "vietnormalizer": "vietnormalizer",
    "sea_g2p": "sea-g2p Normalizer",
}

# Dropdown choices for pipeline steps (bước 2–3 có thể chọn "none")
NORMALIZE_STEP_CHOICES: dict[str, str] = {
    "none": "— Không —",
    "vinorm": NORMALIZE_BACKENDS["vinorm"],
    "vietnormalizer": NORMALIZE_BACKENDS["vietnormalizer"],
    "sea_g2p": NORMALIZE_BACKENDS["sea_g2p"],
}

_sea_g2p_normalizer = None
_vietnamese_normalizer = None


def _strip_sea_g2p_en_tags(text: str) -> str:
    """sea-g2p bọc tiếng Anh bằng <en>; espeak ZipVoice không hiểu tag này."""
    return re.sub(r"</?en>", "", text)


def _normalize_vinorm(text: str) -> str:
    from vinorm import TTSnorm

    return TTSnorm(text)


def _normalize_vietnormalizer(text: str) -> str:
    global _vietnamese_normalizer
    if _vietnamese_normalizer is None:
        from vietnormalizer import VietnameseNormalizer

        _vietnamese_normalizer = VietnameseNormalizer()
    return _vietnamese_normalizer.normalize(text)


def _normalize_sea_g2p(text: str) -> str:
    global _sea_g2p_normalizer
    if _sea_g2p_normalizer is None:
        from sea_g2p import Normalizer

        _sea_g2p_normalizer = Normalizer()
    out = _sea_g2p_normalizer.normalize(text)
    if isinstance(out, list):
        out = out[0] if out else text
    return _strip_sea_g2p_en_tags(str(out))


def normalize_text(text: str, backend: str = "vinorm") -> str:
    """Chuẩn hóa NSW (số, ngày, ký hiệu...) trước khi đưa vào EspeakTokenizer."""
    key = (backend or "vinorm").strip().lower()
    if key in ("none", "off", "không"):
        return text
    try:
        if key == "vinorm":
            return _normalize_vinorm(text)
        if key == "vietnormalizer":
            return _normalize_vietnormalizer(text)
        if key == "sea_g2p":
            return _normalize_sea_g2p(text)
        logger.warning("Unknown normalize backend %r — skip", backend)
        return text
    except ImportError as exc:
        pkg = {
            "vinorm": "vinorm",
            "vietnormalizer": "vietnormalizer",
            "sea_g2p": "sea-g2p",
        }.get(key, key)
        raise ImportError(
            f"Chưa cài `{pkg}`. Chạy: pip install {pkg}"
        ) from exc
    except Exception:
        logger.exception("normalize_text failed | backend=%s", key)
        return text


def build_normalize_pipeline(step1: str, step2: str, step3: str) -> list[str]:
    """
    Ghép tối đa 3 bước chuẩn hóa theo thứ tự.
    Mỗi thư viện chỉ dùng Normalizer (đầu ra là text), không G2P — có thể xếp chuỗi.
    """
    pipeline: list[str] = []
    seen: set[str] = set()
    for raw in (step1, step2, step3):
        key = (raw or "none").strip().lower()
        if key in ("none", "off", "không", ""):
            continue
        if key not in NORMALIZE_BACKENDS or key == "none":
            raise ValueError(f"Backend chuẩn hóa không hợp lệ: {raw!r}")
        if key in seen:
            raise ValueError(
                f"Trùng bước chuẩn hóa: {NORMALIZE_BACKENDS.get(key, key)}"
            )
        seen.add(key)
        pipeline.append(key)
    return pipeline


def format_normalize_pipeline(backends: list[str]) -> str:
    if not backends:
        return NORMALIZE_BACKENDS["none"]
    return " → ".join(NORMALIZE_BACKENDS.get(b, b) for b in backends)


def normalize_text_pipeline(text: str, backends: list[str]) -> str:
    result = text
    for backend in backends:
        result = normalize_text(result, backend)
    return result


PREVIEW_MAX_CHARS = 20_000


def preview_normalize_output(
    text: str,
    step1: str,
    step2: str,
    step3: str,
    chunk_max_chars: int = 135,
    max_preview_chars: int = PREVIEW_MAX_CHARS,
) -> str:
    """
    Chạy pipeline chuẩn hóa trên văn bản (ô 3) — xem trước khi TTS.
    Không cần load model inference.
    """
    raw = (text or "").strip()
    if not raw:
        return "(Chưa có văn bản — nhập ô số 3 hoặc upload file .txt / .md)"

    pipeline = build_normalize_pipeline(step1, step2, step3)
    label = format_normalize_pipeline(pipeline)
    normalized = prepare_tts_text(raw, pipeline)
    chunks = split_text_for_tts(raw, max_chars=int(chunk_max_chars))

    lines = [
        f"Pipeline: {label}",
        f"Gốc: {len(raw):,} ký tự → sau chuẩn hóa: {len(normalized):,} ký tự",
        f"Chunks TTS (max {int(chunk_max_chars)} ký tự/chunk): {len(chunks)}",
        "",
        "── Text sau pipeline (lowercase + dọn dấu câu) ──",
    ]

    if len(normalized) > max_preview_chars:
        lines.append(normalized[:max_preview_chars])
        lines.append(
            f"\n… (còn {len(normalized) - max_preview_chars:,} ký tự — "
            "bấm Tổng hợp để xử lý toàn bộ)"
        )
    else:
        lines.append(normalized)

    if len(chunks) > 1:
        lines.extend(["", "── Xem trước từng chunk (5 chunk đầu) ──"])
        for i, ch in enumerate(chunks[:5]):
            chunk_norm = prepare_tts_text(ch.text, pipeline)
            lines.append(f"[{i + 1}/{len(chunks)}] {chunk_norm}")
        if len(chunks) > 5:
            lines.append(f"… và {len(chunks) - 5} chunk nữa")

    return "\n".join(lines)


def prepare_tts_text(text: str, backend: str | list[str] = "vinorm") -> str:
    if isinstance(backend, list):
        normalized = (
            normalize_text_pipeline(text, backend) if backend else text
        )
    else:
        normalized = normalize_text(text, backend)
    return post_process_text(normalized).lower()


def normalize_vietnamese(text: str) -> str:
    """Giữ tương thích code cũ — mặc định vinorm."""
    return normalize_text(text, "vinorm")

