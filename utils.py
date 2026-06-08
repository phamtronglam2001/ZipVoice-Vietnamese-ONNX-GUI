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
from typing import List, Literal

NormalizeInputMode = Literal["raw", "prepared"]

INPUT_MODE_CHOICES: dict[str, str] = {
    "raw": "Văn bản gốc",
    "prepared": "Đã chuẩn hóa (bỏ qua pipeline)",
}

import numpy as np
from pydub import AudioSegment, silence
from scipy.io import wavfile
from scipy.signal import resample_poly

from config import OUTPUT_DIR, apply_cpu_env, ensure_ffmpeg_on_path, set_offline_env

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

PAUSE_SENTENCE_DEFAULT = 0.35
PAUSE_PARAGRAPH_DEFAULT = 0.65
PAUSE_CHAPTER_DEFAULT = 2.0
PAUSE_ENUM_DEFAULT = 1.0
PAUSE_FORCED_SPLIT_DEFAULT = 0.28
CROSSFADE_FORCED_SPLIT_S = 0.040

RE_NEWLINE = re.compile(r"[\r\n]+")
RE_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
RE_MINOR_PUNCT = re.compile(r"(?<=[,;:\-–—])\s+")
RE_CHAPTER_HEADING = re.compile(
    r"^(?:"
    r"(?:chương|phần|mục|chapter|part)\s+[\dIVXLCivxlc]+"
    r"|chương\s+thứ\s+"
    r"(?:nhất|một|hai|ba|tư|năm|sáu|bảy|tám|chín|mười(?:\s+một|\s+hai)?)"
    r"|lời\s+nói\s+đầu"
    r"|phụ\s+lục"
    r")",
    re.IGNORECASE,
)
RE_MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+")
RE_HR_LINE = re.compile(r"^[\-*_]{3,}\s*$")

# ZipVoice ONNX trims prompt mel frames after ODE; chunks at/below this length (or a
# lone short word) often yield 0 mel frames or vocoder noise — merge before synthesis.
MIN_TTS_CHUNK_CHARS = 12
# Single-word chunks up to this length are merged even when above MIN_TTS_CHUNK_CHARS.
MIN_TTS_SINGLE_WORD_MAX_CHARS = 16

_PLACEHOLDER_PREFIX = "\x00VNPROT"
_SENTENCE_PROTECTED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\d+\.\d+"), "DEC"),
    (re.compile(r"(?:PGS\.TS|PGS|TS|tp|Tp|Dr|Mr|Mrs)\.", re.IGNORECASE), "ABBR"),
    (re.compile(r"v\.v(?:\.|…)"), "VV"),
    (re.compile(r"(?<![\n\r])(?:số|Số)\s+\d+(?:[.,]\d+)?"), "SO"),
    (
        re.compile(
            r"(?i)\bchương\s+(?:thứ\s+)?(?:nhất|một|hai|ba|tư|năm|sáu|bảy|tám|chín|mười)",
        ),
        "CH",
    ),
]


@dataclass
class TtsChunk:
    text: str
    pause_after: float = PAUSE_SENTENCE_DEFAULT
    is_sentence_end: bool = True
    is_paragraph_end: bool = False
    is_chapter_break: bool = False
    is_forced_split: bool = False


def _is_chapter_heading(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    return bool(
        RE_CHAPTER_HEADING.match(s)
        or RE_MARKDOWN_HEADING.match(s)
        or RE_HR_LINE.match(s)
    )


def split_sentences_vn(text: str) -> list[str]:
    """Tách câu tiếng Việt — bảo vệ số thập phân, viết tắt, v.v."""
    if not text or not text.strip():
        return []

    placeholders: dict[str, str] = {}
    protected = text

    def _make_placeholder(match: re.Match[str]) -> str:
        key = f"{_PLACEHOLDER_PREFIX}{len(placeholders)}__"
        placeholders[key] = match.group(0)
        return key

    for pattern, _kind in _SENTENCE_PROTECTED_PATTERNS:
        protected = pattern.sub(_make_placeholder, protected)

    raw_parts = RE_SENTENCE_END.split(protected)
    sentences: list[str] = []
    for part in raw_parts:
        restored = part
        for key, original in placeholders.items():
            restored = restored.replace(key, original)
        restored = restored.strip()
        if restored:
            sentences.append(restored)
    return sentences if sentences else [text.strip()]


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


def _should_not_merge_into(prev: TtsChunk) -> bool:
    """Không gộp chunk nhỏ vào điểm nghỉ cấu trúc (đoạn/chương/enum)."""
    return (
        prev.pause_after >= 1.0
        or prev.is_paragraph_end
        or prev.is_chapter_break
    )


def _is_too_short_for_synthesis(text: str) -> bool:
    """True when chunk text is likely to produce 0/noisy mel after ONNX ODE trim."""
    t = text.strip()
    if not t:
        return True
    if len(t) <= MIN_TTS_CHUNK_CHARS:
        return True
    words = t.split()
    if len(words) == 1 and len(t) <= MIN_TTS_SINGLE_WORD_MAX_CHARS:
        return True
    return False


def _preview_chunk_text(text: str, limit: int = 60) -> str:
    t = text.strip()
    return t if len(t) <= limit else f"{t[: limit - 1]}…"


def _merge_tiny_chunks(
    chunks: list[TtsChunk],
    merge_log: list[str] | None = None,
) -> list[TtsChunk]:
    """Gộp chunk quá ngắn vào chunk liền kề — tránh mel T=0 / noise sau ODE trim."""
    if len(chunks) <= 1:
        return chunks

    def _log(msg: str) -> None:
        logger.info(msg)
        if merge_log is not None:
            merge_log.append(msg)

    merged: list[TtsChunk] = []
    for ch in chunks:
        text = ch.text.strip()
        if not text:
            continue
        if (
            merged
            and _is_too_short_for_synthesis(text)
            and not ch.is_chapter_break
            and not _should_not_merge_into(merged[-1])
        ):
            prev = merged[-1]
            new_text = f"{prev.text} {ch.text}".strip()
            merged[-1] = TtsChunk(
                text=new_text,
                pause_after=ch.pause_after,
                is_sentence_end=ch.is_sentence_end,
                is_paragraph_end=ch.is_paragraph_end,
                is_chapter_break=ch.is_chapter_break,
                is_forced_split=prev.is_forced_split,
            )
            _log(
                "Gộp chunk ngắn "
                f"({len(text)} ký tự) «{_preview_chunk_text(text, 40)}» "
                f"vào chunk trước → «{_preview_chunk_text(new_text)}»"
            )
        elif (
            merged
            and _is_too_short_for_synthesis(text)
            and not ch.is_chapter_break
            and _should_not_merge_into(merged[-1])
        ):
            _log(
                "Giữ chunk ngắn "
                f"({len(text)} ký tự) «{_preview_chunk_text(text, 40)}» "
                f"— không gộp vào chunk trước (nghỉ đoạn/chương "
                f"{merged[-1].pause_after}s)"
            )
            merged.append(ch)
        else:
            merged.append(ch)

    if len(merged) <= 1:
        return merged

    i = 0
    while i < len(merged) - 1:
        ch = merged[i]
        text = ch.text.strip()
        if _is_too_short_for_synthesis(text) and not ch.is_chapter_break:
            if _should_not_merge_into(ch):
                _log(
                    "Giữ chunk ngắn "
                    f"({len(text)} ký tự) «{_preview_chunk_text(text, 40)}» "
                    f"— không gộp qua nghỉ đoạn/chương ({ch.pause_after}s)"
                )
                i += 1
                continue
            nxt = merged[i + 1]
            new_text = f"{ch.text} {nxt.text}".strip()
            merged[i + 1] = TtsChunk(
                text=new_text,
                pause_after=nxt.pause_after,
                is_sentence_end=nxt.is_sentence_end,
                is_paragraph_end=nxt.is_paragraph_end,
                is_chapter_break=nxt.is_chapter_break,
                is_forced_split=ch.is_forced_split or nxt.is_forced_split,
            )
            _log(
                "Gộp chunk ngắn "
                f"({len(text)} ký tự) «{_preview_chunk_text(text, 40)}» "
                f"vào chunk sau → «{_preview_chunk_text(new_text)}»"
            )
            merged.pop(i)
            continue
        i += 1

    return merged


def split_text_for_tts(
    text: str,
    max_chars: int = 135,
    pause_sentence: float = PAUSE_SENTENCE_DEFAULT,
    pause_paragraph: float = PAUSE_PARAGRAPH_DEFAULT,
    pause_chapter: float = PAUSE_CHAPTER_DEFAULT,
    pause_enum_item: float = PAUSE_ENUM_DEFAULT,
    pause_forced_split: float = PAUSE_FORCED_SPLIT_DEFAULT,
    merge_log: list[str] | None = None,
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
        sentences = split_sentences_vn(block)
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
                    forced = not sent_end
                    block_chunks.append(
                        TtsChunk(
                            text=frag,
                            is_sentence_end=sent_end,
                            pause_after=pause_forced_split if forced else pause_sentence,
                            is_forced_split=forced,
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
            from period_linebreak import is_enumeration_only_block

            last.pause_after = (
                pause_enum_item
                if is_enumeration_only_block(block)
                else pause_paragraph
            )
        elif last.is_sentence_end:
            last.pause_after = pause_sentence

        chunks.extend(block_chunks)

    if not chunks:
        return [TtsChunk(text=text.strip(), pause_after=0.0)]

    chunks[-1].pause_after = 0.0
    return _merge_tiny_chunks(chunks, merge_log=merge_log)


def _linear_crossfade(
    a: np.ndarray, b: np.ndarray, fade_samples: int
) -> np.ndarray:
    if fade_samples <= 0 or a.size == 0 or b.size == 0:
        return np.concatenate([a, b])
    fade_samples = min(fade_samples, a.size, b.size)
    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    blended = a[-fade_samples:] * fade_out + b[:fade_samples] * fade_in
    return np.concatenate([a[:-fade_samples], blended, b[fade_samples:]])


def join_tts_audio_chunks(
    wave_chunks: list[np.ndarray],
    tts_chunks: list[TtsChunk],
    sample_rate: int,
) -> np.ndarray:
    if not wave_chunks:
        return np.array([], dtype=np.float32)

    normalized = [
        w.astype(np.float32)
        if w is not None and getattr(w, "size", 0) > 0
        else np.array([], dtype=np.float32)
        for w in wave_chunks
    ]
    if len(normalized) == 1:
        return normalized[0]

    final = normalized[0]
    for i in range(1, len(normalized)):
        pause_s = 0.0
        prev_meta: TtsChunk | None = None
        if i - 1 < len(tts_chunks):
            prev_meta = tts_chunks[i - 1]
            pause_s = prev_meta.pause_after
        segment = normalized[i]

        if (
            prev_meta
            and prev_meta.is_forced_split
            and final.size > 0
            and segment.size > 0
        ):
            fade_n = int(sample_rate * CROSSFADE_FORCED_SPLIT_S)
            final = _linear_crossfade(final, segment, fade_n)
            remaining = max(0.0, pause_s - CROSSFADE_FORCED_SPLIT_S)
            if remaining > 0:
                gap = int(sample_rate * remaining)
                final = np.concatenate(
                    [final, np.zeros(gap, dtype=np.float32)]
                )
        else:
            gap = int(sample_rate * max(0.0, pause_s))
            if gap > 0:
                final = np.concatenate(
                    [final, np.zeros(gap, dtype=np.float32), segment]
                )
            else:
                final = np.concatenate([final, segment])
    return final


def format_tts_timing_line(
    elapsed_sec: float,
    num_chunks: int,
    quant_mode: str,
    *,
    chunks_synthesized: int | None = None,
    parallel_workers: int | None = None,
) -> str:
    """One-line Vietnamese TTS benchmark summary for GUI status and logs."""
    quant = str(quant_mode).strip().lower() or "?"
    per_denom = (
        chunks_synthesized
        if chunks_synthesized is not None and chunks_synthesized > 0
        else num_chunks
    )
    if per_denom <= 0:
        per_denom = 1
    per_chunk = elapsed_sec / per_denom
    parts = [
        f"Hoàn tất trong {elapsed_sec:.1f}s",
        f"{num_chunks} chunk",
        quant,
        f"~{per_chunk:.1f}s/chunk",
    ]
    if parallel_workers is not None and parallel_workers > 1:
        parts.insert(-1, f"{parallel_workers} worker")
    return " | ".join(parts)


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


def post_process_text(text: str, *, apply_lower: bool = True) -> str:
    text = " " + text + " "
    for bad, good in [
        (" . . ", " . "),
        (" .. ", " . "),
        (" , , ", " , "),
        (" ,, ", " , "),
    ]:
        text = text.replace(bad, good)
    text = text.replace('"', "")
    lines = [" ".join(line.split()) for line in text.splitlines()]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.lower() if apply_lower else cleaned


def parse_input_mode(mode: str | bool | None) -> NormalizeInputMode:
    if mode is True:
        return "prepared"
    key = (str(mode or "raw")).strip().lower()
    if key in ("prepared", "skip", "skip_normalize", "input_prepared"):
        return "prepared"
    return "raw"


def prepare_tts_text_passthrough(text: str) -> str:
    """Prepared input: strip + light punctuation cleanup, no lowercase."""
    text = (text or "").strip()
    if not text:
        return ""
    return post_process_text(text, apply_lower=False)


def prepare_for_tts(
    text: str,
    pipeline: list[str] | str | None = None,
    mode: NormalizeInputMode = "raw",
    *,
    already_normalized: bool = False,
) -> str:
    """Run pipeline on a text fragment. Prefer normalize_full_document + split for TTS."""
    if already_normalized or parse_input_mode(mode) == "prepared":
        return prepare_tts_text_passthrough(text)
    return prepare_tts_text(text, pipeline or [])


def normalize_full_document(
    text: str,
    pipeline: list[str] | str | None,
    mode: NormalizeInputMode = "raw",
) -> str:
    """Full-document normalize for export / preview textbox."""
    raw = (text or "").strip()
    if not raw:
        return ""
    if parse_input_mode(mode) == "prepared":
        return prepare_tts_text_passthrough(raw)
    steps = build_normalize_pipeline(pipeline)
    normalized = normalize_text_pipeline(raw, steps) if steps else raw
    return post_process_text(normalized, apply_lower=True)


def default_normalized_export_path(
    source_hint: str = "text",
    output_dir: Path | None = None,
) -> Path:
    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(source_hint).stem if source_hint else "text"
    stem = re.sub(r"[^\w\-.]+", "_", stem, flags=re.UNICODE).strip("._") or "text"
    return out / f"{stem}_normalized.txt"


def export_normalized_text_file(
    text: str,
    pipeline: list[str] | str | None,
    mode: NormalizeInputMode = "raw",
    output_path: str | Path | None = None,
    source_hint: str = "text",
) -> Path:
    full = normalize_full_document(text, pipeline, mode)
    if not full:
        raise ValueError("Không có văn bản để xuất sau chuẩn hóa.")
    path = Path(output_path) if output_path else default_normalized_export_path(source_hint)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(full, encoding="utf-8")
    logger.info("exported normalized text | path=%s | chars=%d | mode=%s", path, len(full), mode)
    return path.resolve()


NORMALIZE_BACKENDS: dict[str, str] = {
    "none": "Không (chỉ dọn dấu câu)",
    "vieneu": "VieNeu (dọn punctuation/noise)",
    "join_soft_breaks": "Gộp xuống dòng PDF (dòng ngắn, viết thường)",
    "newline_sentence": "Xuống dòng → ranh giới câu (thêm chấm)",
    "period_break": "Cấu trúc TTS (ngoặc→phẩy, số+chấm→xuống dòng)",
    "sea_g2p": "sea-g2p Normalizer",
}

# Dropdown choices for pipeline steps (bước 2–3 có thể chọn "none")
NORMALIZE_STEP_CHOICES: dict[str, str] = {
    "none": "— Không —",
    "vieneu": NORMALIZE_BACKENDS["vieneu"],
    "join_soft_breaks": NORMALIZE_BACKENDS["join_soft_breaks"],
    "newline_sentence": NORMALIZE_BACKENDS["newline_sentence"],
    "period_break": NORMALIZE_BACKENDS["period_break"],
    "sea_g2p": NORMALIZE_BACKENDS["sea_g2p"],
}

# Backends available in "Chọn loại chuẩn hóa" (excludes "none")
NORMALIZE_ADD_CHOICES: dict[str, str] = {
    k: v for k, v in NORMALIZE_STEP_CHOICES.items() if k != "none"
}

DEFAULT_NORMALIZE_PIPELINE: list[str] = []
# Full pipeline: all add-dropdown steps, bottom-to-top UI order (sea_g2p runs first).
AUDIOBOOK_PRESET_PIPELINE: list[str] = list(reversed(list(NORMALIZE_ADD_CHOICES)))

_sea_g2p_normalizer = None


def _strip_sea_g2p_en_tags(text: str) -> str:
    """sea-g2p bọc tiếng Anh bằng <en>; espeak ZipVoice không hiểu tag này."""
    return re.sub(r"</?en>", "", text)


def _normalize_per_line(text: str, normalize_fn) -> str:
    """Run NSW normalizer per line so user paragraph breaks (\\n) survive."""
    if not text or "\n" not in text:
        return normalize_fn(text)
    out: list[str] = []
    for line in text.split("\n"):
        out.append(normalize_fn(line) if line.strip() else "")
    return "\n".join(out)


def _normalize_vieneu(text: str) -> str:
    from vieneu_text import clean_text_noise

    return clean_text_noise(text)


def _normalize_period_break(text: str) -> str:
    from period_linebreak import prepare_tts_structure

    return prepare_tts_structure(text)


def _normalize_newline_sentence(text: str) -> str:
    from period_linebreak import newline_sentence_boundary

    return newline_sentence_boundary(text)


def _normalize_join_soft_breaks(text: str) -> str:
    from period_linebreak import join_soft_breaks

    return join_soft_breaks(text)


def _normalize_sea_g2p(text: str) -> str:
    global _sea_g2p_normalizer
    if _sea_g2p_normalizer is None:
        from sea_g2p import Normalizer

        _sea_g2p_normalizer = Normalizer()

    def _run(line: str) -> str:
        out = _sea_g2p_normalizer.normalize(line)
        if isinstance(out, list):
            out = out[0] if out else line
        return _strip_sea_g2p_en_tags(str(out))

    return _normalize_per_line(text, _run)


def normalize_text(text: str, backend: str = "none") -> str:
    """Chuẩn hóa NSW (số, ngày, ký hiệu...) trước khi đưa vào EspeakTokenizer."""
    key = (backend or "none").strip().lower()
    if key in ("none", "off", "không"):
        return text
    try:
        if key == "vieneu":
            return _normalize_vieneu(text)
        if key == "period_break":
            return _normalize_period_break(text)
        if key == "newline_sentence":
            return _normalize_newline_sentence(text)
        if key == "join_soft_breaks":
            return _normalize_join_soft_breaks(text)
        if key == "sea_g2p":
            return _normalize_sea_g2p(text)
        logger.warning("Unknown normalize backend %r — skip", backend)
        return text
    except ImportError as exc:
        pkg = {
            "sea_g2p": "sea-g2p",
        }.get(key, key)
        raise ImportError(
            f"Chưa cài `{pkg}` (không có sẵn trong ZipVoice — package PyPI riêng). "
            f"Chạy: pip install {pkg}  hoặc dùng bước `VieNeu` / `Không`."
        ) from exc
    except Exception:
        logger.exception("normalize_text failed | backend=%s", key)
        return text


def build_normalize_pipeline(steps: list[str] | str | None) -> list[str]:
    """
    Validate ordered normalizer keys (any length).
    Each backend outputs text only (no G2P) — steps can be chained.
    Duplicate keys are skipped with a warning.
    """
    if steps is None:
        return []
    if isinstance(steps, str):
        key = (steps or "none").strip().lower()
        if key in ("none", "off", "không", ""):
            return []
        if key not in NORMALIZE_BACKENDS or key == "none":
            raise ValueError(f"Backend chuẩn hóa không hợp lệ: {steps!r}")
        return [key]

    pipeline: list[str] = []
    seen: set[str] = set()
    for raw in steps:
        key = (raw or "none").strip().lower()
        if key in ("none", "off", "không", ""):
            continue
        if key not in NORMALIZE_BACKENDS or key == "none":
            raise ValueError(f"Backend chuẩn hóa không hợp lệ: {raw!r}")
        if key in seen:
            logger.warning(
                "Trùng bước chuẩn hóa (bỏ qua): %s",
                NORMALIZE_BACKENDS.get(key, key),
            )
            continue
        seen.add(key)
        pipeline.append(key)
    return pipeline


def format_normalize_pipeline(backends: list[str]) -> str:
    if not backends:
        return NORMALIZE_BACKENDS["none"]
    return " → ".join(NORMALIZE_BACKENDS.get(b, b) for b in backends)


def format_normalize_pipeline_list(backends: list[str]) -> str:
    """Numbered markdown for the Gradio pipeline builder."""
    if not backends:
        return (
            "**Chưa có bước** — khi TTS chỉ áp dụng dọn dấu câu (post-process)."
        )
    lines = [f"**{i}.** {NORMALIZE_BACKENDS.get(k, k)}" for i, k in enumerate(backends, 1)]
    return f"{format_normalize_pipeline(backends)}\n\n" + "\n".join(lines)


def pipeline_selector_choices(steps: list[str]) -> list[tuple[str, str]]:
    return [
        (f"{i + 1}. {NORMALIZE_BACKENDS.get(k, k)}", str(i))
        for i, k in enumerate(steps)
    ]


def _parse_pipeline_index(index: int | str | None) -> int | None:
    if index is None or index == "":
        return None
    try:
        return int(index)
    except (TypeError, ValueError):
        return None


def pipeline_add_step(steps: list[str] | None, new_step: str) -> list[str]:
    result = list(steps or [])
    key = (new_step or "").strip().lower()
    if not key or key == "none":
        return result
    if key not in NORMALIZE_BACKENDS:
        raise ValueError(f"Backend chuẩn hóa không hợp lệ: {new_step!r}")
    if key in result:
        logger.warning(
            "Trùng bước — không thêm lại: %s",
            NORMALIZE_BACKENDS.get(key, key),
        )
        return result
    result.append(key)
    return result


def pipeline_remove_at(steps: list[str] | None, index: int | str | None) -> list[str]:
    result = list(steps or [])
    idx = _parse_pipeline_index(index)
    if idx is None or not (0 <= idx < len(result)):
        return result
    result.pop(idx)
    return result


def pipeline_move(
    steps: list[str] | None, index: int | str | None, direction: int
) -> list[str]:
    result = list(steps or [])
    idx = _parse_pipeline_index(index)
    if idx is None:
        return result
    new_idx = idx + direction
    if 0 <= idx < len(result) and 0 <= new_idx < len(result):
        result[idx], result[new_idx] = result[new_idx], result[idx]
    return result


def normalize_text_pipeline(text: str, backends: list[str]) -> str:
    """Apply pipeline steps sequentially: text₀ → step₁ → text₁ → …"""
    result = text
    for backend in backends:
        result = normalize_text(result, backend)
    return result


PREVIEW_MAX_CHARS = 20_000


def preview_normalize_output(
    text: str,
    pipeline_steps: list[str] | str | None,
    chunk_max_chars: int = 135,
    mode: NormalizeInputMode = "raw",
    max_preview_chars: int | None = None,
) -> str:
    """
    Chạy pipeline chuẩn hóa trên văn bản (ô 3) — xem trước khi TTS.
    Không cần load model inference. Hiển thị toàn bộ text đã chuẩn hóa.
    """
    raw = (text or "").strip()
    if not raw:
        return "(Chưa có văn bản — nhập ô số 3 hoặc upload file .txt / .md)"

    input_mode = parse_input_mode(mode)
    pipeline = build_normalize_pipeline(pipeline_steps)
    label = format_normalize_pipeline(pipeline)
    normalized = normalize_full_document(raw, pipeline, input_mode)
    chunks = split_text_for_tts(normalized, max_chars=int(chunk_max_chars))

    mode_label = INPUT_MODE_CHOICES.get(input_mode, input_mode)
    lines = [
        f"Chế độ nhập: {mode_label}",
        f"Pipeline: {label}" + (
            " (bỏ qua khi TTS)" if input_mode == "prepared" else ""
        ),
        f"Gốc: {len(raw):,} ký tự → sau xử lý: {len(normalized):,} ký tự",
        f"Chunks TTS (max {int(chunk_max_chars)} ký tự/chunk): {len(chunks)}",
        "",
        "── Text đầy đủ sau chuẩn hóa ──",
    ]

    if max_preview_chars is not None and len(normalized) > max_preview_chars:
        lines.append(normalized[:max_preview_chars])
        lines.append(
            f"\n… (còn {len(normalized) - max_preview_chars:,} ký tự — "
            "bấm Xuất .txt hoặc Tổng hợp để xử lý toàn bộ)"
        )
    else:
        lines.append(normalized)

    if len(chunks) > 1:
        lines.extend(["", "── Xem trước từng chunk (5 chunk đầu) ──"])
        for i, ch in enumerate(chunks[:5]):
            lines.append(f"[{i + 1}/{len(chunks)}] {ch.text}")
        if len(chunks) > 5:
            lines.append(f"… và {len(chunks) - 5} chunk nữa")

    return "\n".join(lines)


def prepare_tts_text(text: str, backend: str | list[str] = "none") -> str:
    if isinstance(backend, list):
        normalized = (
            normalize_text_pipeline(text, backend) if backend else text
        )
    else:
        normalized = normalize_text(text, backend)
    return post_process_text(normalized, apply_lower=True)


def normalize_vietnamese(text: str) -> str:
    """Giữ tương thích code cũ — mặc định sea-g2p Normalizer."""
    return normalize_text(text, "sea_g2p")

