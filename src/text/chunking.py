"""Long-text chunking for ZipVoice TTS synthesis."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from text.normalizers.period_linebreak import is_enumeration_only_block

logger = logging.getLogger("zipvoice_gui")

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
    r"|chương\s+(?:thứ\s+)?"
    r"(?:nhất|một|hai|ba|tư|năm|sáu|bảy|tám|chín|mười(?:\s+một|\s+hai)?)"
    r"|lời\s+nói\s+đầu"
    r"|phụ\s+lục"
    r")",
    re.IGNORECASE,
)
RE_MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+")
RE_HR_LINE = re.compile(r"^[\-*_]{3,}\s*$")

MIN_TTS_CHUNK_CHARS = 12
DEFAULT_CHUNK_MIN_CHARS = 70

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
    leading_pause: float = 0.0
    merged_boundary_pause: float | None = None
    merged_prefix_len: int = 0
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
    return (
        prev.pause_after >= 1.0
        or prev.is_paragraph_end
        or prev.is_chapter_break
    )


def _is_too_short_for_synthesis(
    text: str,
    min_chars: int = DEFAULT_CHUNK_MIN_CHARS,
) -> bool:
    t = text.strip()
    if not t:
        return True
    floor = max(int(min_chars), MIN_TTS_CHUNK_CHARS + 1)
    return len(t) < floor


def _preview_chunk_text(text: str, limit: int = 60) -> str:
    t = text.strip().replace("\n", "\\n")
    return t if len(t) <= limit else f"{t[: limit - 1]}…"


def _join_merged_chunk_parts(left: str, right: str) -> str:
    left = left.rstrip()
    right = right.lstrip()
    if not left:
        return right
    if not right:
        return left
    return f"{left}\n{right}"


def _merge_tiny_chunks(
    chunks: list[TtsChunk],
    min_chars: int = DEFAULT_CHUNK_MIN_CHARS,
    merge_log: list[str] | None = None,
) -> list[TtsChunk]:
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
            and _is_too_short_for_synthesis(text, min_chars)
            and not ch.is_chapter_break
            and not _should_not_merge_into(merged[-1])
        ):
            prev = merged[-1]
            new_text = _join_merged_chunk_parts(prev.text, ch.text)
            merged[-1] = TtsChunk(
                text=new_text,
                pause_after=ch.pause_after,
                leading_pause=prev.leading_pause,
                is_sentence_end=ch.is_sentence_end,
                is_paragraph_end=ch.is_paragraph_end,
                is_chapter_break=ch.is_chapter_break,
                is_forced_split=prev.is_forced_split,
            )
            _log(
                "Gộp micro-chunk "
                f"({len(text)} ký tự) «{_preview_chunk_text(text, 40)}» "
                f"vào chunk trước (\\n) → «{_preview_chunk_text(new_text)}»"
            )
        elif (
            merged
            and _is_too_short_for_synthesis(text, min_chars)
            and not ch.is_chapter_break
            and _should_not_merge_into(merged[-1])
        ):
            _log(
                "Giữ micro-chunk "
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
        if _is_too_short_for_synthesis(text, min_chars):
            if ch.is_chapter_break:
                i += 1
                continue
            nxt = merged[i + 1]
            new_text = _join_merged_chunk_parts(ch.text, nxt.text)
            merged[i + 1] = TtsChunk(
                text=new_text,
                pause_after=nxt.pause_after,
                leading_pause=nxt.leading_pause,
                is_sentence_end=nxt.is_sentence_end,
                is_paragraph_end=nxt.is_paragraph_end,
                is_chapter_break=nxt.is_chapter_break,
                is_forced_split=ch.is_forced_split or nxt.is_forced_split,
            )
            _log(
                "Gộp micro-chunk "
                f"({len(text)} ký tự) «{_preview_chunk_text(text, 40)}» "
                f"vào chunk sau (\\n) → «{_preview_chunk_text(new_text)}»"
            )
            merged.pop(i)
            continue
        i += 1

    if len(merged) >= 2:
        last = merged[-1]
        last_text = last.text.strip()
        if _is_too_short_for_synthesis(last_text, min_chars) and not last.is_chapter_break:
            prev = merged[-2]
            if prev.is_chapter_break:
                return merged
            new_text = _join_merged_chunk_parts(prev.text, last.text)
            merged[-2] = TtsChunk(
                text=new_text,
                pause_after=last.pause_after,
                leading_pause=last.leading_pause,
                is_sentence_end=last.is_sentence_end,
                is_paragraph_end=last.is_paragraph_end,
                is_chapter_break=last.is_chapter_break,
                is_forced_split=prev.is_forced_split or last.is_forced_split,
            )
            _log(
                "Gộp micro-chunk cuối "
                f"({len(last_text)} ký tự) «{_preview_chunk_text(last_text, 40)}» "
                f"vào chunk trước (\\n) → «{_preview_chunk_text(new_text)}»"
            )
            merged.pop()

    return merged


def split_text_for_tts(
    text: str,
    max_chars: int = 135,
    min_chars: int = DEFAULT_CHUNK_MIN_CHARS,
    pause_sentence: float = PAUSE_SENTENCE_DEFAULT,
    pause_paragraph: float = PAUSE_PARAGRAPH_DEFAULT,
    pause_chapter: float = PAUSE_CHAPTER_DEFAULT,
    pause_enum_item: float = PAUSE_ENUM_DEFAULT,
    pause_forced_split: float = PAUSE_FORCED_SPLIT_DEFAULT,
    merge_log: list[str] | None = None,
) -> list[TtsChunk]:
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
    return _merge_tiny_chunks(chunks, min_chars=int(min_chars), merge_log=merge_log)


def chunk_text(text: str, max_chars: int = 135) -> list[str]:
    return [c.text for c in split_text_for_tts(text, max_chars=max_chars)]


CHUNK_PREVIEW_NL_TAG = "[NL]"


def _chunk_preview_flags(ch: TtsChunk) -> list[str]:
    flags: list[str] = []
    if ch.is_chapter_break:
        flags.append("chapter")
    if ch.is_paragraph_end:
        flags.append("paragraph")
    if ch.is_forced_split:
        flags.append("forced split")
    if ch.is_sentence_end and not ch.is_paragraph_end and not ch.is_chapter_break:
        if ch.pause_after > 0 and ch.pause_after < PAUSE_PARAGRAPH_DEFAULT:
            flags.append("sentence")
    return flags


def _chunk_preview_visible_text(text: str, *, newline_tag: str = CHUNK_PREVIEW_NL_TAG) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", newline_tag)


def format_chunks_preview(
    tts_chunks: list[TtsChunk],
    *,
    show_micro_merge: bool = True,
    newline_tag: str = CHUNK_PREVIEW_NL_TAG,
) -> str:
    """Human-readable preview of final TTS chunks (post micro-merge)."""
    if not tts_chunks:
        return "(Không có chunk — văn bản trống)"

    total = len(tts_chunks)
    lines = [f"Tổng: {total} chunk sẽ tổng hợp", ""]
    for i, ch in enumerate(tts_chunks):
        flags = _chunk_preview_flags(ch)
        meta = (
            f"Chunk {i + 1}/{total} ({len(ch.text)} chars, "
            f"pause_after={ch.pause_after:.2f}s"
        )
        if ch.leading_pause > 0:
            meta += f", leading_pause={ch.leading_pause:.2f}s"
        if flags:
            meta += f", {', '.join(flags)}"
        meta += ")"
        if show_micro_merge and "\n" in ch.text:
            meta += " [micro-merged]"
        lines.append(meta)
        visible = _chunk_preview_visible_text(ch.text, newline_tag=newline_tag)
        lines.append(f"  {visible}")
        if i < total - 1:
            lines.append("")
    return "\n".join(lines)
