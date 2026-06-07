"""
Chuẩn hóa cấu trúc câu cho TTS:
- Ngoặc () [] {} → dấu phẩy (ngắt hơi): "mẫu (mẹ)" → "mẫu, mẹ"
- Số/chữ + chấm + space → xuống dòng: "một. đọc" → "một.\nđọc"
"""
from __future__ import annotations

import re

_VI_NUM_WORDS = (
    "một|hai|ba|bốn|tư|năm|sáu|bảy|tám|chín|"
    "mười|mười một|mười hai|mười ba|mười bốn|mười lăm|"
    "mười sáu|mười bảy|mười tám|mười chín|"
    "hai mươi|ba mươi|bốn mươi|năm mươi"
)

_RE_DIGIT_PERIOD = re.compile(r"(\d{1,4})\.(?!\d)\s+")
_RE_WORD_PERIOD = re.compile(
    rf"\b({_VI_NUM_WORDS})\s*\.\s+",
    re.IGNORECASE | re.UNICODE,
)

RE_ENUM_ONLY_LINE = re.compile(
    rf"^(?:\d{{1,4}}|{_VI_NUM_WORDS})\s*\.\s*$",
    re.IGNORECASE | re.UNICODE,
)

# (mẹ) [x] {y} — không hỗ trợ ngoặc lồng nhau
_RE_BRACKET_GROUP = re.compile(
    r"[\(\[\{]\s*([^\(\)\[\]\{\}]+?)\s*[\)\]\}]",
    re.UNICODE,
)


def parentheses_to_commas(text: str) -> str:
    """
    Thay nội dung trong ngoặc bằng cụm sau dấu phẩy.
    "mẫu (mẹ)" → "mẫu, mẹ"
    """
    if not text or not text.strip():
        return text

    def _repl(match: re.Match) -> str:
        inner = match.group(1).strip()
        return f", {inner}" if inner else ""

    out = _RE_BRACKET_GROUP.sub(_repl, text)
    out = re.sub(r",\s*,", ", ", out)
    out = re.sub(r"\s+,", ",", out)
    out = re.sub(r",\s+", ", ", out)
    return out.strip()


def insert_period_linebreaks(text: str) -> str:
    """Sau «số/chữ + . + space» chèn xuống dòng để TTS nghỉ giữa các mục."""
    if not text or not text.strip():
        return text
    out = _RE_DIGIT_PERIOD.sub(r"\1.\n", text)
    out = _RE_WORD_PERIOD.sub(lambda m: f"{m.group(1)}.\n", out)
    return out


def prepare_tts_structure(text: str) -> str:
    """Pipeline cấu trúc TTS: ngoặc → phẩy, rồi số+chấm → xuống dòng."""
    out = parentheses_to_commas(text)
    return insert_period_linebreaks(out)


def is_enumeration_only_block(block: str) -> bool:
    return bool(RE_ENUM_ONLY_LINE.match(block.strip()))


_TERMINAL_PUNCT = re.compile(r'[.!?…]["\'""»)\]]*\s*$')
_RE_LOWERCASE_START = re.compile(r"^[a-zà-ỹ0-9]", re.UNICODE)
_MAX_SOFT_JOIN_LINE = 120


def newline_sentence_boundary(text: str) -> str:
    """
    newline_sentence — mid-text line breaks without terminal punctuation become
    sentence/paragraph boundaries for TTS chunking.

    Example: "Chương 1\\nNội dung" → "Chương 1.\\nNội dung"
    (appends '.' so split_text_for_tts treats the break as a paragraph end)
    """
    if not text or not text.strip():
        return text
    lines = text.split("\n")
    out: list[str] = []
    for i, line in enumerate(lines):
        stripped = line.rstrip()
        if not stripped:
            out.append("")
            continue
        if i < len(lines) - 1 and not _TERMINAL_PUNCT.search(stripped):
            stripped = stripped.rstrip(".") + "."
        out.append(stripped)
    return "\n".join(out)


def _should_join_soft_break(prev: str, nxt: str) -> bool:
    """True when nxt looks like a PDF-wrapped continuation of prev."""
    if _TERMINAL_PUNCT.search(prev):
        return False
    if len(prev) > _MAX_SOFT_JOIN_LINE or len(nxt) > _MAX_SOFT_JOIN_LINE:
        return False
    if not _RE_LOWERCASE_START.match(nxt):
        return False
    return True


def join_soft_breaks(text: str) -> str:
    """
    join_soft_breaks — merge lines likely split by PDF extraction (opposite of newline_sentence).

    Heuristics: previous line has no terminal punctuation, next line is short-ish and
    starts lowercase/digit (mid-sentence continuation).

    Example: "câu bị ngắt giữa chừng\\nở đây tiếp" → "câu bị ngắt giữa chừng ở đây tiếp"
    """
    if not text or "\n" not in text:
        return text
    lines = text.split("\n")
    merged: list[str] = []
    buf = ""
    for line in lines:
        if not line.strip():
            if buf:
                merged.append(buf)
                buf = ""
            merged.append("")
            continue
        piece = line.strip()
        if not buf:
            buf = piece
            continue
        if _should_join_soft_break(buf, piece):
            buf = f"{buf} {piece}"
        else:
            merged.append(buf)
            buf = piece
    if buf:
        merged.append(buf)
    return "\n".join(merged)
