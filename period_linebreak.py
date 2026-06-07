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
