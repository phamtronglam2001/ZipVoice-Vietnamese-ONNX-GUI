"""Read uploaded or local text files for TTS input."""
from __future__ import annotations

from pathlib import Path


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
