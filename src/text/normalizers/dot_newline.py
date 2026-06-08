"""Normalizer step: period + space → newline (optional pipeline step)."""
from __future__ import annotations

import re

# Period not preceded by digit, followed by whitespace → newline.
# Skips decimals like "3.14" (no space) and "3. 14" (digit before dot).
_RE_DOT_SPACE = re.compile(r"(?<![0-9])\.\s+")


def dot_space_to_newline(text: str) -> str:
    """Convert each «. » (period + space) pair to a newline."""
    if not text:
        return text
    return _RE_DOT_SPACE.sub(".\n", text)
