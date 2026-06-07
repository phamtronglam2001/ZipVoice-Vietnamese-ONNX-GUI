"""
VieNeu-TTS text hygiene — punctuation / spacing cleanup on raw text.

Ported from VieNeu-TTS `vieneu_utils/core_utils.py` (`_clean_phoneme_noise`).
Applies to plain Vietnamese text before Espeak (not G2P, not NSW expansion).
"""
from __future__ import annotations

import re
from typing import List, Tuple

_NOISE_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"([.!?])[.,;:]+"), r"\1"),
    (re.compile(r"[.,;:]+([.!?])"), r"\1"),
    (re.compile(r"\s+[,;]\s+"), " "),
    (re.compile(r" {2,}"), " "),
]
_MULTI_PUNCT = re.compile(r"([.!?])\s*[.!?]+")


def _pick_strongest(m: re.Match) -> str:
    s = m.group(0)
    if "!" in s:
        return "!"
    if "?" in s:
        return "?"
    return "."


def clean_text_noise(text: str) -> str:
    """Dọn lỗi dấu câu lặp / khoảng trắng thừa (VieNeu v2 noise rules)."""
    for pattern, repl in _NOISE_RULES:
        text = pattern.sub(repl, text)
    return _MULTI_PUNCT.sub(_pick_strongest, text).strip()
