"""
VieNeu-TTS text hygiene — punctuation / spacing cleanup on raw text.

Ported from VieNeu-TTS `vieneu_utils/core_utils.py` (`_clean_phoneme_noise`).
Applies to plain Vietnamese text before Espeak (not G2P, not NSW expansion).
"""
from __future__ import annotations

import re
from typing import List, Tuple

_NOISE_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r" {2,}"), " "),
]


def clean_text_noise(text: str) -> str:
    """Dọn khoảng trắng thừa (VieNeu v2 noise rules, punctuation preserved)."""
    for pattern, repl in _NOISE_RULES:
        text = pattern.sub(repl, text)
    return text.strip()
