"""Step-by-step pipeline debug — no pydub dependency."""
from __future__ import annotations

import json
import re
from pathlib import Path

from text.normalizers import build_normalize_pipeline, normalize_text
from text.normalizers.period_linebreak import (
    join_soft_breaks,
    newline_sentence_boundary,
    prepare_tts_structure,
)
from text.normalizers.vieneu_text import clean_text_noise
from text.pipeline import post_process_text

RAW = """1. Top các mẫu viết đoạn văn cảm nhận về sự giàu đẹp của tiếng Việt chọn lọc hay nhất
Mẫu số 1 (Ngắn gọn 5–6 câu)"""

sach = json.loads(Path("profiles/sach.json").read_text(encoding="utf-8"))
pipeline = sach["pipeline"]

STEP_FNS = {
    "period_break": prepare_tts_structure,
    "newline_sentence": newline_sentence_boundary,
    "join_soft_breaks": join_soft_breaks,
    "vieneu": clean_text_noise,
}

print("Pipeline:", " -> ".join(pipeline))
print("=" * 60)
print("RAW repr:", repr(RAW))
print("=" * 60)

result = RAW
for step in pipeline:
    fn = STEP_FNS.get(step)
    if fn is None:
        print(f"\n[{step}] SKIP (not installed in debug)")
        continue
    nxt = fn(result)
    has_nl = "\n" in nxt
    print(f"\n--- after {step} ---")
    print(f"has newline: {has_nl}")
    print(repr(nxt[:400]))
    result = nxt

# post_process simulation
text = " " + result + " "
lines = [" ".join(line.split()) for line in text.splitlines()]
cleaned = "\n".join(lines).lower()
print("\n--- after post_process_text (simulated) ---")
print(f"has newline: {'\\n' in repr(cleaned)}")
print(repr(cleaned[:400]))

# Test join_soft_breaks hypothesis on lowercase merged text
print("\n--- join_soft_breaks on lowercase without period ---")
low = "tốp các mẫu viết đoạn văn cảm nhận về sự giàu đẹp của tiếng việt chọn lọc hay nhất\nmẫu số một, ngắn gọn 5–6 câu"
out = join_soft_breaks(low)
print(repr(out))
