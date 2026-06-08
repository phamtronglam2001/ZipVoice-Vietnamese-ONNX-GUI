"""Step-by-step pipeline debug for newline bug."""
from __future__ import annotations

import json
from pathlib import Path

from text.normalizers import build_normalize_pipeline, normalize_text
from text.pipeline import post_process_text

RAW = """1. Top các mẫu viết đoạn văn cảm nhận về sự giàu đẹp của tiếng Việt chọn lọc hay nhất
Mẫu số 1 (Ngắn gọn 5–6 câu)"""

sach = json.loads(Path("profiles/sach.json").read_text(encoding="utf-8"))
pipeline = build_normalize_pipeline(sach["pipeline"])
print("Pipeline:", " -> ".join(pipeline))
print("=" * 60)
print("RAW repr:", repr(RAW))
print("=" * 60)

result = RAW
for step in pipeline:
    try:
        nxt = normalize_text(result, step)
    except Exception as e:
        print(f"\n[{step}] ERROR: {e}")
        break
    has_nl = "\\n" in repr(nxt)
    print(f"\n--- after {step} ---")
    print(f"has newline: {has_nl}")
    print(repr(nxt[:300]))
    result = nxt

print("\n" + "=" * 60)
final = post_process_text(result, apply_lower=True)
print("--- after post_process_text ---")
print(f"has newline: {'\\n' in repr(final)}")
print(repr(final[:400]))
