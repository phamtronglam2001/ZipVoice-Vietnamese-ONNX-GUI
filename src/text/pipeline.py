"""Text normalization orchestration and TTS text preparation."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal

from config import OUTPUT_DIR
from text.chunking import DEFAULT_CHUNK_MIN_CHARS, split_text_for_tts
from text.normalizers import (
    build_normalize_pipeline,
    format_normalize_pipeline,
    normalize_text,
    normalize_text_pipeline,
)

logger = logging.getLogger("zipvoice_gui")

NormalizeInputMode = Literal["raw", "prepared"]

INPUT_MODE_CHOICES: dict[str, str] = {
    "raw": "Văn bản gốc",
    "prepared": "Đã chuẩn hóa (bỏ qua pipeline)",
}

PREVIEW_MAX_CHARS = 20_000


def post_process_text(text: str, *, apply_lower: bool = True) -> str:
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
    if already_normalized or parse_input_mode(mode) == "prepared":
        return prepare_tts_text_passthrough(text)
    return prepare_tts_text(text, pipeline or [])


def normalize_full_document(
    text: str,
    pipeline: list[str] | str | None,
    mode: NormalizeInputMode = "raw",
) -> str:
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


def preview_normalize_output(
    text: str,
    pipeline_steps: list[str] | str | None,
    chunk_max_chars: int = 135,
    chunk_min_chars: int = DEFAULT_CHUNK_MIN_CHARS,
    mode: NormalizeInputMode = "raw",
    max_preview_chars: int | None = None,
) -> str:
    raw = (text or "").strip()
    if not raw:
        return "(Chưa có văn bản — nhập ô số 3 hoặc upload file .txt / .md)"

    input_mode = parse_input_mode(mode)
    pipeline = build_normalize_pipeline(pipeline_steps)
    label = format_normalize_pipeline(pipeline)
    normalized = normalize_full_document(raw, pipeline, input_mode)
    chunks = split_text_for_tts(
        normalized,
        max_chars=int(chunk_max_chars),
        min_chars=int(chunk_min_chars),
    )

    mode_label = INPUT_MODE_CHOICES.get(input_mode, input_mode)
    lines = [
        f"Chế độ nhập: {mode_label}",
        f"Pipeline: {label}" + (
            " (bỏ qua khi TTS)" if input_mode == "prepared" else ""
        ),
        f"Gốc: {len(raw):,} ký tự → sau xử lý: {len(normalized):,} ký tự",
        (
            f"Chunks TTS (min {int(chunk_min_chars)}, max {int(chunk_max_chars)} "
            f"ký tự/chunk): {len(chunks)}"
        ),
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
        normalized = normalize_text_pipeline(text, backend) if backend else text
    else:
        normalized = normalize_text(text, backend)
    return post_process_text(normalized, apply_lower=True)


def format_tts_timing_line(
    elapsed_sec: float,
    num_chunks: int,
    quant_mode: str,
    *,
    chunks_synthesized: int | None = None,
    parallel_workers: int | None = None,
) -> str:
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
