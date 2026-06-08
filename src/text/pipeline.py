"""Text normalization orchestration and TTS text preparation."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal

from config import OUTPUT_DIR
from text.chunking import (
    DEFAULT_CHUNK_MIN_CHARS,
    PAUSE_CHAPTER_DEFAULT,
    PAUSE_ENUM_DEFAULT,
    PAUSE_FORCED_SPLIT_DEFAULT,
    PAUSE_PARAGRAPH_DEFAULT,
    PAUSE_SENTENCE_DEFAULT,
    TtsChunk,
    format_chunks_preview,
    split_text_for_tts,
)
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
    normalized = normalize_text_pipeline(raw, steps)
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


def split_tts_chunks_for_preview(
    text: str,
    pipeline_steps: list[str] | str | None,
    *,
    chunk_max_chars: int = 135,
    chunk_min_chars: int = DEFAULT_CHUNK_MIN_CHARS,
    mode: NormalizeInputMode = "raw",
    pause_sentence: float = PAUSE_SENTENCE_DEFAULT,
    pause_paragraph: float = PAUSE_PARAGRAPH_DEFAULT,
    pause_chapter: float = PAUSE_CHAPTER_DEFAULT,
    pause_enum_item: float = PAUSE_ENUM_DEFAULT,
    pause_forced_split: float = PAUSE_FORCED_SPLIT_DEFAULT,
    merge_log: list[str] | None = None,
) -> tuple[str, list[TtsChunk]]:
    """Normalize document and split into final TTS chunks (post micro-merge)."""
    input_mode = parse_input_mode(mode)
    pipeline = build_normalize_pipeline(pipeline_steps)
    normalized = normalize_full_document(text, pipeline, input_mode)
    chunks = split_text_for_tts(
        normalized,
        max_chars=int(chunk_max_chars),
        min_chars=int(chunk_min_chars),
        pause_sentence=float(pause_sentence),
        pause_paragraph=float(pause_paragraph),
        pause_chapter=float(pause_chapter),
        pause_enum_item=float(pause_enum_item),
        pause_forced_split=float(pause_forced_split),
        merge_log=merge_log,
    )
    return normalized, chunks


def preview_chunks_output(
    text: str,
    pipeline_steps: list[str] | str | None,
    *,
    chunk_max_chars: int = 135,
    chunk_min_chars: int = DEFAULT_CHUNK_MIN_CHARS,
    mode: NormalizeInputMode = "raw",
    pause_sentence: float = PAUSE_SENTENCE_DEFAULT,
    pause_paragraph: float = PAUSE_PARAGRAPH_DEFAULT,
    pause_chapter: float = PAUSE_CHAPTER_DEFAULT,
    pause_enum_item: float = PAUSE_ENUM_DEFAULT,
    pause_forced_split: float = PAUSE_FORCED_SPLIT_DEFAULT,
    show_micro_merge: bool = True,
) -> str:
    raw = (text or "").strip()
    if not raw:
        return "(Chưa có văn bản — nhập ô số 3 hoặc upload file .txt / .md)"

    input_mode = parse_input_mode(mode)
    pipeline = build_normalize_pipeline(pipeline_steps)
    label = format_normalize_pipeline(pipeline)
    merge_log: list[str] = []
    normalized, chunks = split_tts_chunks_for_preview(
        raw,
        pipeline,
        chunk_max_chars=int(chunk_max_chars),
        chunk_min_chars=int(chunk_min_chars),
        mode=input_mode,
        pause_sentence=float(pause_sentence),
        pause_paragraph=float(pause_paragraph),
        pause_chapter=float(pause_chapter),
        pause_enum_item=float(pause_enum_item),
        pause_forced_split=float(pause_forced_split),
        merge_log=merge_log,
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
        (
            f"Nghỉ: câu={pause_sentence}s, đoạn={pause_paragraph}s, "
            f"chương={pause_chapter}s, enum={pause_enum_item}s, cắt={pause_forced_split}s"
        ),
        "",
        "── Chunk sẽ tổng hợp (sau gộp micro-chunk) ──",
        "",
        format_chunks_preview(chunks, show_micro_merge=show_micro_merge),
    ]
    if merge_log:
        lines.extend(["", "── Nhật ký gộp micro-chunk ──"])
        lines.extend(merge_log)
    return "\n".join(lines)


def preview_normalize_output(
    text: str,
    pipeline_steps: list[str] | str | None,
    chunk_max_chars: int = 135,
    chunk_min_chars: int = DEFAULT_CHUNK_MIN_CHARS,
    mode: NormalizeInputMode = "raw",
    max_preview_chars: int | None = None,
    pause_sentence: float = PAUSE_SENTENCE_DEFAULT,
    pause_paragraph: float = PAUSE_PARAGRAPH_DEFAULT,
    pause_chapter: float = PAUSE_CHAPTER_DEFAULT,
    pause_enum_item: float = PAUSE_ENUM_DEFAULT,
    pause_forced_split: float = PAUSE_FORCED_SPLIT_DEFAULT,
    include_chunk_preview: bool = True,
) -> str:
    raw = (text or "").strip()
    if not raw:
        return "(Chưa có văn bản — nhập ô số 3 hoặc upload file .txt / .md)"

    input_mode = parse_input_mode(mode)
    pipeline = build_normalize_pipeline(pipeline_steps)
    label = format_normalize_pipeline(pipeline)
    normalized, chunks = split_tts_chunks_for_preview(
        raw,
        pipeline,
        chunk_max_chars=int(chunk_max_chars),
        chunk_min_chars=int(chunk_min_chars),
        mode=input_mode,
        pause_sentence=float(pause_sentence),
        pause_paragraph=float(pause_paragraph),
        pause_chapter=float(pause_chapter),
        pause_enum_item=float(pause_enum_item),
        pause_forced_split=float(pause_forced_split),
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

    if include_chunk_preview and chunks:
        lines.extend([
            "",
            "── Chunk sẽ tổng hợp (5 chunk đầu, [NL]=xuống dòng) ──",
            "",
        ])
        preview_limit = 5
        lines.append(
            format_chunks_preview(chunks[:preview_limit], show_micro_merge=True)
        )
        if len(chunks) > preview_limit:
            lines.append(f"… và {len(chunks) - preview_limit} chunk nữa")

    return "\n".join(lines)


def prepare_tts_text(text: str, backend: str | list[str] = "none") -> str:
    if isinstance(backend, list):
        steps = backend
    else:
        steps = build_normalize_pipeline(backend)
    normalized = normalize_text_pipeline(text, steps)
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
