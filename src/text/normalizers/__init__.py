"""Normalizer registry and pipeline step helpers."""
from __future__ import annotations

import logging
import re
from collections.abc import Callable

from text.normalizers.dot_newline import dot_space_to_newline
from text.normalizers.period_linebreak import (
    join_soft_breaks,
    newline_sentence_boundary,
    prepare_tts_structure,
)
from text.normalizers.vieneu_text import clean_text_noise

logger = logging.getLogger("zipvoice_gui")

NormalizerFn = Callable[[str], str]

NORMALIZE_BACKENDS: dict[str, str] = {
    "none": "Không (chỉ dọn dấu câu)",
    "vieneu": "VieNeu (dọn punctuation/noise)",
    "join_soft_breaks": "Gộp xuống dòng PDF (dòng ngắn, viết thường)",
    "newline_sentence": "Xuống dòng → ranh giới câu (thêm chấm)",
    "period_break": "Cấu trúc TTS (ngoặc→xuống dòng, số+chấm→xuống dòng)",
    "dot_newline": "Newline (chấm + space → xuống dòng)",
    "sea_g2p": "sea-g2p Normalizer",
}

NORMALIZE_STEP_CHOICES: dict[str, str] = {
    "none": "— Không —",
    "vieneu": NORMALIZE_BACKENDS["vieneu"],
    "join_soft_breaks": NORMALIZE_BACKENDS["join_soft_breaks"],
    "newline_sentence": NORMALIZE_BACKENDS["newline_sentence"],
    "period_break": NORMALIZE_BACKENDS["period_break"],
    "dot_newline": NORMALIZE_BACKENDS["dot_newline"],
    "sea_g2p": NORMALIZE_BACKENDS["sea_g2p"],
}

NORMALIZE_ADD_CHOICES: dict[str, str] = {
    k: v for k, v in NORMALIZE_STEP_CHOICES.items() if k != "none"
}

DEFAULT_NORMALIZE_PIPELINE: list[str] = []
AUDIOBOOK_PRESET_PIPELINE: list[str] = list(reversed(list(NORMALIZE_ADD_CHOICES)))

_sea_g2p_normalizer = None


def _strip_sea_g2p_en_tags(text: str) -> str:
    """sea-g2p bọc tiếng Anh bằng <en>; espeak ZipVoice không hiểu tag này."""
    return re.sub(r"</?en>", "", text)


def _normalize_per_line(text: str, normalize_fn: NormalizerFn) -> str:
    """Run NSW normalizer per line so user paragraph breaks (\\n) survive."""
    if not text or "\n" not in text:
        return normalize_fn(text)
    out: list[str] = []
    for line in text.split("\n"):
        out.append(normalize_fn(line) if line.strip() else "")
    return "\n".join(out)


def _normalize_sea_g2p(text: str) -> str:
    global _sea_g2p_normalizer
    if _sea_g2p_normalizer is None:
        from sea_g2p import Normalizer

        _sea_g2p_normalizer = Normalizer()

    def _run(line: str) -> str:
        out = _sea_g2p_normalizer.normalize(line)
        if isinstance(out, list):
            out = out[0] if out else line
        return _strip_sea_g2p_en_tags(str(out))

    return _normalize_per_line(text, _run)


# Plugin-style registry: add a module + entry here to expose a new normalizer step.
def _normalize_vieneu(text: str) -> str:
    """Per-line space collapse so paragraph breaks (\\n) survive."""
    return _normalize_per_line(text, clean_text_noise)


NORMALIZERS: dict[str, NormalizerFn] = {
    "vieneu": _normalize_vieneu,
    "dot_newline": dot_space_to_newline,
    "period_break": prepare_tts_structure,
    "newline_sentence": newline_sentence_boundary,
    "join_soft_breaks": join_soft_breaks,
    "sea_g2p": _normalize_sea_g2p,
}


def register_normalizer(key: str, fn: NormalizerFn, *, label: str) -> None:
    """Register a custom normalizer step at runtime."""
    key = key.strip().lower()
    NORMALIZERS[key] = fn
    NORMALIZE_BACKENDS[key] = label
    if key != "none":
        NORMALIZE_STEP_CHOICES[key] = label
        NORMALIZE_ADD_CHOICES[key] = label


def normalize_text(text: str, backend: str = "none") -> str:
    """Chuẩn hóa NSW (số, ngày, ký hiệu...) trước khi đưa vào EspeakTokenizer."""
    key = (backend or "none").strip().lower()
    if key in ("none", "off", "không"):
        return text
    fn = NORMALIZERS.get(key)
    if fn is None:
        logger.warning("Unknown normalize backend %r — skip", backend)
        return text
    try:
        return fn(text)
    except ImportError as exc:
        pkg = {"sea_g2p": "sea-g2p"}.get(key, key)
        raise ImportError(
            f"Chưa cài `{pkg}` (không có sẵn trong ZipVoice — package PyPI riêng). "
            f"Chạy: pip install {pkg}  hoặc dùng bước `VieNeu` / `Không`."
        ) from exc
    except Exception:
        logger.exception("normalize_text failed | backend=%s", key)
        return text


def build_normalize_pipeline(steps: list[str] | str | None) -> list[str]:
    """
    Validate ordered normalizer keys (any length).
    Each backend outputs text only (no G2P) — steps can be chained.
    Duplicate keys are skipped with a warning.
    """
    if steps is None:
        return []
    if isinstance(steps, str):
        key = (steps or "none").strip().lower()
        if key in ("none", "off", "không", ""):
            return []
        if key not in NORMALIZE_BACKENDS or key == "none":
            raise ValueError(f"Backend chuẩn hóa không hợp lệ: {steps!r}")
        return [key]

    pipeline: list[str] = []
    seen: set[str] = set()
    for raw in steps:
        key = (raw or "none").strip().lower()
        if key in ("none", "off", "không", ""):
            continue
        if key not in NORMALIZE_BACKENDS or key == "none":
            raise ValueError(f"Backend chuẩn hóa không hợp lệ: {raw!r}")
        if key in seen:
            logger.warning(
                "Trùng bước chuẩn hóa (bỏ qua): %s",
                NORMALIZE_BACKENDS.get(key, key),
            )
            continue
        seen.add(key)
        pipeline.append(key)
    return pipeline


def format_normalize_pipeline(backends: list[str]) -> str:
    if not backends:
        return NORMALIZE_BACKENDS["none"]
    return " → ".join(NORMALIZE_BACKENDS.get(b, b) for b in backends)


def format_normalize_pipeline_list(backends: list[str]) -> str:
    """Numbered markdown for the Gradio pipeline builder."""
    if not backends:
        return (
            "**Chưa có bước** — khi TTS chỉ áp dụng dọn dấu câu (post-process)."
        )
    lines = [f"**{i}.** {NORMALIZE_BACKENDS.get(k, k)}" for i, k in enumerate(backends, 1)]
    return f"{format_normalize_pipeline(backends)}\n\n" + "\n".join(lines)


def pipeline_selector_choices(steps: list[str]) -> list[tuple[str, str]]:
    return [
        (f"{i + 1}. {NORMALIZE_BACKENDS.get(k, k)}", str(i))
        for i, k in enumerate(steps)
    ]


def _parse_pipeline_index(index: int | str | None) -> int | None:
    if index is None or index == "":
        return None
    try:
        return int(index)
    except (TypeError, ValueError):
        return None


def pipeline_add_step(steps: list[str] | None, new_step: str) -> list[str]:
    result = list(steps or [])
    key = (new_step or "").strip().lower()
    if not key or key == "none":
        return result
    if key not in NORMALIZE_BACKENDS:
        raise ValueError(f"Backend chuẩn hóa không hợp lệ: {new_step!r}")
    if key in result:
        logger.warning(
            "Trùng bước — không thêm lại: %s",
            NORMALIZE_BACKENDS.get(key, key),
        )
        return result
    result.append(key)
    return result


def pipeline_remove_at(steps: list[str] | None, index: int | str | None) -> list[str]:
    result = list(steps or [])
    idx = _parse_pipeline_index(index)
    if idx is None or not (0 <= idx < len(result)):
        return result
    result.pop(idx)
    return result


def pipeline_move(
    steps: list[str] | None, index: int | str | None, direction: int
) -> list[str]:
    result = list(steps or [])
    idx = _parse_pipeline_index(index)
    if idx is None:
        return result
    new_idx = idx + direction
    if 0 <= idx < len(result) and 0 <= new_idx < len(result):
        result[idx], result[new_idx] = result[new_idx], result[idx]
    return result


def normalize_text_pipeline(text: str, backends: list[str]) -> str:
    """Apply pipeline steps in order: text₀ → step₁ → text₁ → …"""
    result = text
    for backend in backends:
        result = normalize_text(result, backend)
    return result


def normalize_vietnamese(text: str) -> str:
    """Giữ tương thích code cũ — mặc định sea-g2p Normalizer."""
    return normalize_text(text, "sea_g2p")
