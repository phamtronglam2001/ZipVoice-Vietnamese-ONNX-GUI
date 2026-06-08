"""
Per-stage inference timing and provider diagnostics.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from onnx_providers import session_active_provider


@dataclass
class StageTimings:
    """Wall-clock seconds per inference stage (one chunk or one batch)."""

    mel_extract: float = 0.0
    tokenize: float = 0.0
    text_encoder: float = 0.0
    fm_decoder: float = 0.0
    vocoder: float = 0.0
    istft: float = 0.0
    total: float = 0.0
    fm_decoder_steps: int = 0
    batch_size: int = 1

    def as_dict(self) -> dict[str, float | int]:
        return {
            "mel_extract_s": self.mel_extract,
            "tokenize_s": self.tokenize,
            "text_encoder_s": self.text_encoder,
            "fm_decoder_s": self.fm_decoder,
            "vocoder_s": self.vocoder,
            "istft_s": self.istft,
            "total_s": self.total,
            "fm_decoder_steps": self.fm_decoder_steps,
            "batch_size": self.batch_size,
        }


@dataclass
class InferenceProfileReport:
    """Aggregated profile for one or more chunks."""

    stages: list[StageTimings] = field(default_factory=list)
    providers: dict[str, str] = field(default_factory=dict)
    quant_mode: str = ""
    use_gpu: bool = False
    notes: list[str] = field(default_factory=list)

    def add(self, timing: StageTimings) -> None:
        self.stages.append(timing)

    @property
    def chunk_count(self) -> int:
        return len(self.stages)

    def totals(self) -> StageTimings:
        out = StageTimings()
        if not self.stages:
            return out
        for s in self.stages:
            out.mel_extract += s.mel_extract
            out.tokenize += s.tokenize
            out.text_encoder += s.text_encoder
            out.fm_decoder += s.fm_decoder
            out.vocoder += s.vocoder
            out.istft += s.istft
            out.total += s.total
            out.fm_decoder_steps += s.fm_decoder_steps
        out.batch_size = max(s.batch_size for s in self.stages)
        return out

    def format_summary(self) -> str:
        t = self.totals()
        lines = [
            "=== Inference profile ===",
            f"chunks/batches: {self.chunk_count} | quant={self.quant_mode} | gpu={self.use_gpu}",
        ]
        if self.providers:
            lines.append(
                "providers: "
                + ", ".join(f"{k}={v}" for k, v in sorted(self.providers.items()))
            )
        for note in self.notes:
            lines.append(f"note: {note}")
        if not self.stages:
            lines.append("(no timings recorded)")
            return "\n".join(lines)
        lines.extend(
            [
                f"total: {t.total:.3f}s",
                f"  mel_extract: {t.mel_extract:.3f}s",
                f"  tokenize:    {t.tokenize:.3f}s",
                f"  text_encoder:{t.text_encoder:.3f}s",
                f"  fm_decoder:  {t.fm_decoder:.3f}s ({t.fm_decoder_steps} step-runs)",
                f"  vocoder:     {t.vocoder:.3f}s",
                f"  istft:       {t.istft:.3f}s",
            ]
        )
        if t.fm_decoder > 0 and t.total > 0:
            pct = 100.0 * t.fm_decoder / t.total
            lines.append(f"  fm_decoder share: {pct:.1f}%")
        return "\n".join(lines)


class StageTimer:
    """Context manager accumulating into StageTimings fields."""

    def __init__(self, timings: StageTimings, field_name: str) -> None:
        self._timings = timings
        self._field = field_name
        self._t0 = 0.0

    def __enter__(self) -> StageTimer:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        elapsed = time.perf_counter() - self._t0
        current = getattr(self._timings, self._field)
        setattr(self._timings, self._field, current + elapsed)


def collect_provider_info(engine: Any) -> dict[str, str]:
    """Active ORT EP per model component."""
    return {
        "text_encoder": session_active_provider(engine.model.text_encoder),
        "fm_decoder": session_active_provider(engine.model.fm_decoder),
        "vocoder": session_active_provider(engine.vocoder),
    }


def analyze_providers(report: InferenceProfileReport) -> None:
    """Append notes when GPU requested but CPU fallback detected."""
    fm = report.providers.get("fm_decoder", "")
    if report.use_gpu and fm == "CPUExecutionProvider":
        report.notes.append(
            "fm_decoder on CPU — INT4/INT8 MatMulNBits may be CPU-only on this ORT build; "
            "cài onnxruntime-gpu mới hơn hoặc dùng CPU-only."
        )
