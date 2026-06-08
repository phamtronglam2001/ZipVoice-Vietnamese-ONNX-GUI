"""In-memory status log buffer for Gradio synthesis progress."""
from __future__ import annotations

import time
from datetime import datetime


class StatusLog:
    """Append-only log with timestamps for GUI status textbox."""

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._stage_starts: dict[str, float] = {}

    def clear(self) -> None:
        self._lines.clear()
        self._stage_starts.clear()

    @staticmethod
    def _ts() -> str:
        return datetime.now().strftime("%H:%M:%S")

    def info(self, msg: str) -> None:
        self._lines.append(f"[{self._ts()}] {msg}")

    def warn(self, msg: str) -> None:
        self._lines.append(f"[{self._ts()}] ⚠ {msg}")

    def error(self, msg: str) -> None:
        self._lines.append(f"[{self._ts()}] ✗ {msg}")

    def blank(self) -> None:
        self._lines.append("")

    def heading(self, title: str) -> None:
        self._lines.append(f"[{self._ts()}] ── {title} ──")

    def stage_begin(self, name: str) -> None:
        self._stage_starts[name] = time.perf_counter()
        self.info(f"{name}...")

    def stage_end(self, name: str, extra: str = "") -> None:
        t0 = self._stage_starts.pop(name, None)
        elapsed = ""
        if t0 is not None:
            elapsed = f" ({time.perf_counter() - t0:.2f}s)"
        suffix = f" — {extra}" if extra else ""
        self.info(f"{name} xong{elapsed}{suffix}")

    def text(self) -> str:
        return "\n".join(self._lines)
