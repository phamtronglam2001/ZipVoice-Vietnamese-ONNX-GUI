"""In-memory status log buffer for Gradio synthesis progress."""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime

logger = logging.getLogger("zipvoice_gui")

LogNotifyFn = Callable[[], None]


class StatusLog:
    """Append-only log with timestamps for GUI status textbox."""

    def __init__(self, *, mirror_console: bool = True, on_notify: LogNotifyFn | None = None) -> None:
        self._lines: list[str] = []
        self._stage_starts: dict[str, float] = {}
        self._mirror_console = mirror_console
        self._on_notify = on_notify

    def clear(self) -> None:
        self._lines.clear()
        self._stage_starts.clear()

    @staticmethod
    def _ts() -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _append(self, line: str) -> None:
        self._lines.append(line)
        if self._mirror_console:
            logger.info("status_log | %s", line)
            print(f"status_log | {line}", flush=True)
        if self._on_notify is not None:
            self._on_notify()

    def info(self, msg: str) -> None:
        self._append(f"[{self._ts()}] {msg}")

    def warn(self, msg: str) -> None:
        self._append(f"[{self._ts()}] ⚠ {msg}")

    def error(self, msg: str) -> None:
        self._append(f"[{self._ts()}] ✗ {msg}")

    def blank(self) -> None:
        self._append("")

    def heading(self, title: str) -> None:
        self._append(f"[{self._ts()}] ── {title} ──")

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
