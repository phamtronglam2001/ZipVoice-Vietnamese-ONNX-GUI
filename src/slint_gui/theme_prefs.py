"""Persist Slint GUI theme preference (dark / light)."""
from __future__ import annotations

import json
from pathlib import Path

from config import ROOT

THEME_PREF_PATH = ROOT / "profiles" / "slint_ui.json"


def load_dark_mode(default: bool = False) -> bool:
    path = THEME_PREF_PATH
    if not path.is_file():
        return default
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    if isinstance(data, dict) and "dark_mode" in data:
        return bool(data["dark_mode"])
    return default


def save_dark_mode(dark_mode: bool) -> None:
    path = THEME_PREF_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"dark_mode": bool(dark_mode)}, indent=2),
        encoding="utf-8",
    )
