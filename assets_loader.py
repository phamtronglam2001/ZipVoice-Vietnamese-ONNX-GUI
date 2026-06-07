"""
Load reference voices from assets/ref_info.json.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from config import ASSETS_DIR, REF_INFO_JSON, ROOT

logger = logging.getLogger("zipvoice_gui")

MANUAL_CHOICE = "— Upload thủ công —"


@dataclass(frozen=True)
class RefVoice:
    id: str
    label: str
    audio_path: str
    transcript: str
    folder: str


def _resolve_audio_path(raw_path: str) -> Path | None:
    p = Path(raw_path)
    if p.is_absolute() and p.is_file():
        return p

    candidates = [
        ASSETS_DIR / raw_path,
        ROOT / raw_path,
        ASSETS_DIR / "ref_audio" / Path(raw_path).name,
        ROOT / "data" / "ref_audio" / Path(raw_path).name,
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return None


def _load_ref_info_json(path: Path | None = None) -> dict:
    info_path = path or REF_INFO_JSON
    if not info_path.is_file():
        return {}
    try:
        data = json.loads(info_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.error("Cannot read %s: %s", info_path, exc)
        return {}


def scan_ref_voices(assets_dir: Path | None = None) -> list[RefVoice]:
    """Build voice list from assets/ref_info.json."""
    _ = assets_dir  # kept for API compatibility
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    voices: list[RefVoice] = []
    ref_data = _load_ref_info_json()

    if not ref_data:
        logger.warning("No entries in %s", REF_INFO_JSON)
        return voices

    for voice_id, entry in ref_data.items():
        if not isinstance(entry, dict):
            continue

        name = str(entry.get("name") or voice_id).strip()
        transcript = str(entry.get("text") or entry.get("transcript") or "").strip()
        raw_audio = str(entry.get("audio_path") or entry.get("audio") or "").strip()

        if not raw_audio:
            logger.warning("ref_info.json[%s]: missing audio_path", voice_id)
            continue

        audio = _resolve_audio_path(raw_audio)
        if audio is None:
            logger.warning(
                "ref_info.json[%s]: audio not found: %s", voice_id, raw_audio
            )
            continue

        label = name
        if transcript:
            preview = transcript[:40] + ("…" if len(transcript) > 40 else "")
            label = f"{name} — {preview}"

        voices.append(
            RefVoice(
                id=voice_id,
                label=label,
                audio_path=str(audio),
                transcript=transcript,
                folder=str(audio.parent.relative_to(ROOT)).replace("\\", "/")
                if audio.is_relative_to(ROOT)
                else str(audio.parent),
            )
        )

    voices.sort(key=lambda v: v.label.lower())
    logger.info("Loaded %d voice(s) from %s", len(voices), REF_INFO_JSON.name)
    return voices


def dropdown_choices(voices: list[RefVoice] | None = None) -> list[tuple[str, str]]:
    voices = voices if voices is not None else scan_ref_voices()
    choices: list[tuple[str, str]] = [(MANUAL_CHOICE, MANUAL_CHOICE)]
    for v in voices:
        choices.append((v.label, v.id))
    return choices


def get_voice_by_id(voice_id: str, voices: list[RefVoice] | None = None) -> RefVoice | None:
    if not voice_id or voice_id == MANUAL_CHOICE:
        return None
    voices = voices if voices is not None else scan_ref_voices()
    for v in voices:
        if v.id == voice_id:
            return v
    return None
