"""
Load reference voices from assets/ref_info.json and paired audio+.txt folders.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from config import (
    ASSETS_DIR,
    REF_AUDIO_EXTENSIONS,
    REF_AUDIO_PAIR_DIRS,
    REF_INFO_JSON,
    ROOT,
)

logger = logging.getLogger("zipvoice_gui")

MANUAL_CHOICE = "— Upload thủ công —"


@dataclass(frozen=True)
class RefVoice:
    id: str
    label: str
    audio_path: str
    transcript: str
    folder: str
    source: str = "json"


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


def _voice_folder_label(audio: Path) -> str:
    if audio.is_relative_to(ROOT):
        return str(audio.parent.relative_to(ROOT)).replace("\\", "/")
    return str(audio.parent)


def _make_label(name: str, transcript: str) -> str:
    if not transcript:
        return name
    preview = transcript[:40] + ("…" if len(transcript) > 40 else "")
    return f"{name} — {preview}"


def _pair_voice_id(rel_dir: str, stem: str) -> str:
    folder_key = rel_dir.replace("\\", "/").replace("/", "__")
    return f"{folder_key}__{stem}"


def _read_pair_transcript(txt_path: Path) -> str:
    try:
        return txt_path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        logger.warning("Cannot read transcript %s: %s", txt_path, exc)
        return ""


def _scan_ref_info_voices() -> list[RefVoice]:
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

        voices.append(
            RefVoice(
                id=voice_id,
                label=_make_label(name, transcript),
                audio_path=str(audio),
                transcript=transcript,
                folder=_voice_folder_label(audio),
                source="json",
            )
        )

    return voices


def _scan_audio_txt_pairs(
    pair_dir: Path,
    assets_root: Path = ASSETS_DIR,
) -> list[RefVoice]:
    """Scan a folder for paired audio + .txt files sharing the same stem."""
    voices: list[RefVoice] = []
    if not pair_dir.is_dir():
        return voices

    try:
        rel_dir = pair_dir.relative_to(assets_root).as_posix()
    except ValueError:
        rel_dir = pair_dir.name

    for audio_path in sorted(pair_dir.iterdir()):
        if not audio_path.is_file():
            continue
        if audio_path.suffix.lower() not in REF_AUDIO_EXTENSIONS:
            continue

        txt_path = audio_path.with_suffix(".txt")
        if not txt_path.is_file():
            logger.warning(
                "%s: missing paired transcript %s", audio_path.name, txt_path.name
            )
            continue

        stem = audio_path.stem
        transcript = _read_pair_transcript(txt_path)
        voice_id = _pair_voice_id(rel_dir, stem)
        audio = audio_path.resolve()

        voices.append(
            RefVoice(
                id=voice_id,
                label=_make_label(stem, transcript),
                audio_path=str(audio),
                transcript=transcript,
                folder=_voice_folder_label(audio),
                source="pair",
            )
        )

    return voices


def scan_ref_voices(assets_dir: Path | None = None) -> list[RefVoice]:
    """Build voice list from ref_info.json and configured audio+.txt folders."""
    assets_root = assets_dir or ASSETS_DIR
    assets_root.mkdir(parents=True, exist_ok=True)

    voices: list[RefVoice] = []
    seen_audio: set[str] = set()

    for voice in _scan_ref_info_voices():
        if voice.audio_path in seen_audio:
            continue
        voices.append(voice)
        seen_audio.add(voice.audio_path)

    pair_count = 0
    for pair_dir in REF_AUDIO_PAIR_DIRS:
        resolved_pair_dir = pair_dir
        if assets_dir is not None:
            try:
                resolved_pair_dir = assets_root / pair_dir.relative_to(ASSETS_DIR)
            except ValueError:
                resolved_pair_dir = pair_dir
        for voice in _scan_audio_txt_pairs(resolved_pair_dir, assets_root=assets_root):
            if voice.audio_path in seen_audio:
                continue
            voices.append(voice)
            seen_audio.add(voice.audio_path)
            pair_count += 1

    voices.sort(key=lambda v: v.label.lower())
    json_count = sum(1 for v in voices if v.source == "json")
    logger.info(
        "Loaded %d voice(s): %d from %s, %d from audio+.txt folders",
        len(voices),
        json_count,
        REF_INFO_JSON.name,
        pair_count,
    )
    return voices


def format_voice_load_summary(voices: list[RefVoice] | None = None) -> str:
    """Human-readable counts for GUI status lines."""
    voices = voices if voices is not None else scan_ref_voices()
    if not voices:
        return (
            "Chưa có giọng hợp lệ. Thêm `assets/ref_info.json` "
            "hoặc cặp audio+.txt trong `assets/sample_audio/`."
        )

    json_count = sum(1 for v in voices if v.source == "json")
    pair_count = sum(1 for v in voices if v.source == "pair")
    parts: list[str] = [f"**{len(voices)}** giọng"]
    if json_count:
        parts.append(f"**{json_count}** từ `ref_info.json`")
    if pair_count:
        parts.append(f"**{pair_count}** từ `sample_audio/` (audio + .txt)")
    return "Đã load " + " · ".join(parts)


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
