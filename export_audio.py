"""
Save generated audio to output/ as WAV or MP3 (ffmpeg).
"""
from __future__ import annotations

import logging
import re
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from config import OUTPUT_DIR, ensure_ffmpeg_on_path

logger = logging.getLogger("zipvoice_gui")

SAMPLE_RATE = 24000
EXPORT_CHOICES = {
    "WAV 24kHz": ("wav", None),
    "MP3 32kbps 24kHz": ("mp3", "32k"),
    "MP3 128kbps 24kHz": ("mp3", "128k"),
}


def _slug(text: str, max_len: int = 40) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[\s_]+", "-", text).strip("-")
    return (text[:max_len] or "output").strip("-") or "output"


def _ffmpeg_exe() -> str:
    ensure_ffmpeg_on_path()
    return "ffmpeg"


def save_output(
    audio: np.ndarray,
    sample_rate: int,
    export_format: str,
    voice_label: str = "",
    text_preview: str = "",
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fmt, bitrate = EXPORT_CHOICES.get(export_format, ("wav", None))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _slug(text_preview or voice_label or "tts")
    base = OUTPUT_DIR / f"{ts}_{slug}"

    wav_path = base.with_suffix(".wav")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    sf.write(str(wav_path), audio, sample_rate, subtype="PCM_16")

    if fmt == "wav":
        logger.info("Saved %s", wav_path)
        return wav_path

    out_path = base.with_suffix(".mp3")
    cmd = [
        _ffmpeg_exe(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(wav_path),
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "1",
        "-b:a",
        bitrate or "128k",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    wav_path.unlink(missing_ok=True)
    logger.info("Saved %s", out_path)
    return out_path
