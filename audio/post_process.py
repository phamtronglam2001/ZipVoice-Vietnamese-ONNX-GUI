"""Join synthesized TTS wave chunks with pauses and crossfade."""
from __future__ import annotations

import numpy as np

from text.chunking import CROSSFADE_FORCED_SPLIT_S, TtsChunk


def prepend_leading_pause(
    wave: np.ndarray,
    leading_pause: float,
    sample_rate: int,
) -> np.ndarray:
    gap = int(sample_rate * max(0.0, leading_pause))
    if gap <= 0 or wave.size == 0:
        return wave
    return np.concatenate([np.zeros(gap, dtype=np.float32), wave])


def apply_chunk_wave_pauses(
    wave: np.ndarray,
    chunk: TtsChunk,
    sample_rate: int,
) -> np.ndarray:
    return prepend_leading_pause(wave, chunk.leading_pause, sample_rate)


def _linear_crossfade(
    a: np.ndarray, b: np.ndarray, fade_samples: int
) -> np.ndarray:
    if fade_samples <= 0 or a.size == 0 or b.size == 0:
        return np.concatenate([a, b])
    fade_samples = min(fade_samples, a.size, b.size)
    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    blended = a[-fade_samples:] * fade_out + b[:fade_samples] * fade_in
    return np.concatenate([a[:-fade_samples], blended, b[fade_samples:]])


def join_tts_audio_chunks(
    wave_chunks: list[np.ndarray],
    tts_chunks: list[TtsChunk],
    sample_rate: int,
) -> np.ndarray:
    if not wave_chunks:
        return np.array([], dtype=np.float32)

    normalized = [
        w.astype(np.float32)
        if w is not None and getattr(w, "size", 0) > 0
        else np.array([], dtype=np.float32)
        for w in wave_chunks
    ]
    if len(normalized) == 1:
        if tts_chunks and len(tts_chunks) > 0:
            return apply_chunk_wave_pauses(
                normalized[0], tts_chunks[0], sample_rate
            )
        return normalized[0]

    chunk0 = tts_chunks[0] if tts_chunks else TtsChunk(text="")
    final = apply_chunk_wave_pauses(normalized[0], chunk0, sample_rate)
    for i in range(1, len(normalized)):
        pause_s = 0.0
        prev_meta: TtsChunk | None = None
        if i - 1 < len(tts_chunks):
            prev_meta = tts_chunks[i - 1]
            pause_s = prev_meta.pause_after
        chunk_meta = (
            tts_chunks[i] if i < len(tts_chunks) else TtsChunk(text="")
        )
        segment = apply_chunk_wave_pauses(
            normalized[i], chunk_meta, sample_rate
        )

        if (
            prev_meta
            and prev_meta.is_forced_split
            and final.size > 0
            and segment.size > 0
        ):
            fade_n = int(sample_rate * CROSSFADE_FORCED_SPLIT_S)
            final = _linear_crossfade(final, segment, fade_n)
            remaining = max(0.0, pause_s - CROSSFADE_FORCED_SPLIT_S)
            if remaining > 0:
                gap = int(sample_rate * remaining)
                final = np.concatenate(
                    [final, np.zeros(gap, dtype=np.float32)]
                )
        else:
            gap = int(sample_rate * max(0.0, pause_s))
            if gap > 0:
                final = np.concatenate(
                    [final, np.zeros(gap, dtype=np.float32), segment]
                )
            else:
                final = np.concatenate([final, segment])
    return final
