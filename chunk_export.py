"""
Shared per-chunk WAV export for Gradio debug and CLI script.

Writes chunk_NNN.wav + manifest.txt under a target directory using the same
normalize → split → ONNX synthesis path as production TTS.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from config import OUTPUT_DIR
from onnx_engine import OnnxTTSEngine
from audio.post_process import apply_chunk_wave_pauses
from audio.ref_audio import preprocess_ref_audio_text
from text.chunking import split_text_for_tts
from text.pipeline import (
    normalize_full_document,
    parse_input_mode,
    prepare_for_tts,
    prepare_tts_text,
)

CHUNK_EXPORT_DIR = OUTPUT_DIR / "chunk_test"
PREVIEW_LEN = 80
_WRITE_RETRIES = 3
_WRITE_RETRY_DELAY_S = 0.5
# Debug export: flag chunks that often correlate with int4 artifacts / voice drift.
WARN_MEL_FRAMES_THRESHOLD = 80
WARN_CHARS_THRESHOLD = 12


@dataclass
class ChunkExportConfig:
    norm_pipeline: list[str]
    input_mode: str
    chunk_max_chars: int
    chunk_min_chars: int
    pause_sentence: float
    pause_paragraph: float
    pause_chapter: float
    pause_enum_item: float
    pause_forced: float
    speed: float
    synth_num_step: int
    synth_guidance_scale: float
    synth_t_shift: float
    quant_mode: str
    use_gpu: bool = False
    resume: bool = False
    ode_seed: int = 42
    use_fixed_seed: bool = True
    same_seed_all_chunks: bool = False


@dataclass
class ChunkMeta:
    index: int
    chars: int
    mel_frames: int
    pause_after: float
    leading_pause: float
    text: str
    wav: str
    flags: str = ""

    @property
    def gen_chars(self) -> int:
        return self.chars


@dataclass
class ChunkExportResult:
    out_dir: Path
    manifest_path: Path
    saved_wavs: list[str]
    chunks_meta: list[ChunkMeta] = field(default_factory=list)
    elapsed: float = 0.0


def write_chunk_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write WAV with retries (Windows sometimes returns transient 'System error')."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + "._tmp.wav")
    last_err: Exception | None = None
    for attempt in range(1, _WRITE_RETRIES + 1):
        try:
            if tmp.exists():
                tmp.unlink()
            sf.write(str(tmp), audio, sample_rate, subtype="PCM_16")
            if path.exists():
                path.unlink()
            tmp.replace(path)
            return
        except OSError as exc:
            last_err = exc
            if attempt < _WRITE_RETRIES:
                time.sleep(_WRITE_RETRY_DELAY_S * attempt)
    raise RuntimeError(f"Error opening {path!s}: {last_err}") from last_err


def chunk_quality_flags(*, chars: int, mel_frames: int) -> str:
    """Comma-separated debug flags for manifest (warn only, never skip synthesis)."""
    flags: list[str] = []
    if chars < WARN_CHARS_THRESHOLD:
        flags.append("SHORT_CHARS")
    if 0 < mel_frames < WARN_MEL_FRAMES_THRESHOLD:
        flags.append("LOW_MEL")
    if mel_frames == 0:
        flags.append("NO_MEL")
    return ",".join(flags)


def write_chunk_manifest(
    path: Path,
    *,
    input_label: str,
    voice_id: str,
    ref_audio: str,
    ref_text: str,
    profile_label: str,
    quant: str,
    chunks_meta: list[ChunkMeta],
) -> None:
    lines = [
        f"input={input_label}",
        f"voice_id={voice_id}",
        f"ref_audio={ref_audio}",
        f"ref_transcript={ref_text[:120]}{'…' if len(ref_text) > 120 else ''}",
        f"profile={profile_label}",
        f"quant={quant}",
        f"chunk_count={len(chunks_meta)}",
        "",
        "index\tgen_chars\tmel_frames\tpause_after\tleading_pause\tflags\tpreview\twav",
    ]
    for m in chunks_meta:
        preview = m.text.replace("\t", " ").replace("\n", " ")
        if len(preview) > PREVIEW_LEN:
            preview = preview[:PREVIEW_LEN] + "…"
        lines.append(
            f"{m.index:03d}\t{m.gen_chars}\t{m.mel_frames}\t{m.pause_after:.2f}\t"
            f"{m.leading_pause:.2f}\t{m.flags}\t{preview}\t{m.wav}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_chunks_to_dir(
    *,
    gen_text: str,
    ref_audio_path: str,
    ref_text: str,
    out_dir: Path,
    config: ChunkExportConfig,
    voice_id: str = "",
    input_label: str = "",
    profile_label: str = "",
    progress: Callable[[float, str | None], None] | None = None,
    log: Callable[[str], None] | None = None,
) -> ChunkExportResult:
    """Synthesize each TTS chunk to a separate WAV under *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tts_input_mode = parse_input_mode(config.input_mode)

    normalized_doc = normalize_full_document(
        gen_text, config.norm_pipeline, tts_input_mode
    )
    tts_chunks = split_text_for_tts(
        normalized_doc,
        max_chars=int(config.chunk_max_chars),
        min_chars=int(config.chunk_min_chars),
        pause_sentence=float(config.pause_sentence),
        pause_paragraph=float(config.pause_paragraph),
        pause_chapter=float(config.pause_chapter),
        pause_enum_item=float(config.pause_enum_item),
        pause_forced_split=float(config.pause_forced),
    )
    if not tts_chunks:
        raise ValueError("Không có chunk sau chuẩn hóa / chia đoạn.")

    ref_audio, resolved_ref_text = preprocess_ref_audio_text(
        ref_audio_path,
        ref_text,
        show_info=log or (lambda _: None),
    )
    prompt_normalized = prepare_tts_text(resolved_ref_text, config.norm_pipeline)
    engine = OnnxTTSEngine.get(
        quant_mode=config.quant_mode, use_gpu=config.use_gpu
    )
    sample_rate = engine.sampling_rate

    chunks_meta: list[ChunkMeta] = []
    saved: list[str] = []
    total = len(tts_chunks)
    t0 = time.perf_counter()

    for i, tts_chunk in enumerate(tts_chunks):
        chunk_num = i + 1
        if progress:
            progress((i + 0.5) / total, f"Chunk {chunk_num}/{total}")

        normalized = prepare_for_tts(
            tts_chunk.text,
            config.norm_pipeline,
            tts_input_mode,
            already_normalized=True,
        )
        if not normalized.strip():
            if log:
                log(f"  [{chunk_num:03d}] SKIP — empty after normalize")
            continue

        wav_name = f"chunk_{chunk_num:03d}.wav"
        wav_path = out_dir / wav_name
        if (
            config.resume
            and wav_path.is_file()
            and wav_path.stat().st_size > 0
        ):
            if log:
                log(f"  [{chunk_num:03d}] EXISTS — {wav_path}")
            meta = ChunkMeta(
                index=chunk_num,
                chars=len(normalized),
                mel_frames=0,
                pause_after=tts_chunk.pause_after,
                leading_pause=tts_chunk.leading_pause,
                text=normalized,
                wav=wav_name,
                flags="RESUMED",
            )
            chunks_meta.append(meta)
            saved.append(wav_name)
            continue

        if log:
            log(
                f"  Synthesizing chunk {chunk_num}/{total} "
                f"({len(normalized)} chars)…"
            )

        chunk_seed_index = None if config.same_seed_all_chunks else i
        wav, mel_frames = engine.generate(
            prompt_text=prompt_normalized,
            prompt_wav=ref_audio,
            text=normalized,
            speed=float(config.speed),
            num_step=int(config.synth_num_step),
            guidance_scale=float(config.synth_guidance_scale),
            t_shift=float(config.synth_t_shift),
            ode_seed=int(config.ode_seed),
            use_fixed_seed=bool(config.use_fixed_seed),
            chunk_index=chunk_seed_index,
            return_mel_frames=True,
        )
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=0)
        wav = apply_chunk_wave_pauses(wav, tts_chunk, sample_rate)

        write_chunk_wav(wav_path, wav, sample_rate)
        flags = chunk_quality_flags(chars=len(normalized), mel_frames=mel_frames)
        meta = ChunkMeta(
            index=chunk_num,
            chars=len(normalized),
            mel_frames=mel_frames,
            pause_after=tts_chunk.pause_after,
            leading_pause=tts_chunk.leading_pause,
            text=normalized,
            wav=wav_name,
            flags=flags,
        )
        chunks_meta.append(meta)
        saved.append(wav_name)

        if log:
            flag_note = f" flags={flags}" if flags else ""
            log(
                f"  Saved: {wav_path} ({len(wav) / sample_rate:.2f}s, "
                f"mel={mel_frames}){flag_note}"
            )

    manifest_path = out_dir / "manifest.txt"
    write_chunk_manifest(
        manifest_path,
        input_label=input_label or "(gui)",
        voice_id=voice_id or "(upload)",
        ref_audio=ref_audio_path,
        ref_text=resolved_ref_text,
        profile_label=profile_label,
        quant=config.quant_mode,
        chunks_meta=chunks_meta,
    )

    elapsed = time.perf_counter() - t0
    return ChunkExportResult(
        out_dir=out_dir,
        manifest_path=manifest_path,
        saved_wavs=saved,
        chunks_meta=chunks_meta,
        elapsed=elapsed,
    )


def format_chunk_export_status(
    result: ChunkExportResult,
    *,
    quant_mode: str,
    use_gpu: bool,
    ode_seed: int = 42,
    use_fixed_seed: bool = True,
    same_seed_all_chunks: bool = False,
    max_listed: int = 12,
) -> str:
    from onnx_engine import format_ode_seed_log

    seed_line = format_ode_seed_log(
        ode_seed=ode_seed,
        use_fixed_seed=use_fixed_seed,
        same_seed_all_chunks=same_seed_all_chunks,
    )
    lines = [
        f"**Xong** — {len(result.saved_wavs)} WAV trong `{result.out_dir}` "
        f"({result.elapsed:.1f}s)",
        f"- Quant: `{quant_mode}` · GPU: `{use_gpu}` · {seed_line}",
        f"- Manifest: `{result.manifest_path}`",
    ]

    warn_rows: list[str] = []
    short_char_chunks = 0
    low_mel_chunks = 0
    for m in result.chunks_meta:
        if m.flags == "RESUMED":
            continue
        notes: list[str] = []
        if m.chars < WARN_CHARS_THRESHOLD:
            short_char_chunks += 1
            notes.append(f"chars={m.chars}<{WARN_CHARS_THRESHOLD}")
        if 0 < m.mel_frames < WARN_MEL_FRAMES_THRESHOLD:
            low_mel_chunks += 1
            notes.append(f"mel={m.mel_frames}<{WARN_MEL_FRAMES_THRESHOLD}")
        if notes:
            preview = m.text.replace("\n", " ")[:50]
            warn_rows.append(
                f"  - chunk {m.index:03d}: {', '.join(notes)} — «{preview}…»"
            )

    if warn_rows:
        lines.append("")
        lines.append("**⚠ Cảnh báo chất lượng (debug)**")
        lines.extend(warn_rows)
        if short_char_chunks and quant_mode == "int4":
            lines.append(
                f"  - {short_char_chunks} chunk ngắn (<{WARN_CHARS_THRESHOLD} ký tự) — "
                "chunk ngắn có thể cần gộp thêm (tăng min_chars hoặc kiểm tra chuẩn hóa)."
            )
        suspect = short_char_chunks + low_mel_chunks
        if suspect >= 2 and quant_mode == "int4":
            lines.append(
                "  - **Gợi ý A/B:** thử `int8` (dropdown ONNX quant) rồi export lại — "
                "int4 đôi khi lệch giọng trên một số cụm âm (số, enum, câu ngắn)."
            )

    for name in result.saved_wavs[:max_listed]:
        lines.append(f"- `{result.out_dir / name}`")
    if len(result.saved_wavs) > max_listed:
        lines.append(f"- … và {len(result.saved_wavs) - max_listed} file khác")
    return "\n".join(lines)
