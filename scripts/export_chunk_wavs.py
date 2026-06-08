"""
Export each TTS chunk as a separate WAV for quality comparison.

Run from repo root:
  .venv\\Scripts\\python.exe scripts\\export_chunk_wavs.py --input assets\\example\\test.txt --voice kim_cuc

Uses the same normalize + split pipeline as production (profile sach by default).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from assets_loader import get_voice_by_id, scan_ref_voices
from chunk_export import CHUNK_EXPORT_DIR, ChunkExportConfig, export_chunks_to_dir
from config import ensure_ffmpeg_on_path, models_ready, onnx_ready, vocoder_onnx_ready
from onnx_quant import normalize_quant_mode
from preset_io import load_preset
from runtime_log import setup_runtime_logging
from text.io import read_text_file

logger = setup_runtime_logging(name="export_chunk_wavs")

VOICE_ALIASES: dict[str, str] = {
    "kim_cuc": "nsnd_kim_cuc",
    "kim-cuc": "nsnd_kim_cuc",
    "nsnd_kim_cuc": "nsnd_kim_cuc",
    "nsnd-kim-cuc": "nsnd_kim_cuc",
}

DEFAULT_INPUT = ROOT / "assets" / "example" / "test.txt"
DEFAULT_OUTPUT_DIR = CHUNK_EXPORT_DIR
DEFAULT_PROFILE = "sach"


def _resolve_voice(voice_arg: str) -> tuple[str, str, str]:
    """Return (voice_id, ref_audio_path, ref_transcript)."""
    key = (voice_arg or "").strip().lower().replace(" ", "_")
    voice_id = VOICE_ALIASES.get(key, voice_arg.strip())
    voices = scan_ref_voices()
    voice = get_voice_by_id(voice_id, voices)
    if voice is None:
        ids = ", ".join(v.id for v in voices)
        raise ValueError(f"Voice not found: {voice_arg!r}. Available: {ids}")
    if not voice.transcript.strip():
        raise ValueError(f"Voice {voice_id} has no transcript in ref_info.json")
    return voice.id, voice.audio_path, voice.transcript


def _resolve_input(path: str | None) -> Path:
    if not path:
        candidates = [
            DEFAULT_INPUT,
            Path(r"D:\CodeApp\ZipVoice-Vietnamese-GUI\assets\example\test.txt"),
        ]
        for c in candidates:
            if c.is_file():
                return c.resolve()
        raise FileNotFoundError(
            "test.txt not found. Pass --input or place file at assets/example/test.txt"
        )
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Input file not found: {p}")
    return p.resolve()


def _ensure_runtime() -> None:
    if not models_ready():
        raise RuntimeError("Runtime not ready. Run install_cpu.bat first.")
    if not onnx_ready():
        raise RuntimeError("Missing ONNX files in models/onnx/")
    if not vocoder_onnx_ready():
        from config import vocoder_deploy_instructions

        raise RuntimeError(
            "Missing ONNX vocoder in models/vocoder/ (mel_spec_24khz.onnx).\n"
            + vocoder_deploy_instructions()
        )


def main(argv: list[str] | None = None) -> int:
    ensure_ffmpeg_on_path()

    parser = argparse.ArgumentParser(
        description="Synthesize each TTS chunk separately for quality review."
    )
    parser.add_argument(
        "--input",
        "-i",
        help="Path to input .txt (default: assets/example/test.txt)",
    )
    parser.add_argument(
        "--voice",
        "-v",
        default="kim_cuc",
        help="Bundled voice id or alias (default: kim_cuc → nsnd_kim_cuc)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for chunk WAVs and manifest (default: output/chunk_test)",
    )
    parser.add_argument(
        "--quant",
        choices=("int8", "int4"),
        default="int8",
        help="ONNX quant mode (default: int8)",
    )
    parser.add_argument(
        "--profile",
        "-p",
        default=DEFAULT_PROFILE,
        help=f"Normalize/split profile (default: {DEFAULT_PROFILE})",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use ONNX Runtime GPU providers",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip chunks whose WAV already exists in output dir",
    )
    args = parser.parse_args(argv)

    input_path = _resolve_input(args.input)
    voice_id, ref_audio_path, ref_text = _resolve_voice(args.voice)
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (ROOT / out_dir).resolve()

    quant_mode = normalize_quant_mode(args.quant)
    preset = load_preset(args.profile)

    print(f"Input:      {input_path}")
    print(f"Voice:      {voice_id} (NSND Kim Cúc)")
    print(f"Ref audio:  {ref_audio_path}")
    print(f"Ref text:   {ref_text[:80]}{'…' if len(ref_text) > 80 else ''}")
    print(f"Profile:    {args.profile} | pipeline={preset.pipeline}")
    print(f"Quant:      {quant_mode}")
    print(f"Output dir: {out_dir}")

    raw_text = read_text_file(str(input_path))
    if not raw_text.strip():
        raise ValueError("Input text is empty")

    _ensure_runtime()

    config = ChunkExportConfig(
        norm_pipeline=preset.pipeline,
        input_mode=preset.input_mode,
        chunk_max_chars=preset.chunk_max_chars,
        chunk_min_chars=preset.chunk_min_chars,
        pause_sentence=preset.pause_sentence,
        pause_paragraph=preset.pause_paragraph,
        pause_chapter=preset.pause_chapter,
        pause_enum_item=preset.pause_enum_item,
        pause_forced=preset.pause_forced_split,
        speed=preset.speed,
        synth_num_step=int(preset.num_step),
        synth_guidance_scale=float(preset.guidance_scale),
        synth_t_shift=float(preset.t_shift),
        quant_mode=quant_mode,
        use_gpu=bool(args.gpu),
        resume=bool(args.resume),
    )

    result = export_chunks_to_dir(
        gen_text=raw_text,
        ref_audio_path=ref_audio_path,
        ref_text=ref_text,
        out_dir=out_dir,
        config=config,
        voice_id=voice_id,
        input_label=str(input_path),
        profile_label=args.profile,
        log=print,
    )

    logger.info(
        "export_chunk_wavs | input=%s | voice=%s | chunks=%d | quant=%s",
        input_path,
        voice_id,
        len(result.saved_wavs),
        quant_mode,
    )

    print()
    print(f"Done in {result.elapsed:.1f}s — {len(result.saved_wavs)} WAV(s)")
    print(f"Manifest: {result.manifest_path}")
    for name in result.saved_wavs:
        print(f"  {out_dir / name}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
