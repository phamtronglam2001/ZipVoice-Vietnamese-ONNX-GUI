"""
ZipVoice Vietnamese ONNX — CLI (profile-driven, no pipeline editing).
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from assets_loader import get_voice_by_id, scan_ref_voices
from config import OUTPUT_DIR, ensure_ffmpeg_on_path, models_ready, onnx_ready
from export_audio import save_output
from preset_io import (
    PRESETS_DIR,
    PresetConfig,
    load_preset,
    list_presets,
    preset_to_dict,
)
from runtime_log import setup_runtime_logging

ensure_ffmpeg_on_path()
logger = setup_runtime_logging(name="zipvoice_cli")

DEFAULT_PROFILE = "none"


def _ensure_runtime() -> None:
    if not models_ready():
        print("[ERROR] Runtime not ready. Run install_cpu.bat first.", file=sys.stderr)
        if not onnx_ready():
            print("[ERROR] Missing ONNX files in models/onnx/", file=sys.stderr)
        sys.exit(1)


def _resolve_ref_from_preset(preset: PresetConfig, voices: list) -> tuple[str, str]:
    if preset.voice_mode == "bundled" and preset.voice_id:
        voice = get_voice_by_id(preset.voice_id, voices)
        if voice is None:
            raise ValueError(f"Voice not found in assets: {preset.voice_id}")
        ref_path = voice.audio_path
        ref_text = voice.transcript or ""
        if not ref_text.strip():
            raise ValueError(
                f"Voice `{preset.voice_id}` has no transcript in ref_info.json"
            )
        return ref_path, ref_text

    ref_path = preset.ref_wav.strip()
    ref_text = preset.ref_text.strip()
    if not ref_path:
        raise ValueError("Preset uses manual voice but ref_wav is empty")
    if not Path(ref_path).is_file():
        raise ValueError(f"ref_wav not found: {ref_path}")
    if not ref_text:
        raise ValueError("Preset uses manual voice but ref_text is empty")
    return ref_path, ref_text


def _read_input_text(args: argparse.Namespace) -> str:
    if args.text:
        return args.text
    if args.file:
        p = Path(args.file)
        if not p.is_file():
            raise ValueError(f"Input file not found: {args.file}")
        return p.read_text(encoding="utf-8")
    raise ValueError("Provide --text or --file")


def _load_profile_arg(profile: str | None) -> PresetConfig:
    name = profile or DEFAULT_PROFILE
    return load_preset(name)


def cmd_list_voices(_args: argparse.Namespace) -> int:
    voices = scan_ref_voices()
    if not voices:
        print("No voices in assets/ref_info.json")
        return 0
    for v in voices:
        print(f"{v.id}\t{v.label}")
    return 0


def cmd_profile_list(_args: argparse.Namespace) -> int:
    paths = list_presets()
    if not paths:
        print(f"No profiles in {PRESETS_DIR}")
        return 0
    for p in paths:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            name = data.get("name") or p.stem
            desc = data.get("description") or ""
            pipeline = data.get("pipeline") or []
            print(f"{p.stem}\t{name}\t{','.join(pipeline)}\t{desc}")
        except Exception as exc:
            print(f"{p.stem}\t(error: {exc})")
    return 0


def cmd_profile_show(args: argparse.Namespace) -> int:
    preset = load_preset(args.name)
    print(json.dumps(preset_to_dict(preset), ensure_ascii=False, indent=2))
    return 0


def _resolve_input_mode(args: argparse.Namespace, preset: PresetConfig) -> str:
    if getattr(args, "skip_normalize", False):
        return "prepared"
    return preset.input_mode or "raw"


def cmd_preview(args: argparse.Namespace) -> int:
    from utils import export_normalized_text_file, preview_normalize_output

    preset = _load_profile_arg(args.profile)
    text = _read_input_text(args)
    mode = _resolve_input_mode(args, preset)
    out = preview_normalize_output(
        text,
        preset.pipeline,
        chunk_max_chars=preset.chunk_max_chars,
        mode=mode,
    )
    print(out)
    if args.output_normalized:
        path = export_normalized_text_file(
            text,
            preset.pipeline,
            mode=mode,
            output_path=args.output_normalized,
            source_hint=args.file or "text",
        )
        print(f"\nSaved normalized text: {path}")
    return 0


def cmd_synthesize(args: argparse.Namespace) -> int:
    import gc
    import numpy as np

    from config import is_force_cpu
    from onnx_engine import OnnxTTSEngine
    from utils import (
        compute_tts_checkpoint_key,
        export_normalized_text_file,
        join_tts_audio_chunks,
        load_tts_checkpoint_chunks,
        normalize_full_document,
        prepare_for_tts,
        prepare_tts_text,
        preprocess_ref_audio_text,
        preview_normalize_output,
        save_tts_checkpoint_chunk,
        split_text_for_tts,
    )

    preset = _load_profile_arg(args.profile)
    gen_text = _read_input_text(args)
    if not gen_text.strip():
        raise ValueError("Input text is empty")

    tts_input_mode = _resolve_input_mode(args, preset)
    if args.normalize_only:
        full = normalize_full_document(gen_text, preset.pipeline, tts_input_mode)
        if args.output_normalized:
            path = export_normalized_text_file(
                gen_text,
                preset.pipeline,
                mode=tts_input_mode,
                output_path=args.output_normalized,
                source_hint=args.file or "text",
            )
            print(f"Saved: {path} ({len(full):,} chars)")
        else:
            print(preview_normalize_output(
                gen_text,
                preset.pipeline,
                chunk_max_chars=preset.chunk_max_chars,
                mode=tts_input_mode,
            ))
        return 0

    _ensure_runtime()

    voices = scan_ref_voices()
    ref_path, ref_text = _resolve_ref_from_preset(preset, voices)
    voice_label = preset.voice_id or "manual"

    if tts_input_mode == "prepared" and preset.pipeline:
        logger.warning(
            "input_mode=prepared — skipping normalize pipeline: %s",
            preset.pipeline,
        )

    logger.info(
        "CLI synthesize | profile=%s | gen_len=%d | mode=%s | int8=%s",
        args.profile or DEFAULT_PROFILE,
        len(gen_text),
        tts_input_mode,
        preset.use_int8,
    )

    engine = OnnxTTSEngine.get(use_int8=preset.use_int8)
    tts_chunks = split_text_for_tts(
        gen_text,
        max_chars=preset.chunk_max_chars,
        pause_sentence=preset.pause_sentence,
        pause_paragraph=preset.pause_paragraph,
        pause_chapter=preset.pause_chapter,
        pause_enum_item=preset.pause_enum_item,
        pause_forced_split=preset.pause_forced_split,
    )

    checkpoint_key = compute_tts_checkpoint_key(
        gen_text,
        preset.pipeline,
        preset.chunk_max_chars,
        speed=preset.speed,
        pause_sentence=preset.pause_sentence,
        pause_paragraph=preset.pause_paragraph,
        pause_chapter=preset.pause_chapter,
        pause_forced=preset.pause_forced_split,
        asset_voice_id=preset.voice_id or "",
        input_mode=tts_input_mode,
    )
    manifest = {
        "checkpoint_key": checkpoint_key,
        "num_chunks": len(tts_chunks),
        "pipeline": preset.pipeline,
        "input_mode": tts_input_mode,
        "chunk_max_chars": preset.chunk_max_chars,
        "speed": preset.speed,
        "pause_sentence": preset.pause_sentence,
        "pause_paragraph": preset.pause_paragraph,
        "pause_chapter": preset.pause_chapter,
        "pause_forced": preset.pause_forced_split,
        "asset_voice_id": preset.voice_id or "",
    }
    cached_chunks, resume_from = load_tts_checkpoint_chunks(
        OUTPUT_DIR, checkpoint_key, len(tts_chunks)
    )
    wave_parts: list[np.ndarray] = [
        cached_chunks.get(i, np.array([], dtype=np.float32))
        for i in range(len(tts_chunks))
    ]

    ref_audio = ""
    resolved_ref_text = ""
    for i, tts_chunk in enumerate(tts_chunks):
        if i < resume_from and wave_parts[i].size > 0:
            continue
        print(f"[{i + 1}/{len(tts_chunks)}] synthesizing...", flush=True)

        if i == 0:
            ref_audio, resolved_ref_text = preprocess_ref_audio_text(
                ref_path, ref_text, show_info=logger.info
            )
        else:
            resolved_ref_text = ref_text.strip() or resolved_ref_text

        prompt_normalized = prepare_tts_text(resolved_ref_text, preset.pipeline)
        normalized = prepare_for_tts(
            tts_chunk.text, preset.pipeline, tts_input_mode
        )
        if not normalized.strip():
            wave_parts[i] = np.array([], dtype=np.float32)
            continue

        wav = engine.generate(
            prompt_text=prompt_normalized,
            prompt_wav=ref_audio,
            text=normalized,
            speed=preset.speed,
            num_step=preset.num_step,
            guidance_scale=preset.guidance_scale,
            t_shift=preset.t_shift,
        )
        wave_parts[i] = wav
        save_tts_checkpoint_chunk(
            OUTPUT_DIR, i, wav, engine.sampling_rate, manifest
        )
        del wav
        if is_force_cpu():
            gc.collect()

    final_wave = join_tts_audio_chunks(wave_parts, tts_chunks, engine.sampling_rate)
    if final_wave is None or final_wave.size == 0:
        raise ValueError("No audio generated")

    saved = save_output(
        final_wave,
        engine.sampling_rate,
        preset.export_format,
        voice_label=voice_label,
        text_preview=gen_text,
    )
    if args.output:
        import shutil

        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(saved, out_path)
        saved = out_path

    print(f"Saved: {saved}")
    if args.output_normalized:
        path = export_normalized_text_file(
            gen_text,
            preset.pipeline,
            mode=tts_input_mode,
            output_path=args.output_normalized,
            source_hint=args.file or "text",
        )
        print(f"Saved normalized text: {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli_tts",
        description="ZipVoice Vietnamese ONNX CLI — load full config from profiles/",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-voices", help="List bundled voices from assets/ref_info.json")

    profile_p = sub.add_parser("profile", help="Profile management")
    profile_sub = profile_p.add_subparsers(dest="profile_cmd", required=True)
    profile_sub.add_parser("list", help="List profiles/*.json")
    show_p = profile_sub.add_parser("show", help="Show profile JSON")
    show_p.add_argument("name", help="Profile name or path")

    preview_p = sub.add_parser("preview", help="Preview normalize pipeline on text")
    preview_p.add_argument(
        "--profile", "-p", default=DEFAULT_PROFILE, help="Profile name (default: none)"
    )
    preview_p.add_argument("--text", "-t", help="Input text")
    preview_p.add_argument("--file", "-f", help="Input .txt file")
    preview_p.add_argument(
        "--output-normalized",
        help="Save normalized text to this .txt path",
    )
    preview_p.add_argument(
        "--skip-normalize",
        action="store_true",
        help="Input is already normalized (skip pipeline)",
    )
    preview_p.add_argument(
        "--input-prepared",
        action="store_true",
        dest="skip_normalize",
        help="Alias for --skip-normalize",
    )

    synth_p = sub.add_parser("synthesize", help="Synthesize speech from profile")
    synth_p.add_argument(
        "--profile", "-p", default=DEFAULT_PROFILE, help="Profile name (default: none)"
    )
    synth_p.add_argument("--text", "-t", help="Input text")
    synth_p.add_argument("--file", "-f", help="Input .txt file")
    synth_p.add_argument(
        "--output", "-o", help="Output file path (only CLI flag that overrides preset)"
    )
    synth_p.add_argument(
        "--normalize-only",
        action="store_true",
        help="Run normalize pipeline only; print/save .txt, no TTS",
    )
    synth_p.add_argument(
        "--output-normalized",
        help="Save normalized text to this .txt path",
    )
    synth_p.add_argument(
        "--skip-normalize",
        action="store_true",
        help="Input is already normalized (skip pipeline)",
    )
    synth_p.add_argument(
        "--input-prepared",
        action="store_true",
        dest="skip_normalize",
        help="Alias for --skip-normalize",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "list-voices":
            return cmd_list_voices(args)
        if args.command == "profile":
            if args.profile_cmd == "list":
                return cmd_profile_list(args)
            if args.profile_cmd == "show":
                return cmd_profile_show(args)
        if args.command == "preview":
            return cmd_preview(args)
        if args.command == "synthesize":
            return cmd_synthesize(args)
        parser.error(f"Unknown command: {args.command}")
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.error("CLI failed:\n%s", traceback.format_exc())
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
