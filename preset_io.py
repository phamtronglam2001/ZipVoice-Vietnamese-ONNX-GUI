"""
Unified preset/profile I/O for ZipVoice Vietnamese GUI and CLI.
Schema v1 — JSON files under profiles/
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from assets_loader import MANUAL_CHOICE
from config import ROOT
from export_audio import EXPORT_CHOICES
from utils import (
    AUDIOBOOK_PRESET_PIPELINE,
    DEFAULT_NORMALIZE_PIPELINE,
    PAUSE_CHAPTER_DEFAULT,
    PAUSE_ENUM_DEFAULT,
    PAUSE_FORCED_SPLIT_DEFAULT,
    PAUSE_PARAGRAPH_DEFAULT,
    PAUSE_SENTENCE_DEFAULT,
    build_normalize_pipeline,
    pipeline_selector_choices,
    format_normalize_pipeline_list,
)

PRESETS_DIR = ROOT / "profiles"
SCHEMA_VERSION = 1

VALID_PIPELINE_STEPS = frozenset(
    {
        "vieneu",
        "join_soft_breaks",
        "newline_sentence",
        "period_break",
        "sea_g2p",
    }
)


@dataclass
class PresetConfig:
    schema_version: int = SCHEMA_VERSION
    name: str = ""
    description: str = ""
    voice_mode: str = "manual"  # "bundled" | "manual"
    voice_id: str = ""
    ref_wav: str = ""
    ref_text: str = ""
    pipeline: list[str] = field(default_factory=lambda: list(DEFAULT_NORMALIZE_PIPELINE))
    chunk_max_chars: int = 135
    pause_sentence: float = PAUSE_SENTENCE_DEFAULT
    pause_paragraph: float = PAUSE_PARAGRAPH_DEFAULT
    pause_chapter: float = PAUSE_CHAPTER_DEFAULT
    pause_enum_item: float = PAUSE_ENUM_DEFAULT
    pause_forced_split: float = PAUSE_FORCED_SPLIT_DEFAULT
    onnx_quant_mode: str = "int8"
    num_step: int = 16
    speed: float = 1.0
    guidance_scale: float = 1.0
    t_shift: float = 0.5
    export_format: str = "WAV 24kHz"
    input_mode: str = "raw"  # "raw" | "prepared"


def _slug_name(name: str) -> str:
    s = name.strip()
    s = re.sub(r"[^\w\s\-]", "", s, flags=re.UNICODE)
    s = re.sub(r"[\s_]+", "_", s).strip("_")
    return s or "preset"


def _resolve_preset_path(path_or_name: str | Path) -> Path:
    p = Path(path_or_name)
    if p.is_file():
        return p.resolve()
    stem = Path(path_or_name).stem if str(path_or_name).endswith(".json") else str(path_or_name)
    candidate = PRESETS_DIR / f"{stem}.json"
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"Preset not found: {path_or_name}")


def validate_preset(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["Root must be a JSON object"]

    ver = data.get("schema_version")
    if ver is not None and ver != SCHEMA_VERSION:
        errors.append(f"Unsupported schema_version: {ver} (expected {SCHEMA_VERSION})")

    voice = data.get("voice") or {}
    voice_manual = data.get("voice_manual") or {}
    bundled_id = str(voice.get("voice_id") or "").strip()
    manual_wav = str(voice_manual.get("ref_wav") or "").strip()
    has_bundled = bool(bundled_id)
    has_manual = bool(manual_wav)
    if has_bundled and has_manual:
        errors.append("Only one voice mode allowed: bundled voice_id OR manual ref_wav")

    pipeline = data.get("pipeline")
    if pipeline is not None:
        if not isinstance(pipeline, list):
            errors.append("pipeline must be a list")
        else:
            for step in pipeline:
                if step not in VALID_PIPELINE_STEPS:
                    errors.append(f"Unknown pipeline step: {step}")

    chunk = data.get("chunk_max_chars")
    if chunk is not None and (not isinstance(chunk, (int, float)) or chunk < 1):
        errors.append("chunk_max_chars must be a positive number")

    pause = data.get("pause") or {}
    for key in ("sentence", "paragraph", "chapter", "enum_item", "forced_split"):
        val = pause.get(key)
        if val is not None and not isinstance(val, (int, float)):
            errors.append(f"pause.{key} must be a number")

    synth = data.get("synthesis") or {}
    for key in ("speed", "guidance_scale", "t_shift"):
        val = synth.get(key)
        if val is not None and not isinstance(val, (int, float)):
            errors.append(f"synthesis.{key} must be a number")
    if "num_step" in synth and not isinstance(synth["num_step"], int):
        errors.append("synthesis.num_step must be an integer")

    export = data.get("export") or {}
    fmt = export.get("format")
    if fmt is not None and fmt not in EXPORT_CHOICES:
        errors.append(f"export.format must be one of: {list(EXPORT_CHOICES.keys())}")

    input_mode = data.get("input_mode")
    if input_mode is not None and input_mode not in ("raw", "prepared"):
        errors.append("input_mode must be 'raw' or 'prepared'")

    return errors


def _resolve_preset_quant_mode(synth: dict[str, Any]) -> str:
    from onnx_quant import QUANT_MODE_CHOICES, normalize_quant_mode

    raw = synth.get("onnx_quant_mode")
    if raw and str(raw).strip().lower() in QUANT_MODE_CHOICES:
        return str(raw).strip().lower()
    return normalize_quant_mode(None, use_int8=bool(synth.get("use_int8", True)))


def _preset_from_dict(data: dict[str, Any]) -> PresetConfig:
    errors = validate_preset(data)
    if errors:
        raise ValueError("; ".join(errors))

    voice = data.get("voice") or {}
    voice_manual = data.get("voice_manual") or {}
    bundled_id = str(voice.get("voice_id") or "").strip()
    manual_wav = str(voice_manual.get("ref_wav") or "").strip()
    manual_text = str(voice_manual.get("ref_text") or "")

    if bundled_id:
        voice_mode = "bundled"
        voice_id = bundled_id
        ref_wav = ""
        ref_text = ""
    else:
        voice_mode = "manual"
        voice_id = ""
        ref_wav = manual_wav
        ref_text = manual_text

    pause = data.get("pause") or {}
    synth = data.get("synthesis") or {}
    export = data.get("export") or {}

    pipeline_raw = data.get("pipeline", DEFAULT_NORMALIZE_PIPELINE)
    pipeline = build_normalize_pipeline(pipeline_raw if isinstance(pipeline_raw, list) else [])

    return PresetConfig(
        schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
        name=str(data.get("name") or ""),
        description=str(data.get("description") or ""),
        voice_mode=voice_mode,
        voice_id=voice_id,
        ref_wav=ref_wav,
        ref_text=ref_text,
        pipeline=pipeline,
        chunk_max_chars=int(data.get("chunk_max_chars", 135)),
        pause_sentence=float(pause.get("sentence", PAUSE_SENTENCE_DEFAULT)),
        pause_paragraph=float(pause.get("paragraph", PAUSE_PARAGRAPH_DEFAULT)),
        pause_chapter=float(pause.get("chapter", PAUSE_CHAPTER_DEFAULT)),
        pause_enum_item=float(pause.get("enum_item", PAUSE_ENUM_DEFAULT)),
        pause_forced_split=float(pause.get("forced_split", PAUSE_FORCED_SPLIT_DEFAULT)),
        onnx_quant_mode=_resolve_preset_quant_mode(synth),
        num_step=int(synth.get("num_step", 16)),
        speed=float(synth.get("speed", 1.0)),
        guidance_scale=float(synth.get("guidance_scale", 1.0)),
        t_shift=float(synth.get("t_shift", 0.5)),
        export_format=str(export.get("format", "WAV 24kHz")),
        input_mode=str(data.get("input_mode", "raw")),
    )


def preset_to_dict(preset: PresetConfig) -> dict[str, Any]:
    if preset.voice_mode == "bundled":
        voice = {"mode": "bundled", "voice_id": preset.voice_id}
        voice_manual = {"mode": "manual", "ref_wav": "", "ref_text": ""}
    else:
        voice = {"mode": "bundled", "voice_id": ""}
        voice_manual = {
            "mode": "manual",
            "ref_wav": preset.ref_wav,
            "ref_text": preset.ref_text,
        }

    return {
        "schema_version": preset.schema_version,
        "name": preset.name,
        "description": preset.description,
        "voice": voice,
        "voice_manual": voice_manual,
        "pipeline": list(preset.pipeline),
        "chunk_max_chars": int(preset.chunk_max_chars),
        "pause": {
            "sentence": preset.pause_sentence,
            "paragraph": preset.pause_paragraph,
            "chapter": preset.pause_chapter,
            "enum_item": preset.pause_enum_item,
            "forced_split": preset.pause_forced_split,
        },
        "synthesis": {
            "onnx_quant_mode": preset.onnx_quant_mode,
            "use_int8": preset.onnx_quant_mode == "int8",
            "num_step": preset.num_step,
            "speed": preset.speed,
            "guidance_scale": preset.guidance_scale,
            "t_shift": preset.t_shift,
        },
        "export": {"format": preset.export_format},
        "input_mode": preset.input_mode,
    }


def load_preset(path_or_name: str | Path) -> PresetConfig:
    path = _resolve_preset_path(path_or_name)
    data = json.loads(path.read_text(encoding="utf-8"))
    preset = _preset_from_dict(data)
    if not preset.name:
        preset.name = path.stem
    return preset


def save_preset(path: str | Path, preset: PresetConfig) -> Path:
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    p = Path(path)
    if p.suffix != ".json":
        p = PRESETS_DIR / f"{_slug_name(str(path))}.json"
    elif not p.is_absolute():
        p = PRESETS_DIR / p.name
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = preset_to_dict(preset)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return p.resolve()


def list_presets() -> list[Path]:
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(PRESETS_DIR.glob("*.json"), key=lambda x: x.stem.lower())


def preset_dropdown_choices() -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for p in list_presets():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            label = str(data.get("name") or p.stem)
            desc = str(data.get("description") or "").strip()
            if desc:
                label = f"{label} — {desc[:40]}"
        except Exception:
            label = p.stem
        choices.append((label, p.stem))
    return choices


def preset_from_gui_state(
    *,
    voice_dropdown: str,
    ref_audio_path: str | None,
    ref_text: str,
    norm_pipeline: list[str] | None,
    chunk_max_chars: int,
    pause_sentence: float,
    pause_paragraph: float,
    pause_chapter: float,
    pause_enum_item: float,
    pause_forced: float,
    speed: float,
    export_format: str,
    onnx_quant_mode: str = "int8",
    num_step: int = 16,
    guidance_scale: float = 1.0,
    t_shift: float = 0.5,
    preset_name: str = "",
    description: str = "",
    input_mode: str = "raw",
) -> PresetConfig:
    """Alias for collect_gui_state."""
    return collect_gui_state(
        voice_dropdown=voice_dropdown,
        ref_audio_path=ref_audio_path,
        ref_text=ref_text,
        norm_pipeline=norm_pipeline,
        chunk_max_chars=chunk_max_chars,
        pause_sentence=pause_sentence,
        pause_paragraph=pause_paragraph,
        pause_chapter=pause_chapter,
        pause_enum_item=pause_enum_item,
        pause_forced=pause_forced,
        speed=speed,
        export_format=export_format,
        onnx_quant_mode=onnx_quant_mode,
        num_step=num_step,
        guidance_scale=guidance_scale,
        t_shift=t_shift,
        preset_name=preset_name,
        description=description,
        input_mode=input_mode,
    )


def collect_gui_state(
    *,
    voice_dropdown: str,
    ref_audio_path: str | None,
    ref_text: str,
    norm_pipeline: list[str] | None,
    chunk_max_chars: int,
    pause_sentence: float,
    pause_paragraph: float,
    pause_chapter: float,
    pause_enum_item: float,
    pause_forced: float,
    speed: float,
    export_format: str,
    onnx_quant_mode: str = "int8",
    num_step: int = 16,
    guidance_scale: float = 1.0,
    t_shift: float = 0.5,
    preset_name: str = "",
    description: str = "",
    input_mode: str = "raw",
) -> PresetConfig:
    pipeline = build_normalize_pipeline(norm_pipeline)
    is_manual = not voice_dropdown or voice_dropdown == MANUAL_CHOICE
    mode = str(input_mode or "raw")

    if is_manual:
        return PresetConfig(
            name=preset_name,
            description=description,
            voice_mode="manual",
            voice_id="",
            ref_wav=str(ref_audio_path or ""),
            ref_text=ref_text or "",
            pipeline=pipeline,
            chunk_max_chars=int(chunk_max_chars),
            pause_sentence=float(pause_sentence),
            pause_paragraph=float(pause_paragraph),
            pause_chapter=float(pause_chapter),
            pause_enum_item=float(pause_enum_item),
            pause_forced_split=float(pause_forced),
            onnx_quant_mode=str(onnx_quant_mode),
            num_step=int(num_step),
            speed=float(speed),
            guidance_scale=float(guidance_scale),
            t_shift=float(t_shift),
            export_format=export_format,
            input_mode=mode,
        )

    return PresetConfig(
        name=preset_name,
        description=description,
        voice_mode="bundled",
        voice_id=voice_dropdown,
        ref_wav="",
        ref_text="",
        pipeline=pipeline,
        chunk_max_chars=int(chunk_max_chars),
        pause_sentence=float(pause_sentence),
        pause_paragraph=float(pause_paragraph),
        pause_chapter=float(pause_chapter),
        pause_enum_item=float(pause_enum_item),
        pause_forced_split=float(pause_forced),
        onnx_quant_mode=str(onnx_quant_mode),
        num_step=int(num_step),
        speed=float(speed),
        guidance_scale=float(guidance_scale),
        t_shift=float(t_shift),
        export_format=export_format,
        input_mode=mode,
    )


def apply_preset_to_gui(preset: PresetConfig) -> dict[str, Any]:
    """
    Map PresetConfig → Gradio component updates (gr.update values).
    Caller merges pipeline state refresh separately if needed.
    """
    import gradio as gr

    if preset.voice_mode == "bundled" and preset.voice_id:
        voice_value = preset.voice_id
        ref_audio_value = None
        ref_text_value = ""
    else:
        voice_value = MANUAL_CHOICE
        ref_audio_value = preset.ref_wav or None
        ref_text_value = preset.ref_text or ""

    pipeline = build_normalize_pipeline(preset.pipeline)
    sel_choices = pipeline_selector_choices(pipeline)
    sel_value = sel_choices[0][1] if sel_choices else None

    export_val = preset.export_format
    if export_val not in EXPORT_CHOICES:
        export_val = "WAV 24kHz"

    return {
        "voice_dropdown": gr.update(value=voice_value),
        "ref_audio": gr.update(value=ref_audio_value),
        "ref_text": gr.update(value=ref_text_value),
        "norm_pipeline_state": pipeline,
        "norm_pipeline_display": format_normalize_pipeline_list(pipeline),
        "norm_sel_step": gr.update(choices=sel_choices, value=sel_value),
        "chunk_max_chars": gr.update(value=int(preset.chunk_max_chars)),
        "pause_sentence": gr.update(value=float(preset.pause_sentence)),
        "pause_paragraph": gr.update(value=float(preset.pause_paragraph)),
        "pause_chapter": gr.update(value=float(preset.pause_chapter)),
        "pause_enum_item": gr.update(value=float(preset.pause_enum_item)),
        "pause_forced": gr.update(value=float(preset.pause_forced_split)),
        "speed": gr.update(value=float(preset.speed)),
        "export_format": gr.update(value=export_val),
        "onnx_quant_mode": gr.update(value=str(preset.onnx_quant_mode)),
        "synth_num_step": int(preset.num_step),
        "synth_guidance_scale": float(preset.guidance_scale),
        "synth_t_shift": float(preset.t_shift),
        "input_mode": gr.update(value=preset.input_mode),
        "preset_status": (
            f"Đã tải preset **{preset.name}**"
            + (f" — {preset.description}" if preset.description else "")
        ),
    }


def default_none_preset() -> PresetConfig:
    return PresetConfig(
        name="none",
        description="Pipeline trống, giọng upload thủ công, mặc định nghỉ/chunk",
        voice_mode="manual",
        pipeline=list(DEFAULT_NORMALIZE_PIPELINE),
    )


def default_sach_preset() -> PresetConfig:
    return PresetConfig(
        name="Sách — Ái Vy",
        description="Audiobook: toàn bộ chuẩn hóa (sea-g2p → … → VieNeu), giọng Ái Vy",
        voice_mode="bundled",
        voice_id="ai_vy",
        pipeline=list(AUDIOBOOK_PRESET_PIPELINE),
        chunk_max_chars=135,
    )
