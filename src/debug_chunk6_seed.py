"""Compare unseeded vs seeded ODE init for chunk 6 (wheezy artifact investigation)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from assets_loader import get_voice_by_id, scan_ref_voices
from config import ONNX_TOKENS
from espeak_tokenizer import EspeakTokenizer
from onnx_engine import (
    FEAT_SCALE,
    OnnxTTSEngine,
    SAMPLING_RATE,
    _get_time_steps,
    _load_wav_24k,
    resolve_ode_seed,
    _vocos_decode_onnx,
)
from preset_io import load_preset
from audio.ref_audio import preprocess_ref_audio_text
from text.chunking import split_text_for_tts
from text.io import read_text_file
from text.pipeline import normalize_full_document, prepare_for_tts, prepare_tts_text

os.environ.setdefault("ZIPVOICE_FORCE_CPU", "1")


def _metrics(wav: np.ndarray) -> dict:
    rms = float(np.sqrt(np.mean(wav**2)))
    peak = float(np.max(np.abs(wav)))
    zc = int(np.sum(np.diff(np.signbit(wav)).astype(int)))
    fft = np.abs(np.fft.rfft(wav))
    hf = float(fft[len(fft) // 2 :].sum())
    lf = float(fft[: len(fft) // 4].sum() + 1e-9)
    return {
        "samples": len(wav),
        "dur": len(wav) / SAMPLING_RATE,
        "rms": rms,
        "peak": peak,
        "zc": zc,
        "hf_lf": hf / lf,
    }


def _synthesize_with_seed(
    engine: OnnxTTSEngine,
    *,
    text: str,
    prompt: str,
    ref_audio: str,
    ode_seed: int | None,
    preset,
) -> np.ndarray:
    """Run fm_decoder ODE with explicit or unseeded noise init."""
    tokens = engine.tokenizer.texts_to_token_ids([text])
    prompt_tokens = engine.tokenizer.texts_to_token_ids([prompt])
    tokens_np = np.array(tokens, dtype=np.int64)
    prompt_tokens_np = np.array(prompt_tokens, dtype=np.int64)

    prompt_audio = _load_wav_24k(ref_audio, engine.sampling_rate)
    prompt_rms = float(np.sqrt(np.mean(np.square(prompt_audio))))
    target_rms = 0.1
    if prompt_rms < target_rms:
        prompt_audio = prompt_audio * (target_rms / prompt_rms)

    prompt_features = engine.feature_extractor.extract(
        prompt_audio, sampling_rate=engine.sampling_rate
    )
    prompt_features = prompt_features * FEAT_SCALE
    if prompt_features.ndim == 2:
        prompt_features = prompt_features[np.newaxis, ...]
    prompt_len = np.array([prompt_features.shape[1]], dtype=np.int64)
    speed_np = np.array(preset.speed, dtype=np.float32)

    text_condition = engine.model.run_text_encoder(
        tokens_np, prompt_tokens_np, prompt_len, speed_np
    )
    batch_size, num_frames, _ = text_condition.shape
    feat_dim = engine.model.feat_dim
    timesteps = _get_time_steps(0.0, 1.0, int(preset.num_step), float(preset.t_shift))

    if ode_seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(ode_seed)

    x = rng.standard_normal((batch_size, num_frames, feat_dim), dtype=np.float32)
    pad_frames = num_frames - prompt_features.shape[1]
    if pad_frames > 0:
        speech_condition = np.pad(
            prompt_features,
            ((0, 0), (0, pad_frames), (0, 0)),
            mode="constant",
        )
    else:
        speech_condition = prompt_features[:, :num_frames, :]
    guidance_np = np.array(preset.guidance_scale, dtype=np.float32)

    for step in range(int(preset.num_step)):
        t_step = np.array(timesteps[step], dtype=np.float32).reshape(())
        v = engine.model.run_fm_decoder(
            t_step, x, text_condition, speech_condition, guidance_np
        )
        x = x + v * (timesteps[step + 1] - timesteps[step])

    pred = x[:, int(prompt_len[0]) :, :]
    mel = np.transpose(pred, (0, 2, 1)) / FEAT_SCALE
    wav = _vocos_decode_onnx(engine.vocoder, mel.astype(np.float32))
    if prompt_rms < target_rms:
        wav = wav * (prompt_rms / target_rms)
    return wav.astype(np.float32)


def main() -> int:
    raw = read_text_file(str(ROOT / "assets" / "example" / "test.txt"))
    preset = load_preset("sach")
    norm = normalize_full_document(raw, preset.pipeline, preset.input_mode)
    chunks = split_text_for_tts(
        norm,
        max_chars=preset.chunk_max_chars,
        min_chars=preset.chunk_min_chars,
        pause_sentence=preset.pause_sentence,
        pause_paragraph=preset.pause_paragraph,
        pause_chapter=preset.pause_chapter,
        pause_enum_item=preset.pause_enum_item,
        pause_forced_split=preset.pause_forced_split,
    )
    text6 = prepare_for_tts(
        chunks[5].text, preset.pipeline, preset.input_mode, already_normalized=True
    )

    voice = get_voice_by_id("nsnd_kim_cuc", scan_ref_voices())
    ref_audio, ref_text = preprocess_ref_audio_text(voice.audio_path, voice.transcript)
    prompt = prepare_tts_text(ref_text, preset.pipeline)

    tok = EspeakTokenizer(token_file=str(ONNX_TOKENS), lang="vi")
    t6 = np.array(tok.texts_to_token_ids([text6]), dtype=np.int64)
    pt = np.array(tok.texts_to_token_ids([prompt]), dtype=np.int64)
    ode_seed, _ = resolve_ode_seed(ode_seed=42, use_fixed_seed=True, chunk_index=5)

    out_dir = ROOT / "output" / "chunk_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"chunk6: {text6!r}")
    print(f"ode_seed (ZIPVOICE_SEED XOR tokens): {ode_seed} (0x{ode_seed:08x})")
    print(f"mel_frames (from prior log): 255 for int4 sequential run")
    print()

    engine = OnnxTTSEngine.get(quant_mode="int4", use_gpu=False)

    # Seeded (production path)
    seeded_wavs = []
    for run in range(2):
        wav = engine.generate(
            prompt_text=prompt,
            prompt_wav=ref_audio,
            text=text6,
            speed=preset.speed,
            num_step=int(preset.num_step),
            guidance_scale=float(preset.guidance_scale),
            t_shift=float(preset.t_shift),
        )
        m = _metrics(wav)
        print(f"seeded_run{run + 1}: {m}")
        seeded_wavs.append(wav)
        if run == 0:
            sf.write(str(out_dir / "chunk_006_seeded.wav"), wav, SAMPLING_RATE)

    print(
        f"seeded reproducibility max_diff: "
        f"{float(np.max(np.abs(seeded_wavs[1] - seeded_wavs[0]))):.6f}"
    )
    print()

    # Unseeded (old committed behavior)
    print("--- unseeded (2 runs, old behavior) ---")
    unseeded = []
    for run in range(2):
        wav = _synthesize_with_seed(
            engine,
            text=text6,
            prompt=prompt,
            ref_audio=ref_audio,
            ode_seed=None,
            preset=preset,
        )
        m = _metrics(wav)
        print(f"unseeded_run{run + 1}: {m}")
        unseeded.append(wav)
        if run == 0:
            sf.write(str(out_dir / "chunk_006_unseeded_run1.wav"), wav, SAMPLING_RATE)

    diff = float(np.max(np.abs(unseeded[1] - unseeded[0])))
    print(f"unseeded max diff run1 vs run2: {diff:.4f}")
    hf_ratios = [_metrics(w)["hf_lf"] for w in unseeded]
    print(f"unseeded hf/lf spread: min={min(hf_ratios):.3f} max={max(hf_ratios):.3f}")
    print(f"seeded hf/lf: {_metrics(seeded_wavs[0])['hf_lf']:.3f}")

    sf.write(str(out_dir / "chunk_006.wav"), seeded_wavs[0], SAMPLING_RATE)
    print(f"\nWrote {out_dir / 'chunk_006.wav'} (seeded fix)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
