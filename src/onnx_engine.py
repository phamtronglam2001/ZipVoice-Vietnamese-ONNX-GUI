"""
ZipVoice Vietnamese ONNX inference engine.
ZipVoice encoder/decoder + Vocos vocoder: ONNX Runtime (numpy/librosa).

Audio stack:
  - Ref WAV load/resample: soundfile + scipy.signal.resample_poly.
  - Prompt mel: VocosFbank — librosa melspectrogram (htk=True, norm=None).
  - ZipVoice mel: ONNX text_encoder + fm_decoder (FEAT_SCALE=0.1).
  - Vocoder: mel_spec_24khz.onnx (100 mel → mag/x/y) + vocos_istft (librosa).
"""
from __future__ import annotations

import gc
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample_poly

import config
from config import (
    VOCODER_MEL_CHANNELS,
    VOCODER_ONNX,
    VOCODER_RUNTIME_LABEL,
    apply_cpu_env,
    is_force_cpu,
    is_onnx_gpu_env,
    ode_solver_default,
    onnx_files,
    onnx_num_threads,
    onnx_quant_mode,
)
from inference_profile import StageTimings, StageTimer
from onnx_providers import (
    create_inference_session,
    ensure_cuda_runtime_on_path,
    provider_status_message,
)
from onnx_session_opts import build_session_options

ensure_cuda_runtime_on_path()
from onnx_quant import format_sizes
from espeak_tokenizer import EspeakTokenizer
from vocos_fbank import VocosFbank
from vocos_istft import vocos_istft

apply_cpu_env()

SAMPLING_RATE = 24000
FEAT_SCALE = 0.1
TARGET_RMS = 0.1
MIN_VOCODER_MEL_FRAMES = 1

OdeSolver = Literal["euler", "heun", "midpoint"]
ODE_SOLVERS: tuple[str, ...] = ("euler", "heun", "midpoint")

logger = logging.getLogger("zipvoice_onnx_gui")

# Flow-matching ODE starts from Gaussian noise; unseeded draws cause rare noisy/wheezy
# chunks. Use a fixed base seed (+ chunk index) for reproducible, stable synthesis.
_DEFAULT_ODE_SEED = 42


def _env_ode_seed_base() -> int:
    raw = os.environ.get("ZIPVOICE_SEED", str(_DEFAULT_ODE_SEED)).strip()
    try:
        return int(raw, 0)
    except ValueError:
        logger.warning("Invalid ZIPVOICE_SEED=%r — using %d", raw, _DEFAULT_ODE_SEED)
        return _DEFAULT_ODE_SEED


def resolve_ode_seed(
    *,
    ode_seed: int | None = None,
    use_fixed_seed: bool = True,
    chunk_index: int | None = None,
) -> tuple[int | None, str]:
    """Return (seed, mode). seed=None means unseeded random per call."""
    if use_fixed_seed:
        base = int(ode_seed) if ode_seed is not None else _env_ode_seed_base()
        if chunk_index is not None:
            seed = int((base + chunk_index) & 0xFFFFFFFF)
        else:
            seed = int(base & 0xFFFFFFFF)
        return seed, "fixed"
    return None, "random"


def format_ode_seed_log(
    *,
    ode_seed: int | None = None,
    use_fixed_seed: bool = True,
    same_seed_all_chunks: bool = False,
) -> str:
    if use_fixed_seed:
        base = int(ode_seed) if ode_seed is not None else _env_ode_seed_base()
        if same_seed_all_chunks:
            return f"ode_seed={base} (fixed, same every chunk)"
        return f"ode_seed={base} (fixed, +chunk_index)"
    return "ode_seed=random"


def _get_time_steps(
    t_start: float = 0.0,
    t_end: float = 1.0,
    num_step: int = 10,
    t_shift: float = 1.0,
) -> np.ndarray:
    timesteps = np.linspace(t_start, t_end, num_step + 1, dtype=np.float32)
    return (t_shift * timesteps / (1 + (t_shift - 1) * timesteps)).astype(np.float32)


def _normalize_ode_solver(solver: str | None) -> OdeSolver:
    raw = (solver or ode_solver_default()).strip().lower()
    if raw in ODE_SOLVERS:
        return raw  # type: ignore[return-value]
    return "euler"


def _pad_token_batch(token_rows: List[List[int]]) -> np.ndarray:
    """Pad variable-length token lists to (batch, max_len)."""
    batch_size = len(token_rows)
    max_len = max(len(row) for row in token_rows)
    out = np.zeros((batch_size, max_len), dtype=np.int64)
    for i, row in enumerate(token_rows):
        out[i, : len(row)] = row
    return out


@dataclass
class PromptState:
    """Cached reference-audio conditioning (fixed for one synthesis run)."""

    prompt_text: str = ""
    prompt_wav: str = ""
    prompt_tokens: List[List[int]] = field(default_factory=list)
    prompt_features: np.ndarray | None = None
    prompt_rms: float = 1.0


@dataclass
class BatchGenerateResult:
    waveforms: list[np.ndarray]
    mel_frames: list[int] = field(default_factory=list)
    timing: StageTimings | None = None


def _run_ode_loop(
    model: "OnnxModel",
    *,
    x: np.ndarray,
    text_condition: np.ndarray,
    speech_condition: np.ndarray,
    guidance_np: np.ndarray,
    timesteps: np.ndarray,
    num_step: int,
    solver: OdeSolver,
    timings: StageTimings | None = None,
) -> np.ndarray:
    """Flow-matching ODE integration (Euler / Heun / midpoint)."""
    for step in range(num_step):
        t0 = timesteps[step]
        dt = float(timesteps[step + 1] - t0)
        t_step = np.array(t0, dtype=np.float32).reshape(())

        if solver == "euler":
            t_fm = time.perf_counter() if timings else 0.0
            v = model.run_fm_decoder(
                t_step, x, text_condition, speech_condition, guidance_np
            )
            if timings:
                timings.fm_decoder += time.perf_counter() - t_fm
                timings.fm_decoder_steps += 1
            x = x + v * dt
            continue

        t_fm = time.perf_counter() if timings else 0.0
        v0 = model.run_fm_decoder(
            t_step, x, text_condition, speech_condition, guidance_np
        )
        if timings:
            timings.fm_decoder += time.perf_counter() - t_fm
            timings.fm_decoder_steps += 1

        if solver == "midpoint":
            t_mid = np.array(t0 + dt * 0.5, dtype=np.float32).reshape(())
            x_mid = x + v0 * (dt * 0.5)
            t_fm = time.perf_counter() if timings else 0.0
            v_mid = model.run_fm_decoder(
                t_mid, x_mid, text_condition, speech_condition, guidance_np
            )
            if timings:
                timings.fm_decoder += time.perf_counter() - t_fm
                timings.fm_decoder_steps += 1
            x = x + v_mid * dt
        else:  # heun
            x_euler = x + v0 * dt
            t1 = np.array(timesteps[step + 1], dtype=np.float32).reshape(())
            t_fm = time.perf_counter() if timings else 0.0
            v1 = model.run_fm_decoder(
                t1, x_euler, text_condition, speech_condition, guidance_np
            )
            if timings:
                timings.fm_decoder += time.perf_counter() - t_fm
                timings.fm_decoder_steps += 1
            x = x + (v0 + v1) * (dt * 0.5)

    return x


def _load_wav_24k(path: str, target_sr: int = SAMPLING_RATE) -> np.ndarray:
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    audio = audio.astype(np.float32)
    if sr != target_sr:
        audio = resample_poly(audio, target_sr, sr).astype(np.float32)
    return audio


class OnnxModel:
    def __init__(
        self,
        text_encoder_path: str,
        fm_decoder_path: str,
        num_thread: int | None = None,
        *,
        use_gpu: bool = False,
        quant_mode: str | None = None,
    ):
        opts = build_session_options(num_threads=num_thread)
        self.text_encoder = create_inference_session(
            text_encoder_path,
            sess_options=opts,
            use_gpu=use_gpu,
            quant_mode=quant_mode,
            component="text_encoder",
        )
        self.fm_decoder = create_inference_session(
            fm_decoder_path,
            sess_options=opts,
            use_gpu=use_gpu,
            quant_mode=quant_mode,
            component="fm_decoder",
        )
        meta = self.fm_decoder.get_modelmeta().custom_metadata_map
        self.feat_dim = int(meta["feat_dim"])

    def run_text_encoder(
        self,
        tokens: np.ndarray,
        prompt_tokens: np.ndarray,
        prompt_features_len: np.ndarray,
        speed: np.ndarray,
        *,
        timings: StageTimings | None = None,
    ) -> np.ndarray:
        te_in = self.text_encoder.get_inputs()
        t0 = time.perf_counter() if timings else 0.0
        out = self.text_encoder.run(
            [self.text_encoder.get_outputs()[0].name],
            {
                te_in[0].name: tokens,
                te_in[1].name: prompt_tokens,
                te_in[2].name: prompt_features_len,
                te_in[3].name: speed,
            },
        )
        if timings:
            timings.text_encoder += time.perf_counter() - t0
        return out[0]

    def run_fm_decoder(
        self,
        t: np.ndarray,
        x: np.ndarray,
        text_condition: np.ndarray,
        speech_condition: np.ndarray,
        guidance_scale: np.ndarray,
    ) -> np.ndarray:
        fm_in = self.fm_decoder.get_inputs()
        out = self.fm_decoder.run(
            [self.fm_decoder.get_outputs()[0].name],
            {
                fm_in[0].name: t,
                fm_in[1].name: x,
                fm_in[2].name: text_condition,
                fm_in[3].name: speech_condition,
                fm_in[4].name: guidance_scale,
            },
        )
        return out[0]


def onnx_sample(
    model: OnnxModel,
    tokens: List[List[int]],
    prompt_tokens: List[List[int]],
    prompt_features: np.ndarray,
    speed: float = 1.0,
    t_shift: float = 0.5,
    guidance_scale: float = 1.0,
    num_step: int = 16,
    *,
    ode_seed: int | None = None,
    use_fixed_seed: bool = True,
    chunk_index: int | None = None,
    ode_solver: str | None = None,
    timings: StageTimings | None = None,
) -> np.ndarray:
    """
    Flow-matching sample for one or more texts (batch dimension = len(tokens)).

    Returns pred mel (batch, time, feat) after trimming prompt frames.
    """
    batch_size = len(tokens)
    if batch_size != len(prompt_tokens):
        raise ValueError("tokens and prompt_tokens batch size must match")

    tokens_np = _pad_token_batch(tokens)
    prompt_np = _pad_token_batch(prompt_tokens)
    if prompt_np.shape[0] == 1 and batch_size > 1:
        prompt_np = np.repeat(prompt_np, batch_size, axis=0)

    if prompt_features.ndim == 2:
        prompt_features = prompt_features[np.newaxis, ...]
    if prompt_features.shape[0] == 1 and batch_size > 1:
        prompt_features = np.repeat(prompt_features, batch_size, axis=0)

    prompt_len = np.array([prompt_features.shape[1]], dtype=np.int64)
    speed_np = np.array(speed, dtype=np.float32)

    text_condition = model.run_text_encoder(
        tokens_np, prompt_np, prompt_len, speed_np, timings=timings
    )
    _, num_frames, feat_dim = text_condition.shape

    timesteps = _get_time_steps(0.0, 1.0, num_step, t_shift)
    solver = _normalize_ode_solver(ode_solver)
    seed, seed_mode = resolve_ode_seed(
        ode_seed=ode_seed,
        use_fixed_seed=use_fixed_seed,
        chunk_index=chunk_index,
    )
    if seed is None:
        rng = np.random.default_rng()
        logger.debug(
            "onnx_sample ode_seed=random (%s, mel_frames=%d, batch=%d)",
            seed_mode,
            num_frames,
            batch_size,
        )
    else:
        rng = np.random.default_rng(seed)
        logger.debug(
            "onnx_sample ode_seed=%d (%s, mel_frames=%d, batch=%d)",
            seed,
            seed_mode,
            num_frames,
            batch_size,
        )
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

    guidance_np = np.array(guidance_scale, dtype=np.float32)

    if timings:
        timings.batch_size = batch_size

    x = _run_ode_loop(
        model,
        x=x,
        text_condition=text_condition,
        speech_condition=speech_condition,
        guidance_np=guidance_np,
        timesteps=timesteps,
        num_step=num_step,
        solver=solver,
        timings=timings,
    )

    trim = int(prompt_len[0])
    pred = x[:, trim:, :]
    if pred.shape[1] == 0:
        logger.warning(
            "onnx_sample: 0 mel frames after trim (num_frames=%d, prompt_len=%d)",
            num_frames,
            trim,
        )
    return pred


def _load_vocoder_onnx(*, use_gpu: bool = False) -> ort.InferenceSession:
    opts = build_session_options(num_threads=onnx_num_threads())
    session = create_inference_session(
        str(VOCODER_ONNX),
        sess_options=opts,
        use_gpu=use_gpu,
        component="vocoder",
    )
    mel_input = next(
        (inp for inp in session.get_inputs() if inp.name == "mels"),
        session.get_inputs()[0],
    )
    mel_shape = mel_input.shape
    logger.info(
        "Vocoder ONNX loaded | file=%s | input=%s shape=%s | expected_mel=%d",
        VOCODER_ONNX.name,
        mel_input.name,
        mel_shape,
        VOCODER_MEL_CHANNELS,
    )
    if len(mel_shape) >= 2 and isinstance(mel_shape[1], int):
        if mel_shape[1] != VOCODER_MEL_CHANNELS:
            logger.warning(
                "Vocoder mel channels %s != ZipVoice feat_dim %d — audio may be corrupted "
                "(expected bundled 100-mel export from ZipVoice-Vietnamese-GUI)",
                mel_shape[1],
                VOCODER_MEL_CHANNELS,
            )
    return session


def _vocos_decode_onnx(
    vocoder: ort.InferenceSession,
    mel_bct: np.ndarray,
    *,
    timings: StageTimings | None = None,
) -> np.ndarray:
    """mel (batch, channels, time) -> 1d waveform float32 (batch 0 only if batch>1)."""
    mel_frames = int(mel_bct.shape[2]) if mel_bct.ndim == 3 else 0
    logger.debug("mel shape before vocoder: %s", mel_bct.shape)
    if mel_frames < MIN_VOCODER_MEL_FRAMES:
        logger.warning("skip vocoder: mel frames=%d", mel_frames)
        return np.array([], dtype=np.float32)
    t0 = time.perf_counter() if timings else 0.0
    mag, x, y = vocoder.run(None, {"mels": mel_bct.astype(np.float32)})
    if timings:
        timings.vocoder += time.perf_counter() - t0
    t1 = time.perf_counter() if timings else 0.0
    wav = vocos_istft(mag, x, y)
    if timings:
        timings.istft += time.perf_counter() - t1
    return wav


def _decode_mel_batch(
    vocoder: ort.InferenceSession,
    pred_features: np.ndarray,
    *,
    prompt_rms: float,
    timings: StageTimings | None = None,
) -> tuple[list[np.ndarray], list[int]]:
    """Decode batched mel predictions to waveforms."""
    waveforms: list[np.ndarray] = []
    mel_frames: list[int] = []
    batch_size = pred_features.shape[0]
    for b in range(batch_size):
        mel_frames.append(int(pred_features.shape[1]))
        if mel_frames[-1] < MIN_VOCODER_MEL_FRAMES:
            waveforms.append(np.array([], dtype=np.float32))
            continue
        mel = np.transpose(pred_features[b : b + 1], (0, 2, 1)) / FEAT_SCALE
        wav = _vocos_decode_onnx(vocoder, mel.astype(np.float32), timings=timings)
        if prompt_rms < TARGET_RMS:
            wav = wav * (prompt_rms / TARGET_RMS)
        waveforms.append(wav.astype(np.float32))
    return waveforms, mel_frames


class OnnxTTSEngine:
    _instance: Optional["OnnxTTSEngine"] = None
    _models_dir: Path | None = None

    def __init__(
        self,
        quant_mode: str | None = None,
        *,
        use_int8: bool | None = None,
        use_gpu: bool | None = None,
    ) -> None:
        self.use_gpu = is_onnx_gpu_env() if use_gpu is None else bool(use_gpu)
        if is_force_cpu():
            self.use_gpu = False
        self.quant_mode = onnx_quant_mode() if quant_mode is None else quant_mode
        if use_int8 is not None and quant_mode is None:
            from onnx_quant import normalize_quant_mode

            self.quant_mode = normalize_quant_mode(None, use_int8=use_int8)
        te_name, fm_name = onnx_files(self.quant_mode, use_int8=use_int8)
        num_thread = onnx_num_threads()

        with open(config.ONNX_MODEL_JSON, encoding="utf-8") as f:
            model_config = json.load(f)

        self.tokenizer = EspeakTokenizer(token_file=str(config.ONNX_TOKENS), lang="vi")
        self.feature_extractor = VocosFbank()
        self.model = OnnxModel(
            str(config.ONNX_DIR / te_name),
            str(config.ONNX_DIR / fm_name),
            num_thread=num_thread,
            use_gpu=self.use_gpu,
            quant_mode=self.quant_mode,
        )
        self.vocoder = _load_vocoder_onnx(use_gpu=self.use_gpu)
        self.sampling_rate = model_config["feature"]["sampling_rate"]
        self._prompt_state: PromptState | None = None
        size_note = format_sizes(config.ONNX_DIR, (te_name, fm_name))
        provider_note = provider_status_message(self.use_gpu)
        logger.info(
            "OnnxTTSEngine ready | mode=%s | device=%s | %s | vocoder=%s",
            self.quant_mode,
            provider_note,
            size_note,
            VOCODER_RUNTIME_LABEL,
        )

    @property
    def use_int8(self) -> bool:
        """Backward-compatible alias."""
        return self.quant_mode == "int8"

    @classmethod
    def get(
        cls,
        quant_mode: str | None = None,
        *,
        use_int8: bool | None = None,
        use_gpu: bool | None = None,
    ) -> "OnnxTTSEngine":
        from onnx_quant import normalize_quant_mode

        want_mode = onnx_quant_mode() if quant_mode is None else quant_mode
        if use_int8 is not None and quant_mode is None:
            want_mode = normalize_quant_mode(None, use_int8=use_int8)
        want_gpu = is_onnx_gpu_env() if use_gpu is None else bool(use_gpu)
        if is_force_cpu():
            want_gpu = False

        if (
            cls._instance is None
            or cls._instance.quant_mode != want_mode
            or cls._instance.use_gpu != want_gpu
        ):
            device = provider_status_message(want_gpu)
            print(
                f"[ZipVoice ONNX] Loading (mode={want_mode}, device={device}, "
                f"vocoder={VOCODER_RUNTIME_LABEL})..."
            )
            cls._instance = cls(
                quant_mode=want_mode,
                use_int8=use_int8,
                use_gpu=want_gpu,
            )
            print("[ZipVoice ONNX] Ready.")
        return cls._instance

    def clear_prompt_cache(self) -> None:
        self._prompt_state = None

    def prepare_prompt(self, prompt_text: str, prompt_wav: str) -> PromptState:
        """Load and cache reference mel + tokens (reuse across chunks)."""
        state = PromptState(prompt_text=prompt_text, prompt_wav=prompt_wav)
        state.prompt_tokens = self.tokenizer.texts_to_token_ids([prompt_text])
        prompt_audio = _load_wav_24k(prompt_wav, self.sampling_rate)
        state.prompt_rms = float(np.sqrt(np.mean(np.square(prompt_audio))))
        if state.prompt_rms < TARGET_RMS:
            prompt_audio = prompt_audio * (TARGET_RMS / state.prompt_rms)
        state.prompt_features = (
            self.feature_extractor.extract(prompt_audio, sampling_rate=self.sampling_rate)
            * FEAT_SCALE
        )
        self._prompt_state = state
        return state

    def _resolve_prompt(
        self,
        prompt_text: str,
        prompt_wav: str,
        *,
        use_prompt_cache: bool,
        timings: StageTimings | None,
    ) -> PromptState:
        if (
            use_prompt_cache
            and self._prompt_state is not None
            and self._prompt_state.prompt_text == prompt_text
            and self._prompt_state.prompt_wav == prompt_wav
            and self._prompt_state.prompt_features is not None
        ):
            return self._prompt_state

        state = PromptState(prompt_text=prompt_text, prompt_wav=prompt_wav)
        if timings:
            with StageTimer(timings, "tokenize"):
                state.prompt_tokens = self.tokenizer.texts_to_token_ids([prompt_text])
        else:
            state.prompt_tokens = self.tokenizer.texts_to_token_ids([prompt_text])

        if timings:
            with StageTimer(timings, "mel_extract"):
                prompt_audio = _load_wav_24k(prompt_wav, self.sampling_rate)
                state.prompt_rms = float(np.sqrt(np.mean(np.square(prompt_audio))))
                if state.prompt_rms < TARGET_RMS:
                    prompt_audio = prompt_audio * (TARGET_RMS / state.prompt_rms)
                state.prompt_features = (
                    self.feature_extractor.extract(
                        prompt_audio, sampling_rate=self.sampling_rate
                    )
                    * FEAT_SCALE
                )
        else:
            prompt_audio = _load_wav_24k(prompt_wav, self.sampling_rate)
            state.prompt_rms = float(np.sqrt(np.mean(np.square(prompt_audio))))
            if state.prompt_rms < TARGET_RMS:
                prompt_audio = prompt_audio * (TARGET_RMS / state.prompt_rms)
            state.prompt_features = (
                self.feature_extractor.extract(
                    prompt_audio, sampling_rate=self.sampling_rate
                )
                * FEAT_SCALE
            )

        if use_prompt_cache:
            self._prompt_state = state
        return state

    def generate(
        self,
        prompt_text: str,
        prompt_wav: str,
        text: str,
        speed: float = 1.0,
        num_step: int = 16,
        guidance_scale: float = 1.0,
        t_shift: float = 0.5,
        *,
        ode_seed: int | None = None,
        use_fixed_seed: bool = True,
        chunk_index: int | None = None,
        return_mel_frames: bool = False,
        use_prompt_cache: bool = True,
        ode_solver: str | None = None,
        profile: bool = False,
        pre_tokens: List[List[int]] | None = None,
    ) -> np.ndarray | tuple[np.ndarray, int] | tuple[np.ndarray, StageTimings]:
        logger.info("ONNX generate chunk (%d chars)...", len(text))
        timings = StageTimings() if profile else None
        t_total = time.perf_counter() if profile else 0.0

        prompt = self._resolve_prompt(
            prompt_text,
            prompt_wav,
            use_prompt_cache=use_prompt_cache,
            timings=timings,
        )

        if pre_tokens is not None:
            tokens = pre_tokens
        elif timings:
            with StageTimer(timings, "tokenize"):
                tokens = self.tokenizer.texts_to_token_ids([text])
        else:
            tokens = self.tokenizer.texts_to_token_ids([text])

        pred_features = onnx_sample(
            model=self.model,
            tokens=tokens,
            prompt_tokens=prompt.prompt_tokens,
            prompt_features=prompt.prompt_features,
            speed=speed,
            t_shift=t_shift,
            guidance_scale=guidance_scale,
            num_step=num_step,
            ode_seed=ode_seed,
            use_fixed_seed=use_fixed_seed,
            chunk_index=chunk_index,
            ode_solver=ode_solver,
            timings=timings,
        )

        mel_frames = int(pred_features.shape[1]) if pred_features.ndim == 3 else 0
        if mel_frames < MIN_VOCODER_MEL_FRAMES:
            logger.warning(
                "skip vocoder: mel frames=%d (chunk %d chars)",
                mel_frames,
                len(text),
            )
            empty = np.array([], dtype=np.float32)
            if profile and timings is not None:
                timings.total = time.perf_counter() - t_total
                return empty, timings
            if return_mel_frames:
                return empty, mel_frames
            return empty

        mel = np.transpose(pred_features, (0, 2, 1)) / FEAT_SCALE
        wav = _vocos_decode_onnx(self.vocoder, mel.astype(np.float32), timings=timings)

        if prompt.prompt_rms < TARGET_RMS:
            wav = wav * (prompt.prompt_rms / TARGET_RMS)

        if is_force_cpu():
            gc.collect()
        wav = wav.astype(np.float32)
        if profile and timings is not None:
            timings.total = time.perf_counter() - t_total
            return wav, timings
        if return_mel_frames:
            return wav, mel_frames
        return wav

    def generate_batch(
        self,
        prompt_text: str,
        prompt_wav: str,
        texts: list[str],
        speed: float = 1.0,
        num_step: int = 16,
        guidance_scale: float = 1.0,
        t_shift: float = 0.5,
        *,
        ode_seed: int | None = None,
        use_fixed_seed: bool = True,
        chunk_indices: list[int] | None = None,
        use_prompt_cache: bool = True,
        ode_solver: str | None = None,
        profile: bool = False,
        pre_token_rows: list[List[List[int]]] | None = None,
    ) -> BatchGenerateResult:
        """Synthesize multiple texts in one ONNX batch."""
        if not texts:
            return BatchGenerateResult(waveforms=[])

        timings = StageTimings() if profile else None
        t_total = time.perf_counter() if profile else 0.0

        prompt = self._resolve_prompt(
            prompt_text,
            prompt_wav,
            use_prompt_cache=use_prompt_cache,
            timings=timings,
        )

        if pre_token_rows is not None:
            token_rows = [row[0] for row in pre_token_rows]
        elif timings:
            with StageTimer(timings, "tokenize"):
                token_rows = [
                    self.tokenizer.texts_to_token_ids([t])[0] for t in texts
                ]
        else:
            token_rows = [self.tokenizer.texts_to_token_ids([t])[0] for t in texts]

        prompt_token_rows = [prompt.prompt_tokens[0]] * len(texts)
        chunk_index = chunk_indices[0] if chunk_indices else None

        pred_features = onnx_sample(
            model=self.model,
            tokens=token_rows,
            prompt_tokens=prompt_token_rows,
            prompt_features=prompt.prompt_features,
            speed=speed,
            t_shift=t_shift,
            guidance_scale=guidance_scale,
            num_step=num_step,
            ode_seed=ode_seed,
            use_fixed_seed=use_fixed_seed,
            chunk_index=chunk_index,
            ode_solver=ode_solver,
            timings=timings,
        )

        waveforms, mel_frames = _decode_mel_batch(
            self.vocoder,
            pred_features,
            prompt_rms=prompt.prompt_rms,
            timings=timings,
        )

        if is_force_cpu():
            gc.collect()

        if profile and timings is not None:
            timings.total = time.perf_counter() - t_total

        return BatchGenerateResult(
            waveforms=waveforms,
            mel_frames=mel_frames,
            timing=timings,
        )
