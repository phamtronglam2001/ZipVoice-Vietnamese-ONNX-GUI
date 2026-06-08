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
from typing import List, Optional

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample_poly

from config import (
    ONNX_DIR,
    ONNX_MODEL_JSON,
    ONNX_TOKENS,
    VOCODER_MEL_CHANNELS,
    VOCODER_ONNX,
    VOCODER_RUNTIME_LABEL,
    apply_cpu_env,
    is_force_cpu,
    is_onnx_gpu_env,
    onnx_files,
    onnx_quant_mode,
)
from onnx_providers import (
    create_inference_session,
    ensure_cuda_runtime_on_path,
    provider_status_message,
)

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

logger = logging.getLogger("zipvoice_onnx_gui")


def _get_time_steps(
    t_start: float = 0.0,
    t_end: float = 1.0,
    num_step: int = 10,
    t_shift: float = 1.0,
) -> np.ndarray:
    timesteps = np.linspace(t_start, t_end, num_step + 1, dtype=np.float32)
    return (t_shift * timesteps / (1 + (t_shift - 1) * timesteps)).astype(np.float32)


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
        num_thread: int = 1,
        *,
        use_gpu: bool = False,
        quant_mode: str | None = None,
    ):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = num_thread
        opts.intra_op_num_threads = num_thread
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
    ) -> np.ndarray:
        te_in = self.text_encoder.get_inputs()
        out = self.text_encoder.run(
            [self.text_encoder.get_outputs()[0].name],
            {
                te_in[0].name: tokens,
                te_in[1].name: prompt_tokens,
                te_in[2].name: prompt_features_len,
                te_in[3].name: speed,
            },
        )
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
) -> np.ndarray:
    assert len(tokens) == len(prompt_tokens) == 1
    tokens_np = np.array(tokens, dtype=np.int64)
    prompt_tokens_np = np.array(prompt_tokens, dtype=np.int64)
    # prompt_features: (time, feat) -> batch (1, time, feat)
    if prompt_features.ndim == 2:
        prompt_features = prompt_features[np.newaxis, ...]
    prompt_len = np.array([prompt_features.shape[1]], dtype=np.int64)
    speed_np = np.array(speed, dtype=np.float32)

    text_condition = model.run_text_encoder(
        tokens_np, prompt_tokens_np, prompt_len, speed_np
    )
    batch_size, num_frames, _ = text_condition.shape
    feat_dim = model.feat_dim

    timesteps = _get_time_steps(0.0, 1.0, num_step, t_shift)
    rng = np.random.default_rng()
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

    for step in range(num_step):
        t_step = np.array(timesteps[step], dtype=np.float32).reshape(())
        v = model.run_fm_decoder(
            t_step,
            x,
            text_condition,
            speech_condition,
            guidance_np,
        )
        x = x + v * (timesteps[step + 1] - timesteps[step])

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
    opts = ort.SessionOptions()
    num_thread = int(os.environ.get("ZIPVOICE_ONNX_THREADS", "1"))
    opts.inter_op_num_threads = num_thread
    opts.intra_op_num_threads = num_thread
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


def _vocos_decode_onnx(vocoder: ort.InferenceSession, mel_bct: np.ndarray) -> np.ndarray:
    """mel (batch, channels, time) -> waveform numpy 1d."""
    mel_frames = int(mel_bct.shape[2]) if mel_bct.ndim == 3 else 0
    logger.debug("mel shape before vocoder: %s", mel_bct.shape)
    if mel_frames < MIN_VOCODER_MEL_FRAMES:
        logger.warning("skip vocoder: mel frames=%d", mel_frames)
        return np.array([], dtype=np.float32)
    mag, x, y = vocoder.run(None, {"mels": mel_bct.astype(np.float32)})
    return vocos_istft(mag, x, y)


class OnnxTTSEngine:
    _instance: Optional["OnnxTTSEngine"] = None

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
        num_thread = int(os.environ.get("ZIPVOICE_ONNX_THREADS", "1"))

        with open(ONNX_MODEL_JSON, encoding="utf-8") as f:
            model_config = json.load(f)

        self.tokenizer = EspeakTokenizer(token_file=str(ONNX_TOKENS), lang="vi")
        self.feature_extractor = VocosFbank()
        self.model = OnnxModel(
            str(ONNX_DIR / te_name),
            str(ONNX_DIR / fm_name),
            num_thread=num_thread,
            use_gpu=self.use_gpu,
            quant_mode=self.quant_mode,
        )
        self.vocoder = _load_vocoder_onnx(use_gpu=self.use_gpu)
        self.sampling_rate = model_config["feature"]["sampling_rate"]
        size_note = format_sizes(ONNX_DIR, (te_name, fm_name))
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

    def generate(
        self,
        prompt_text: str,
        prompt_wav: str,
        text: str,
        speed: float = 1.0,
        num_step: int = 16,
        guidance_scale: float = 1.0,
        t_shift: float = 0.5,
    ) -> np.ndarray:
        logger.info("ONNX generate chunk (%d chars)...", len(text))

        tokens = self.tokenizer.texts_to_token_ids([text])
        prompt_tokens = self.tokenizer.texts_to_token_ids([prompt_text])

        prompt_audio = _load_wav_24k(prompt_wav, self.sampling_rate)
        prompt_rms = float(np.sqrt(np.mean(np.square(prompt_audio))))
        if prompt_rms < TARGET_RMS:
            prompt_audio = prompt_audio * (TARGET_RMS / prompt_rms)

        prompt_features = self.feature_extractor.extract(
            prompt_audio, sampling_rate=self.sampling_rate
        )
        prompt_features = prompt_features * FEAT_SCALE

        pred_features = onnx_sample(
            model=self.model,
            tokens=tokens,
            prompt_tokens=prompt_tokens,
            prompt_features=prompt_features,
            speed=speed,
            t_shift=t_shift,
            guidance_scale=guidance_scale,
            num_step=num_step,
        )

        mel_frames = int(pred_features.shape[1]) if pred_features.ndim == 3 else 0
        if mel_frames < MIN_VOCODER_MEL_FRAMES:
            logger.warning(
                "skip vocoder: mel frames=%d (chunk %d chars)",
                mel_frames,
                len(text),
            )
            return np.array([], dtype=np.float32)

        # (B, T, C) -> (B, C, T) for Vocos
        mel = np.transpose(pred_features, (0, 2, 1)) / FEAT_SCALE
        wav = _vocos_decode_onnx(self.vocoder, mel.astype(np.float32))

        if prompt_rms < TARGET_RMS:
            wav = wav * (prompt_rms / TARGET_RMS)

        if is_force_cpu():
            gc.collect()
        return wav.astype(np.float32)
