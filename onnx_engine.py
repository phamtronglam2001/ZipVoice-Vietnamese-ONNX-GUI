"""
ZipVoice Vietnamese ONNX inference engine.
Text encoder + flow-matching decoder run via ONNX Runtime;
Vocos vocoder stays PyTorch (upstream design).
"""
from __future__ import annotations

import gc
import json
import logging
import os
from typing import List, Optional, Tuple

import onnxruntime as ort
import torch
import torchaudio
from vocos import Vocos

from config import (
    ONNX_DIR,
    ONNX_MODEL_JSON,
    ONNX_TOKENS,
    VOCODER_DIR,
    apply_cpu_env,
    ensure_vendor_on_path,
    is_force_cpu,
    onnx_files,
    set_offline_env,
    use_onnx_int8,
)

apply_cpu_env()
ensure_vendor_on_path()
set_offline_env()

from zipvoice.tokenizer.tokenizer import EspeakTokenizer  # noqa: E402
from zipvoice.utils.feature import VocosFbank  # noqa: E402

SAMPLING_RATE = 24000
FEAT_SCALE = 0.1
TARGET_RMS = 0.1

logger = logging.getLogger("zipvoice_onnx_gui")

if is_force_cpu():
    threads = os.environ.get("ZIPVOICE_CPU_THREADS", "4")
    try:
        torch.set_num_threads(int(threads))
    except Exception:
        pass


def _get_time_steps(
    t_start: float = 0.0,
    t_end: float = 1.0,
    num_step: int = 10,
    t_shift: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    timesteps = torch.linspace(t_start, t_end, num_step + 1).to(device)
    return t_shift * timesteps / (1 + (t_shift - 1) * timesteps)


class OnnxModel:
    def __init__(
        self,
        text_encoder_path: str,
        fm_decoder_path: str,
        num_thread: int = 1,
    ):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = num_thread
        session_opts.intra_op_num_threads = num_thread
        self.session_opts = session_opts
        self.init_text_encoder(text_encoder_path)
        self.init_fm_decoder(fm_decoder_path)

    def init_text_encoder(self, model_path: str):
        self.text_encoder = ort.InferenceSession(
            model_path,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

    def init_fm_decoder(self, model_path: str):
        self.fm_decoder = ort.InferenceSession(
            model_path,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )
        meta = self.fm_decoder.get_modelmeta().custom_metadata_map
        self.feat_dim = int(meta["feat_dim"])

    def run_text_encoder(
        self,
        tokens: torch.Tensor,
        prompt_tokens: torch.Tensor,
        prompt_features_len: torch.Tensor,
        speed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.text_encoder.run(
            [self.text_encoder.get_outputs()[0].name],
            {
                self.text_encoder.get_inputs()[0].name: tokens.numpy(),
                self.text_encoder.get_inputs()[1].name: prompt_tokens.numpy(),
                self.text_encoder.get_inputs()[2].name: prompt_features_len.numpy(),
                self.text_encoder.get_inputs()[3].name: speed.numpy(),
            },
        )
        return torch.from_numpy(out[0])

    def run_fm_decoder(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        text_condition: torch.Tensor,
        speech_condition: torch.Tensor,
        guidance_scale: torch.Tensor,
    ) -> torch.Tensor:
        out = self.fm_decoder.run(
            [self.fm_decoder.get_outputs()[0].name],
            {
                self.fm_decoder.get_inputs()[0].name: t.numpy(),
                self.fm_decoder.get_inputs()[1].name: x.numpy(),
                self.fm_decoder.get_inputs()[2].name: text_condition.numpy(),
                self.fm_decoder.get_inputs()[3].name: speech_condition.numpy(),
                self.fm_decoder.get_inputs()[4].name: guidance_scale.numpy(),
            },
        )
        return torch.from_numpy(out[0])


def onnx_sample(
    model: OnnxModel,
    tokens: List[List[int]],
    prompt_tokens: List[List[int]],
    prompt_features: torch.Tensor,
    speed: float = 1.0,
    t_shift: float = 0.5,
    guidance_scale: float = 1.0,
    num_step: int = 16,
) -> torch.Tensor:
    assert len(tokens) == len(prompt_tokens) == 1
    tokens_t = torch.tensor(tokens, dtype=torch.int64)
    prompt_tokens_t = torch.tensor(prompt_tokens, dtype=torch.int64)
    prompt_features_len = torch.tensor(prompt_features.size(1), dtype=torch.int64)
    speed_t = torch.tensor(speed, dtype=torch.float32)

    text_condition = model.run_text_encoder(
        tokens_t, prompt_tokens_t, prompt_features_len, speed_t
    )

    batch_size, num_frames, _ = text_condition.shape
    feat_dim = model.feat_dim

    timesteps = _get_time_steps(
        t_start=0.0, t_end=1.0, num_step=num_step, t_shift=t_shift
    )
    x = torch.randn(batch_size, num_frames, feat_dim)
    speech_condition = torch.nn.functional.pad(
        prompt_features, (0, 0, 0, num_frames - prompt_features.shape[1])
    )
    guidance_scale_t = torch.tensor(guidance_scale, dtype=torch.float32)

    for step in range(num_step):
        v = model.run_fm_decoder(
            t=timesteps[step],
            x=x,
            text_condition=text_condition,
            speech_condition=speech_condition,
            guidance_scale=guidance_scale_t,
        )
        x = x + v * (timesteps[step + 1] - timesteps[step])

    return x[:, prompt_features_len.item() :, :]


def _load_vocoder() -> Vocos:
    vocoder = Vocos.from_hparams(str(VOCODER_DIR / "config.yaml"))
    state_dict = torch.load(
        VOCODER_DIR / "pytorch_model.bin",
        weights_only=True,
        map_location="cpu",
    )
    vocoder.load_state_dict(state_dict)
    return vocoder.eval()


class OnnxTTSEngine:
    """Lazy-loaded singleton for Gradio ONNX inference."""

    _instance: Optional["OnnxTTSEngine"] = None

    def __init__(self, use_int8: bool | None = None) -> None:
        self.use_int8 = use_onnx_int8() if use_int8 is None else use_int8
        te_name, fm_name = onnx_files(self.use_int8)
        num_thread = int(os.environ.get("ZIPVOICE_ONNX_THREADS", "1"))

        with open(ONNX_MODEL_JSON, encoding="utf-8") as f:
            model_config = json.load(f)

        self.tokenizer = EspeakTokenizer(token_file=str(ONNX_TOKENS), lang="vi")
        self.feature_extractor = VocosFbank()
        self.model = OnnxModel(
            str(ONNX_DIR / te_name),
            str(ONNX_DIR / fm_name),
            num_thread=num_thread,
        )
        self.vocoder = _load_vocoder()
        self.sampling_rate = model_config["feature"]["sampling_rate"]
        logger.info(
            "OnnxTTSEngine ready | int8=%s | te=%s | fm=%s",
            self.use_int8,
            te_name,
            fm_name,
        )

    @classmethod
    def get(cls, use_int8: bool | None = None) -> "OnnxTTSEngine":
        want_int8 = use_onnx_int8() if use_int8 is None else use_int8
        if cls._instance is None or cls._instance.use_int8 != want_int8:
            print(
                f"[ZipVoice ONNX] Loading models (int8={want_int8}, first run may take a minute)..."
            )
            cls._instance = cls(use_int8=want_int8)
            print("[ZipVoice ONNX] Models ready.")
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
    ) -> torch.Tensor:
        logger.info("ONNX generate chunk (%d chars)...", len(text))
        with torch.inference_mode():
            tokens = self.tokenizer.texts_to_token_ids([text])
            prompt_tokens = self.tokenizer.texts_to_token_ids([prompt_text])

            prompt_audio, prompt_sr = torchaudio.load(prompt_wav)
            if prompt_sr != self.sampling_rate:
                prompt_audio = torchaudio.transforms.Resample(
                    orig_freq=prompt_sr, new_freq=self.sampling_rate
                )(prompt_audio)

            prompt_rms = torch.sqrt(torch.mean(torch.square(prompt_audio)))
            if prompt_rms < TARGET_RMS:
                prompt_audio = prompt_audio * TARGET_RMS / prompt_rms

            prompt_features = (
                self.feature_extractor.extract(
                    prompt_audio, sampling_rate=self.sampling_rate
                ).unsqueeze(0)
                * FEAT_SCALE
            )

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

            pred_features = pred_features.permute(0, 2, 1) / FEAT_SCALE
            wav = self.vocoder.decode(pred_features).squeeze(1).clamp(-1, 1)

            if prompt_rms < TARGET_RMS:
                wav = wav * prompt_rms / TARGET_RMS

            out = wav.cpu()
            del prompt_audio, prompt_features, pred_features, wav
            if is_force_cpu():
                gc.collect()
            return out
