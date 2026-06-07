"""
Mel filterbank extractor for ZipVoice prompt features (VocosFbank-compatible).
Standalone — no lhotse dependency.
"""
from __future__ import annotations

import numpy as np
import torch
import torchaudio


class VocosFbank:
    SAMPLING_RATE = 24000
    N_MELS = 100
    N_FFT = 1024
    HOP_LENGTH = 256

    def __init__(self) -> None:
        self.fbank = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.SAMPLING_RATE,
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH,
            n_mels=self.N_MELS,
            center=True,
            power=1,
        )

    def extract(
        self,
        samples: np.ndarray | torch.Tensor,
        sampling_rate: int,
    ) -> torch.Tensor:
        assert sampling_rate == self.SAMPLING_RATE, (
            f"Expected {self.SAMPLING_RATE} Hz, got {sampling_rate}"
        )
        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(np.asarray(samples, dtype=np.float32))

        if samples.ndim == 1:
            samples = samples.unsqueeze(0)
        elif samples.ndim == 2 and samples.shape[0] == 2:
            samples = samples.mean(dim=0, keepdim=True)

        mel = self.fbank(samples)
        logmel = mel.clamp(min=1e-7).log()
        logmel = logmel.reshape(-1, logmel.shape[-1]).t()

        num_frames = samples.shape[1] // self.HOP_LENGTH + 1
        if logmel.shape[0] > num_frames:
            logmel = logmel[:num_frames]
        elif logmel.shape[0] < num_frames:
            logmel = torch.nn.functional.pad(
                logmel.unsqueeze(0),
                (0, 0, 0, num_frames - logmel.shape[0]),
                mode="replicate",
            ).squeeze(0)
        return logmel
