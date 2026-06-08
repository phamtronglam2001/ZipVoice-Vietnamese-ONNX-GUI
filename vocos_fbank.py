"""
Mel filterbank for ZipVoice prompt features (VocosFbank-compatible).
Matches k2-fsa/ZipVoice zipvoice/utils/feature.py (torchaudio MelSpectrogram)
via librosa with htk=True, norm=None (RMSE ~0.017 vs torchaudio reference).

Not used for vocoder decode (ONNX mag/x/y + librosa ISTFT in vocos_istft.py).
"""
from __future__ import annotations

import math

import librosa
import numpy as np


class VocosFbank:
    SAMPLING_RATE = 24000
    N_MELS = 100
    N_FFT = 1024
    HOP_LENGTH = 256

    @staticmethod
    def _num_frames(num_samples: int, sampling_rate: int, hop_length: int) -> int:
        """Same frame count as lhotse compute_num_frames (ZipVoice upstream)."""
        frame_shift = hop_length / sampling_rate
        duration = num_samples / sampling_rate
        return int(math.ceil(duration / frame_shift))

    def extract(
        self,
        samples: np.ndarray,
        sampling_rate: int,
    ) -> np.ndarray:
        assert sampling_rate == self.SAMPLING_RATE, (
            f"Expected {self.SAMPLING_RATE} Hz, got {sampling_rate}"
        )
        audio = np.asarray(samples, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.SAMPLING_RATE,
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH,
            n_mels=self.N_MELS,
            center=True,
            power=1.0,
            norm=None,
            htk=True,
        )
        logmel = np.log(np.maximum(mel, 1e-7)).T

        num_frames = self._num_frames(len(audio), sampling_rate, self.HOP_LENGTH)
        if logmel.shape[0] > num_frames:
            logmel = logmel[:num_frames]
        elif logmel.shape[0] < num_frames:
            pad = num_frames - logmel.shape[0]
            logmel = np.pad(logmel, ((0, pad), (0, 0)), mode="edge")
        return logmel.astype(np.float32)
