"""
Mel filterbank for ZipVoice prompt features (VocosFbank-compatible).
NumPy + librosa — no torchaudio.
"""
from __future__ import annotations

import numpy as np
import librosa


class VocosFbank:
    SAMPLING_RATE = 24000
    N_MELS = 100
    N_FFT = 1024
    HOP_LENGTH = 256

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
        )
        logmel = np.log(np.maximum(mel, 1e-7))
        logmel = logmel.T  # (time, n_mels)

        num_frames = len(audio) // self.HOP_LENGTH + 1
        if logmel.shape[0] > num_frames:
            logmel = logmel[:num_frames]
        elif logmel.shape[0] < num_frames:
            pad = num_frames - logmel.shape[0]
            logmel = np.pad(logmel, ((0, pad), (0, 0)), mode="edge")
        return logmel.astype(np.float32)
