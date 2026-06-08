"""
Vocos ONNX decode: ISTFT from magnitude + complex parts (x, y).

Matches ZipVoice-Vietnamese-GUI vocos_export.decode_mag_xy_with_librosa
and charactr/vocos-mel-24khz ISTFT: center=True, hann, hop=256, win=1024, n_fft=1024.

Pure numpy/librosa — no torchaudio at decode time.
"""
from __future__ import annotations

import librosa
import numpy as np

N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024


def vocos_istft(mag: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """(batch, 513, time) mag/x/y -> 1d waveform float32 clipped to [-1, 1]."""
    if mag.ndim == 3:
        mag, x, y = mag[0], x[0], y[0]
    stft = mag * (x + 1j * y)
    wav = librosa.istft(
        stft,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_fft=N_FFT,
        window="hann",
        center=True,
    )
    return np.clip(wav, -1.0, 1.0).astype(np.float32)
