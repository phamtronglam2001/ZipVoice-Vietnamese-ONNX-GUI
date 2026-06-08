"""
Reference audio preprocessing for ZipVoice voice cloning.
Transcript giọng mẫu bắt buộc nhập thủ công — không auto ASR.
"""
from __future__ import annotations

import logging
import tempfile

import numpy as np
from pydub import AudioSegment, silence
from scipy.io import wavfile
from scipy.signal import resample_poly

from config import apply_cpu_env, ensure_ffmpeg_on_path, set_offline_env

ensure_ffmpeg_on_path()
apply_cpu_env()
set_offline_env()

logger = logging.getLogger("zipvoice_gui")


def resample_to_24khz(input_path: str, output_path: str) -> None:
    orig_sr, audio = wavfile.read(input_path)
    if len(audio.shape) == 2:
        audio = audio.mean(axis=1)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    resampled = resample_poly(audio, 24000, orig_sr)
    resampled_int16 = (resampled * 32767).astype(np.int16)
    wavfile.write(output_path, 24000, resampled_int16)


def _remove_silence_edges(audio: AudioSegment, silence_threshold: int = -42) -> AudioSegment:
    non_silent_start = silence.detect_leading_silence(
        audio, silence_threshold=silence_threshold
    )
    audio = audio[non_silent_start:]
    end_ms = audio.duration_seconds
    for ms in reversed(audio):
        if ms.dBFS > silence_threshold:
            break
        end_ms -= 0.001
    return audio[: int(end_ms * 1000)]


def preprocess_ref_audio_text(
    ref_audio_orig: str,
    ref_text: str = "",
    clip_short: bool = True,
    show_info=print,
) -> tuple[str, str]:
    ref_text = ref_text.strip()
    if not ref_text:
        raise ValueError(
            "Bắt buộc nhập transcript giọng mẫu (ô số 2 hoặc trường `text` "
            "trong ref_info.json). App không tự nhận dạng giọng nói."
        )

    show_info("Đang xử lý file giọng mẫu...")
    logger.info("Using manual transcript | len=%d", len(ref_text))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        aseg = AudioSegment.from_file(ref_audio_orig)

        if clip_short:
            non_silent_segs = silence.split_on_silence(
                aseg,
                min_silence_len=1000,
                silence_thresh=-50,
                keep_silence=1000,
                seek_step=10,
            )
            non_silent_wave = AudioSegment.silent(duration=0)
            for seg in non_silent_segs:
                if len(non_silent_wave) > 6000 and len(non_silent_wave + seg) > 15000:
                    show_info("Audio over 15s — clipping (pass 1).")
                    break
                non_silent_wave += seg

            if len(non_silent_wave) > 15000:
                non_silent_segs = silence.split_on_silence(
                    aseg,
                    min_silence_len=100,
                    silence_thresh=-40,
                    keep_silence=1000,
                    seek_step=10,
                )
                non_silent_wave = AudioSegment.silent(duration=0)
                for seg in non_silent_segs:
                    if len(non_silent_wave) > 6000 and len(non_silent_wave + seg) > 15000:
                        show_info("Audio over 15s — clipping (pass 2).")
                        break
                    non_silent_wave += seg
                aseg = non_silent_wave

            if len(aseg) > 15000:
                aseg = aseg[:15000]
                show_info("Audio over 15s — hard clip.")

        aseg = _remove_silence_edges(aseg) + AudioSegment.silent(duration=50)
        aseg.export(f.name, format="wav")
        ref_audio = f.name

    if not ref_text.endswith(". ") and not ref_text.endswith("。"):
        ref_text = ref_text + (". " if not ref_text.endswith(".") else " ")

    return ref_audio, ref_text
