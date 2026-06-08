"""

Compare VocosFbank: librosa (production) vs optional torchaudio legacy reference.



1. RMSE of log-mel features on a reference voice WAV.

2. A/B synthesis WAVs — librosa (default) vs optional torchaudio if installed.



torch/torchaudio are optional — install only for legacy A/B comparison:

  pip install torch torchaudio



Usage (from repo root):

  .venv\\Scripts\\python.exe scripts/compare_fbank_audio.py

  ZIPVOICE_ONNX_GPU=1 .venv\\Scripts\\python.exe scripts/compare_fbank_audio.py

"""

from __future__ import annotations



import math

import sys

from pathlib import Path



import numpy as np

import soundfile as sf



ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:

    sys.path.insert(0, str(ROOT))



from assets_loader import get_voice_by_id, scan_ref_voices

from config import OUTPUT_DIR, models_ready, onnx_ready

from onnx_engine import OnnxTTSEngine, _load_wav_24k

from vocos_fbank import VocosFbank



VOICE_ID = "an_nhi"

GEN_TEXT = (

    "Xin chào, đây là bài kiểm tra so sánh fbank. "

    "Chúng ta nghe thử chất lượng giọng nói."

)

OUT_TORCH = OUTPUT_DIR / "compare_fbank_torch.wav"

OUT_LIBROSA = OUTPUT_DIR / "compare_fbank_librosa.wav"



SYNTH_KWARGS = dict(

    speed=1.0,

    num_step=16,

    guidance_scale=1.0,

    t_shift=0.5,

)



_TORCH_MEL = None





def _num_frames(num_samples: int, sampling_rate: int, hop_length: int) -> int:

    frame_shift = hop_length / sampling_rate

    duration = num_samples / sampling_rate

    return int(math.ceil(duration / frame_shift))





def _align_mel(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    n = min(a.shape[0], b.shape[0])

    return a[:n], b[:n]





def rmse(a: np.ndarray, b: np.ndarray) -> float:

    aa, bb = _align_mel(a, b)

    return float(np.sqrt(np.mean((aa - bb) ** 2)))





def extract_librosa(fbank: VocosFbank, audio: np.ndarray, sr: int) -> np.ndarray:

    return fbank.extract(audio, sr)





def extract_torch_legacy(audio: np.ndarray, sampling_rate: int) -> np.ndarray | None:

    """Optional torchaudio reference (legacy compare only). Returns None if torch missing."""

    global _TORCH_MEL

    try:

        import torch

        import torchaudio

    except ImportError:

        return None



    if _TORCH_MEL is None:

        _TORCH_MEL = torchaudio.transforms.MelSpectrogram(

            sample_rate=VocosFbank.SAMPLING_RATE,

            n_fft=VocosFbank.N_FFT,

            hop_length=VocosFbank.HOP_LENGTH,

            n_mels=VocosFbank.N_MELS,

            center=True,

            power=1,

        )



    samples = torch.from_numpy(audio).unsqueeze(0)

    mel = _TORCH_MEL(samples).clamp(min=1e-7).log()

    logmel = mel.reshape(-1, mel.shape[-1]).t()



    num_frames = _num_frames(len(audio), sampling_rate, VocosFbank.HOP_LENGTH)

    if logmel.shape[0] > num_frames:

        logmel = logmel[:num_frames]

    elif logmel.shape[0] < num_frames:

        logmel = torch.nn.functional.pad(

            logmel.unsqueeze(0),

            (0, 0, 0, num_frames - logmel.shape[0]),

            mode="replicate",

        ).squeeze(0)



    return logmel.cpu().numpy().astype(np.float32)





def extract_librosa_slaney(audio: np.ndarray) -> np.ndarray:

    import librosa



    mel = librosa.feature.melspectrogram(

        y=audio,

        sr=VocosFbank.SAMPLING_RATE,

        n_fft=VocosFbank.N_FFT,

        hop_length=VocosFbank.HOP_LENGTH,

        n_mels=VocosFbank.N_MELS,

        center=True,

        power=1.0,

    )

    logmel = np.log(np.maximum(mel, 1e-7)).T

    num_frames = _num_frames(len(audio), VocosFbank.SAMPLING_RATE, VocosFbank.HOP_LENGTH)

    if logmel.shape[0] > num_frames:

        logmel = logmel[:num_frames]

    elif logmel.shape[0] < num_frames:

        pad = num_frames - logmel.shape[0]

        logmel = np.pad(logmel, ((0, pad), (0, 0)), mode="edge")

    return logmel.astype(np.float32)





def compare_mel_features(ref_wav: Path) -> dict[str, float | None]:

    audio = _load_wav_24k(str(ref_wav), VocosFbank.SAMPLING_RATE)

    fbank = VocosFbank()

    sr = VocosFbank.SAMPLING_RATE



    librosa_htk = extract_librosa(fbank, audio, sr)

    librosa_slaney = extract_librosa_slaney(audio)

    torch_mel = extract_torch_legacy(audio, sr)



    stats: dict[str, float | None] = {

        "torch_vs_librosa_htk": None,

        "torch_vs_librosa_slaney": None,

        "frames_torch": float(torch_mel.shape[0]) if torch_mel is not None else None,

        "frames_librosa_htk": float(librosa_htk.shape[0]),

        "frames_librosa_slaney": float(librosa_slaney.shape[0]),

    }

    if torch_mel is not None:

        stats["torch_vs_librosa_htk"] = rmse(torch_mel, librosa_htk)

        stats["torch_vs_librosa_slaney"] = rmse(torch_mel, librosa_slaney)

    return stats





def _patch_fbank_backend(engine: OnnxTTSEngine, backend: str) -> None:

    fbank = engine.feature_extractor



    if backend == "torch":



        def extract(samples, sampling_rate):

            audio = np.asarray(samples, dtype=np.float32)

            if audio.ndim > 1:

                audio = audio.mean(axis=-1)

            mel = extract_torch_legacy(audio, sampling_rate)

            if mel is None:

                raise ImportError(

                    "torch/torchaudio not installed — pip install torch torchaudio for legacy compare"

                )

            return mel



    elif backend == "librosa":



        def extract(samples, sampling_rate):

            return fbank.extract(samples, sampling_rate)



    else:

        raise ValueError(f"Unknown backend: {backend}")



    fbank.extract = extract  # type: ignore[method-assign]





def synthesize_with_backend(

    engine: OnnxTTSEngine,

    backend: str,

    prompt_text: str,

    prompt_wav: str,

    text: str,

) -> np.ndarray:

    _patch_fbank_backend(engine, backend)

    return engine.generate(

        prompt_text=prompt_text,

        prompt_wav=prompt_wav,

        text=text,

        **SYNTH_KWARGS,

    )





def save_wav(path: Path, audio: np.ndarray, sample_rate: int = 24000) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    audio = np.asarray(audio, dtype=np.float32)

    if audio.size == 0:

        raise RuntimeError(f"No audio generated for {path.name}")

    sf.write(str(path), audio, sample_rate, subtype="PCM_16")





def main() -> int:

    if not models_ready() or not onnx_ready():

        print("[ERROR] Models not ready. Run install_cpu.bat first.")

        return 1



    voice = get_voice_by_id(VOICE_ID, scan_ref_voices())

    if voice is None:

        print(f"[ERROR] Voice not found: {VOICE_ID}")

        return 1



    ref_wav = Path(voice.audio_path)

    ref_text = voice.transcript

    print(f"Reference voice: {voice.id} ({ref_wav.name})")

    print(f"Gen text ({len(GEN_TEXT)} chars): {GEN_TEXT!r}")

    print()



    print("=== Mel feature RMSE (production = librosa htk=True, norm=None) ===")

    mel_stats = compare_mel_features(ref_wav)

    has_torch = mel_stats["torch_vs_librosa_htk"] is not None

    if has_torch:

        print(

            f"  torch vs librosa (htk=True, norm=None):  "

            f"RMSE = {mel_stats['torch_vs_librosa_htk']:.6f}"

        )

        print(

            f"  torch vs librosa (Slaney default):       "

            f"RMSE = {mel_stats['torch_vs_librosa_slaney']:.6f}"

        )

        print(

            f"  frames: torch={int(mel_stats['frames_torch'])}, "

            f"htk={int(mel_stats['frames_librosa_htk'])}, "

            f"slaney={int(mel_stats['frames_librosa_slaney'])}"

        )

    else:

        print("  torch/torchaudio not installed — skipping torch reference RMSE.")

        print(

            f"  frames: librosa htk={int(mel_stats['frames_librosa_htk'])}, "

            f"slaney={int(mel_stats['frames_librosa_slaney'])}"

        )

    print()



    print("=== Synthesis A/B (int8 ONNX, same settings) ===")

    OnnxTTSEngine._instance = None

    engine = OnnxTTSEngine.get(quant_mode="int8")

    device = "GPU" if engine.use_gpu else "CPU"

    print(f"Engine: quant=int8, device={device}")

    print()



    print("Synthesizing with librosa fbank (htk=True, norm=None)...")

    wav_librosa = synthesize_with_backend(

        engine, "librosa", ref_text, str(ref_wav), GEN_TEXT

    )

    save_wav(OUT_LIBROSA, wav_librosa)

    print(f"  Saved: {OUT_LIBROSA} ({len(wav_librosa) / 24000:.2f}s)")



    if has_torch:

        print("Synthesizing with torchaudio fbank (legacy reference)...")

        wav_torch = synthesize_with_backend(

            engine, "torch", ref_text, str(ref_wav), GEN_TEXT

        )

        save_wav(OUT_TORCH, wav_torch)

        print(f"  Saved: {OUT_TORCH} ({len(wav_torch) / 24000:.2f}s)")

    else:

        print("Skipping torchaudio synthesis (torch not installed).")

    print()



    print("=== Summary ===")

    if has_torch:

        htk_rmse = mel_stats["torch_vs_librosa_htk"]

        slaney_rmse = mel_stats["torch_vs_librosa_slaney"]

        print(f"RMSE torch vs librosa HTK:    {htk_rmse:.6f}")

        print(f"RMSE torch vs librosa Slaney: {slaney_rmse:.6f}")

        print(f"Listen A (torch):    {OUT_TORCH.resolve()}")

        print(f"Listen B (librosa):  {OUT_LIBROSA.resolve()}")

        print()

        if htk_rmse is not None and htk_rmse < 0.05:

            print(

                "Conclusion: librosa (htk=True, norm=None) closely matches torchaudio "

                f"(RMSE ~{htk_rmse:.3f}). Production uses librosa only."

            )

        else:

            print(

                f"Conclusion: librosa HTK RMSE ({htk_rmse:.3f}) is higher than expected; "

                "listen to A/B WAVs before relying on librosa."

            )

        if slaney_rmse is not None and slaney_rmse > 1.0:

            print(

                f"Slaney default RMSE (~{slaney_rmse:.1f}) confirms Slaney must NOT be used "

                "— use htk=True, norm=None only."

            )

    else:

        print(f"Librosa output: {OUT_LIBROSA.resolve()}")

        print(

            "Install torch+torchaudio for optional legacy A/B comparison against torchaudio."

        )

    return 0





if __name__ == "__main__":

    raise SystemExit(main())


