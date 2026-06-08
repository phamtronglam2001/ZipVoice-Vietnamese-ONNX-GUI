"""
Per-stage ONNX inference profiler.

Run from repo root:
  set PYTHONPATH=src
  python scripts/profile_inference.py [--gpu] [--quant int4] [--chunks 3] [--batch 1]

Prints wall-clock per stage and active ORT execution providers.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from assets_loader import scan_ref_voices  # noqa: E402
from config import ASSETS_DIR  # noqa: E402
from inference_profile import InferenceProfileReport, analyze_providers  # noqa: E402
from onnx_engine import OnnxTTSEngine  # noqa: E402


_SAMPLE_TEXTS = [
    "Đây là câu thử nghiệm đầu tiên cho profiler hiệu năng.",
    "Câu thứ hai dài hơn một chút để kiểm tra batch và biến thiên độ dài chunk.",
    "Câu ngắn.",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile ZipVoice ONNX inference stages")
    p.add_argument("--gpu", action="store_true", help="Enable CUDA/DirectML (ZIPVOICE_ONNX_GPU=1)")
    p.add_argument("--quant", default="", help="int4 | int8 (default: auto)")
    p.add_argument("--chunks", type=int, default=3, help="Number of sample chunks")
    p.add_argument("--batch", type=int, default=1, help="inference_batch_size")
    p.add_argument("--steps", type=int, default=16, help="ODE num_step")
    p.add_argument("--ref-wav", default="", help="Reference WAV path")
    p.add_argument("--ref-text", default="", help="Reference transcript")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.gpu:
        os.environ["ZIPVOICE_ONNX_GPU"] = "1"
    else:
        os.environ["ZIPVOICE_ONNX_GPU"] = "0"

    ref_wav = args.ref_wav
    if not ref_wav:
        voices = scan_ref_voices(ASSETS_DIR)
        if voices:
            ref_wav = voices[0].audio_path
        else:
            print("No ref wav — pass --ref-wav", file=sys.stderr)
            return 1
    ref_text = args.ref_text or "Xin chào, đây là giọng mẫu thử nghiệm."

    quant = args.quant.strip() or None
    engine = OnnxTTSEngine.get(quant_mode=quant, use_gpu=args.gpu)
    engine.prepare_prompt(ref_text, ref_wav)

    texts = _SAMPLE_TEXTS[: max(1, args.chunks)]
    report = InferenceProfileReport(
        quant_mode=engine.quant_mode,
        use_gpu=engine.use_gpu,
    )
    from inference_profile import collect_provider_info

    report.providers = collect_provider_info(engine)
    analyze_providers(report)

    batch_size = max(1, args.batch)
    i = 0
    while i < len(texts):
        batch_texts = texts[i : i + batch_size]
        if len(batch_texts) == 1:
            _wav, timing = engine.generate(
                prompt_text=ref_text,
                prompt_wav=ref_wav,
                text=batch_texts[0],
                num_step=args.steps,
                use_prompt_cache=True,
                profile=True,
            )
            report.add(timing)
        else:
            results = engine.generate_batch(
                prompt_text=ref_text,
                prompt_wav=ref_wav,
                texts=batch_texts,
                num_step=args.steps,
                use_prompt_cache=True,
                profile=True,
            )
            if results.timing is not None:
                report.add(results.timing)
        i += batch_size

    print(report.format_summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
