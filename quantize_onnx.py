"""
Build quantized ONNX variants from existing FP32 baseline in models/onnx/.

Usage:
  .venv\\Scripts\\python quantize_onnx.py --mode int4
  .venv\\Scripts\\python quantize_onnx.py --mode int8
"""
from __future__ import annotations

import argparse
import sys

from config import ONNX_DIR
from onnx_quant import (
    QUANT_MODE_CHOICES,
    export_quant_variants,
    format_sizes,
    needed_fp32_baselines,
    onnx_filenames,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantize FP32 ONNX models in models/onnx/")
    parser.add_argument(
        "--mode",
        choices=[m for m in QUANT_MODE_CHOICES if m not in ("fp32", "mixed")],
        default="int4",
        help="Quant mode to build from FP32 baseline (default: int4)",
    )
    parser.add_argument(
        "--keep-fp32",
        action="store_true",
        help="Giữ text_encoder.onnx + fm_decoder.onnx sau khi quant (mặc định: xóa nếu không cần inference)",
    )
    args = parser.parse_args()

    for base in ("text_encoder.onnx", "fm_decoder.onnx"):
        if not (ONNX_DIR / base).is_file():
            print(f"[ERROR] Missing FP32 baseline: {ONNX_DIR / base}")
            print("Export FP32 from PyTorch GUI first, or copy models/onnx/ from export repo.")
            return 1

    keep_fp32 = args.keep_fp32
    print(f"Quantizing mode={args.mode} in {ONNX_DIR} (keep_fp32={keep_fp32}) ...")
    created = export_quant_variants(
        ONNX_DIR,
        args.mode,  # type: ignore[arg-type]
        keep_fp32_baseline=keep_fp32,
    )
    if not keep_fp32:
        needed = needed_fp32_baselines(args.mode)
        if needed:
            print(f"FP32 intermediate removed; kept: {', '.join(sorted(needed))}")
        else:
            print(f"FP32 intermediate removed; shipped {args.mode} only.")
    te, fm = onnx_filenames(args.mode)
    print(f"Created: {', '.join(created)}")
    print(f"Inference files: {format_sizes(ONNX_DIR, (te, fm))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
