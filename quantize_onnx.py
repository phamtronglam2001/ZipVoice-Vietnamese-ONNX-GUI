"""
Build quantized ONNX variants from unquantized export in models/onnx/.

Requires text_encoder.onnx + fm_decoder.onnx as build input (from PyTorch GUI export).
These baseline files are removed by default after quant — only int4/int8 are kept.

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
    baseline_filename,
    export_quant_variants,
    format_sizes,
    onnx_filenames,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quantize unquantized ONNX exports in models/onnx/ to int4 or int8"
    )
    parser.add_argument(
        "--mode",
        choices=list(QUANT_MODE_CHOICES),
        default="int8",
        help="Quant mode to build (default: int8)",
    )
    parser.add_argument(
        "--keep-baseline",
        action="store_true",
        help="Giữ text_encoder.onnx + fm_decoder.onnx sau khi quant (mặc định: xóa)",
    )
    args = parser.parse_args()

    for comp in ("text_encoder", "fm_decoder"):
        base = ONNX_DIR / baseline_filename(comp)
        if not base.is_file():
            print(f"[ERROR] Missing baseline export: {base}")
            print("Export from PyTorch GUI first, or copy models/onnx/ from export repo.")
            return 1

    keep_baseline = args.keep_baseline
    print(f"Quantizing mode={args.mode} in {ONNX_DIR} (keep_baseline={keep_baseline}) ...")
    created = export_quant_variants(
        ONNX_DIR,
        args.mode,  # type: ignore[arg-type]
        keep_baseline=keep_baseline,
    )
    if not keep_baseline:
        print(f"Baseline exports removed; shipped {args.mode} only.")
    te, fm = onnx_filenames(args.mode)
    print(f"Created: {', '.join(created)}")
    print(f"Inference files: {format_sizes(ONNX_DIR, (te, fm))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
