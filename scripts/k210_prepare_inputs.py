"""Converts a handful of PlantVillage images into raw uint8 input buffers for
the K210 C SDK benchmark (docs/k210_riscv_c_sdk_benchmark.md, Step 2).

Produces plain resized RGB pixel bytes -- no ToTensor/Normalize. Mean/std
normalization is baked into the .kmodel itself at nncase-compile time via
--input-mean/--input-std, so the buffers the firmware feeds kpu_run_kmodel()
must be raw 0-255 pixels, not already-normalized floats.

Usage:
    python scripts/k210_prepare_inputs.py \\
        --manifest outputs/pi_export/mobilenet_v3_small/manifest.json \\
        --num-images 20 --layout hwc --output-dir k210_inputs
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image


def convert_image(path: Path, image_size: int, layout: str) -> bytes:
    # Direct resize to (image_size, image_size), matching
    # src/data/plantvillage.py's transforms.Resize((image_size, image_size))
    # exactly (no aspect-preserving crop).
    img = Image.open(path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)  # HWC
    if layout == "chw":
        arr = arr.transpose(2, 0, 1)  # CHW
    return arr.tobytes()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="outputs/pi_export/mobilenet_v3_small/manifest.json")
    parser.add_argument("--images-dir", default="data/PlantVillage")
    parser.add_argument("--num-images", type=int, default=20)
    parser.add_argument(
        "--layout",
        choices=["hwc", "chw"],
        default="hwc",
        help="hwc matches `ncc compile --input-layout NHWC` (recommended, matches the doc's "
        "compile command); use chw only if you compiled without that flag (ONNX/nncase default).",
    )
    parser.add_argument("--output-dir", default="k210_inputs")
    parser.add_argument("--seed", type=int, default=42, help="Same seed as the doc's calibration set for reproducibility")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    image_size = manifest["image_size"]

    images_dir = Path(args.images_dir)
    all_images = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not all_images:
        raise SystemExit(f"No images found under {images_dir}")

    random.Random(args.seed).shuffle(all_images)
    chosen = all_images[: args.num_images]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    header_lines = ['#pragma once', '#include "incbin.h"', ""]
    array_entries = []
    bytes_per_image = None
    for i, img_path in enumerate(chosen):
        raw = convert_image(img_path, image_size, args.layout)
        bytes_per_image = len(raw)
        bin_path = output_dir / f"input_{i}.bin"
        bin_path.write_bytes(raw)
        header_lines.append(f'INCBIN(input{i}, "{bin_path.name}");')
        array_entries.append(f"input{i}_data")
        print(f"{bin_path}  <-  {img_path}  ({len(raw)} bytes, layout={args.layout})")

    header_lines.append("")
    header_lines.append(f"static const uint8_t *k210_bench_inputs[] = {{ {', '.join(array_entries)} }};")
    header_lines.append(f"static const size_t k210_bench_num_inputs = {len(array_entries)};")
    (output_dir / "inputs_incbin.h").write_text("\n".join(header_lines) + "\n")

    print(f"\nWrote {len(chosen)} raw buffers + inputs_incbin.h to {output_dir}/")
    print(f"image_size={image_size} layout={args.layout} bytes_per_image={bytes_per_image}")
    print("Copy this whole folder next to main.c, then in main.c:")
    print('  #include "inputs_incbin.h"')
    print("and replace the manual INCBIN block + inputs[] array in the benchmark firmware with these generated ones.")


if __name__ == "__main__":
    main()
