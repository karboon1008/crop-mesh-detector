"""Exports a trained checkpoint to a small, self-contained bundle a
Raspberry Pi can run without the training stack (no torch/torchvision/timm
needed on the Pi — only `onnxruntime`, `numpy`, `Pillow`).

By default picks the (architecture, node) with the highest average
crop/disease test accuracy in outputs/results_summary.json — same selection
`src/predict.py` uses — override with --arch/--node for a specific one
(e.g. if the best-scoring architecture turns out too slow on real hardware).

Run (from the repo root, after `python -m src.train`):
    python scripts/export_for_pi.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.plantvillage import IMAGENET_MEAN, IMAGENET_STD, load_full_dataset
from src.model_selection import pick_best_arch_node
from src.models.factory import build_model


def export_onnx(model: torch.nn.Module, image_size: int, output_path: Path) -> None:
    model.eval()
    dummy_input = torch.zeros(1, 3, image_size, image_size)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["image"],
        output_names=["crop_logits", "disease_logits"],
        opset_version=17,  # mobilevit_xxs's attention op needs >=14
    )


def check_parity(model: torch.nn.Module, session: onnxruntime.InferenceSession, samples) -> int:
    """Compares PyTorch vs. quantized-ONNX top-1 predictions on a few real
    images. Returns the number of mismatches (0 == safe to deploy).
    """
    input_name = session.get_inputs()[0].name
    mismatches = 0
    with torch.no_grad():
        for image, _, _ in samples:
            batch = image.unsqueeze(0)
            torch_crop, torch_disease = model(batch)
            onnx_crop, onnx_disease = session.run(None, {input_name: batch.numpy()})
            if torch_crop.argmax(1).item() != onnx_crop.argmax(1).item():
                mismatches += 1
            elif torch_disease.argmax(1).item() != onnx_disease.argmax(1).item():
                mismatches += 1
    return mismatches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", default="outputs/checkpoints")
    parser.add_argument("--results", default="outputs/results_summary.json")
    parser.add_argument("--arch", default=None, help="Override: which architecture to export")
    parser.add_argument("--node", default=None, help="Override: which node's checkpoint to export, e.g. node_0")
    parser.add_argument("--data-root", default="data/PlantVillage", help="Used only for the parity check, if present")
    parser.add_argument("--num-parity-samples", type=int, default=8)
    parser.add_argument("--output-dir", default="outputs/pi_export")
    args = parser.parse_args()

    checkpoints_dir = Path(args.checkpoints_dir)
    if args.arch and args.node:
        arch, node_id = args.arch, args.node
    else:
        arch, node_id = pick_best_arch_node(Path(args.results))

    classes = json.loads((checkpoints_dir / "classes.json").read_text())
    crop_classes = classes["crop_classes"]
    disease_classes = classes["disease_classes"]
    image_size = classes["image_size"]

    model = build_model(arch, len(crop_classes), len(disease_classes), pretrained=False)
    state_dict = torch.load(checkpoints_dir / arch / f"{node_id}.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = output_dir / "model_fp32.onnx"
    quantized_path = output_dir / "model.onnx"

    print(f"Exporting {arch}/{node_id} to ONNX...")
    export_onnx(model, image_size, fp32_path)

    print("Quantizing to int8...")
    # Dynamic quantization of Conv layers produces ConvInteger ops many onnxruntime
    # builds lack a CPU kernel for. Restrict to the linear heads (MatMul/Gemm) —
    # same scope as factory.py's own quantize_dynamic() helper, which also only
    # quantizes nn.Linear and leaves the conv backbone in fp32.
    quantize_dynamic(
        str(fp32_path), str(quantized_path), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"]
    )
    fp32_path.unlink()

    manifest = {
        "arch": arch,
        "node_id": node_id,
        "crop_classes": crop_classes,
        "disease_classes": disease_classes,
        "image_size": image_size,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    data_root = Path(args.data_root)
    if data_root.exists():
        print(f"Checking PyTorch-vs-ONNX prediction parity on {args.num_parity_samples} samples from {data_root}...")
        dataset = load_full_dataset(data_root, image_size)
        step = max(1, len(dataset) // args.num_parity_samples)
        samples = [dataset[i] for i in range(0, len(dataset), step)][: args.num_parity_samples]
        session = onnxruntime.InferenceSession(str(quantized_path))
        mismatches = check_parity(model, session, samples)
        if mismatches:
            print(
                f"WARNING: {mismatches}/{len(samples)} samples disagree between PyTorch and the "
                f"quantized ONNX model. Consider re-running with --arch/--node to pick a different "
                f"checkpoint, or inspect model_fp32 export for {arch} before deploying."
            )
        else:
            print(f"Parity OK: all {len(samples)} sampled predictions match.")
    else:
        print(f"{data_root} not found locally — skipping parity check (export itself still succeeded).")

    print(f"\nDone. Bundle ready at {output_dir}/ (model.onnx + manifest.json) — copy this whole folder to the Pi.")


if __name__ == "__main__":
    main()
