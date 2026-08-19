"""Exports trained checkpoint(s) to small, self-contained bundle(s) a
Raspberry Pi can run without the training stack (no torch/torchvision/timm
needed on the Pi — only `onnxruntime`, `numpy`, `Pillow`).

Single-model mode (default) picks the (architecture, node) with the highest
average crop/disease test accuracy in outputs/results_summary.json — same
selection `src/predict.py` uses — override with --arch/--node for a specific
one (e.g. if the best-scoring architecture turns out too slow on real
hardware):
    python scripts/export_for_pi.py
    python scripts/export_for_pi.py --arch mobilenet_v3_small --node node_0

--all mode exports every architecture present in results_summary.json (each
using its own best-scoring node) into its own subfolder, so you end up with
all trained models ready to deploy and can pick between them after comparing
accuracy/latency on the actual Pi:
    python scripts/export_for_pi.py --all

To export from a stage-1-only run (`python -m src.train_local`, before any
mesh training) instead, point at its output dir with --stage1-dir — node
selection then scores by local_eval rather than mesh_eval:
    python scripts/export_for_pi.py --stage1-dir outputs/stage1_local
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import onnxruntime
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.plantvillage import IMAGENET_MEAN, IMAGENET_STD, load_full_dataset
from src.model_selection import list_architectures, pick_best_arch_node, pick_best_node_for_arch
from src.models.factory import build_model


def export_onnx(model: torch.nn.Module, image_size: int, output_path: Path) -> None:
    model.eval()
    dummy_input = torch.zeros(1, 3, image_size, image_size)
    export_kwargs = dict(
        input_names=["image"],
        output_names=["crop_logits", "disease_logits"],
        opset_version=17,  # mobilevit_xxs's attention op needs >=14
    )
    try:
        # Recent torch defaults to the newer torch.export-based ("dynamo") ONNX
        # exporter, which has real bugs with this model's shape inference
        # (mismatched dims during quantization) and an opset-downgrade path
        # that silently fails. Force the older TorchScript-based exporter,
        # which is what was actually validated against all 3 architectures.
        torch.onnx.export(model, dummy_input, str(output_path), dynamo=False, **export_kwargs)
    except TypeError:
        # Older torch (pre-dynamo-exporter) doesn't have this kwarg at all —
        # it only has the legacy exporter anyway, so just call without it.
        torch.onnx.export(model, dummy_input, str(output_path), **export_kwargs)


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


def export_one(
    arch: str,
    node_id: str,
    checkpoints_dir: Path,
    classes_path: Path,
    output_dir: Path,
    data_root: Path,
    num_parity_samples: int,
) -> None:
    classes = json.loads(classes_path.read_text())
    crop_classes = classes["crop_classes"]
    disease_classes = classes["disease_classes"]
    image_size = classes["image_size"]

    model = build_model(arch, len(crop_classes), len(disease_classes), pretrained=False)
    state_dict = torch.load(checkpoints_dir / arch / f"{node_id}.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

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

    if data_root.exists():
        print(f"Checking PyTorch-vs-ONNX prediction parity on {num_parity_samples} samples from {data_root}...")
        dataset = load_full_dataset(data_root, image_size)
        step = max(1, len(dataset) // num_parity_samples)
        samples = [dataset[i] for i in range(0, len(dataset), step)][:num_parity_samples]
        session = onnxruntime.InferenceSession(str(quantized_path))
        mismatches = check_parity(model, session, samples)
        if mismatches:
            print(
                f"WARNING: {mismatches}/{len(samples)} samples disagree between PyTorch and the "
                f"quantized ONNX model. Consider a different checkpoint for {arch}, or inspect the "
                f"fp32 export before deploying."
            )
        else:
            print(f"Parity OK: all {len(samples)} sampled predictions match.")
    else:
        print(f"{data_root} not found locally — skipping parity check (export itself still succeeded).")

    print(f"Bundle ready at {output_dir}/ (model.onnx + manifest.json).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", default="outputs/checkpoints")
    parser.add_argument("--results", default=None)
    parser.add_argument(
        "--stage1-dir", default=None,
        help="Export from a `python -m src.train_local` (stage 1, local-only) run instead of a stage-2 mesh "
        "run — e.g. --stage1-dir outputs/stage1_local. Overrides --checkpoints-dir/--results and scores nodes "
        "by local_eval instead of mesh_eval.",
    )
    parser.add_argument("--arch", default=None, help="Override: which architecture to export")
    parser.add_argument("--node", default=None, help="Override: which node's checkpoint to export, e.g. node_0")
    parser.add_argument(
        "--all", action="store_true", help="Export every architecture in results_summary.json, each to its own subfolder"
    )
    parser.add_argument("--data-root", default="data/PlantVillage", help="Used only for the parity check, if present")
    parser.add_argument("--num-parity-samples", type=int, default=8)
    parser.add_argument("--output-dir", default="outputs/pi_export")
    args = parser.parse_args()

    if args.stage1_dir:
        stage1_dir = Path(args.stage1_dir)
        checkpoints_dir = stage1_dir / "checkpoints"
        classes_path = stage1_dir / "classes.json"
        results_path = Path(args.results) if args.results else stage1_dir / "results_summary.json"
        eval_key = "local_eval"
    else:
        checkpoints_dir = Path(args.checkpoints_dir)
        classes_path = checkpoints_dir / "classes.json"
        results_path = Path(args.results) if args.results else Path("outputs/results_summary.json")
        eval_key = "mesh_eval"
    output_dir = Path(args.output_dir)
    data_root = Path(args.data_root)

    if args.all:
        archs = list_architectures(results_path)
        print(f"Exporting all {len(archs)} architecture(s): {archs}")
        summary = []
        for arch in archs:
            node_id, score = pick_best_node_for_arch(results_path, arch, eval_key=eval_key)
            print(f"\n=== {arch} (best node: {node_id}, avg test accuracy {score:.4f}) ===")
            export_one(arch, node_id, checkpoints_dir, classes_path, output_dir / arch, data_root, args.num_parity_samples)
            summary.append((arch, node_id, score, output_dir / arch))
        print("\nAll exports done:")
        for arch, node_id, score, path in summary:
            print(f"  {arch:20s} node={node_id:8s} avg_accuracy={score:.4f}  ->  {path}/")
        print(f"\nCopy whichever bundle(s) you want to the Pi — see docs/raspberry_pi_deployment.md.")
        return

    if args.arch and args.node:
        arch, node_id = args.arch, args.node
    else:
        arch, node_id = pick_best_arch_node(results_path, eval_key=eval_key)
    export_one(arch, node_id, checkpoints_dir, classes_path, output_dir, data_root, args.num_parity_samples)
    print(f"\nDone. Copy the whole {output_dir}/ folder to the Pi.")


if __name__ == "__main__":
    main()
