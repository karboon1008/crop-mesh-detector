"""Exports a lower-opset FP32 ONNX graph specifically for the K210/nncase
path (scripts/export_for_k210.py).

scripts/export_for_pi.py's own ONNX export uses opset 17 (needed for
mobilevit_xxs's attention op), which Kendryte's 2021-era nncase 1.0 ONNX
importer can't parse — it raises "Bad optional access" on an opset-17
Shape-op construct, and onnx's own version_converter can't downgrade an
already-exported graph either (no registered adapter for that Shape
change). Re-exporting directly from PyTorch at a lower opset avoids the
problem entirely.

Only meaningful for CNN-only architectures (mobilenet_v3_small,
efficientnet_lite0) — mobilevit_xxs needs opset>=14 for its attention op
and isn't a realistic K210 target anyway (the KPU doesn't support
self-attention at all).

Run in the MAIN .venv (this needs torch/timm), after training and after
scripts/export_for_pi.py has produced checkpoints/results:
    python scripts/export_onnx_k210.py --arch efficientnet_lite0 --node node_0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model_selection import pick_best_node_for_arch  # noqa: E402
from src.models.factory import build_model  # noqa: E402

K210_OPSET = 11  # predates the opset-17 Shape-op construct nncase 1.0 can't parse


def export(arch: str, node_id: str, checkpoints_dir: Path, pi_export_dir: Path) -> Path:
    classes = json.loads((checkpoints_dir / "classes.json").read_text())
    crop_classes = classes["crop_classes"]
    disease_classes = classes["disease_classes"]
    image_size = classes["image_size"]

    model = build_model(arch, len(crop_classes), len(disease_classes), pretrained=False)
    state_dict = torch.load(checkpoints_dir / arch / f"{node_id}.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    output_path = pi_export_dir / arch / "model_fp32_k210.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.zeros(1, 3, image_size, image_size)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["image"],
        output_names=["crop_logits", "disease_logits"],
        opset_version=K210_OPSET,
        dynamo=False,
    )
    print(f"Wrote {output_path} (opset {K210_OPSET}, for nncase/K210)")
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True)
    parser.add_argument("--node", default=None)
    parser.add_argument("--results", default="outputs/results_summary.json")
    parser.add_argument("--checkpoints-dir", default="outputs/checkpoints")
    parser.add_argument("--pi-export-dir", default="outputs/pi_export")
    args = parser.parse_args()

    node_id = args.node
    if not node_id:
        node_id, score = pick_best_node_for_arch(Path(args.results), args.arch)
        print(f"Auto-selected node={node_id} (avg test accuracy {score:.4f})")

    export(args.arch, node_id, Path(args.checkpoints_dir), Path(args.pi_export_dir))


if __name__ == "__main__":
    main()
