"""Feed one or more images through a stage-1 (local-only) checkpoint from
`python -m src.train_local`, for a quick sanity check before spending
compute on stage 2 (mesh) training.

Auto-selects whichever (architecture, node) pair scored highest average
local-eval crop/disease accuracy in outputs/stage1_local/results_summary.json
— override with --arch/--node to pick a specific one.

Run:
    python -m src.infer_stage1_local --images leaf1.jpg leaf2.jpg
    python -m src.infer_stage1_local --images leaf1.jpg --arch mobilenet_v3_small --node node_3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from src.model_selection import pick_best_arch_node
from src.models.factory import build_model
from src.predict import build_transform


def load_model(stage1_dir: Path, arch: str | None, node_id: str | None):
    if not (arch and node_id):
        arch, node_id = pick_best_arch_node(stage1_dir / "results_summary.json", eval_key="local_eval")

    classes = json.loads((stage1_dir / "classes.json").read_text())
    crop_classes = classes["crop_classes"]
    disease_classes = classes["disease_classes"]
    image_size = classes["image_size"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(arch, len(crop_classes), len(disease_classes), pretrained=False)
    ckpt_path = stage1_dir / "checkpoints" / arch / f"{node_id}.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()
    return model, crop_classes, disease_classes, image_size, device


def classify(model, transform, device, image: Image.Image) -> dict:
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        crop_logits, disease_logits = model(tensor)
    crop_probs = F.softmax(crop_logits, dim=1)[0]
    disease_probs = F.softmax(disease_logits, dim=1)[0]
    crop_conf, crop_idx = crop_probs.max(dim=0)
    disease_conf, disease_idx = disease_probs.max(dim=0)
    return {
        "crop_idx": crop_idx.item(),
        "crop_confidence": round(crop_conf.item(), 4),
        "disease_idx": disease_idx.item(),
        "disease_confidence": round(disease_conf.item(), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", required=True, help="One or more image files to classify")
    parser.add_argument("--stage1-dir", default="outputs/stage1_local", help="Stage-1 output dir")
    parser.add_argument("--arch", default=None, help="Override: which architecture's checkpoint to use")
    parser.add_argument("--node", default=None, help="Override: which node's checkpoint to use, e.g. node_0")
    args = parser.parse_args()

    stage1_dir = Path(args.stage1_dir)
    model, crop_classes, disease_classes, image_size, device = load_model(stage1_dir, args.arch, args.node)
    transform = build_transform(image_size)

    for path in args.images:
        image = Image.open(path)
        raw = classify(model, transform, device, image)
        crop = crop_classes[raw["crop_idx"]]
        disease = disease_classes[raw["disease_idx"]]
        print(
            f"{path} -> crop: {crop} ({raw['crop_confidence']:.2%}), "
            f"disease: {disease} ({raw['disease_confidence']:.2%})"
        )


if __name__ == "__main__":
    main()
