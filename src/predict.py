"""Batch inference on a folder of images using a mesh-trained checkpoint
from a completed `python -m src.train` run.

By default, auto-selects whichever (architecture, node) pair scored the
highest average of crop/disease test accuracy in outputs/results_summary.json
— override with --arch/--node to pick a specific one.

If images are laid out one-folder-per-class like PlantVillage itself
(e.g. "Tomato___Bacterial_spot/img1.jpg"), true labels are parsed from the
folder name and an accuracy report is printed alongside the CSV. A flat
folder of images (no class subfolders) just gets predictions, no accuracy.

Run:
    python -m src.predict --folder path/to/images --checkpoints-dir outputs/checkpoints
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.data.plantvillage import IMAGENET_MEAN, IMAGENET_STD, _parse_crop_disease
from src.model_selection import pick_best_arch_node, pick_best_node_for_arch
from src.models.factory import build_model

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class InferenceImage:
    def __init__(self, path: Path, true_crop: str | None, true_disease: str | None):
        self.path = path
        self.true_crop = true_crop
        self.true_disease = true_disease


def discover_images(folder: Path) -> list[InferenceImage]:
    subdirs = [p for p in folder.iterdir() if p.is_dir()]
    images: list[InferenceImage] = []
    if subdirs:
        for subdir in subdirs:
            true_crop, true_disease = _parse_crop_disease(subdir.name)
            for path in sorted(subdir.rglob("*")):
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    images.append(InferenceImage(path, true_crop, true_disease))
    else:
        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(InferenceImage(path, None, None))
    return images


def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class InferenceDataset(Dataset):
    def __init__(self, images: list[InferenceImage], image_size: int):
        self.images = images
        self.transform = build_transform(image_size)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = Image.open(self.images[idx].path).convert("RGB")
        return self.transform(image), idx


def load_model(checkpoints_dir: Path, results_path: Path, arch: str | None, node_id: str | None):
    """Loads a trained checkpoint plus its class lists, auto-selecting the
    best-scoring (arch, node) from results_path if arch/node_id aren't both
    given. Shared by batch folder inference (this module) and interactive
    single-image/camera inference (src/infer.py).
    """
    if arch and not node_id:
        node_id, _ = pick_best_node_for_arch(results_path, arch)
    elif not arch:
        arch, node_id = pick_best_arch_node(results_path)

    classes = json.loads((checkpoints_dir / "classes.json").read_text())
    crop_classes = classes["crop_classes"]
    disease_classes = classes["disease_classes"]
    image_size = classes["image_size"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(arch, len(crop_classes), len(disease_classes), pretrained=False)
    state_dict = torch.load(checkpoints_dir / arch / f"{node_id}.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, crop_classes, disease_classes, image_size, device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="Folder of images to classify")
    parser.add_argument("--checkpoints-dir", default="outputs/checkpoints")
    parser.add_argument("--results", default="outputs/results_summary.json")
    parser.add_argument("--arch", default=None, help="Override: which architecture's checkpoint to use")
    parser.add_argument("--node", default=None, help="Override: which node's checkpoint to use, e.g. node_0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="outputs/predictions.csv")
    args = parser.parse_args()

    checkpoints_dir = Path(args.checkpoints_dir)
    model, crop_classes, disease_classes, image_size, device = load_model(
        checkpoints_dir, Path(args.results), args.arch, args.node
    )

    images = discover_images(Path(args.folder))
    if not images:
        raise ValueError(f"No images found under {args.folder}")
    has_labels = images[0].true_crop is not None
    print(f"Found {len(images)} images{' (labeled by folder name)' if has_labels else ''}.")

    dataset = InferenceDataset(images, image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    rows = []
    correct_crop, correct_disease = 0, 0
    with torch.no_grad():
        for batch_images, batch_idx in loader:
            batch_images = batch_images.to(device)
            crop_logits, disease_logits = model(batch_images)
            crop_probs = F.softmax(crop_logits, dim=1)
            disease_probs = F.softmax(disease_logits, dim=1)
            crop_conf, crop_pred = crop_probs.max(dim=1)
            disease_conf, disease_pred = disease_probs.max(dim=1)

            for i, idx in enumerate(batch_idx.tolist()):
                img = images[idx]
                pred_crop = crop_classes[crop_pred[i].item()]
                pred_disease = disease_classes[disease_pred[i].item()]
                row = {
                    "file": str(img.path),
                    "predicted_crop": pred_crop,
                    "crop_confidence": round(crop_conf[i].item(), 4),
                    "predicted_disease": pred_disease,
                    "disease_confidence": round(disease_conf[i].item(), 4),
                }
                if has_labels:
                    crop_ok = pred_crop == img.true_crop
                    disease_ok = pred_disease == img.true_disease
                    row.update(
                        true_crop=img.true_crop,
                        true_disease=img.true_disease,
                        crop_correct=crop_ok,
                        disease_correct=disease_ok,
                    )
                    correct_crop += int(crop_ok)
                    correct_disease += int(disease_ok)
                rows.append(row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} predictions to {output_path}")
    if has_labels:
        n = len(rows)
        print(f"crop_accuracy: {correct_crop / n:.4f}  disease_accuracy: {correct_disease / n:.4f}")


if __name__ == "__main__":
    main()
