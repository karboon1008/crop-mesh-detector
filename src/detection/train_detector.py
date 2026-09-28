"""Fine-tunes the stage-1 leaf detector from COCO weights on a VOC leaf-box
dataset, and evaluates it (mAP@0.5 etc., src/detection/metrics.py).

Train (e.g. on the PlantDoc object-detection dataset's TRAIN/ and TEST/
folders — every box is treated as "leaf"):
    python -m src.detection.train_detector \
        --train-dir data/PlantDoc-OD/TRAIN --test-dir data/PlantDoc-OD/TEST

Evaluate an existing checkpoint only:
    python -m src.detection.train_detector --eval-only \
        --test-dir data/PlantDoc-OD/TEST --checkpoint outputs/detector/leaf_ssdlite.pt

Writes the best checkpoint (by mAP@0.5 on --test-dir) to --checkpoint and
the per-epoch metrics + compute energy to <checkpoint dir>/detector_metrics.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.detection.dataset import LeafBoxDataset, detection_collate, load_voc_folder
from src.detection.leaf_detector import LEAF_LABEL, build_leaf_detector, load_detector, save_detector
from src.detection.metrics import detection_report
from src.energy.tracker import ComputeEnergyTracker


@torch.no_grad()
def evaluate_detector(model, loader: DataLoader, device: str, score_threshold: float = 0.5) -> dict:
    model.eval()
    predictions, targets = [], []
    for images, batch_targets in loader:
        outputs = model([img.to(device) for img in images])
        for out, target in zip(outputs, batch_targets):
            keep = out["labels"] == LEAF_LABEL
            predictions.append({"boxes": out["boxes"][keep].cpu(), "scores": out["scores"][keep].cpu()})
            targets.append({"boxes": target["boxes"]})
    return detection_report(predictions, targets, score_threshold)


def train_one_epoch(model, loader: DataLoader, optimizer, device: str) -> float:
    model.train()
    total, steps = 0.0, 0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        losses = model(images, targets)  # {"bbox_regression", "classification"}
        loss = sum(losses.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
        steps += 1
    return total / max(steps, 1)


def train_detector(
    train_dir: Path,
    test_dir: Path,
    checkpoint_path: Path,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 0.01,
    weight_decay: float = 4e-5,
    score_threshold: float = 0.5,
    pretrained: bool = True,
    device: str | None = None,
    num_workers: int = 2,
    energy_enabled: bool = True,
) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        LeafBoxDataset(load_voc_folder(train_dir), train=True), batch_size=batch_size, shuffle=True,
        collate_fn=detection_collate, num_workers=num_workers,
    )
    test_loader = DataLoader(
        LeafBoxDataset(load_voc_folder(test_dir)), batch_size=batch_size, shuffle=False,
        collate_fn=detection_collate, num_workers=num_workers,
    )

    model = build_leaf_detector(pretrained=pretrained).to(device)
    # SGD + cosine is torchvision's own ssdlite recipe; lr is scaled down
    # from its from-scratch 0.15 because the backbone starts COCO-trained.
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=lr, momentum=0.9, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    tracker = ComputeEnergyTracker(enabled=energy_enabled, output_dir=checkpoint_path.parent)

    history, best = [], None
    for epoch in range(epochs):
        with tracker.track(f"detector_epoch_{epoch}_train") as train_record:
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        with tracker.track(f"detector_epoch_{epoch}_eval"):
            metrics = evaluate_detector(model, test_loader, device, score_threshold)
        row = {"epoch": epoch, "train_loss": round(train_loss, 4), **metrics,
               "train_energy_kwh": train_record["energy_kwh"]}
        history.append(row)
        print(f"epoch {epoch}: loss {train_loss:.4f}  mAP@0.5 {metrics['map_50']:.4f}  "
              f"P {metrics['precision_at_threshold']:.3f}  R {metrics['recall_at_threshold']:.3f}")
        if best is None or metrics["map_50"] > best["map_50"]:
            best = row
            save_detector(model, checkpoint_path, {"epoch": epoch, "metrics": metrics, "score_threshold": score_threshold})

    report = {
        "checkpoint": str(checkpoint_path),
        "train_dir": str(train_dir),
        "test_dir": str(test_dir),
        "num_train_images": len(train_loader.dataset),
        "num_test_images": len(test_loader.dataset),
        "best": best,
        "history": history,
        "energy": {**tracker.summary(), "method": tracker.measurement_method},
    }
    (checkpoint_path.parent / "detector_metrics.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default=None, help="Folder of images + VOC .xml files")
    parser.add_argument("--test-dir", required=True, help="Folder of images + VOC .xml files")
    parser.add_argument("--checkpoint", default="outputs/detector/leaf_ssdlite.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--score-threshold", type=float, default=0.5,
                        help="Score cut-off used for precision/recall (and later by inference)")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-pretrained", action="store_true", help="Start from random weights instead of COCO")
    parser.add_argument("--no-energy", action="store_true", help="Disable CodeCarbon (wall-clock proxy only)")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate --checkpoint on --test-dir")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if args.eval_only:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_detector(checkpoint_path, device)
        loader = DataLoader(
            LeafBoxDataset(load_voc_folder(Path(args.test_dir))), batch_size=args.batch_size,
            collate_fn=detection_collate, num_workers=args.num_workers,
        )
        print(json.dumps(evaluate_detector(model, loader, device, args.score_threshold), indent=2))
        return

    if not args.train_dir:
        parser.error("--train-dir is required unless --eval-only")
    report = train_detector(
        Path(args.train_dir), Path(args.test_dir), checkpoint_path, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, score_threshold=args.score_threshold,
        pretrained=not args.no_pretrained, num_workers=args.num_workers, energy_enabled=not args.no_energy,
    )
    print(f"Best epoch {report['best']['epoch']}: mAP@0.5 {report['best']['map_50']:.4f} -> {checkpoint_path}")


if __name__ == "__main__":
    main()
