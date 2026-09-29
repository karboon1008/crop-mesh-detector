"""Fine-tunes the stage-1 leaf detector from COCO weights on a VOC leaf-box
dataset, and evaluates it (mAP@0.5 etc., src/detection/metrics.py).

Train (e.g. on the PlantDoc object-detection dataset's TRAIN/ and TEST/
folders — every box is treated as "leaf"):
    python -m src.detection.train_detector \
        --train-dir data/PlantDoc-OD/TRAIN --test-dir data/PlantDoc-OD/TEST

Teach it what is NOT a leaf (people, faces, rooms): add a folder of photos
with no leaves in them. They need no annotation (scripts/capture_negatives.py
collects them with a webcam). 15% are held out to measure false alarms:
    python -m src.detection.train_detector \
        --train-dir data/PlantDoc-OD/TRAIN --test-dir data/PlantDoc-OD/TEST \
        --negatives-dir data/negatives

Evaluate an existing checkpoint only:
    python -m src.detection.train_detector --eval-only \
        --test-dir data/PlantDoc-OD/TEST --checkpoint outputs/detector/leaf_ssdlite.pt

Writes the best checkpoint to --checkpoint and the per-epoch metrics + compute
energy to <checkpoint dir>/detector_metrics.json. "Best" is mAP@0.5 on --test-dir;
with negatives it is mAP@0.5 x (1 - false-alarm rate on the held-out negatives),
so an epoch that finds leaves well but also boxes faces doesn't win.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

from src.detection.dataset import LeafBoxDataset, detection_collate, load_negative_folder, load_voc_folder
from src.detection.leaf_detector import LEAF_LABEL, build_leaf_detector, load_detector, save_detector
from src.detection.metrics import detection_report, false_alarm_report
from src.energy.tracker import ComputeEnergyTracker


@torch.no_grad()
def _predict(model, loader: DataLoader, device: str) -> tuple[list[dict], list[dict]]:
    model.eval()
    predictions, targets = [], []
    for images, batch_targets in loader:
        outputs = model([img.to(device) for img in images])
        for out, target in zip(outputs, batch_targets):
            keep = out["labels"] == LEAF_LABEL
            predictions.append({"boxes": out["boxes"][keep].cpu(), "scores": out["scores"][keep].cpu()})
            targets.append({"boxes": target["boxes"]})
    return predictions, targets


def evaluate_detector(model, loader: DataLoader, device: str, score_threshold: float = 0.5,
                      negatives_loader: DataLoader | None = None) -> dict:
    report = detection_report(*_predict(model, loader, device), score_threshold)
    if negatives_loader is not None:
        report.update(false_alarm_report(_predict(model, negatives_loader, device)[0], score_threshold))
        report["selection_score"] = round(report["map_50"] * (1 - report["false_alarm_image_rate"]), 4)
    else:
        report["selection_score"] = report["map_50"]
    return report


def split_negatives(negatives_dir: Path | None, negatives_test_dir: Path | None, holdout: float = 0.15, seed: int = 0):
    """(train, test) negative images. Without a separate test folder, a fixed
    random `holdout` share of --negatives-dir is kept out of training for measuring false alarms.
    """
    train = load_negative_folder(negatives_dir) if negatives_dir else []
    if negatives_test_dir:
        return train, load_negative_folder(negatives_test_dir)
    if not train:
        return [], []
    shuffled = train[:]
    random.Random(seed).shuffle(shuffled)
    n_test = max(1, round(len(shuffled) * holdout)) if len(shuffled) > 1 else 0
    return shuffled[n_test:], shuffled[:n_test]


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
    negatives_dir: Path | None = None,
    negatives_test_dir: Path | None = None,
    init_checkpoint: Path | None = None,
) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    negatives_train, negatives_test = split_negatives(negatives_dir, negatives_test_dir)
    train_set = LeafBoxDataset(load_voc_folder(train_dir), train=True)
    if negatives_train:
        train_set = ConcatDataset([train_set, LeafBoxDataset(negatives_train, train=True)])
    # drop_last: a final batch of one image crashes the batch-norm layers in training mode.
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, collate_fn=detection_collate, num_workers=num_workers,
        drop_last=len(train_set) > batch_size,
    )
    negatives_loader = DataLoader(
        LeafBoxDataset(negatives_test), batch_size=batch_size, shuffle=False,
        collate_fn=detection_collate, num_workers=num_workers,
    ) if negatives_test else None
    test_loader = DataLoader(
        LeafBoxDataset(load_voc_folder(test_dir)), batch_size=batch_size, shuffle=False,
        collate_fn=detection_collate, num_workers=num_workers,
    )

    if init_checkpoint:
        # Continue from an already-trained leaf detector (e.g. to add negatives) instead of COCO.
        model = load_detector(init_checkpoint, device)
    else:
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
            metrics = evaluate_detector(model, test_loader, device, score_threshold, negatives_loader)
        row = {"epoch": epoch, "train_loss": round(train_loss, 4), **metrics,
               "train_energy_kwh": train_record["energy_kwh"]}
        history.append(row)
        false_alarms = (f"  false alarms {metrics['false_alarm_image_rate']:.1%} of no-leaf images"
                        if "false_alarm_image_rate" in metrics else "")
        print(f"epoch {epoch}: loss {train_loss:.4f}  mAP@0.5 {metrics['map_50']:.4f}  "
              f"P {metrics['precision_at_threshold']:.3f}  R {metrics['recall_at_threshold']:.3f}{false_alarms}")
        if best is None or metrics["selection_score"] > best["selection_score"]:
            best = row
            save_detector(model, checkpoint_path, {"epoch": epoch, "metrics": metrics, "score_threshold": score_threshold})

    report = {
        "checkpoint": str(checkpoint_path),
        "train_dir": str(train_dir),
        "test_dir": str(test_dir),
        "num_train_images": len(train_loader.dataset),
        "num_train_negatives": len(negatives_train),
        "num_test_negatives": len(negatives_test),
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
    parser.add_argument("--negatives-dir", default=None,
                        help="Photos with NO leaves (people, faces, rooms...), no annotation needed; "
                             "15%% are held out to measure false alarms unless --negatives-test-dir is given")
    parser.add_argument("--negatives-test-dir", default=None, help="Held-out no-leaf photos for the false-alarm rate")
    parser.add_argument("--init-checkpoint", default=None,
                        help="Start from this trained leaf detector (.pt) instead of COCO weights, "
                             "e.g. outputs/detector/leaf_ssdlite.pt, to add negatives in a few epochs")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if args.eval_only:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_detector(checkpoint_path, device)
        loader = DataLoader(
            LeafBoxDataset(load_voc_folder(Path(args.test_dir))), batch_size=args.batch_size,
            collate_fn=detection_collate, num_workers=args.num_workers,
        )
        negatives = args.negatives_test_dir or args.negatives_dir  # eval-only: every negative is unseen
        negatives_loader = DataLoader(
            LeafBoxDataset(load_negative_folder(Path(negatives))), batch_size=args.batch_size,
            collate_fn=detection_collate, num_workers=args.num_workers,
        ) if negatives else None
        print(json.dumps(evaluate_detector(model, loader, device, args.score_threshold, negatives_loader), indent=2))
        return

    if not args.train_dir:
        parser.error("--train-dir is required unless --eval-only")
    report = train_detector(
        Path(args.train_dir), Path(args.test_dir), checkpoint_path, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, score_threshold=args.score_threshold,
        pretrained=not args.no_pretrained, num_workers=args.num_workers, energy_enabled=not args.no_energy,
        negatives_dir=Path(args.negatives_dir) if args.negatives_dir else None,
        negatives_test_dir=Path(args.negatives_test_dir) if args.negatives_test_dir else None,
        init_checkpoint=Path(args.init_checkpoint) if args.init_checkpoint else None,
    )
    print(f"Best epoch {report['best']['epoch']}: mAP@0.5 {report['best']['map_50']:.4f} -> {checkpoint_path}")


if __name__ == "__main__":
    main()
