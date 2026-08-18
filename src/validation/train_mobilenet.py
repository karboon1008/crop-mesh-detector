"""Trains mobilenet_v3_small on node_1's dedup-aware split (see
node1_dataset.py) with an improved recipe over config.yaml's current
defaults: more epochs, weight decay, a cosine LR schedule, and disease-head
class weighting -- see docs/superpowers/specs/2026-08-18-
node1-validation-pipeline-design.md for why each of these was added.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.data.plantvillage import PlantVillageDataset
from src.models.factory import build_model


def disease_labels_for_indices(dataset: PlantVillageDataset, indices: list[int]) -> list[int]:
    return [dataset.labels.class_to_crop_disease[dataset.base.targets[idx]][1] for idx in indices]


def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights, rescaled so the mean weight across
    *present* classes is 1.0 (keeps the loss magnitude comparable to
    unweighted cross-entropy). Classes absent from `labels` get weight 0 --
    node_1 only ever has samples for the diseases of its own crops, but the
    disease head still spans the full global disease label space.
    """
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for label in labels:
        counts[label] += 1
    weights = torch.zeros(num_classes, dtype=torch.float32)
    present = counts > 0
    weights[present] = 1.0 / counts[present]
    if present.any():
        weights[present] = weights[present] * (present.sum() / weights[present].sum())
    return weights


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    disease_class_weights: torch.Tensor,
    device: str,
) -> float:
    model.train()
    total_loss, total_batches = 0.0, 0
    weights = disease_class_weights.to(device)
    for images, crop_labels, disease_labels in loader:
        images = images.to(device)
        crop_labels = crop_labels.to(device)
        disease_labels = disease_labels.to(device)

        optimizer.zero_grad()
        crop_logits, disease_logits = model(images)
        loss = F.cross_entropy(crop_logits, crop_labels) + F.cross_entropy(
            disease_logits, disease_labels, weight=weights
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_batches += 1
    return total_loss / max(1, total_batches)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    correct_crop, correct_disease, total = 0, 0, 0
    for images, crop_labels, disease_labels in loader:
        images = images.to(device)
        crop_labels = crop_labels.to(device)
        disease_labels = disease_labels.to(device)
        crop_logits, disease_logits = model(images)
        correct_crop += (crop_logits.argmax(dim=1) == crop_labels).sum().item()
        correct_disease += (disease_logits.argmax(dim=1) == disease_labels).sum().item()
        total += images.shape[0]
    total = max(1, total)
    return {"crop_accuracy": correct_crop / total, "disease_accuracy": correct_disease / total}


def run_training(
    train_ds: PlantVillageDataset,
    train_idx: list[int],
    eval_ds: PlantVillageDataset,
    test_idx: list[int],
    num_crop_classes: int,
    num_disease_classes: int,
    output_dir: Path,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    pretrained: bool = True,
    device: str = "cpu",
) -> dict:
    disease_labels = disease_labels_for_indices(train_ds, train_idx)
    class_weights = compute_class_weights(disease_labels, num_disease_classes)

    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=batch_size, shuffle=True, drop_last=True)
    eval_loader = DataLoader(Subset(eval_ds, test_idx), batch_size=batch_size, shuffle=False)

    model = build_model("mobilenet_v3_small", num_crop_classes, num_disease_classes, pretrained=pretrained).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    output_dir.mkdir(parents=True, exist_ok=True)
    best_disease_accuracy = -1.0
    log_entries = []
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, class_weights, device)
        scheduler.step()
        eval_metrics = evaluate(model, eval_loader, device)
        log_entries.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_crop_accuracy": eval_metrics["crop_accuracy"],
                "test_disease_accuracy": eval_metrics["disease_accuracy"],
            }
        )
        if eval_metrics["disease_accuracy"] > best_disease_accuracy:
            best_disease_accuracy = eval_metrics["disease_accuracy"]
            torch.save(model.state_dict(), output_dir / "checkpoint.pt")

    (output_dir / "training_log.json").write_text(json.dumps(log_entries, indent=2))
    return {"log_entries": log_entries, "best_disease_accuracy": best_disease_accuracy}
