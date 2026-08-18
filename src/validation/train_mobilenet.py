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
