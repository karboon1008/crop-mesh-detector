"""What an edge node reads from its own disk.

Each round, a farm's new photos arrive as an inbox manifest,
`<edge_dir>/inbox/<node_id>/round_<NNNN>.json`:

    {
      "batch_idx": 3,
      "node_id": "node_0",
      "train":     [{"path": ".../leaf_001.jpg", "class": "Tomato___Early_blight"}, ...],
      "unlabeled": [{"path": ".../leaf_002.jpg", "hidden_class": null}, ...],
      "test":      [{"path": ".../leaf_003.jpg", "class": "Tomato___healthy"}, ...],
      "early_warning": [{"path": ".../leaf_004.jpg", "class": "Tomato___Late_blight"}, ...]
    }

`train` and `test` carry the labels the farmer gave. `unlabeled` photos are
pseudo-labelled by the node; `hidden_class` is only filled in by the
simulation feeder, to measure pseudo-label accuracy, and is null on a real
farm. `early_warning` (simulation only, optional) lists photos of diseases
on this farm's crops that only other farms have seen, to test before and
after distillation whether the co-op taught the farm a disease before it
arrived. The round's probe images (public, the same for every farm) are listed
by the knowledge server. Class names map to crop/disease indices through the
co-op's shared label space (classes.json), so every farm's model heads line
up.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import default_loader

from src.data.plantvillage import (
    _inverse_frequency_weights,
    build_eval_transform,
    build_strong_transform,
    build_train_transform,
)
from src.data.splits import NodeBatchLoaders


def inbox_path(edge_dir: str | Path, node_id: str, batch_idx: int) -> Path:
    return Path(edge_dir) / "inbox" / node_id / f"round_{batch_idx:04d}.json"


def read_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


@dataclass
class LabelSpace:
    """The co-op's shared class list (classes.json)."""

    crop_classes: list[str]
    disease_classes: list[str]
    name_to_crop_disease: dict[str, tuple[int, int]]
    image_size: int

    @classmethod
    def from_dict(cls, data: dict) -> "LabelSpace":
        return cls(
            crop_classes=list(data["crop_classes"]),
            disease_classes=list(data["disease_classes"]),
            name_to_crop_disease={k: tuple(v) for k, v in data["name_to_crop_disease"].items()},
            image_size=int(data["image_size"]),
        )

    def to_dict(self) -> dict:
        return {
            "crop_classes": self.crop_classes,
            "disease_classes": self.disease_classes,
            "name_to_crop_disease": {k: list(v) for k, v in self.name_to_crop_disease.items()},
            "image_size": self.image_size,
        }

    @property
    def pair_class_names(self) -> list[str]:
        return list(self.name_to_crop_disease)

    @property
    def class_to_crop_disease(self) -> dict[int, tuple[int, int]]:
        return {i: pair for i, pair in enumerate(self.name_to_crop_disease.values())}

    def pair(self, class_name: str | None) -> tuple[int, int]:
        """(crop_idx, disease_idx); (-1, -1) for an unknown label."""
        if class_name is None:
            return -1, -1
        if class_name not in self.name_to_crop_disease:
            raise ValueError(f"Class {class_name!r} is not in the co-op's label space (classes.json)")
        return self.name_to_crop_disease[class_name]


class FileListDataset(Dataset):
    """(image, crop_idx, disease_idx) for a list of (path, class name or None)."""

    def __init__(self, items: list[tuple[str, str | None]], labels: LabelSpace, transform):
        self.items = items
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, class_name = self.items[i]
        crop_idx, disease_idx = self.labels.pair(class_name)
        return self.transform(default_loader(path)), crop_idx, disease_idx

    def class_counts(self) -> tuple[dict[int, int], dict[int, int]]:
        """Per-class counts from the labels alone (no image loading)."""
        crop_counts: dict[int, int] = {}
        disease_counts: dict[int, int] = {}
        for _, class_name in self.items:
            if class_name is None:
                continue
            c, d = self.labels.pair(class_name)
            crop_counts[c] = crop_counts.get(c, 0) + 1
            disease_counts[d] = disease_counts.get(d, 0) + 1
        return crop_counts, disease_counts


class TwoViewFileDataset(Dataset):
    """Unlabelled photos for pseudo-labelling: (plain view, strong view,
    crop_idx, disease_idx) — the indices are -1 unless the simulation knows
    the hidden label.
    """

    def __init__(self, items: list[tuple[str, str | None]], labels: LabelSpace):
        self.items = items
        self.labels = labels
        self.weak = build_eval_transform(labels.image_size)
        self.strong = build_strong_transform(labels.image_size)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, hidden_class = self.items[i]
        image = default_loader(path)
        crop_idx, disease_idx = self.labels.pair(hidden_class)
        return self.weak(image), self.strong(image), crop_idx, disease_idx


def _class_weights(counts: dict[int, int], num_classes: int) -> torch.Tensor:
    tensor = torch.zeros(num_classes)
    for cls, n in counts.items():
        tensor[cls] = n
    return _inverse_frequency_weights(tensor)


def build_node_loaders(cfg, manifest: dict, labels: LabelSpace) -> NodeBatchLoaders | None:
    """Loaders for one farm's round, or None when it got too little to take
    part (fewer than 2 labelled train photos — BatchNorm — or no test photo).
    """
    train = [(e["path"], e["class"]) for e in manifest.get("train", [])]
    test = [(e["path"], e["class"]) for e in manifest.get("test", [])]
    unlabeled = [(e["path"], e.get("hidden_class")) for e in manifest.get("unlabeled", [])]
    if len(train) < 2 or not test:
        return None

    batch_size = cfg.get("training.batch_size", 32)
    train_ds = FileListDataset(train, labels, build_train_transform(labels.image_size))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=len(train) % batch_size == 1)
    test_loader = DataLoader(
        FileListDataset(test, labels, build_eval_transform(labels.image_size)), batch_size=batch_size, shuffle=False,
    )
    unlabeled_loader = None
    # a lone unlabelled photo can't form a batch (BatchNorm), and would leave an empty loader
    if len(unlabeled) >= 2:
        unlabeled_bs = batch_size * cfg.get("continual.unlabeled_batch_ratio", 7)
        unlabeled_loader = DataLoader(
            TwoViewFileDataset(unlabeled, labels), batch_size=unlabeled_bs, shuffle=True,
            drop_last=len(unlabeled) % unlabeled_bs == 1,
        )
    crop_counts, disease_counts = train_ds.class_counts()
    crop_weights = (
        _class_weights(crop_counts, len(labels.crop_classes))
        if cfg.get("training.crop_class_balanced", False) else None
    )
    disease_weights = (
        _class_weights(disease_counts, len(labels.disease_classes))
        if cfg.get("training.disease_class_balanced", True) else None
    )
    early_warning = [(e["path"], e["class"]) for e in manifest.get("early_warning", [])]
    early_warning_loader = DataLoader(
        FileListDataset(early_warning, labels, build_eval_transform(labels.image_size)), batch_size=batch_size,
        shuffle=False,
    ) if early_warning else None
    return NodeBatchLoaders(
        train_loader, test_loader, unlabeled_loader, crop_weights, disease_weights, early_warning_loader,
    )


def build_probe_loader(cfg, probe_paths: list[str], labels: LabelSpace) -> DataLoader:
    # unshuffled: a probe image's position is how its logits line up across farms
    return DataLoader(
        FileListDataset([(p, None) for p in probe_paths], labels, build_eval_transform(labels.image_size)),
        batch_size=cfg.get("training.batch_size", 32), shuffle=False,
    )
