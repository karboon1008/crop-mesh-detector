"""Combines PlantVillage (lab photos) and PlantDoc (real-world photos) into
ONE dataset over their shared crop+disease classes, at a fixed 1:1 domain
ratio per class: every class contributes min(pv_count, pd_count) images
from each side — in practice pd_count, since PlantDoc is always the
smaller side — so the model sees an already domain-balanced dataset by
construction, rather than PlantDoc being a secondary pool mixed into
training batches at a fixed ratio (the old approach).

Exposes the same (image, crop_idx, disease_idx) / `.base` / `.labels` /
`.targets` shape as PlantVillageDataset, so it drops straight into the
existing probe/global-test carve, node partitioning, and train/val split
pipeline (carve_public_probe_set, carve_global_test_set, partition_nodes,
train_test_split_indices, make_subset, ...) unchanged — including that
those eval splits stay unaugmented and only the train split gets
build_train_transform, exactly as for a PlantVillage-only dataset.
"""

from __future__ import annotations
import random
from pathlib import Path

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

from src.data.plantdoc import PLANTDOC_TO_PLANTVILLAGE, overlapping_plantvillage_classes
from src.data.plantvillage import (
    PlantVillageDataset,
    _FilteredImageFolder,
    build_eval_transform,
    build_train_transform,
)


class _MergedImageSource:
    """`.base` for MergedPlantDataset: the minimal ImageFolder-like
    interface (`.classes`, `.targets`, `__len__`, `__getitem__`) that every
    downstream PlantVillageDataset helper relies on.
    """

    def __init__(self, pv_root: str, pd_root: str, seed: int):
        classes = sorted(overlapping_plantvillage_classes())
        self.classes = classes
        class_to_idx = {c: i for i, c in enumerate(classes)}

        pv_folder = _FilteredImageFolder(pv_root, allowed_classes=set(classes))
        pv_by_class: dict[int, list[str]] = {i: [] for i in range(len(classes))}
        for path, pv_cls_idx in pv_folder.samples:
            pv_by_class[class_to_idx[pv_folder.classes[pv_cls_idx]]].append(path)

        pd_folder = ImageFolder(pd_root)
        pd_by_class: dict[int, list[str]] = {i: [] for i in range(len(classes))}
        for path, raw_idx in pd_folder.samples:
            mapped = PLANTDOC_TO_PLANTVILLAGE.get(pd_folder.classes[raw_idx])
            if mapped is None:
                continue  # e.g. the spider-mites folder, dropped from PLANTDOC_TO_PLANTVILLAGE
            crop, disease = mapped
            pd_by_class[class_to_idx[f"{crop}___{disease}"]].append(path)

        rng = random.Random(seed)
        samples: list[tuple[str, int]] = []
        for cls_idx, cls_name in enumerate(classes):
            pd_paths = pd_by_class[cls_idx]
            pv_paths = pv_by_class[cls_idx]
            if not pd_paths:
                raise ValueError(f"MergedPlantDataset: class {cls_name!r} has zero PlantDoc images")
            n = min(len(pd_paths), len(pv_paths))
            pv_sample = pv_paths.copy()
            rng.shuffle(pv_sample)
            pd_sample = pd_paths.copy()
            rng.shuffle(pd_sample)
            samples.extend((p, cls_idx) for p in pv_sample[:n])
            samples.extend((p, cls_idx) for p in pd_sample[:n])
        rng.shuffle(samples)

        self.samples = samples
        self.targets = [c for _, c in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, cls_idx = self.samples[idx]
        return default_loader(path), cls_idx


class MergedPlantDataset(Dataset):
    """PlantVillage + PlantDoc combined 1:1 per class. Same public shape as
    PlantVillageDataset (image, crop_idx, disease_idx via `.labels`), so
    every existing carve/partition/split helper works on it unchanged.
    """

    def __init__(self, pv_root: str | Path, pd_root: str | Path, image_size: int = 160, seed: int = 42):
        self.transform = build_eval_transform(image_size)
        self.train_transform = build_train_transform(image_size)
        self.base = _MergedImageSource(str(pv_root), str(pd_root), seed)
        self.labels = PlantVillageDataset._build_label_maps(self.base.classes)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        image, class_idx = self.base[idx]
        image = self.transform(image)
        crop_idx, disease_idx = self.labels.class_to_crop_disease[class_idx]
        return image, crop_idx, disease_idx

    @property
    def targets(self) -> list[int]:
        return self.base.targets


def load_merged_dataset(
    pv_root: str | Path, pd_root: str | Path, image_size: int = 160, seed: int = 42
) -> MergedPlantDataset:
    pv_root, pd_root = Path(pv_root), Path(pd_root)
    if not pv_root.exists():
        raise FileNotFoundError(
            f"PlantVillage data not found at {pv_root}. Run "
            f"'python scripts/download_plantvillage.py' first, or point "
            f"config.yaml's data.root at your existing copy."
        )
    if not pd_root.exists():
        raise FileNotFoundError(
            f"PlantDoc data not found at {pd_root}. Run "
            f"'python scripts/download_plantdoc.py' first, or point "
            f"config.yaml's data.plantdoc_root at your existing copy."
        )
    return MergedPlantDataset(pv_root, pd_root, image_size=image_size, seed=seed)
