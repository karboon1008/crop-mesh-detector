"""Orchestrates the "one_node_one_dataset" strategy: node_0/1/2 each own one
whole dataset domain (PlantVillage / PlantDoc / PlantWild) instead of a
non-IID shard of a single dataset (see src/data/plantvillage.py's
`partition_nodes` for that older, still-supported strategy family).

The shared/public probe set and the held-out global test set are stratified
by (crop, disease) and drawn from all three datasets, so the cross-node
reference point spans every domain rather than PlantVillage alone. Whatever
is left per dataset, after those two carve-outs, becomes that dataset's own
private train/validation split — train gets the augmentation transform,
validation the plain eval transform, same convention as every other split in
this codebase.

`build_one_node_one_dataset_loaders` returns exactly the shapes
src.train.run_baseline/run_mesh already consume, so nothing in
src/federated/ needs to change — the mesh/node/aggregation layer is already
generic over "however many nodes, whatever loaders".
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import ConcatDataset, DataLoader

from src.config import Config
from src.data import plantdoc, plantwild
from src.data.plantvillage import (
    _stratified_carve_by_group,
    crop_class_weights_from_pairs,
    disease_class_weights_from_pairs,
    load_full_dataset,
    make_subset as pv_make_subset,
    train_test_split_indices,
)

NODE_ORDER = ("plantvillage", "plantdoc", "plantwild")


def _pair_group_key(pair_labels: list[tuple[int, int]], num_disease: int) -> np.ndarray:
    return np.array([c * num_disease + d for c, d in pair_labels])


def _load_datasets(cfg: Config):
    image_size = cfg.get("data.image_size", 160)
    pv = load_full_dataset(cfg.get("data.root"), image_size)

    plantdoc_root = cfg.get("data.plantdoc_root")
    doc = plantdoc.load_plantdoc_dataset(
        plantdoc_root, pv.labels.crop_classes, pv.labels.disease_classes,
        pv.labels.class_to_crop_disease, image_size,
    )
    if doc is None:
        raise FileNotFoundError(
            f"data.non_iid_strategy 'one_node_one_dataset' requires PlantDoc data at "
            f"{plantdoc_root!r} — run 'python scripts/download_plantdoc.py' first, or "
            f"point data.plantdoc_root at your existing copy."
        )

    plantwild_root = cfg.get("data.plantwild_root")
    wild = plantwild.load_plantwild_dataset(
        plantwild_root, pv.labels.crop_classes, pv.labels.disease_classes,
        pv.labels.class_to_crop_disease, image_size,
    )
    if wild is None:
        raise FileNotFoundError(
            f"data.non_iid_strategy 'one_node_one_dataset' requires PlantWild data at "
            f"{plantwild_root!r} — run 'python scripts/download_plantwild.py' first, or "
            f"point data.plantwild_root at your existing copy."
        )

    return {"plantvillage": pv, "plantdoc": doc, "plantwild": wild}


_MAKE_SUBSET_FNS = {
    "plantvillage": pv_make_subset,
    "plantdoc": plantdoc.make_subset,
    "plantwild": plantwild.make_subset,
}


def build_one_node_one_dataset_loaders(cfg: Config):
    """Returns (probe_loader, global_test_loader, node_loaders, crop_classes,
    disease_classes, crop_class_weights, disease_class_weights).

    node_loaders is a list of (train_loader, validation_loader) tuples,
    index-aligned to NODE_ORDER (plantvillage, plantdoc, plantwild) — the
    same [(train, test), ...] shape the by_crop/by_disease/dirichlet/manual
    strategies already produce, so run_baseline/run_mesh need no changes.
    """
    num_nodes = cfg.get("data.num_nodes", 3)
    if num_nodes != len(NODE_ORDER):
        raise ValueError(
            f"non_iid_strategy 'one_node_one_dataset' fixes node count to "
            f"{len(NODE_ORDER)} ({', '.join(NODE_ORDER)}) — data.num_nodes is {num_nodes}"
        )

    datasets = _load_datasets(cfg)
    pv = datasets["plantvillage"]
    crop_classes = pv.labels.crop_classes
    disease_classes = pv.labels.disease_classes
    num_crop, num_disease = len(crop_classes), len(disease_classes)

    seed = cfg.get("data.seed", 42)
    probe_fraction = cfg.get("data.probe_set_fraction", 0.05)
    global_test_fraction = cfg.get("data.global_test_fraction", 0.05)
    large_class_threshold = cfg.get("data.probe_set_large_class_threshold", 200)
    min_samples_small_class = cfg.get("data.probe_set_min_samples_small_class", 8)
    max_fraction_small_class = cfg.get("data.probe_set_max_fraction_small_class", 0.2)
    test_fraction = cfg.get("data.test_fraction", 0.15)
    batch_size = cfg.get("training.batch_size", 32)

    probe_parts, global_test_parts = [], []
    node_loaders = []
    crop_class_weights, disease_class_weights = [], []

    for name in NODE_ORDER:
        dataset = datasets[name]
        make_subset_fn = _MAKE_SUBSET_FNS[name]
        group_key_full = _pair_group_key(dataset.pair_labels, num_disease)
        all_indices = list(range(len(dataset)))

        probe_idx, remaining_idx = _stratified_carve_by_group(
            group_key_full[all_indices], all_indices, probe_fraction, seed,
            large_class_threshold, min_samples_small_class, max_fraction_small_class,
        )
        gtest_idx, node_pool_idx = _stratified_carve_by_group(
            group_key_full[remaining_idx], remaining_idx, global_test_fraction, seed,
            large_class_threshold, min_samples_small_class, max_fraction_small_class,
        )
        probe_parts.append(make_subset_fn(dataset, probe_idx, train=False))
        global_test_parts.append(make_subset_fn(dataset, gtest_idx, train=False))

        train_idx, val_idx = train_test_split_indices(node_pool_idx, test_fraction, seed)
        train_loader = DataLoader(make_subset_fn(dataset, train_idx, train=True), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(make_subset_fn(dataset, val_idx, train=False), batch_size=batch_size, shuffle=False)
        node_loaders.append((train_loader, val_loader))

        train_pairs = [dataset.pair_labels[i] for i in train_idx]
        crop_class_weights.append(
            crop_class_weights_from_pairs(train_pairs, num_crop)
            if cfg.get("training.crop_class_balanced", False) else None
        )
        disease_class_weights.append(
            disease_class_weights_from_pairs(train_pairs, num_disease)
            if cfg.get("training.disease_class_balanced", True) else None
        )

    probe_loader = DataLoader(ConcatDataset(probe_parts), batch_size=batch_size, shuffle=False)
    global_test_loader = DataLoader(ConcatDataset(global_test_parts), batch_size=batch_size, shuffle=False)

    return (
        probe_loader, global_test_loader, node_loaders,
        crop_classes, disease_classes, crop_class_weights, disease_class_weights,
    )
