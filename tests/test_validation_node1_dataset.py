from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.plantvillage import PlantVillageDataset
from src.validation.node1_dataset import (
    compute_image_hashes,
    dedup_aware_split,
    get_node1_indices,
    group_duplicates,
)


def test_get_node1_indices_returns_only_the_configured_node1_crop(node1_scoped_config):
    cfg, root = node1_scoped_config
    dataset = PlantVillageDataset(root, image_size=32)

    node1_indices = get_node1_indices(dataset, cfg)

    assert len(node1_indices) == 16  # 8 Potato___Early_blight + 8 Potato___healthy
    for idx in node1_indices:
        _, class_idx = dataset.base.samples[idx]
        crop_idx, _ = dataset.labels.class_to_crop_disease[class_idx]
        assert dataset.labels.crop_classes[crop_idx] == "Potato"


def test_compute_image_hashes_returns_one_hash_per_index(node1_scoped_config):
    cfg, root = node1_scoped_config
    dataset = PlantVillageDataset(root, image_size=32)
    node1_indices = get_node1_indices(dataset, cfg)

    hashes = compute_image_hashes(dataset, node1_indices)

    assert set(hashes.keys()) == set(node1_indices)
    assert all(isinstance(h, int) for h in hashes.values())


def test_group_duplicates_merges_near_identical_hashes_and_separates_distinct_ones():
    hashes = {0: 0b0000_0000, 1: 0b0000_0001, 2: 0b1111_1111, 3: 0b1111_1110}

    groups = group_duplicates([0, 1, 2, 3], hashes, threshold=1)

    assert groups[0] == groups[1]
    assert groups[2] == groups[3]
    assert groups[0] != groups[2]


def test_dedup_aware_split_keeps_duplicate_groups_on_one_side():
    indices = [0, 1, 2, 3, 4, 5]
    # (0, 1) near-duplicates; (2, 3) near-duplicates; 4 and 5 are singletons far from everything.
    hashes = {0: 0, 1: 1, 2: 0b1111_0000, 3: 0b1111_0001, 4: 0b0101_0101, 5: 0b1010_1010}

    train_idx, test_idx = dedup_aware_split(indices, hashes, test_fraction=0.34, seed=0, threshold=1)

    train_set, test_set = set(train_idx), set(test_idx)
    assert train_set.isdisjoint(test_set)
    assert train_set | test_set == set(indices)
    assert (0 in train_set) == (1 in train_set)
    assert (2 in train_set) == (3 in train_set)


from torchvision import transforms

from src.validation.node1_dataset import build_train_eval_datasets, prepare_node1_data


def test_build_train_eval_datasets_augments_train_only(node1_scoped_config):
    cfg, root = node1_scoped_config

    train_ds, eval_ds = build_train_eval_datasets(root, image_size=32)

    train_types = [type(t) for t in train_ds.transform.transforms]
    eval_types = [type(t) for t in eval_ds.transform.transforms]

    assert transforms.RandomHorizontalFlip in train_types
    assert transforms.ColorJitter in train_types
    assert eval_types == [transforms.Resize, transforms.ToTensor, transforms.Normalize]


def test_prepare_node1_data_returns_disjoint_indices_and_full_label_space(node1_scoped_config):
    cfg, root = node1_scoped_config

    train_ds, train_idx, eval_ds, test_idx = prepare_node1_data(cfg)

    assert set(train_idx).isdisjoint(set(test_idx))
    assert len(train_idx) + len(test_idx) == 16
    # both crops are in the label space even though only Potato samples were
    # selected for node_1 -- the model must predict among the full global
    # label set, matching how the real pipeline sizes its heads.
    assert set(eval_ds.labels.crop_classes) == {"Tomato", "Potato"}
