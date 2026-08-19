from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.plantvillage import PlantVillageDataset
from src.validation.corn_mesh_dataset import (
    CORN_DISEASE_ORDER,
    build_corn_label_map,
    get_corn_disease_indices,
    get_corn_indices,
    split_healthy_3way,
)


def test_build_corn_label_map_has_one_crop_and_four_diseases():
    label_map = build_corn_label_map()

    assert label_map.crop_classes == ["Corn"]
    assert label_map.disease_classes == CORN_DISEASE_ORDER
    assert label_map.name_to_disease_idx["healthy"] == 0
    assert label_map.name_to_disease_idx["Common_rust"] == 1


def test_get_corn_indices_excludes_other_crops(corn_scoped_config):
    cfg, root = corn_scoped_config
    dataset = PlantVillageDataset(root, image_size=32)

    corn_indices = get_corn_indices(dataset)

    assert len(corn_indices) == 12 + 6 + 6 + 6  # healthy + 3 diseases, no Tomato
    for idx in corn_indices:
        crop_idx, _ = dataset.labels.class_to_crop_disease[dataset.base.targets[idx]]
        assert dataset.labels.crop_classes[crop_idx] == "Corn"


def test_get_corn_disease_indices_filters_to_named_disease(corn_scoped_config):
    cfg, root = corn_scoped_config
    dataset = PlantVillageDataset(root, image_size=32)
    corn_indices = get_corn_indices(dataset)

    rust_indices = get_corn_disease_indices(dataset, corn_indices, "Common_rust")
    healthy_indices = get_corn_disease_indices(dataset, corn_indices, "healthy")

    assert len(rust_indices) == 6
    assert len(healthy_indices) == 12
    assert set(rust_indices).isdisjoint(set(healthy_indices))


def test_split_healthy_3way_produces_disjoint_complete_shares(corn_scoped_config):
    cfg, root = corn_scoped_config
    dataset = PlantVillageDataset(root, image_size=32)
    corn_indices = get_corn_indices(dataset)
    healthy_indices = get_corn_disease_indices(dataset, corn_indices, "healthy")

    shares = split_healthy_3way(dataset, healthy_indices, num_nodes=3, seed=0, threshold=5)

    assert len(shares) == 3
    all_assigned = [idx for share in shares for idx in share]
    assert sorted(all_assigned) == sorted(healthy_indices)
    for i in range(3):
        for j in range(i + 1, 3):
            assert set(shares[i]).isdisjoint(set(shares[j]))
    # 12 healthy images, real (non-duplicate) random noise -> roughly balanced
    assert max(len(s) for s in shares) - min(len(s) for s in shares) <= 2


def test_split_healthy_3way_keeps_duplicate_groups_together():
    # compute_image_hashes opens real image files, so this test drives
    # split_healthy_3way's grouping logic directly via group_duplicates
    # instead of through compute_image_hashes -- verifies groups never
    # split across shares regardless of where the hashes came from.
    from src.validation.node1_dataset import group_duplicates
    from src.validation.corn_mesh_dataset import _assign_groups_to_shares

    indices = [0, 1, 2, 3, 4, 5]
    hashes = {0: 0, 1: 1, 2: 0b1111_0000, 3: 0b1111_0001, 4: 0b0101_0101, 5: 0b1010_1010}
    groups = group_duplicates(indices, hashes, threshold=1)
    group_members: dict[int, list[int]] = {}
    for idx, gid in groups.items():
        group_members.setdefault(gid, []).append(idx)

    # Reimplement just the greedy-assignment half (the part under test)
    # against these hand-built groups, mirroring what split_healthy_3way
    # does internally after grouping.
    shares = _assign_groups_to_shares(list(group_members.values()), num_nodes=3, seed=0)
    idx_to_share = {idx: i for i, share in enumerate(shares) for idx in share}
    assert idx_to_share[0] == idx_to_share[1]
    assert idx_to_share[2] == idx_to_share[3]
