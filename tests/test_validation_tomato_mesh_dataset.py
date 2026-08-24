from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.validation.node1_dataset import group_duplicates
from src.validation.tomato_mesh_dataset import (
    TOMATO_DISEASE_ORDER,
    TomatoMeshData,
    TomatoRawItem,
    _items_from_paths,
    _jensen_shannon_divergence,
    _summarize_partition,
    build_tomato_label_map,
    build_tomato_train_eval_datasets,
    capped_dedup_split,
    carve_merged_probe_set,
    compute_merged_image_hashes,
    enforce_max_group_size,
    load_plantvillage_tomato_items,
    prepare_tomato_mesh_data,
)


def test_tomato_disease_order_has_ten_canonical_classes():
    assert len(TOMATO_DISEASE_ORDER) == 10
    assert "healthy" in TOMATO_DISEASE_ORDER
    assert "Target_Spot" in TOMATO_DISEASE_ORDER


def test_build_tomato_label_map():
    label_map = build_tomato_label_map()
    assert label_map.crop_classes == ["Tomato"]
    assert label_map.disease_classes == TOMATO_DISEASE_ORDER
    assert label_map.name_to_disease_idx["healthy"] == TOMATO_DISEASE_ORDER.index("healthy")


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    arr = rng.randint(0, 255, size=(16, 16, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def test_merged_dataset_getitem_and_duck_typed_surface(tmp_path):
    label_map = build_tomato_label_map()
    p0 = tmp_path / "a.jpg"
    p1 = tmp_path / "b.jpg"
    _write_image(p0)
    _write_image(p1)
    items = [
        TomatoRawItem(source="plantvillage", path=str(p0), canonical_disease_idx=0),
        TomatoRawItem(source="plantdoc", path=str(p1), canonical_disease_idx=2),
    ]
    train_ds, eval_ds = build_tomato_train_eval_datasets(items, label_map, image_size=16)

    assert len(train_ds) == 2
    image, crop_label, disease_label = eval_ds[1]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 16, 16)
    assert crop_label == 0
    assert disease_label == 2

    # duck-typed surface required by evaluate_onnx.run_evaluation / export_onnx.export_checkpoint
    assert eval_ds.base.samples[1] == (str(p1), 2)
    assert eval_ds.base.targets == [0, 2]
    assert eval_ds.labels.class_to_crop_disease[2] == (0, 2)
    assert eval_ds.labels.crop_classes == ["Tomato"]
    assert eval_ds.labels.disease_classes == TOMATO_DISEASE_ORDER

    # train has augmentation, eval doesn't -- same style assertion as
    # tests/test_validation_node1_dataset.py's build_train_eval_datasets test
    train_types = [type(t) for t in train_ds.transform.transforms]
    eval_types = [type(t) for t in eval_ds.transform.transforms]
    assert train_types != eval_types
    assert len(eval_types) == 3  # Resize, ToTensor, Normalize


def _write_pv_class(tmp_path, crop_disease_folder, count):
    cls_dir = tmp_path / "PlantVillage" / crop_disease_folder
    cls_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    for i in range(count):
        arr = rng.randint(0, 255, size=(16, 16, 3), dtype=np.uint8)
        Image.fromarray(arr).save(cls_dir / f"img_{i}.jpg")


def test_load_plantvillage_tomato_items_filters_to_tomato_only(tmp_path):
    label_map = build_tomato_label_map()
    _write_pv_class(tmp_path, "Tomato___Bacterial_spot", 3)
    _write_pv_class(tmp_path, "Tomato___healthy", 2)
    _write_pv_class(tmp_path, "Potato___healthy", 4)  # must be excluded

    items = load_plantvillage_tomato_items(tmp_path / "PlantVillage", label_map)
    assert len(items) == 5
    assert all(item.source == "plantvillage" for item in items)
    canonical_names = {label_map.disease_classes[i.canonical_disease_idx] for i in items}
    assert canonical_names == {"Bacterial_spot", "healthy"}


def test_items_from_paths_maps_canonical_name_to_index():
    label_map = build_tomato_label_map()
    pairs = [("/a.jpg", "healthy"), ("/b.jpg", "Early_blight")]
    items = _items_from_paths(pairs, "plantdoc", label_map)
    assert items[0].canonical_disease_idx == label_map.name_to_disease_idx["healthy"]
    assert items[1].source == "plantdoc"


def test_compute_merged_image_hashes_reads_each_item_path(tmp_path):
    p0, p1 = tmp_path / "a.jpg", tmp_path / "b.jpg"
    _write_image(p0)
    _write_image(p1)
    items = [
        TomatoRawItem(source="plantvillage", path=str(p0), canonical_disease_idx=0),
        TomatoRawItem(source="plantdoc", path=str(p1), canonical_disease_idx=0),
    ]
    hashes = compute_merged_image_hashes(items, [0, 1])
    assert set(hashes.keys()) == {0, 1}
    assert all(isinstance(h, int) for h in hashes.values())


def test_enforce_max_group_size_breaks_up_oversized_group():
    # 5 items, all in canonical class 0, all hashed into ONE group by the
    # caller (hand-built, mirrors test_validation_node1_dataset.py's style).
    # Every member has a distinct exact hash, so breaking up the group
    # results in true singletons (no exact-hash duplicates to preserve).
    items = [TomatoRawItem(source="x", path=f"/{i}.jpg", canonical_disease_idx=0) for i in range(5)]
    indices = [0, 1, 2, 3, 4]
    groups = {i: 0 for i in indices}  # one big group, id=0
    hashes = {i: i for i in indices}  # every member has a unique exact hash

    fixed = enforce_max_group_size(
        indices, groups, items, hashes, max_group_size=2, max_group_fraction_of_class=1.0
    )
    # group of 5 > cap of 2 -> broken into singletons (no shared exact hashes)
    assert len({fixed[i] for i in indices}) == 5
    for i in indices:
        assert fixed[i] == i


def test_enforce_max_group_size_leaves_small_groups_alone():
    items = [TomatoRawItem(source="x", path=f"/{i}.jpg", canonical_disease_idx=0) for i in range(3)]
    indices = [0, 1, 2]
    groups = {0: 0, 1: 0, 2: 2}  # group {0,1} size 2, group {2} size 1
    hashes = {0: 0, 1: 1, 2: 2}

    fixed = enforce_max_group_size(
        indices, groups, items, hashes, max_group_size=10, max_group_fraction_of_class=1.0
    )
    assert fixed == groups


def test_enforce_max_group_size_keeps_exact_hash_duplicates_together():
    # One over-cap pre-existing group of 5 members: 0 and 1 share an EXACT
    # hash (real hash-identical duplicates); 2, 3, 4 each have a unique
    # hash. Breaking up the group must keep 0 and 1 together as one
    # sub-group (not shatter them into full singletons), while 2/3/4 each
    # become their own singleton.
    items = [TomatoRawItem(source="x", path=f"/{i}.jpg", canonical_disease_idx=0) for i in range(5)]
    indices = [0, 1, 2, 3, 4]
    groups = {i: 0 for i in indices}
    hashes = {0: 0b0000, 1: 0b0000, 2: 0b1111, 3: 0b0101, 4: 0b0110}

    fixed = enforce_max_group_size(
        indices, groups, items, hashes, max_group_size=2, max_group_fraction_of_class=1.0
    )
    assert fixed[0] == fixed[1]
    assert fixed[2] == 2
    assert fixed[3] == 3
    assert fixed[4] == 4
    assert len({fixed[0], fixed[2], fixed[3], fixed[4]}) == 4


def test_capped_dedup_split_partitions_all_indices():
    items = [TomatoRawItem(source="x", path=f"/{i}.jpg", canonical_disease_idx=0) for i in range(6)]
    indices = list(range(6))
    hashes = {0: 0b0000, 1: 0b0000, 2: 0b1111, 3: 0b1111, 4: 0b0101, 5: 0b0110}

    train_idx, test_idx = capped_dedup_split(
        indices, hashes, items, test_fraction=0.5, seed=1, threshold=1, max_group_size=10
    )
    assert sorted(train_idx + test_idx) == indices
    assert set(train_idx).isdisjoint(test_idx)


def test_capped_dedup_split_logs_group_size_histogram(caplog):
    items = [TomatoRawItem(source="x", path=f"/{i}.jpg", canonical_disease_idx=0) for i in range(6)]
    indices = list(range(6))
    hashes = {0: 0b0000, 1: 0b0000, 2: 0b1111, 3: 0b1111, 4: 0b0101, 5: 0b0110}

    with caplog.at_level(logging.INFO, logger="src.validation.tomato_mesh_dataset"):
        capped_dedup_split(indices, hashes, items, test_fraction=0.5, seed=1, threshold=1, max_group_size=10)

    assert any("group" in record.message.lower() for record in caplog.records)


def test_enforce_max_group_size_never_exceeds_configured_cap_via_exact_hash_subclusters():
    """Spec requirement: 'a unit test should assert no single group
    exceeds the configured cap.' An over-cap pre-existing group (12
    members) is split by exact-hash equality into 4 clusters of 3 each --
    each cluster fits within max_group_size, so after enforcement no
    resulting group (grouped by post-cap group id) should exceed the cap,
    even though the original group did.
    """
    max_group_size = 3
    group_indices = list(range(12))
    hashes: dict[int, int] = {}
    for cluster in range(4):
        for member in range(3):
            hashes[cluster * 3 + member] = cluster  # 4 distinct exact-hash clusters of size 3
    groups = {i: 0 for i in group_indices}  # one big pre-cap group

    # Pad with already-singleton items of the same class so the
    # max_group_fraction_of_class cap doesn't bind tighter than
    # max_group_size (mirrors a realistically sized class pool).
    padding_indices = list(range(12, 12 + 200))
    for idx in padding_indices:
        groups[idx] = idx
        hashes[idx] = 1000 + idx

    indices = group_indices + padding_indices
    items = [TomatoRawItem(source="x", path=f"/{i}.jpg", canonical_disease_idx=0) for i in indices]

    fixed = enforce_max_group_size(
        indices, groups, items, hashes, max_group_size=max_group_size, max_group_fraction_of_class=0.05
    )

    sizes: dict[int, int] = {}
    for idx in indices:
        sizes[fixed[idx]] = sizes.get(fixed[idx], 0) + 1
    assert max(sizes.values()) <= max_group_size


def test_capped_dedup_split_never_exceeds_configured_cap(caplog):
    """Same spec requirement as above, exercised through the full
    capped_dedup_split entrypoint rather than enforce_max_group_size
    directly.
    """
    max_group_size = 2
    items = [TomatoRawItem(source="x", path=f"/{i}.jpg", canonical_disease_idx=0) for i in range(5)]
    indices = list(range(5))
    # All 5 land in one pre-cap duplicate group (pairwise within
    # threshold), but each has a distinct exact hash, so post-cap they
    # become true singletons rather than one oversized group.
    hashes = {0: 0b00000, 1: 0b00001, 2: 0b00011, 3: 0b00010, 4: 0b00110}

    with caplog.at_level(logging.INFO, logger="src.validation.tomato_mesh_dataset"):
        train_idx, test_idx = capped_dedup_split(
            indices, hashes, items, test_fraction=0.4, seed=1, threshold=5, max_group_size=max_group_size
        )
    assert sorted(train_idx + test_idx) == indices

    groups = group_duplicates(indices, hashes, threshold=5)
    groups = enforce_max_group_size(indices, groups, items, hashes, max_group_size=max_group_size)
    sizes: dict[int, int] = {}
    for idx in indices:
        sizes[groups[idx]] = sizes.get(groups[idx], 0) + 1
    assert max(sizes.values()) <= max_group_size


def test_jensen_shannon_divergence_zero_for_identical_distributions():
    p = np.array([0.5, 0.5])
    assert _jensen_shannon_divergence(p, p) == pytest.approx(0.0, abs=1e-9)


def test_jensen_shannon_divergence_one_for_disjoint_supports():
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert _jensen_shannon_divergence(p, q) == pytest.approx(1.0, abs=1e-9)


def test_carve_merged_probe_set_disjoint_and_covers_all_indices():
    label_map = build_tomato_label_map()
    items = [TomatoRawItem(source="x", path=f"/{i}.jpg", canonical_disease_idx=i % 3) for i in range(30)]
    indices = list(range(30))
    probe_idx, remaining_idx = carve_merged_probe_set(
        items, indices, probe_fraction=0.2, seed=1, min_samples_small_class=1, max_fraction_small_class=0.5
    )
    assert set(probe_idx).isdisjoint(remaining_idx)
    assert sorted(probe_idx + remaining_idx) == indices
    assert len(probe_idx) > 0


def test_summarize_partition_flags_low_representation_classes():
    label_map = build_tomato_label_map()
    items = [TomatoRawItem(source="x", path=f"/{i}.jpg", canonical_disease_idx=0) for i in range(20)]
    items += [TomatoRawItem(source="x", path=f"/h{i}.jpg", canonical_disease_idx=2) for i in range(1)]
    shard = list(range(21))
    summary = _summarize_partition(items, shard, label_map)
    assert summary["num_samples"] == 21
    assert summary["dominant_class"] == label_map.disease_classes[0]
    assert label_map.disease_classes[2] in summary["low_representation_classes"]


def test_summarize_partition_flags_zero_count_class_as_low_representation():
    # A class with ZERO samples for this shard is the most extreme case of
    # low representation, not a non-case -- it must still be flagged.
    label_map = build_tomato_label_map()
    items = [TomatoRawItem(source="x", path=f"/{i}.jpg", canonical_disease_idx=0) for i in range(10)]
    shard = list(range(10))
    summary = _summarize_partition(items, shard, label_map)

    zero_count_class = label_map.disease_classes[5]
    assert summary["per_class_counts"][zero_count_class] == 0
    assert zero_count_class in summary["low_representation_classes"]


def test_prepare_tomato_mesh_data_end_to_end(tomato_scoped_config):
    cfg = tomato_scoped_config
    data = prepare_tomato_mesh_data(cfg)
    assert isinstance(data, TomatoMeshData)
    assert len(data.per_node) == 3

    all_train = [i for shard in data.per_node.values() for i in shard["train_idx"]]
    assert set(all_train).isdisjoint(data.test_idx)
    assert set(all_train).isdisjoint(data.probe_idx)
    # every node's shard is disjoint from every other node's
    node_sets = [set(v["train_idx"]) for v in data.per_node.values()]
    for i in range(len(node_sets)):
        for j in range(i + 1, len(node_sets)):
            assert node_sets[i].isdisjoint(node_sets[j])

    for node_id, diagnostics in data.partition_diagnostics.items():
        assert diagnostics["num_samples"] == len(data.per_node[node_id]["train_idx"])
