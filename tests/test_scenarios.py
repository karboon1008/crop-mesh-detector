"""Tests for the disconnection/class-addition/distribution-shift scenario
simulations, and the core mesh 'active' flag they rely on.
"""

from __future__ import annotations

import json

import pytest
import torch
from torch.utils.data import DataLoader

from src.data.plantvillage import (
    carve_public_probe_set,
    make_subset,
    partition_nodes,
    train_test_split_indices,
)
from src.federated.mesh import MeshSimulator
from src.federated.node import Node
from src.models.factory import build_model
from src.scenarios.harness import ScenarioRoundRecord, _recovery_round, run_scenario, write_scenario_report

from tests.conftest import build_nodes


def test_node_active_defaults_to_true(synthetic_dataset):
    probe_idx, remaining_idx = carve_public_probe_set(synthetic_dataset, 0.1, seed=0)
    shards = partition_nodes(
        synthetic_dataset, remaining_idx, num_nodes=1, strategy="by_crop", dirichlet_alpha=0.3, seed=0
    )
    num_crop = len(synthetic_dataset.labels.crop_classes)
    num_disease = len(synthetic_dataset.labels.disease_classes)
    node = build_nodes(synthetic_dataset, shards, num_crop, num_disease)[0]
    assert node.active is True


def test_inactive_node_excluded_from_broadcast_and_distill(synthetic_dataset):
    # NOTE: deviates from the task-2 brief, which specified
    # strategy="by_crop", seed=1 here. Two issues with that on the shared
    # `synthetic_dataset` fixture (tests/conftest.py, only 2 crop species:
    # Tomato, Potato):
    #   1) "by_crop" round-robins shards across *crop groups*; with only 2
    #      groups it can never populate a 3rd node's shard, for any seed
    #      (deterministic, not seed-luck: verified directly). That leaves
    #      build_nodes() constructing DataLoader(shuffle=True) over an empty
    #      subset, which raises a ValueError in torch's RandomSampler before
    #      the mesh round (the thing under test) is ever exercised.
    #   2) even switching to a strategy that fills all 3 shards (e.g.
    #      "by_disease"), the tiny fixture (32 images total) can produce a
    #      per-node train split whose size % batch_size(=4) == 1. The last
    #      batch of size 1 then hits `model.train()` and crashes inside
    #      MobileNetV3's BatchNorm ("Expected more than 1 value per channel")
    #      — again before the active-flag logic runs.
    # "dirichlet" with seed=2 (found by search) gives 3 balanced, non-empty
    # shards ([7, 6, 7] samples) whose train/probe splits are all clear of
    # that batch-size-1 trap, so the test actually exercises
    # MeshSimulator.run_round's active-flag filtering instead of crashing in
    # unrelated data-plumbing.
    probe_idx, remaining_idx = carve_public_probe_set(synthetic_dataset, 0.2, seed=2)
    shards = partition_nodes(
        synthetic_dataset, remaining_idx, num_nodes=3, strategy="dirichlet", dirichlet_alpha=0.3, seed=2
    )
    probe_loader = DataLoader(make_subset(synthetic_dataset, probe_idx), batch_size=4, shuffle=False)
    num_crop = len(synthetic_dataset.labels.crop_classes)
    num_disease = len(synthetic_dataset.labels.disease_classes)

    nodes = build_nodes(synthetic_dataset, shards, num_crop, num_disease)
    nodes[0].active = False

    mesh = MeshSimulator(
        nodes, probe_loader, aggregation_method="trimmed_mean", trim_fraction=0.0, krum_neighbors=1
    )
    round_log = mesh.run_round(
        0, local_epochs=1, distill_epochs=1, lr=1e-3, distill_lr=1e-3,
        proto_weight=0.5, kd_weight=0.5, temperature=2.0,
    )

    assert round_log.active_nodes == ["node_1", "node_2"]
    # the disconnected node still trained locally and was evaluated...
    assert "node_0" in round_log.per_node_train_loss
    assert "node_0" in round_log.per_node_eval
    # ...but never distilled towards a peer consensus this round.
    assert "node_0" not in round_log.per_node_distill_loss
    # active peers still exchanged and distilled as normal.
    assert "node_1" in round_log.per_node_distill_loss
    assert "node_2" in round_log.per_node_distill_loss


def test_run_scenario_and_write_report(tmp_path, synthetic_dataset):
    probe_idx, remaining_idx = carve_public_probe_set(synthetic_dataset, 0.2, seed=2)
    shards = partition_nodes(
        synthetic_dataset, remaining_idx, num_nodes=2, strategy="by_crop", dirichlet_alpha=0.3, seed=2
    )
    probe_loader = DataLoader(make_subset(synthetic_dataset, probe_idx), batch_size=4, shuffle=False)
    num_crop = len(synthetic_dataset.labels.crop_classes)
    num_disease = len(synthetic_dataset.labels.disease_classes)

    baseline_nodes = build_nodes(synthetic_dataset, shards, num_crop, num_disease)
    mesh_nodes = build_nodes(synthetic_dataset, shards, num_crop, num_disease)
    mesh = MeshSimulator(
        mesh_nodes, probe_loader, aggregation_method="trimmed_mean", trim_fraction=0.0, krum_neighbors=1
    )

    def no_op_hook(round_idx, nodes, mesh_or_none):
        return []

    round_kwargs = {
        "local_epochs": 1, "distill_epochs": 1, "lr": 1e-3, "distill_lr": 1e-3,
        "proto_weight": 0.5, "kd_weight": 0.5, "temperature": 2.0,
    }
    records = run_scenario(baseline_nodes, mesh, num_rounds=2, perturbation_hook=no_op_hook, round_kwargs=round_kwargs)
    assert len(records) == 2
    assert records[0].round_idx == 0 and records[1].round_idx == 1

    report_path = write_scenario_report(
        tmp_path, "unit_test_scenario", "node_0",
        disruption_start_round=1, disruption_end_round=1,
        config_snapshot={"note": "test"}, records=records,
    )
    assert report_path == tmp_path / "scenarios" / "unit_test_scenario.json"
    written = json.loads(report_path.read_text())
    assert written["scenario"] == "unit_test_scenario"
    assert written["target_node"] == "node_0"
    assert len(written["rounds"]) == 2
    assert "recovery_round_mesh" in written["summary"]
    assert "recovery_round_baseline" in written["summary"]


def test_recovery_round_returns_none_for_out_of_range_disruption_start():
    # disruption_start_round is far beyond the available records, so
    # pre_round (disruption_start_round - 1) indexes past the end of the
    # list. This must return None gracefully rather than raise IndexError.
    records = [
        ScenarioRoundRecord(0, {"node_0": {"acc": 0.9}}, {"node_0": {"acc": 0.9}}, {}),
        ScenarioRoundRecord(1, {"node_0": {"acc": 0.9}}, {"node_0": {"acc": 0.9}}, {}),
    ]
    assert _recovery_round(records, "node_0", disruption_start_round=100, disruption_end_round=101, eval_key="mesh_eval") is None
    assert _recovery_round(records, "node_0", disruption_start_round=100, disruption_end_round=101, eval_key="baseline_eval") is None


def test_recovery_round_returns_none_for_empty_records():
    # num_rounds=0 (or any scenario producing no records) must not raise
    # IndexError when looking up the pre-disruption round either.
    assert _recovery_round([], "node_0", disruption_start_round=0, disruption_end_round=1, eval_key="mesh_eval") is None
    assert _recovery_round([], "node_0", disruption_start_round=5, disruption_end_round=6, eval_key="mesh_eval") is None


def test_disconnection_scenario_end_to_end_smoke(tmp_path, synthetic_dataset):
    from src.scenarios.disconnection import make_disconnect_hook

    # NOTE: deviates from the task-4 brief, which specified strategy="by_crop"
    # here. Same root cause already documented above in
    # test_inactive_node_excluded_from_broadcast_and_distill: with only 2 crop
    # groups in the shared `synthetic_dataset` fixture, "by_crop" round-robin
    # can never populate a 3rd node's shard for any seed, which crashes
    # DataLoader(shuffle=True) on the empty shard before this scenario's code
    # ever runs. Reusing the same probe_fraction=0.2/seed=2/"dirichlet"
    # combination that test already verified empirically gives 3 non-empty,
    # balanced shards (sizes [9, 8, 9]) whose per-node train splits (sizes
    # [7, 6, 7]) are also clear of the batch_size=4 trailing-batch-of-1 trap
    # that crashes MobileNetV3's BatchNorm.
    probe_idx, remaining_idx = carve_public_probe_set(synthetic_dataset, 0.2, seed=2)
    shards = partition_nodes(
        synthetic_dataset, remaining_idx, num_nodes=3, strategy="dirichlet", dirichlet_alpha=0.3, seed=2
    )
    probe_loader = DataLoader(make_subset(synthetic_dataset, probe_idx), batch_size=4, shuffle=False)
    num_crop = len(synthetic_dataset.labels.crop_classes)
    num_disease = len(synthetic_dataset.labels.disease_classes)

    baseline_nodes = build_nodes(synthetic_dataset, shards, num_crop, num_disease)
    mesh_nodes = build_nodes(synthetic_dataset, shards, num_crop, num_disease)
    mesh = MeshSimulator(
        mesh_nodes, probe_loader, aggregation_method="trimmed_mean", trim_fraction=0.0, krum_neighbors=1
    )

    hook = make_disconnect_hook("node_1", disconnect_round=1, reconnect_round=2)
    round_kwargs = {
        "local_epochs": 1, "distill_epochs": 1, "lr": 1e-3, "distill_lr": 1e-3,
        "proto_weight": 0.5, "kd_weight": 0.5, "temperature": 2.0,
    }
    records = run_scenario(baseline_nodes, mesh, num_rounds=3, perturbation_hook=hook, round_kwargs=round_kwargs)

    report_path = write_scenario_report(
        tmp_path, "disconnection", "node_1",
        disruption_start_round=1, disruption_end_round=2,
        config_snapshot={"target_node": "node_1", "disconnect_round": 1, "reconnect_round": 2},
        records=records,
    )
    report = json.loads(report_path.read_text())

    event_types = [e["event_type"] for r in report["rounds"] for e in r["events"]]
    assert event_types == ["disconnect", "reconnect"]
    # node_1 is evaluated every round, connected or not
    assert all("node_1" in r["mesh_eval"] for r in report["rounds"])
    assert "recovery_round_mesh" in report["summary"]
    assert "recovery_round_baseline" in report["summary"]

    # Hard evidence that disconnection actually excluded node_1 from the
    # exchange: it's present in active_nodes before/after, absent during,
    # and the disconnected round exchanges fewer total bytes than a fully
    # connected round with the same node set.
    rounds_by_idx = {r["round_idx"]: r for r in report["rounds"]}
    assert "node_1" in rounds_by_idx[0]["active_nodes"]
    assert "node_1" not in rounds_by_idx[1]["active_nodes"]
    assert "node_1" in rounds_by_idx[2]["active_nodes"]
    assert rounds_by_idx[1]["total_bytes_exchanged"] < rounds_by_idx[0]["total_bytes_exchanged"]


def test_find_source_node_returns_owning_node_and_raises_for_unknown_crop():
    from src.scenarios.class_addition import find_source_node

    manual_node_crops = {"node_0": ["Potato"], "node_1": ["Tomato"]}
    assert find_source_node(manual_node_crops, "Tomato") == "node_1"

    with pytest.raises(ValueError, match="not assigned to any node"):
        find_source_node(manual_node_crops, "Corn")


def test_carve_reserve_pool_is_disjoint_from_remaining_source(synthetic_dataset):
    from src.scenarios.class_addition import carve_reserve_pool

    _, remaining_idx = carve_public_probe_set(synthetic_dataset, 0.1, seed=4)
    shards = partition_nodes(
        synthetic_dataset, remaining_idx, num_nodes=2, strategy="manual", dirichlet_alpha=0.3, seed=4,
        manual_node_crops={"node_0": ["Tomato"], "node_1": ["Potato"]},
    )
    source_train_idx, _ = train_test_split_indices(shards[0], test_fraction=0.3, seed=4)  # node_0 grows Tomato

    remaining_source, reserve_train_idx, reserve_test_idx = carve_reserve_pool(
        synthetic_dataset, source_train_idx, source_crop="Tomato", reserve_fraction=0.5, seed=4, test_fraction=0.3,
    )

    reserve_all = set(reserve_train_idx) | set(reserve_test_idx)
    assert reserve_all.isdisjoint(remaining_source)
    assert set(remaining_source) | reserve_all == set(source_train_idx)
    assert len(reserve_all) > 0


def test_class_addition_scenario_end_to_end_smoke(tmp_path, synthetic_dataset):
    from src.scenarios.class_addition import carve_reserve_pool, make_class_addition_hook

    _, remaining_idx = carve_public_probe_set(synthetic_dataset, 0.1, seed=5)
    shards = partition_nodes(
        synthetic_dataset, remaining_idx, num_nodes=2, strategy="manual", dirichlet_alpha=0.3, seed=5,
        manual_node_crops={"node_0": ["Potato"], "node_1": ["Tomato"]},
    )
    probe_idx, _ = carve_public_probe_set(synthetic_dataset, 0.1, seed=5)
    probe_loader = DataLoader(make_subset(synthetic_dataset, probe_idx), batch_size=4, shuffle=False)
    num_crop = len(synthetic_dataset.labels.crop_classes)
    num_disease = len(synthetic_dataset.labels.disease_classes)

    source_train_idx, source_test_idx = train_test_split_indices(shards[1], test_fraction=0.3, seed=5)  # node_1: Tomato
    target_train_idx, target_test_idx = train_test_split_indices(shards[0], test_fraction=0.3, seed=5)  # node_0: Potato

    remaining_source, reserve_train_idx, reserve_test_idx = carve_reserve_pool(
        synthetic_dataset, source_train_idx, source_crop="Tomato", reserve_fraction=0.5, seed=5, test_fraction=0.3,
    )

    def make_loaders(train_idx, test_idx):
        return (
            DataLoader(make_subset(synthetic_dataset, train_idx), batch_size=4, shuffle=True),
            DataLoader(make_subset(synthetic_dataset, test_idx), batch_size=4, shuffle=False),
        )

    node_loaders = [
        make_loaders(target_train_idx, target_test_idx),
        make_loaders(remaining_source, source_test_idx),
    ]

    baseline_nodes = [
        Node(f"node_{i}", build_model("mobilenet_v3_small", num_crop, num_disease, pretrained=False), tl, sl, device="cpu")
        for i, (tl, sl) in enumerate(node_loaders)
    ]
    mesh_nodes = [
        Node(f"node_{i}", build_model("mobilenet_v3_small", num_crop, num_disease, pretrained=False), tl, sl, device="cpu")
        for i, (tl, sl) in enumerate(node_loaders)
    ]
    mesh = MeshSimulator(
        mesh_nodes, probe_loader, aggregation_method="trimmed_mean", trim_fraction=0.0, krum_neighbors=1
    )

    hook = make_class_addition_hook(
        "node_0", "Tomato", inject_round=1, reserve_train_idx=reserve_train_idx,
        reserve_test_idx=reserve_test_idx, dataset=synthetic_dataset, batch_size=4,
    )
    round_kwargs = {
        "local_epochs": 1, "distill_epochs": 1, "lr": 1e-3, "distill_lr": 1e-3,
        "proto_weight": 0.5, "kd_weight": 0.5, "temperature": 2.0,
    }
    records = run_scenario(baseline_nodes, mesh, num_rounds=2, perturbation_hook=hook, round_kwargs=round_kwargs)

    report_path = write_scenario_report(
        tmp_path, "class_addition", "node_0",
        disruption_start_round=1, disruption_end_round=1,
        config_snapshot={"target_node": "node_0", "source_crop": "Tomato", "inject_round": 1},
        records=records,
    )
    report = json.loads(report_path.read_text())
    event_types = [e["event_type"] for r in report["rounds"] for e in r["events"]]
    assert event_types == ["class_added"]
    assert report["rounds"][1]["events"][0]["details"]["crop"] == "Tomato"


def test_corrupted_dataset_is_noop_at_zero_severity_and_differs_otherwise(synthetic_dataset):
    from src.scenarios.distribution_shift import CorruptedDataset

    clean = CorruptedDataset(synthetic_dataset, severity=0.0)
    image_a, crop_a, disease_a = clean[0]
    image_b, crop_b, disease_b = synthetic_dataset[0]
    assert torch.equal(image_a, image_b)
    assert crop_a == crop_b and disease_a == disease_b

    corrupted = CorruptedDataset(synthetic_dataset, severity=0.5)
    image_c, _, _ = corrupted[0]
    assert not torch.equal(image_c, image_b)


def test_distribution_shift_scenario_end_to_end_smoke(tmp_path, synthetic_dataset):
    from src.scenarios.distribution_shift import make_shift_hook

    probe_idx, remaining_idx = carve_public_probe_set(synthetic_dataset, 0.2, seed=6)
    shards = partition_nodes(
        synthetic_dataset, remaining_idx, num_nodes=2, strategy="by_crop", dirichlet_alpha=0.3, seed=6
    )
    probe_loader = DataLoader(make_subset(synthetic_dataset, probe_idx), batch_size=4, shuffle=False)
    num_crop = len(synthetic_dataset.labels.crop_classes)
    num_disease = len(synthetic_dataset.labels.disease_classes)

    baseline_nodes = build_nodes(synthetic_dataset, shards, num_crop, num_disease)
    mesh_nodes = build_nodes(synthetic_dataset, shards, num_crop, num_disease)
    mesh = MeshSimulator(
        mesh_nodes, probe_loader, aggregation_method="trimmed_mean", trim_fraction=0.0, krum_neighbors=1
    )

    hook = make_shift_hook("node_0", shift_round=1, corruption="brightness_blur_noise", severity=0.5, batch_size=4)
    round_kwargs = {
        "local_epochs": 1, "distill_epochs": 1, "lr": 1e-3, "distill_lr": 1e-3,
        "proto_weight": 0.5, "kd_weight": 0.5, "temperature": 2.0,
    }
    records = run_scenario(baseline_nodes, mesh, num_rounds=2, perturbation_hook=hook, round_kwargs=round_kwargs)

    report_path = write_scenario_report(
        tmp_path, "distribution_shift", "node_0",
        disruption_start_round=1, disruption_end_round=1,
        config_snapshot={"target_node": "node_0", "shift_round": 1, "severity": 0.5},
        records=records,
    )
    report = json.loads(report_path.read_text())
    event_types = [e["event_type"] for r in report["rounds"] for e in r["events"]]
    assert event_types == ["shift_applied"]
    assert report["rounds"][1]["events"][0]["details"]["severity"] == 0.5
