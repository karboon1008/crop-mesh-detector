"""End-to-end smoke test on synthetic images — exercises the full
pipeline (label parsing, non-IID partition, model build, one mesh round,
energy/communication accounting) without needing the real PlantVillage
dataset, so `pytest` works right after `pip install -r requirements.txt`.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from src.data.plantvillage import carve_public_probe_set, make_subset, partition_nodes
from src.energy.tracker import CommunicationCostEstimator, ComputeEnergyTracker
from src.evaluate import compute_collaboration_gain
from src.federated.mesh import MeshSimulator, _scale_kd_weight
from src.models.factory import build_model, count_parameters, model_size_mb

from tests.conftest import build_nodes


def test_label_parsing(synthetic_dataset):
    assert set(synthetic_dataset.labels.crop_classes) == {"Tomato", "Potato"}
    assert "healthy" in synthetic_dataset.labels.disease_classes
    assert "Bacterial_spot" in synthetic_dataset.labels.disease_classes


def test_partition_by_crop_is_disjoint_and_complete(synthetic_dataset):
    probe_idx, remaining_idx = carve_public_probe_set(synthetic_dataset, 0.1, seed=0)
    shards = partition_nodes(
        synthetic_dataset, remaining_idx, num_nodes=2, strategy="by_crop", dirichlet_alpha=0.3, seed=0
    )
    assert set(probe_idx).isdisjoint(set(remaining_idx))
    all_shard_idx = [i for shard in shards for i in shard]
    assert sorted(all_shard_idx) == sorted(remaining_idx)


def test_partition_manual_assigns_named_crops_to_named_nodes(synthetic_dataset):
    _, remaining_idx = carve_public_probe_set(synthetic_dataset, 0.1, seed=0)
    shards = partition_nodes(
        synthetic_dataset,
        remaining_idx,
        num_nodes=2,
        strategy="manual",
        dirichlet_alpha=0.3,
        seed=0,
        manual_node_crops={"node_0": ["Tomato"], "node_1": ["Potato"]},
    )
    class_to_crop = {c: cd[0] for c, cd in synthetic_dataset.labels.class_to_crop_disease.items()}
    crops_per_shard = [
        {synthetic_dataset.labels.crop_classes[class_to_crop[synthetic_dataset.targets[i]]] for i in shard}
        for shard in shards
    ]
    assert crops_per_shard[0] == {"Tomato"}
    assert crops_per_shard[1] == {"Potato"}


def test_partition_dirichlet_by_crop_keeps_each_crop_inside_its_node_group(synthetic_dataset):
    _, remaining_idx = carve_public_probe_set(synthetic_dataset, 0.1, seed=0)
    shards = partition_nodes(
        synthetic_dataset,
        remaining_idx,
        num_nodes=4,
        strategy="dirichlet_by_crop",
        dirichlet_alpha=0.5,
        seed=0,
        manual_node_crops={
            "node_0": ["Tomato"], "node_1": ["Tomato"], "node_2": ["Potato"], "node_3": ["Potato"],
        },
    )
    assert sorted(i for shard in shards for i in shard) == sorted(remaining_idx)
    class_to_crop = {c: cd[0] for c, cd in synthetic_dataset.labels.class_to_crop_disease.items()}
    crop_of = lambda i: synthetic_dataset.labels.crop_classes[class_to_crop[synthetic_dataset.targets[i]]]
    assert {crop_of(i) for i in shards[0] + shards[1]} == {"Tomato"}
    assert {crop_of(i) for i in shards[2] + shards[3]} == {"Potato"}


def test_partition_manual_rejects_incomplete_crop_assignment(synthetic_dataset):
    _, remaining_idx = carve_public_probe_set(synthetic_dataset, 0.1, seed=0)
    with pytest.raises(ValueError, match="missing an assignment"):
        partition_nodes(
            synthetic_dataset,
            remaining_idx,
            num_nodes=2,
            strategy="manual",
            dirichlet_alpha=0.3,
            seed=0,
            manual_node_crops={"node_0": ["Tomato"]},
        )


@pytest.mark.parametrize("arch", ["mobilenet_v3_small", "efficientnet_lite0", "mobilevit_xxs"])
def test_model_forward_shapes(synthetic_dataset, arch):
    model = build_model(arch, num_crop_classes=2, num_disease_classes=3, pretrained=False)
    images = torch.stack([synthetic_dataset[i][0] for i in range(4)])
    crop_logits, disease_logits = model(images)
    assert crop_logits.shape == (4, 2)
    assert disease_logits.shape == (4, 3)
    assert count_parameters(model) > 0
    assert model_size_mb(model) > 0


def test_end_to_end_mesh_round_beats_no_exchange_smoke(synthetic_dataset):
    """Not a statistical claim (too little synthetic data for that) —
    just proves the mesh round runs, exchanges only prototypes/logits,
    and produces a well-formed collaboration-gain report.
    """
    probe_idx, remaining_idx = carve_public_probe_set(synthetic_dataset, 0.2, seed=1)
    shards = partition_nodes(
        synthetic_dataset, remaining_idx, num_nodes=2, strategy="by_crop", dirichlet_alpha=0.3, seed=1
    )
    probe_loader = DataLoader(make_subset(synthetic_dataset, probe_idx), batch_size=4, shuffle=False)

    num_crop = len(synthetic_dataset.labels.crop_classes)
    num_disease = len(synthetic_dataset.labels.disease_classes)

    baseline_nodes = build_nodes(synthetic_dataset, shards, num_crop, num_disease)
    baseline_evals = {}
    for node in baseline_nodes:
        node.local_train(epochs=1, lr=1e-3)
        baseline_evals[node.node_id] = node.evaluate()

    mesh_nodes = build_nodes(synthetic_dataset, shards, num_crop, num_disease)
    mesh = MeshSimulator(
        mesh_nodes, probe_loader, aggregation_method="trimmed_mean", trim_fraction=0.0, krum_neighbors=1
    )
    round_log = mesh.run_round(
        0,
        local_epochs=1,
        distill_epochs=1,
        lr=1e-3,
        distill_lr=1e-3,
        proto_weight=0.5,
        kd_weight=0.5,
        temperature=2.0,
    )

    assert round_log.total_bytes_exchanged > 0
    mesh_evals = {node.node_id: node.evaluate() for node in mesh_nodes}

    gain = compute_collaboration_gain(mesh_evals, baseline_evals)
    assert "macro_gain" in gain
    assert "worst_node_gain" in gain
    assert set(gain["per_node"].keys()) == {"node_0", "node_1"}


def test_combined_training_round_runs_without_local_train_and_produces_gain(synthetic_dataset):
    """Smoke test for MeshSimulator(combined_training=True): a round should
    run without ever calling local_train() separately (folded into
    Node.train_round instead), still exchange only prototypes/logits, and
    still produce a well-formed collaboration-gain report -- same shape of
    assertions as the default two-phase path above, so this is a like-for-
    like check that the opt-in path doesn't break the existing contract.
    """
    probe_idx, remaining_idx = carve_public_probe_set(synthetic_dataset, 0.2, seed=1)
    shards = partition_nodes(
        synthetic_dataset, remaining_idx, num_nodes=2, strategy="by_crop", dirichlet_alpha=0.3, seed=1
    )
    probe_loader = DataLoader(make_subset(synthetic_dataset, probe_idx), batch_size=4, shuffle=False)

    num_crop = len(synthetic_dataset.labels.crop_classes)
    num_disease = len(synthetic_dataset.labels.disease_classes)

    baseline_nodes = build_nodes(synthetic_dataset, shards, num_crop, num_disease)
    baseline_evals = {}
    for node in baseline_nodes:
        node.local_train(epochs=1, lr=1e-3)
        baseline_evals[node.node_id] = node.evaluate()

    mesh_nodes = build_nodes(synthetic_dataset, shards, num_crop, num_disease)
    mesh = MeshSimulator(
        mesh_nodes, probe_loader, aggregation_method="trimmed_mean", trim_fraction=0.0, krum_neighbors=1,
        combined_training=True,
    )
    round_log = mesh.run_round(
        0,
        local_epochs=1,
        distill_epochs=1,  # accepted but unused under combined_training
        lr=1e-3,
        distill_lr=1e-3,  # accepted but unused under combined_training
        proto_weight=0.5,
        kd_weight=0.5,
        temperature=2.0,
    )

    assert round_log.total_bytes_exchanged > 0
    assert round_log.per_node_train_loss == {}  # step 1 (local_train) is skipped entirely
    assert set(round_log.per_node_distill_loss.keys()) == {"node_0", "node_1"}
    mesh_evals = {node.node_id: node.evaluate() for node in mesh_nodes}

    gain = compute_collaboration_gain(mesh_evals, baseline_evals)
    assert "macro_gain" in gain
    assert "worst_node_gain" in gain
    assert set(gain["per_node"].keys()) == {"node_0", "node_1"}


def test_scale_kd_weight_pulls_weak_nodes_harder_than_strong_nodes():
    # ahead of peers -> scaled down
    assert _scale_kd_weight(0.7, node_avg=0.8, peer_avg=0.4, min_scale=0.3, max_scale=1.5) == pytest.approx(0.7 * 0.5)
    # behind peers -> scaled up
    assert _scale_kd_weight(0.7, node_avg=0.4, peer_avg=0.8, min_scale=0.3, max_scale=1.5) == pytest.approx(0.7 * 1.5)
    # far behind peers -> capped at max_scale, not scaled without bound
    assert _scale_kd_weight(0.7, node_avg=0.1, peer_avg=0.8, min_scale=0.3, max_scale=1.5) == pytest.approx(0.7 * 1.5)
    # a node with ~zero accuracy has nothing to divide by -> falls back to the flat weight
    assert _scale_kd_weight(0.7, node_avg=0.0, peer_avg=0.5, min_scale=0.3, max_scale=1.5) == 0.7


def test_adaptive_kd_weight_varies_per_node(synthetic_dataset):
    """With adaptive_kd_weight on, a mesh round should record a distinct
    kd_weight per node (scaled by that node's pre-round accuracy relative
    to its peers) instead of every node using the flat config value.
    """
    probe_idx, remaining_idx = carve_public_probe_set(synthetic_dataset, 0.2, seed=1)
    shards = partition_nodes(
        synthetic_dataset, remaining_idx, num_nodes=2, strategy="by_crop", dirichlet_alpha=0.3, seed=1
    )
    probe_loader = DataLoader(make_subset(synthetic_dataset, probe_idx), batch_size=4, shuffle=False)
    num_crop = len(synthetic_dataset.labels.crop_classes)
    num_disease = len(synthetic_dataset.labels.disease_classes)

    nodes = build_nodes(synthetic_dataset, shards, num_crop, num_disease)
    mesh = MeshSimulator(
        nodes, probe_loader, aggregation_method="trimmed_mean", trim_fraction=0.0, krum_neighbors=1,
        adaptive_kd_weight=True,
    )
    round_log = mesh.run_round(
        0, local_epochs=1, distill_epochs=1, lr=1e-3, distill_lr=1e-3, proto_weight=0.5, kd_weight=0.7,
        temperature=2.0,
    )

    assert set(round_log.per_node_kd_weight.keys()) == {"node_0", "node_1"}
    weights = round_log.per_node_kd_weight.values()
    assert all(0.7 * 0.3 <= w <= 0.7 * 1.5 for w in weights)


def test_energy_and_communication_tracking(tmp_path):
    tracker = ComputeEnergyTracker(enabled=False, output_dir=tmp_path, fallback_power_watts=15.0)
    with tracker.track("dummy_block") as record:
        _ = sum(range(1000))
    assert record["method"] == "proxy_wall_power"
    assert record["energy_kwh"] >= 0
    assert tracker.summary()["num_tracked_blocks"] == 1

    estimator = CommunicationCostEstimator(
        radio_energy_j_per_byte={"wifi": 0.00003}, grid_carbon_intensity_gco2_per_kwh=125
    )
    result = estimator.estimate(total_bytes=10_000, radio="wifi")
    assert result["energy_kwh"] > 0
    assert result["co2_kg"] > 0
