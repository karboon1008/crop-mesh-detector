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
# TODO(Task 3): src/scenarios/harness.py doesn't exist yet — Task 3 creates it.
# Uncomment once that module lands.
# from src.scenarios.harness import run_scenario, write_scenario_report

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
