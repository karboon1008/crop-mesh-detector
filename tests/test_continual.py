"""Continual mesh: PlantVillage splits, the shared knowledge database, the
EMA teacher/learner rule, and one end-to-end run on synthetic images."""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest
import torch
from PIL import Image

from src.config import Config
from src.data.plantvillage import PlantVillageDataset
from src.data.splits import build_continual_splits, split_shard_into_batches
from src.energy.tracker import CommunicationCostEstimator, ComputeEnergyTracker
from src.federated.continual import LEARNER, TEACHER, decide_roles, update_ema
from src.federated.knowledge_store import KnowledgeStore
from src.federated.node import KnowledgePayload

CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Apple___Black_rot",
    "Apple___healthy",
]


@pytest.fixture
def pv_root(tmp_path):
    root = tmp_path / "PlantVillage"
    rng = np.random.RandomState(0)
    for cls in CLASSES:
        (root / cls).mkdir(parents=True)
        for i in range(24):
            arr = rng.randint(0, 255, size=(32, 32, 3), dtype=np.uint8)
            Image.fromarray(arr).save(root / cls / f"img_{i}.jpg")
    return root


def _cfg(root, tmp_path, **continual) -> Config:
    return Config({
        "data": {
            "root": str(root), "image_size": 32, "num_nodes": 3, "seed": 0,
            "probe_set_fraction": 0.05, "probe_set_min_samples_small_class": 2,
            "probe_set_max_fraction_small_class": 0.2, "test_fraction": 0.25,
            "non_iid_strategy": "dirichlet", "dirichlet_alpha": 1.0,
        },
        "continual": {"num_batches": 3, "batch_strategy": "stratified", **continual},
        "models": {"architectures": ["mobilenet_v3_small"], "pretrained": False},
        "training": {
            "local_epochs_per_round": 1, "distill_epochs_per_round": 1, "batch_size": 8,
            "lr": 1e-3, "distill_lr": 5e-4, "proto_weight": 0.1, "kd_weight": 0.5, "kd_temperature": 2.0,
        },
        "federated": {"aggregation": "trimmed_mean", "trim_fraction": 0.2, "krum_neighbors": 1},
        "energy": {"track_with_codecarbon": False, "radio_energy_j_per_byte": {"wifi": 0.00003}},
        "output": {"dir": str(tmp_path / "out")},
    })


@pytest.mark.parametrize("strategy", ["stratified", "incremental"])
def test_continual_splits_are_disjoint_and_cover_the_pool(pv_root, tmp_path, strategy):
    dataset = PlantVillageDataset(pv_root, image_size=32)
    splits = build_continual_splits(_cfg(pv_root, tmp_path, batch_strategy=strategy), dataset)

    seen = list(splits.probe_idx)
    for batches in splits.node_batches:
        assert len(batches) == 3
        for split in batches:
            seen += split.train_idx + split.test_idx
    assert sorted(seen) == list(range(len(dataset)))  # every image exactly once
    assert len(splits.probe_idx) == 2 * len(CLASSES)   # 5% per class, min 2


def test_incremental_batches_introduce_classes_over_time(pv_root):
    dataset = PlantVillageDataset(pv_root, image_size=32)
    shard = list(range(len(dataset)))
    batches = split_shard_into_batches(dataset, shard, 3, "incremental", initial_class_fraction=0.5, seed=0)
    targets = np.asarray(dataset.targets)
    classes_per_batch = [set(targets[b].tolist()) for b in batches]

    assert len(classes_per_batch[0]) == 3                     # half the classes at the start
    assert classes_per_batch[0] < classes_per_batch[1] < classes_per_batch[2]  # old classes persist, new ones join
    assert classes_per_batch[2] == set(range(len(CLASSES)))
    assert sorted(i for b in batches for i in b) == shard


def test_decide_roles_follows_the_ema_rule():
    assert decide_roles(0, 0.95, None, 0.8) == [TEACHER, LEARNER]  # batch 0: everyone does both
    assert decide_roles(1, 0.90, 0.85, 0.8) == [TEACHER]           # improving and good
    assert decide_roles(1, 0.70, 0.60, 0.8) == [TEACHER, LEARNER]  # improving but still weak
    assert decide_roles(1, 0.70, 0.75, 0.8) == [LEARNER]           # getting worse and weak
    assert decide_roles(1, 0.85, 0.90, 0.8) == []                  # getting worse but still fine
    assert update_ema(None, 0.6, 0.5) == 0.6
    assert update_ema(0.6, 1.0, 0.5) == pytest.approx(0.8)


def _payload(value: float) -> KnowledgePayload:
    return KnowledgePayload(
        prototypes={("crop", 0): torch.full((4,), value)},
        crop_logits=torch.full((5, 2), value), disease_logits=torch.full((5, 3), value),
        known_crop_classes={0: 3}, known_disease_classes={1: 3},
    )


def test_knowledge_store_replaces_entries_and_excludes_self(tmp_path):
    store = KnowledgeStore(tmp_path / "k.db")
    size = store.upload("node_0", 0, _payload(0.0))
    store.upload("node_1", 0, _payload(1.0))
    store.upload("node_0", 1, _payload(2.0))  # replaces node_0's batch-0 entry

    assert size == _payload(0.0).size_bytes()
    assert [(e["node_id"], e["batch_idx"]) for e in store.entries()] == [("node_0", 1), ("node_1", 0)]

    peers = store.fetch_peers("node_1", batch_idx=1)
    assert list(peers) == ["node_0"]  # never its own entry
    peer_batch, payload = peers["node_0"]
    assert peer_batch == 1 and torch.equal(payload.crop_logits, torch.full((5, 2), 2.0))

    with sqlite3.connect(tmp_path / "k.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM uploads").fetchone()[0] == 3
        assert conn.execute("SELECT from_node, to_node FROM retrievals").fetchall() == [("node_0", "node_1")]


def test_end_to_end_continual_run(pv_root, tmp_path):
    from src.train import build_probe_loader, run_continual

    cfg = _cfg(pv_root, tmp_path, ema_threshold=1.1)  # threshold > 1: every node stays a learner
    dataset = PlantVillageDataset(pv_root, image_size=32)
    splits = build_continual_splits(cfg, dataset)
    out = tmp_path / "out"
    tracker = ComputeEnergyTracker(enabled=False, output_dir=out)
    comm = CommunicationCostEstimator({"wifi": 0.00003}, 125)

    result = run_continual(
        cfg, "mobilenet_v3_small", dataset, splits, build_probe_loader(cfg, dataset, splits.probe_idx),
        tracker, comm, "cpu", out,
    )

    summary = result["continual"]
    assert summary["num_batches"] == 3
    batch0 = summary["per_batch"][0]
    assert batch0["teachers"] == batch0["learners"] == ["node_0", "node_1", "node_2"]
    assert summary["num_distillations"] >= 3  # at least every node in batch 0
    assert summary["total_bytes_uploaded"] > 0 and summary["total_bytes_downloaded"] > 0
    assert summary["total_compute_energy_kwh"] > 0
    assert set(result["mesh_eval"]) == {"node_0", "node_1", "node_2"}

    logs = json.loads((out / "continual" / "mobilenet_v3_small" / "batch_logs.json").read_text())
    first = logs[0]["per_node"]["node_0"]
    assert sorted(first["peers_used"]) == ["node_1", "node_2"]  # own knowledge excluded
    assert first["bytes_uploaded"] > 0 and first["bytes_downloaded"] > 0
    assert {"local_train", "evaluate", "knowledge_extraction", "distill"} <= set(first["energy_kwh"])
    assert (out / "continual" / "mobilenet_v3_small" / "batch_summary.csv").exists()
    assert (out / "checkpoints" / "mobilenet_v3_small" / "node_0.pt").exists()
