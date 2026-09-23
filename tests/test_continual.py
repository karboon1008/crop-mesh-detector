"""Continual mesh: the stratified batch stream, the shared knowledge
database, the EMA teacher/learner rule, and the trigger-by-trigger CLI on
synthetic images."""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter

import numpy as np
import pytest
import torch
from PIL import Image

from src.config import Config
from src.data.plantvillage import PlantVillageDataset
from src.data.splits import split_node_arrival, stratified_sample
from src.data.stream import DataStream
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
        "continual": {"first_batch_size": 60, "next_batch_size": 24, "first_batch_local_epochs": 1, **continual},
        "models": {"architectures": ["mobilenet_v3_small"], "pretrained": False},
        "training": {
            "local_epochs_per_round": 1, "distill_epochs_per_round": 1, "batch_size": 8,
            "lr": 1e-3, "distill_lr": 5e-4, "proto_weight": 0.1, "kd_weight": 0.5, "kd_temperature": 2.0,
        },
        "federated": {"aggregation": "trimmed_mean", "trim_fraction": 0.2, "krum_neighbors": 1},
        "energy": {"track_with_codecarbon": False, "radio_energy_j_per_byte": {"wifi": 0.00003}},
        "output": {"dir": str(tmp_path / "out")},
    })


def test_stratified_sample_is_exact_and_proportional(pv_root):
    dataset = PlantVillageDataset(pv_root, image_size=32)
    pool = list(range(len(dataset)))
    sample = stratified_sample(dataset, pool, 60, seed=0)
    assert len(sample) == len(set(sample)) == 60
    counts = Counter(np.asarray(dataset.targets)[sample].tolist())
    assert set(counts.values()) == {10}  # 6 equal classes -> 10 each
    assert len(stratified_sample(dataset, pool[:5], 60, seed=0)) == 5  # capped at what's left


def test_split_node_arrival_keeps_at_least_one_test_image(pv_root):
    dataset = PlantVillageDataset(pv_root, image_size=32)
    one_per_class = [0, 24, 48]  # a single image of each of three classes
    split = split_node_arrival(dataset, one_per_class, test_fraction=0.2, seed=0)
    assert len(split.test_idx) == 1 and len(split.train_idx) == 2


def test_stream_draws_new_images_each_batch_and_survives_reload(pv_root, tmp_path):
    cfg = _cfg(pv_root, tmp_path)
    dataset = PlantVillageDataset(pv_root, image_size=32)
    path = tmp_path / "stream.json"
    stream = DataStream.create(cfg, dataset, path)
    pool_size = len(stream.remaining())
    first = stream.next_batch(dataset, 60, 0.05, 0.25, seed=0)

    reloaded = DataStream.load(cfg, dataset, path)  # a later trigger is a new process
    second = reloaded.next_batch(dataset, 40, 0.05, 0.25, seed=0)

    def node_images(batch):
        return {i for s in batch["nodes"].values() for i in s["train_idx"] + s["test_idx"]}

    for batch, size, probe_size in ((first, 60, 3), (second, 40, 2)):
        assert batch["size"] == size
        assert len(batch["probe_idx"]) == probe_size          # 5% of THIS batch
        assert not set(batch["probe_idx"]) & node_images(batch)  # probe images never reach a node
        assert len(node_images(batch)) == size - probe_size
    assert not (node_images(first) | set(first["probe_idx"])) & (node_images(second) | set(second["probe_idx"]))
    assert len(reloaded.remaining()) == pool_size - 100
    for node_i, shard in enumerate(stream.node_shards):  # each image goes to its owner
        split = first["nodes"][f"node_{node_i}"]
        assert set(split["train_idx"] + split["test_idx"]) <= set(shard)

    other = _cfg(pv_root, tmp_path)
    other._data["data"]["seed"] = 1
    with pytest.raises(ValueError, match="different dataset/config"):
        DataStream.load(other, dataset, path)


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

    # logits uploaded for another batch's probe set are left out
    stale = store.fetch_peers("node_0", batch_idx=1, logits_batch=1)["node_1"][1]
    assert stale.crop_logits.shape[0] == 0 and stale.prototypes  # prototypes still come through
    assert stale.size_bytes() < _payload(1.0).size_bytes()

    with sqlite3.connect(tmp_path / "k.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM uploads").fetchone()[0] == 3
        assert conn.execute("SELECT from_node, to_node FROM retrievals").fetchall() == [
            ("node_0", "node_1"), ("node_1", "node_0"),
        ]


def _run_cli(monkeypatch, cfg_path, *flags):
    from src import train

    monkeypatch.setattr(sys, "argv", ["train", "--config", str(cfg_path), *flags])
    train.main()


def test_cli_runs_batch_zero_then_one_batch_per_trigger(pv_root, tmp_path, monkeypatch):
    import yaml

    cfg_path = tmp_path / "cfg.yaml"
    cfg = _cfg(pv_root, tmp_path, ema_threshold=1.1, next_batch_size=40)
    cfg_path.write_text(yaml.safe_dump(cfg.as_dict()))
    out = tmp_path / "out"
    run_dir = out / "continual" / "mobilenet_v3_small"

    with pytest.raises(SystemExit):  # nothing to continue yet
        _run_cli(monkeypatch, cfg_path, "--next-batch")

    _run_cli(monkeypatch, cfg_path)  # batch 0
    logs = json.loads((run_dir / "batch_logs.json").read_text())
    assert len(logs) == 1
    first = logs[0]["per_node"]["node_0"]
    assert first["roles"] == ["teacher", "learner"]
    assert sorted(first["peers_used"]) == ["node_1", "node_2"]  # own knowledge excluded
    assert first["bytes_uploaded"] > 0 and first["bytes_downloaded"] > 0
    assert first["probe_images_used"] == 3  # 5% of the 60-image batch
    assert {"local_train", "evaluate", "knowledge_extraction", "distill"} <= set(first["energy_kwh"])
    weights_after_0 = torch.load(run_dir / "nodes" / "node_0.pt")["model"]

    _run_cli(monkeypatch, cfg_path)  # no trigger: nothing new runs
    assert len(json.loads((run_dir / "batch_logs.json").read_text())) == 1

    _run_cli(monkeypatch, cfg_path, "--next-batch")  # batch 1, on top of the saved models
    _run_cli(monkeypatch, cfg_path, "--next-batch")  # batch 2
    logs = json.loads((run_dir / "batch_logs.json").read_text())
    assert [log["batch_idx"] for log in logs] == [0, 1, 2]
    stream = json.loads((out / "continual" / "stream.json").read_text())
    assert [b["size"] for b in stream["batches"]] == [60, 40, 40]
    # probe logits only come from entries uploaded THIS batch (each batch has
    # its own 2-image probe); older entries contribute prototypes only
    for log in logs[1:]:
        for r in log["per_node"].values():
            if r["distilled"]:
                fresh = sorted(p for p, b in r["peers_used"].items() if b == log["batch_idx"])
                assert r["logit_peers"] == fresh
                assert r["probe_images_used"] == (2 if fresh else 0)
    assert json.loads((run_dir / "state.json").read_text())["completed_batches"] == 3
    later = [r for log in logs[1:] for r in log["per_node"].values()]
    assert later and all(r["prev_ema"] is not None for r in later)  # EMA carried across processes
    weights_after_2 = torch.load(run_dir / "nodes" / "node_0.pt")["model"]
    assert any(not torch.equal(weights_after_0[k], weights_after_2[k]) for k in weights_after_0)

    result = json.loads((out / "results_summary.json").read_text())["mobilenet_v3_small"]
    assert result["continual"]["num_batches"] == 3
    assert (run_dir / "batch_summary.csv").exists()
    assert (out / "checkpoints" / "mobilenet_v3_small" / "node_0.pt").exists()

    _run_cli(monkeypatch, cfg_path, "--reset")  # start over from batch 0
    assert len(json.loads((run_dir / "batch_logs.json").read_text())) == 1


def test_distill_without_fresh_peer_logits_uses_prototypes_only(pv_root):
    """No node uploaded this batch -> no probe logits line up with this
    batch's probe set; distillation runs on sup + prototype loss alone."""
    from torch.utils.data import DataLoader

    from src.data.plantvillage import make_subset
    from src.federated.node import Node
    from src.models.factory import build_model

    dataset = PlantVillageDataset(pv_root, image_size=32)
    num_crop, num_disease = len(dataset.labels.crop_classes), len(dataset.labels.disease_classes)
    loader = DataLoader(make_subset(dataset, list(range(16))), batch_size=8)
    node = Node("node_0", build_model("mobilenet_v3_small", num_crop, num_disease, pretrained=False), loader, loader)
    prototypes, _, _ = node.compute_prototypes()

    losses = node.distill(
        prototypes, None, torch.zeros(num_crop, dtype=torch.bool), None, torch.zeros(num_disease, dtype=torch.bool),
        None, epochs=1, lr=1e-3, proto_weight=0.1, kd_weight=0.5, crop_kd_weight=0.5, temperature=2.0,
    )
    assert losses["kd_loss"] == 0.0 and losses["crop_kd_loss"] == 0.0
    assert losses["sup_loss"] > 0
