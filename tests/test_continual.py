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
        "continual": {
            "first_batch_size": 60, "next_batch_size": 24, "first_batch_local_epochs": 1,
            "labeled_fraction": 1.0, **continual,
        },
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


# ---------------------------------------------------------------- self-learning


def test_split_node_arrival_hides_labels_for_most_train_images(pv_root):
    dataset = PlantVillageDataset(pv_root, image_size=32)
    indices = list(range(len(dataset)))
    split = split_node_arrival(dataset, indices, test_fraction=0.15, seed=0, labeled_fraction=0.25)
    full = split_node_arrival(dataset, indices, test_fraction=0.15, seed=0)

    assert split.test_idx == full.test_idx  # test images always keep their labels
    assert sorted(split.train_idx + split.unlabeled_idx) == sorted(full.train_idx)
    assert not set(split.train_idx) & set(split.unlabeled_idx)
    assert len(split.unlabeled_idx) > 2 * len(split.train_idx)
    targets = np.asarray(dataset.targets)
    assert set(targets[split.train_idx].tolist()) == set(range(len(CLASSES)))  # every class keeps a label


def test_prototype_check_rejects_pseudo_labels_the_peers_disagree_with():
    from src.federated.node import _agrees_with_prototypes

    prototypes = {("disease", 0): torch.tensor([1.0, 0.0]), ("disease", 1): torch.tensor([0.0, 1.0])}
    feats = torch.tensor([[0.9, 0.1], [0.9, 0.1], [0.1, 0.9]])
    labels = torch.tensor([0, 1, 2])  # agrees / nearest is class 0 / class 2 has no prototype
    assert _agrees_with_prototypes(feats, labels, prototypes, "disease").tolist() == [True, False, True]
    assert _agrees_with_prototypes(feats, labels, prototypes, "crop").all()  # nothing to check against


def _node_with_unlabeled(pv_root, threshold):
    from torch.utils.data import DataLoader

    from src.data.plantvillage import TwoViewSubset, make_subset
    from src.federated.node import Node
    from src.models.factory import build_model

    dataset = PlantVillageDataset(pv_root, image_size=32)
    num_crop, num_disease = len(dataset.labels.crop_classes), len(dataset.labels.disease_classes)
    labelled = DataLoader(make_subset(dataset, list(range(0, 144, 9)), train=True), batch_size=8)
    unlabelled = DataLoader(TwoViewSubset(dataset, list(range(1, 144, 3))), batch_size=16, shuffle=True)
    node = Node(
        "node_0", build_model("mobilenet_v3_small", num_crop, num_disease, pretrained=False), labelled, labelled,
        pseudo_threshold=threshold,
    )
    return node, unlabelled


def test_local_train_learns_from_confident_pseudo_labels_only(pv_root):
    node, unlabelled = _node_with_unlabeled(pv_root, threshold=0.0)  # everything is "confident"
    node.local_train(1, 1e-3, unlabeled_loader=unlabelled)
    stats = node.reset_pseudo_stats()
    assert stats["seen"] == 32 and stats["disease_accepted"] == 32  # 2 labelled steps x 16 unlabelled
    assert node.pseudo_stats["seen"] == 0  # reset

    node.pseudo_threshold = 1.1  # nothing can be that confident
    node.local_train(1, 1e-3, unlabeled_loader=unlabelled)
    stats = node.reset_pseudo_stats()
    assert stats["seen"] == 32 and stats["crop_accepted"] == stats["disease_accepted"] == 0


def test_cli_later_batches_are_mostly_unlabelled_and_pseudo_labelled(pv_root, tmp_path, monkeypatch):
    import yaml

    cfg = _cfg(pv_root, tmp_path, ema_threshold=1.1, next_batch_size=60,
               labeled_fraction=0.3, pseudo_label_threshold=0.0, unlabeled_batch_ratio=2)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg.as_dict()))
    out = tmp_path / "out"

    _run_cli(monkeypatch, cfg_path)                  # batch 0: fully labelled
    _run_cli(monkeypatch, cfg_path, "--next-batch")  # batch 1: 30% labelled

    stream = json.loads((out / "continual" / "stream.json").read_text())
    assert [b["labeled_fraction"] for b in stream["batches"]] == [1.0, 0.3]
    assert all(not n["unlabeled_idx"] for n in stream["batches"][0]["nodes"].values())
    assert any(n["unlabeled_idx"] for n in stream["batches"][1]["nodes"].values())

    logs = json.loads((out / "continual" / "mobilenet_v3_small" / "batch_logs.json").read_text())
    assert all(not r["pseudo_labels"] for r in logs[0]["per_node"].values())
    later = [r for r in logs[1]["per_node"].values() if r["num_unlabeled"]]
    assert later
    for r in later:
        local, distill = r["pseudo_labels"]["local_train"], r["pseudo_labels"]["distill"]  # all learners
        assert local["seen"] > 0 and local["disease_accepted"] == local["seen"]  # confidence threshold 0
        # during distillation the peer-prototype check can only reject more
        assert distill["seen"] > 0 and distill["disease_accepted"] <= distill["seen"]

    summary = json.loads((out / "results_summary.json").read_text())["mobilenet_v3_small"]["continual"]
    assert 0.0 < summary["pseudo_labels"]["pseudo_disease_accept_rate"] <= 1.0
