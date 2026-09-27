"""Hive experiments: source-based partition (lab vs field photos), offline nodes
(with/without purging their hive entry), a withheld disease, and the extra
per-source / per-class evaluations."""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest
import yaml
from PIL import Image

from src.config import Config
from src.data.multi_source import by_source_shards, load_dataset
from src.data.splits import BatchSplit
from src.federated.knowledge_store import KnowledgeStore
from src.train import scheduled_split
from tests.test_continual import CLASSES, _cfg, _run_cli, pv_root  # noqa: F401 (fixture)


def _images(root, classes, n):
    rng = np.random.RandomState(len(str(root)))
    for cls in classes:
        (root / cls).mkdir(parents=True)
        for i in range(n):
            Image.fromarray(rng.randint(0, 255, size=(32, 32, 3), dtype=np.uint8)).save(root / cls / f"img_{i}.jpg")
    return root


@pytest.fixture
def two_sources(pv_root, tmp_path):
    # a "field" source whose folders map onto the PlantVillage classes (2 of them)
    return pv_root, _images(tmp_path / "PlantDoc", ["Tomato___healthy", "Tomato___Bacterial_spot"], 30)


def _source_cfg(pv_root, pd_root, tmp_path) -> Config:
    cfg = _cfg(pv_root, tmp_path, ema_threshold=1.1)
    cfg._data["data"].update({
        "num_nodes": 2, "non_iid_strategy": "by_source",
        "extra_sources": {"plantdoc": [str(pd_root)]},
        "source_nodes": {"plantvillage": "node_0", "plantdoc": "node_1"},
    })
    return cfg


def test_by_source_gives_each_node_exactly_its_own_source(two_sources, tmp_path):
    pv_root, pd_root = two_sources
    cfg = _source_cfg(pv_root, pd_root, tmp_path)
    dataset = load_dataset(cfg)
    shards = by_source_shards(dataset, cfg)
    paths = [p for p, _ in dataset.base.samples]
    assert len(shards[0]) == 24 * len(CLASSES) and len(shards[1]) == 60
    assert all("PlantVillage" in paths[i] for i in shards[0])
    assert all("PlantDoc" in paths[i] for i in shards[1])


def test_scheduled_split_offline_and_withhold(pv_root, tmp_path):
    cfg = _cfg(pv_root, tmp_path, offline={"node_1": [2, 3]},
               withhold={"node": "node_0", "classes": ["Tomato___healthy"], "until_batch": 2})
    dataset = load_dataset(cfg)
    healthy = dataset.base.class_to_idx["Tomato___healthy"]
    n = len(dataset)
    split = BatchSplit(list(range(0, n, 2)), list(range(1, n, 2)))  # every class, both halves

    assert scheduled_split(cfg, dataset, 2, "node_1", split) is None          # offline
    assert scheduled_split(cfg, dataset, 1, "node_1", split) is split         # online again
    early = scheduled_split(cfg, dataset, 1, "node_0", split)
    assert healthy not in {dataset.targets[i] for i in early.train_idx + early.test_idx}
    assert len(early.train_idx) + len(early.test_idx) == n - 24  # exactly the 24 healthy-tomato images gone
    assert scheduled_split(cfg, dataset, 2, "node_0", split) is split         # the disease has arrived


def test_store_delete_drops_only_the_live_entry(tmp_path):
    from src.federated.node import KnowledgePayload
    import torch

    store = KnowledgeStore(tmp_path / "k.db")
    payload = KnowledgePayload({("crop", 0): torch.ones(4)}, torch.zeros(2, 3), torch.zeros(2, 5), {0: 3}, {1: 3})
    store.upload("node_0", 0, payload)
    store.upload("node_1", 0, payload)
    store.delete("node_1")
    assert [e["node_id"] for e in store.entries()] == ["node_0"]
    with sqlite3.connect(tmp_path / "k.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM uploads").fetchone()[0] == 2  # history kept


def _run_two_batches(monkeypatch, cfg, tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg.as_dict()))
    _run_cli(monkeypatch, cfg_path)
    _run_cli(monkeypatch, cfg_path, "--next-batch")
    return tmp_path / "out" / "continual" / "mobilenet_v3_small"


@pytest.mark.parametrize("purge", [False, True])
def test_offline_node_sits_out_and_purge_removes_its_knowledge(pv_root, tmp_path, monkeypatch, purge):
    cfg = _cfg(pv_root, tmp_path, ema_threshold=1.1, next_batch_size=40,
               offline={"node_1": [1]}, offline_purge=purge)
    run_dir = _run_two_batches(monkeypatch, cfg, tmp_path)
    logs = json.loads((run_dir / "batch_logs.json").read_text())
    assert logs[1]["absent"] == ["node_1"]
    peers = logs[1]["per_node"]["node_0"]["peers_used"]
    assert ("node_1" in peers) is (not purge)  # kept: its batch-0 entry is still served; purged: gone
    pooled = json.loads((run_dir / "pooled_log.json").read_text())
    assert "pair_recall" in pooled[-1]["post"]  # per-class recall for the withheld-disease analysis


def test_by_source_run_scores_every_node_on_every_source(two_sources, tmp_path, monkeypatch):
    pv_root, pd_root = two_sources
    cfg = _source_cfg(pv_root, pd_root, tmp_path)
    cfg._data["continual"].update({"first_batch_size": 80, "next_batch_size": 40})
    run_dir = _run_two_batches(monkeypatch, cfg, tmp_path)
    log = json.loads((run_dir / "pooled_by_source.json").read_text())
    last = [e for e in log if e["batch"] == 1]
    assert {(e["node"], e["source"]) for e in last} == {
        (n, s) for n in ("node_0", "node_1") for s in ("plantdoc", "plantvillage")
    }
    assert all(0.0 <= e["post"]["pair_accuracy"] <= 1.0 for e in last)
