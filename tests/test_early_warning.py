"""Early-warning test: the farm_crops partition, the held-out photos of
diseases on a node's crops that only other nodes have, and the extra
before/after-distillation test on them — in the simulation CLI and on the
edge."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.config import Config
from src.data.plantvillage import PlantVillageDataset, farm_crop_assignment, partition_nodes
from src.data.stream import DataStream, threat_classes

CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Apple___Black_rot",
    "Apple___healthy",
]
PLANTVILLAGE_CROPS = [
    "Apple", "Blueberry", "Cherry_(including_sour)", "Corn_(maize)", "Grape", "Orange", "Peach",
    "Pepper,_bell", "Potato", "Raspberry", "Soybean", "Squash", "Strawberry", "Tomato",
]


@pytest.fixture
def pv_root(tmp_path):
    root = tmp_path / "PlantVillage"
    rng = np.random.RandomState(0)
    for cls in CLASSES:
        (root / cls).mkdir(parents=True)
        for i in range(30):
            arr = rng.randint(0, 255, size=(32, 32, 3), dtype=np.uint8)
            Image.fromarray(arr).save(root / cls / f"img_{i}.jpg")
    return root


def _cfg(root, tmp_path, **data) -> Config:
    return Config({
        "data": {
            "root": str(root), "image_size": 32, "num_nodes": 3, "seed": 0,
            "probe_set_fraction": 0.05, "test_fraction": 0.25,
            "non_iid_strategy": "farm_crops", "farm_crops": {"growers_per_crop": 2, "disease_spread": 0.5},
            "early_warning": {"enabled": True, "holdout_fraction": 0.2, "max_images_per_class": 4},
            **data,
        },
        "continual": {
            "first_batch_size": 70, "next_batch_size": 40, "first_batch_local_epochs": 1, "labeled_fraction": 1.0,
        },
        "models": {"architectures": ["mobilenet_v3_small"], "pretrained": False},
        "training": {
            "local_epochs_per_round": 1, "distill_epochs_per_round": 1, "batch_size": 8,
            "lr": 1e-3, "distill_lr": 5e-4, "proto_weight": 0.1, "kd_weight": 0.5, "kd_temperature": 2.0,
        },
        "federated": {"aggregation": "trimmed_mean", "trim_fraction": 0.2, "krum_neighbors": 1},
        "energy": {"track_with_codecarbon": False, "radio_energy_j_per_byte": {"wifi": 0.00003}},
        "output": {"dir": str(tmp_path / "out")},
        "edge": {"dir": str(tmp_path / "edge"), "round_timeout_s": 0, "poll_interval_s": 0, "export_onnx": False},
    })


def test_farm_crop_assignment_is_balanced_and_mixes_neighbours():
    node_crops = farm_crop_assignment(PLANTVILLAGE_CROPS, num_nodes=6, growers_per_crop=2, seed=42)
    growers = Counter(crop for crops in node_crops for crop in crops)
    assert set(growers) == set(PLANTVILLAGE_CROPS) and set(growers.values()) == {2}
    assert max(map(len, node_crops)) - min(map(len, node_crops)) <= 1
    # no fixed pairs: some farm shares crops with more than one other farm
    partners = [{m for m in range(6) if m != n and set(node_crops[n]) & set(node_crops[m])} for n in range(6)]
    assert max(map(len, partners)) >= 2
    with pytest.raises(ValueError):
        farm_crop_assignment(PLANTVILLAGE_CROPS, num_nodes=3, growers_per_crop=4, seed=0)


def test_farm_crops_partition_spreads_healthy_to_all_growers_and_diseases_to_some(pv_root):
    dataset = PlantVillageDataset(pv_root, image_size=32)
    indices = list(range(len(dataset)))
    shards = partition_nodes(dataset, indices, 3, "farm_crops", 0.5, seed=0,
                             farm_crops={"growers_per_crop": 2, "disease_spread": 0.5})
    assert sorted(i for s in shards for i in s) == indices  # every image exactly once
    node_crops = farm_crop_assignment(dataset.labels.crop_classes, 3, 2, seed=0)
    for cls, name in enumerate(dataset.base.classes):
        crop = name.split("___")[0]
        owners = [n for n, s in enumerate(shards) if any(dataset.targets[i] == cls for i in s)]
        growers = [n for n in range(3) if crop in node_crops[n]]
        assert set(owners) <= set(growers)
        # healthy leaves reach every grower; a disease has reached only one of the two so far
        assert len(owners) == (2 if name.endswith("healthy") else 1)


def test_stream_holds_threat_photos_out_of_every_shard_and_batch(pv_root, tmp_path):
    cfg = _cfg(pv_root, tmp_path)
    dataset = PlantVillageDataset(pv_root, image_size=32)
    stream = DataStream.create(cfg, dataset, tmp_path / "stream.json")
    held = {i for idx in stream.early_warning.values() for i in idx}
    assert held and not held & {i for shard in stream.node_shards for i in shard}

    threats = threat_classes(dataset, stream.node_shards, stream.node_crops)
    for n in range(3):
        node_id = f"node_{n}"
        classes = stream.early_warning_classes(dataset, node_id)
        assert classes == threats[n] and classes  # every node waits for some disease
        owned = {dataset.targets[i] for i in stream.node_shards[n]}
        for c in classes:
            name = dataset.base.classes[c]
            assert name.split("___")[0] in stream.node_crops[n]  # on its own crop
            assert c not in owned and not name.endswith("healthy")  # it never receives it
            assert any(c in {dataset.targets[i] for i in s} for m, s in enumerate(stream.node_shards) if m != n)
        idx = stream.early_warning_idx(dataset, node_id, max_per_class=4)
        assert len(idx) <= 4 * len(classes) and set(idx) <= held

    stream.next_batch(dataset, 70, 0.05, 0.25, seed=0)
    stream.next_batch(dataset, 40, 0.05, 0.25, seed=0)
    used = {i for b in stream.batches for i in b["probe_idx"]}
    used |= {i for b in stream.batches for s in b["nodes"].values()
             for i in s["train_idx"] + s["test_idx"] + s["unlabeled_idx"]}
    assert not used & held  # never trained on, never in a probe
    reloaded = DataStream.load(cfg, dataset, tmp_path / "stream.json")
    assert reloaded.early_warning == stream.early_warning and reloaded.node_crops == stream.node_crops


def test_a_stream_saved_before_early_warning_still_loads(pv_root, tmp_path):
    cfg = _cfg(pv_root, tmp_path, non_iid_strategy="dirichlet", early_warning={"enabled": False})
    dataset = PlantVillageDataset(pv_root, image_size=32)
    stream = DataStream.create(cfg, dataset, tmp_path / "stream.json")
    data = json.loads((tmp_path / "stream.json").read_text())
    del data["node_crops"], data["early_warning"]
    (tmp_path / "stream.json").write_text(json.dumps(data))
    old = DataStream.load(cfg, dataset, tmp_path / "stream.json")
    assert old.early_warning == {} and old.early_warning_idx(dataset, "node_0") == []
    assert old.node_shards == stream.node_shards


def _run_cli(monkeypatch, cfg_path, *flags):
    from src import train

    monkeypatch.setattr(sys, "argv", ["train", "--config", str(cfg_path), *flags])
    train.main()


def test_cli_tests_each_node_on_its_peers_only_diseases_before_and_after_distilling(pv_root, tmp_path, monkeypatch):
    import yaml

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(_cfg(pv_root, tmp_path).as_dict()))
    run_dir = tmp_path / "out" / "continual" / "mobilenet_v3_small"
    _run_cli(monkeypatch, cfg_path)
    _run_cli(monkeypatch, cfg_path, "--next-batch")

    logs = json.loads((run_dir / "batch_logs.json").read_text())
    tested = [r for log in logs for r in log["per_node"].values() if r["num_early_warning"]]
    assert tested
    for r in tested:
        assert {"disease_accuracy", "pair_accuracy"} <= set(r["pre_early_warning_eval"])
        assert {"disease_accuracy", "pair_accuracy"} <= set(r["post_early_warning_eval"])
        if not r["distilled"]:
            assert r["post_early_warning_eval"] == r["pre_early_warning_eval"]
        # the extra test is measurement, not part of the system's energy
        assert r["measurement_energy_kwh"] == r["energy_kwh"]["early_warning_eval"] > 0
        assert r["total_energy_kwh"] == pytest.approx(sum(r["energy_kwh"].values()) - r["measurement_energy_kwh"])

    with open(run_dir / "batch_summary.csv") as f:
        rows = list(csv.DictReader(f))
    assert {"num_early_warning", "pre_ew_disease_accuracy", "post_ew_disease_accuracy", "gain_ew_disease_accuracy"} <= set(rows[0])
    summary = json.loads((tmp_path / "out" / "results_summary.json").read_text())["mobilenet_v3_small"]["continual"]
    assert summary["early_warning"]["node_batches_tested"] == len(tested)


def test_edge_agents_run_the_early_warning_test_from_their_manifest(pv_root, tmp_path):
    from src.edge.client import KnowledgeClient
    from src.edge.data import inbox_path
    from src.edge.feeder import trigger
    from src.edge.node_agent import NodeAgent
    from src.edge.server import create_app

    cfg = _cfg(pv_root, tmp_path)
    client = KnowledgeClient(TestClient(create_app(tmp_path / "server" / "knowledge.db")))
    edge_dir = cfg.get("edge.dir")
    trigger(cfg, client, edge_dir, next_batch=False)
    manifest = json.loads(inbox_path(edge_dir, "node_0", 0).read_text())
    assert manifest["early_warning"] and all(not e["class"].endswith("healthy") for e in manifest["early_warning"])

    agents = [NodeAgent(cfg, f"node_{i}", client, edge_dir, device="cpu") for i in range(3)]
    contexts = [agent.prepare(0) for agent in agents]
    for agent, ctx in zip(agents, contexts):
        agent.train_and_upload(ctx)
    for agent, ctx in zip(agents, contexts):
        record = agent.learn_and_finish(ctx)
        assert record.num_early_warning == len(json.loads(inbox_path(edge_dir, agent.node_id, 0).read_text())["early_warning"])
        assert "disease_accuracy" in record.post_early_warning_eval
