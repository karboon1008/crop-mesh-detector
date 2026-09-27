"""Control groups (FedAvg / D-PSGD on the continual stream): the averaging and byte
accounting, the EMA gating, and the pooled-test log, on synthetic images."""

from __future__ import annotations

import json

import pytest
import torch
import yaml

from src.federated.weight_mesh import ring_neighbors, state_bytes, weighted_average
from tests.test_continual import _cfg, _run_cli, pv_root  # noqa: F401  (pv_root is a fixture)


def test_ring_neighbors():
    assert ring_neighbors(0, 1) == []
    assert ring_neighbors(0, 2) == [1] and ring_neighbors(1, 2) == [0]
    assert ring_neighbors(0, 4) == [1, 3] and ring_neighbors(2, 4) == [1, 3]
    assert ring_neighbors(5, 6) == [0, 4]


def test_weighted_average_is_sample_weighted_and_keeps_counters():
    a = {"w": torch.tensor([0.0, 0.0]), "n": torch.tensor(7)}
    b = {"w": torch.tensor([4.0, 8.0]), "n": torch.tensor(9)}
    merged = weighted_average([a, b], [1, 3])
    assert torch.allclose(merged["w"], torch.tensor([3.0, 6.0]))
    assert merged["n"].item() == 7  # integer buffers are counters, not averaged
    assert a["w"].tolist() == [0.0, 0.0]  # inputs untouched


def _setup(pv_root, tmp_path, monkeypatch, strategy, **continual):  # noqa: F811
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(_cfg(pv_root, tmp_path, next_batch_size=40, **continual).as_dict()))
    run_dir = tmp_path / "out" / "continual" / "mobilenet_v3_small"
    return cfg_path, run_dir, ["--strategy", strategy]


def _model_bytes(run_dir):
    return state_bytes(torch.load(run_dir / "nodes" / "node_0.pt")["model"])


def test_fedavg_every_batch_shares_one_average_and_counts_up_plus_down(pv_root, tmp_path, monkeypatch):  # noqa: F811
    cfg_path, run_dir, flags = _setup(pv_root, tmp_path, monkeypatch, "fedavg")
    _run_cli(monkeypatch, cfg_path, *flags)
    states = [torch.load(run_dir / "nodes" / f"node_{i}.pt")["model"] for i in range(3)]
    for other in states[1:]:  # after batch 0 every node adopted the same averaged weights
        assert all(torch.equal(states[0][k], other[k]) for k in states[0])
    _run_cli(monkeypatch, cfg_path, "--next-batch", *flags)
    logs = json.loads((run_dir / "batch_logs.json").read_text())
    w = _model_bytes(run_dir)
    assert len(logs) == 2
    for log in logs:
        for r in log["per_node"].values():
            assert r["roles"] == ["teacher", "learner"] and r["uploaded"] and r["distilled"]
            assert r["bytes_uploaded"] == r["bytes_downloaded"] == w  # one upload + one download of the full model
            assert {"local_train", "evaluate", "upload", "exchange"} <= set(r["energy_kwh"])


def test_dpsgd_ring_pushes_to_each_neighbour_and_downloads_nothing(pv_root, tmp_path, monkeypatch):  # noqa: F811
    cfg_path, run_dir, flags = _setup(pv_root, tmp_path, monkeypatch, "dpsgd")
    _run_cli(monkeypatch, cfg_path, *flags)
    w = _model_bytes(run_dir)
    for r in json.loads((run_dir / "batch_logs.json").read_text())[0]["per_node"].values():
        assert r["bytes_uploaded"] == 2 * w and r["bytes_downloaded"] == 0  # 3 nodes: two neighbours each
        assert sorted(r["peers_used"]) == sorted(set(f"node_{i}" for i in range(3)) - {r["node_id"]})


def test_gated_variant_follows_the_continual_ema_rules(pv_root, tmp_path, monkeypatch):  # noqa: F811
    # threshold 0: nobody is ever below it, so after batch 0 nobody retrieves; EMA-rose still decides who uploads
    cfg_path, run_dir, flags = _setup(pv_root, tmp_path, monkeypatch, "fedavg_gated", ema_threshold=0.0)
    _run_cli(monkeypatch, cfg_path, *flags)
    _run_cli(monkeypatch, cfg_path, "--next-batch", *flags)
    first, second = json.loads((run_dir / "batch_logs.json").read_text())
    assert all(r["roles"] == ["teacher", "learner"] for r in first["per_node"].values())
    for r in second["per_node"].values():
        assert ("teacher" in r["roles"]) == (r["ema"] > r["prev_ema"]) and "learner" not in r["roles"]
        assert r["distilled"] is False and r["bytes_downloaded"] == 0
        assert r["uploaded"] == ("teacher" in r["roles"])


@pytest.mark.parametrize("strategy", ["continual", "fedavg"])
def test_pooled_log_scores_every_node_pre_and_post(pv_root, tmp_path, monkeypatch, strategy):  # noqa: F811
    cfg_path, run_dir, flags = _setup(pv_root, tmp_path, monkeypatch, strategy)
    _run_cli(monkeypatch, cfg_path, *flags)
    _run_cli(monkeypatch, cfg_path, "--next-batch", *flags)
    pooled = json.loads((run_dir / "pooled_log.json").read_text())
    logs = {(l["batch_idx"], nid): r for l in json.loads((run_dir / "batch_logs.json").read_text())
            for nid, r in l["per_node"].items()}
    assert {(e["batch"], e["node"]) for e in pooled} == set(logs)  # every node that took part, every batch
    for e in pooled:
        assert e["n_pooled"] > 0 and {"pair_accuracy", "crop_accuracy"} <= set(e["pre"]) and set(e["post"])
        if not logs[(e["batch"], e["node"])]["distilled"]:
            assert e["post"] == e["pre"]  # no exchange -> the model did not change


@pytest.mark.parametrize("threshold, later_learners", [(1.1, True), (0.0, False)])
def test_local_extra_sends_nothing_and_only_ema_learners_take_the_extra_epoch(  # noqa: F811
    pv_root, tmp_path, monkeypatch, threshold, later_learners,
):
    cfg_path, run_dir, flags = _setup(pv_root, tmp_path, monkeypatch, "local_extra", ema_threshold=threshold)
    _run_cli(monkeypatch, cfg_path, *flags)
    _run_cli(monkeypatch, cfg_path, "--next-batch", *flags)
    first, second = json.loads((run_dir / "batch_logs.json").read_text())
    for log in (first, second):
        for r in log["per_node"].values():
            assert r["bytes_uploaded"] == r["bytes_downloaded"] == 0 and r["uploaded"] is False
            assert r["distilled"] == ("learner" in r["roles"])  # the extra epoch goes exactly to EMA learners
            assert ("exchange" in r["energy_kwh"]) == r["distilled"]
    assert all(r["distilled"] for r in first["per_node"].values())  # batch 0: everyone is a learner
    assert all(r["distilled"] == later_learners for r in second["per_node"].values())
    states = [torch.load(run_dir / "nodes" / f"node_{i}.pt")["model"] for i in range(3)]
    assert any(not torch.equal(states[0][k], states[1][k]) for k in states[0])  # nothing was averaged
