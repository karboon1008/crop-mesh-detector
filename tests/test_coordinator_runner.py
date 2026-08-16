# tests/test_coordinator_runner.py
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docker" / "coordinator"))

from coordinator_runner import CoordinatorRunner  # noqa: E402
from src.energy import sqlite_store


def _make_runner(tmp_path, post_all, health_check=lambda n: True, num_rounds=1):
    return CoordinatorRunner(
        expected_nodes=["node_0", "node_1"],
        node_base_urls={"node_0": "http://node_0:8000", "node_1": "http://node_1:8000"},
        num_rounds=num_rounds,
        round_timeout_s=30,
        db_path=str(tmp_path / "merged.db"),
        post_all=post_all,
        health_check=health_check,
    )


def test_wait_until_all_online_blocks_until_health_check_passes_for_every_node():
    calls = {"n": 0}

    def health_check(_node_id):
        calls["n"] += 1
        return calls["n"] > 2  # first couple of checks report unhealthy

    runner = CoordinatorRunner(
        expected_nodes=["node_0"],
        node_base_urls={"node_0": "http://node_0:8000"},
        num_rounds=1,
        round_timeout_s=30,
        db_path="unused.db",
        post_all=lambda *a: {},
        health_check=health_check,
    )
    slept = []
    runner.wait_until_all_online(sleep_fn=slept.append, poll_interval_s=0.1)
    assert len(slept) >= 1


def test_run_round_writes_energy_and_eval_rows_from_response_bodies(tmp_path):
    def post_all(node_ids, path, body):
        if path == "/round/start":
            return {
                n: {"energy_kwh": 0.01, "duration_s": 1.0, "energy_method": "proxy_wall_power", "size_bytes": 100}
                for n in node_ids
            }
        assert path == "/round/gather"
        assert body["active_nodes"] == ["node_0", "node_1"]
        return {n: {"crop_accuracy": 0.5, "disease_accuracy": 0.6} for n in node_ids}

    runner = _make_runner(tmp_path, post_all)
    active = runner.run_round(0)

    assert active == ["node_0", "node_1"]
    rows = {r["node_id"]: r for r in sqlite_store.read_all(str(tmp_path / "merged.db"))}
    assert rows["node_0"]["crop_accuracy"] == 0.5
    assert rows["node_1"]["knowledge_bytes_sent"] == 100


def test_run_round_excludes_a_node_that_failed_round_start(tmp_path):
    def post_all(node_ids, path, body):
        if path == "/round/start":
            return {
                "node_0": {"energy_kwh": 0.01, "duration_s": 1.0, "energy_method": "x", "size_bytes": 10},
                "node_1": None,  # timed out / errored
            }
        assert set(body["active_nodes"]) == {"node_0"}
        return {n: {"crop_accuracy": 0.5, "disease_accuracy": 0.5} for n in node_ids}

    runner = _make_runner(tmp_path, post_all)
    active = runner.run_round(0)
    assert active == ["node_0"]


def test_run_all_rounds_calls_run_round_for_every_configured_round(tmp_path):
    seen_rounds = []

    def post_all(node_ids, path, body):
        if path == "/round/start":
            seen_rounds.append(body["round_idx"])
            return {
                n: {"energy_kwh": 0.0, "duration_s": 0.0, "energy_method": "x", "size_bytes": 1} for n in node_ids
            }
        return {n: {"crop_accuracy": 0.0, "disease_accuracy": 0.0} for n in node_ids}

    runner = _make_runner(tmp_path, post_all, num_rounds=3)
    runner.run_all_rounds()
    assert seen_rounds == [0, 1, 2]
