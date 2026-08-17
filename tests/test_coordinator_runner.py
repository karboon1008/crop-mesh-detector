# tests/test_coordinator_runner.py
from __future__ import annotations

import json
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
                n: {
                    "energy_kwh": 0.01, "duration_s": 1.0, "energy_method": "proxy_wall_power", "size_bytes": 100,
                    "baseline_crop_accuracy": 0.0, "baseline_disease_accuracy": 0.0,
                    "baseline_energy_kwh": 0.0, "baseline_duration_s": 0.0,
                }
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


def test_run_round_multiplies_knowledge_bytes_by_the_number_of_fetching_peers(tmp_path):
    # size_bytes is ONE payload's size; in the HTTP pull model every other
    # active peer fetches it independently, so the bytes actually leaving a
    # node are size_bytes * (active_n - 1) -- the same multiplier
    # src/federated/mesh.py applies to RoundLog.total_bytes_exchanged.
    def post_all(node_ids, path, body):
        if path == "/round/start":
            return {
                n: {
                    "energy_kwh": 0.0, "duration_s": 0.0, "energy_method": "x", "size_bytes": 100,
                    "baseline_crop_accuracy": 0.0, "baseline_disease_accuracy": 0.0,
                    "baseline_energy_kwh": 0.0, "baseline_duration_s": 0.0,
                }
                for n in node_ids
            }
        return {n: {"crop_accuracy": 0.0, "disease_accuracy": 0.0} for n in node_ids}

    runner = CoordinatorRunner(
        expected_nodes=["node_0", "node_1", "node_2"],
        node_base_urls={n: f"http://{n}:8000" for n in ["node_0", "node_1", "node_2"]},
        num_rounds=1,
        round_timeout_s=30,
        db_path=str(tmp_path / "merged.db"),
        post_all=post_all,
        health_check=lambda n: True,
    )
    runner.run_round(0)

    rows = sqlite_store.read_all(str(tmp_path / "merged.db"))
    assert [r["knowledge_bytes_sent"] for r in rows] == [200, 200, 200]


def test_run_round_records_zero_knowledge_bytes_when_only_one_node_is_active(tmp_path):
    # A lone active node has nobody to serve its payload to, so nothing is
    # actually transmitted -- mesh.py's max(0, active_n - 1) gives 0 here too.
    def post_all(node_ids, path, body):
        if path == "/round/start":
            return {
                "node_0": {
                    "energy_kwh": 0.0, "duration_s": 0.0, "energy_method": "x", "size_bytes": 100,
                    "baseline_crop_accuracy": 0.0, "baseline_disease_accuracy": 0.0,
                    "baseline_energy_kwh": 0.0, "baseline_duration_s": 0.0,
                },
                "node_1": None,
            }
        return {n: {"crop_accuracy": 0.0, "disease_accuracy": 0.0} for n in node_ids}

    runner = _make_runner(tmp_path, post_all)
    runner.run_round(0)

    rows = sqlite_store.read_all(str(tmp_path / "merged.db"))
    assert len(rows) == 1
    assert rows[0]["knowledge_bytes_sent"] == 0


def test_run_round_excludes_a_node_that_failed_round_start(tmp_path):
    def post_all(node_ids, path, body):
        if path == "/round/start":
            return {
                "node_0": {
                    "energy_kwh": 0.01, "duration_s": 1.0, "energy_method": "x", "size_bytes": 10,
                    "baseline_crop_accuracy": 0.0, "baseline_disease_accuracy": 0.0,
                    "baseline_energy_kwh": 0.0, "baseline_duration_s": 0.0,
                },
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
                n: {
                    "energy_kwh": 0.0, "duration_s": 0.0, "energy_method": "x", "size_bytes": 1,
                    "baseline_crop_accuracy": 0.0, "baseline_disease_accuracy": 0.0,
                    "baseline_energy_kwh": 0.0, "baseline_duration_s": 0.0,
                } for n in node_ids
            }
        return {n: {"crop_accuracy": 0.0, "disease_accuracy": 0.0} for n in node_ids}

    runner = _make_runner(tmp_path, post_all, num_rounds=3)
    runner.run_all_rounds()
    assert seen_rounds == [0, 1, 2]


def _stub_post_all(node_ids, path, body):
    if path == "/round/start":
        return {
            n: {
                "energy_kwh": 0.0, "duration_s": 0.0, "energy_method": "x", "size_bytes": 1,
                "baseline_crop_accuracy": 0.0, "baseline_disease_accuracy": 0.0,
                "baseline_energy_kwh": 0.0, "baseline_duration_s": 0.0,
            } for n in node_ids
        }
    return {n: {"crop_accuracy": 0.0, "disease_accuracy": 0.0} for n in node_ids}


def test_run_round_records_activity_log_entries(tmp_path):
    runner = _make_runner(tmp_path, _stub_post_all)
    runner.run_round(0)

    stages = " ".join(e["message"] for e in runner.activity_log)
    assert "round 0" in stages
    assert all("ts" in e and "message" in e for e in runner.activity_log)


def test_run_all_rounds_writes_status_file_when_status_path_set(tmp_path):
    status_path = tmp_path / "status.json"
    runner = CoordinatorRunner(
        expected_nodes=["node_0", "node_1"],
        node_base_urls={"node_0": "http://node_0:8000", "node_1": "http://node_1:8000"},
        num_rounds=2,
        round_timeout_s=30,
        db_path=str(tmp_path / "merged.db"),
        post_all=_stub_post_all,
        health_check=lambda n: True,
        status_path=str(status_path),
    )
    runner.run_all_rounds()

    assert status_path.exists()
    status = json.loads(status_path.read_text())
    assert status["all_rounds_complete"] is True
    assert status["num_rounds"] == 2
    assert "completed_at" in status


def test_run_all_rounds_skips_status_file_when_status_path_is_none(tmp_path):
    runner = _make_runner(tmp_path, _stub_post_all, num_rounds=1)
    runner.run_all_rounds()
    assert not (tmp_path / "status.json").exists()


def test_log_file_receives_one_json_line_per_log_call(tmp_path):
    log_path = tmp_path / "coordinator.log"
    runner = _make_runner(tmp_path, lambda *a: {})
    runner.log_file = str(log_path)
    runner._log("hello")

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["message"] == "hello"


def test_run_round_persists_baseline_fields_from_round_start_response(tmp_path):
    def post_all(node_ids, path, body):
        if path == "/round/start":
            return {
                n: {
                    "energy_kwh": 0.01, "duration_s": 1.0, "energy_method": "proxy_wall_power",
                    "size_bytes": 100,
                    "baseline_crop_accuracy": 0.55, "baseline_disease_accuracy": 0.45,
                    "baseline_energy_kwh": 0.009, "baseline_duration_s": 0.9,
                }
                for n in node_ids
            }
        return {n: {"crop_accuracy": 0.5, "disease_accuracy": 0.6} for n in node_ids}

    runner = _make_runner(tmp_path, post_all)
    runner.run_round(0)

    rows = {r["node_id"]: r for r in sqlite_store.read_all(str(tmp_path / "merged.db"))}
    assert rows["node_0"]["baseline_crop_accuracy"] == 0.55
    assert rows["node_0"]["baseline_disease_accuracy"] == 0.45
    assert rows["node_0"]["baseline_energy_kwh"] == 0.009
    assert rows["node_0"]["baseline_duration_s"] == 0.9
