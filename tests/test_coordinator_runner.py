# tests/test_coordinator_runner.py
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docker" / "coordinator"))

from coordinator_runner import CoordinatorRunner  # noqa: E402


def _make_runner(tmp_path, post_all, health_check=lambda n: True, num_rounds=1):
    return CoordinatorRunner(
        expected_nodes=["node_0", "node_1"],
        node_base_urls={"node_0": "http://node_0:8000", "node_1": "http://node_1:8000"},
        num_rounds=num_rounds,
        round_timeout_s=30,
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
        post_all=lambda *a: {},
        health_check=health_check,
    )
    slept = []
    runner.wait_until_all_online(sleep_fn=slept.append, poll_interval_s=0.1)
    assert len(slept) >= 1


def test_run_round_dispatches_start_then_gather_to_the_active_nodes(tmp_path):
    # The coordinator only ever sequences /round/start -> /round/gather and
    # reports back which nodes were active -- it never reads energy/accuracy
    # fields out of the response bodies (each node persists those into its
    # own db itself; see test_node_runner.py). No round_metrics db is
    # created by the coordinator at all.
    calls = []

    def post_all(node_ids, path, body):
        calls.append((path, sorted(node_ids)))
        if path == "/round/start":
            return {n: {"ok": True} for n in node_ids}
        assert path == "/round/gather"
        assert body["active_nodes"] == ["node_0", "node_1"]
        assert body["peer_bases"] == {
            "node_0": "http://node_0:8000",
            "node_1": "http://node_1:8000",
        }
        return {n: {"crop_accuracy": 0.5} for n in node_ids}

    runner = _make_runner(tmp_path, post_all)
    active = runner.run_round(0)

    assert active == ["node_0", "node_1"]
    assert calls == [
        ("/round/start", ["node_0", "node_1"]),
        ("/round/gather", ["node_0", "node_1"]),
    ]
    assert not list(tmp_path.iterdir())  # no db files written by the coordinator


def test_run_round_excludes_a_node_that_failed_round_start(tmp_path):
    def post_all(node_ids, path, body):
        if path == "/round/start":
            return {
                "node_0": {"ok": True},
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
            return {n: {"ok": True} for n in node_ids}
        return {n: {"crop_accuracy": 0.0, "disease_accuracy": 0.0} for n in node_ids}

    runner = _make_runner(tmp_path, post_all, num_rounds=3)
    runner.run_all_rounds()
    assert seen_rounds == [0, 1, 2]


def _stub_post_all(node_ids, path, body):
    if path == "/round/start":
        return {n: {"ok": True} for n in node_ids}
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
