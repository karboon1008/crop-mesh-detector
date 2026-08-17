from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docker" / "controller"))

import pytest
from controller_runner import ControllerRunner  # noqa: E402


def _make_runner(start_scenario=None, stop_scenario=None):
    return ControllerRunner(
        start_scenario=start_scenario or (lambda scenario: None),
        stop_scenario=stop_scenario or (lambda: None),
    )


def test_initial_state_is_idle():
    runner = _make_runner()
    assert runner.status() == {
        "running_scenario": None, "state": "idle", "started_at": None, "error_detail": None,
    }


def test_start_moves_to_running_and_records_scenario():
    runner = _make_runner()
    result = runner.start("full_run")
    assert result["state"] == "running"
    assert result["running_scenario"] == "full_run"
    assert result["started_at"] is not None


def test_start_rejects_unknown_scenario():
    runner = _make_runner()
    with pytest.raises(ValueError):
        runner.start("not_a_real_scenario")
    assert runner.status()["state"] == "idle"  # rejected before any state change


def test_start_stops_the_previously_running_scenario_first():
    calls = []
    runner = _make_runner(
        start_scenario=lambda s: calls.append(("start", s)),
        stop_scenario=lambda: calls.append(("stop",)),
    )
    runner.start("full_run")
    runner.start("disconnection")
    assert calls == [("start", "full_run"), ("stop",), ("start", "disconnection")]
    assert runner.status()["running_scenario"] == "disconnection"


def test_stop_when_idle_is_a_noop():
    calls = []
    runner = _make_runner(stop_scenario=lambda: calls.append("stop"))
    result = runner.stop()
    assert result["state"] == "idle"
    assert calls == []


def test_start_failure_moves_to_error_state_with_detail():
    def failing_start(scenario):
        raise RuntimeError("docker compose exploded")

    runner = _make_runner(start_scenario=failing_start)
    with pytest.raises(RuntimeError):
        runner.start("full_run")
    status = runner.status()
    assert status["state"] == "error"
    assert "docker compose exploded" in status["error_detail"]


def test_stop_is_callable_again_after_an_error_and_recovers_to_idle():
    calls = []
    runner = _make_runner(
        start_scenario=lambda s: (_ for _ in ()).throw(RuntimeError("boom")),
        stop_scenario=lambda: calls.append("stop"),
    )
    with pytest.raises(RuntimeError):
        runner.start("full_run")
    assert runner.status()["state"] == "error"

    result = runner.stop()
    assert calls == ["stop"]
    assert result["state"] == "idle"
    assert result["error_detail"] is None
