"""Pure start/stop/mutual-exclusion state machine for the scenario
control plane. Docker/subprocess calls are injected as plain callables
(mirroring docker/node/node_runner.py's fetch_all_knowledge and
docker/coordinator/coordinator_runner.py's post_all) so this class is
unit-testable with fakes -- see docker/controller/main.py for the real
docker-compose-via-subprocess wiring.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

VALID_SCENARIOS = {"full_run", "class_addition", "disconnection", "distribution_shift"}

# Raises on failure; returns nothing on success.
StartScenario = Callable[[str], None]
StopScenario = Callable[[], None]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class ControllerRunner:
    start_scenario: StartScenario
    stop_scenario: StopScenario
    state: str = "idle"  # idle | starting | running | stopping | error
    running_scenario: str | None = None
    started_at: str | None = None
    error_detail: str | None = None

    def status(self) -> dict:
        return {
            "running_scenario": self.running_scenario,
            "state": self.state,
            "started_at": self.started_at,
            "error_detail": self.error_detail,
        }

    def start(self, scenario: str) -> dict:
        if scenario not in VALID_SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}, must be one of {sorted(VALID_SCENARIOS)}")
        if self.state in ("starting", "stopping"):
            raise RuntimeError(f"cannot start while controller is {self.state}")
        if self.state in ("running", "error"):
            self._do_stop()
        self.state = "starting"
        try:
            self.start_scenario(scenario)
        except Exception as exc:
            self.state = "error"
            self.error_detail = str(exc)
            raise
        self.state = "running"
        self.running_scenario = scenario
        self.started_at = _now_iso()
        self.error_detail = None
        return self.status()

    def stop(self) -> dict:
        if self.state in ("starting", "stopping"):
            raise RuntimeError(f"cannot stop while controller is {self.state}")
        if self.state == "idle":
            return self.status()
        self._do_stop()
        return self.status()

    def _do_stop(self) -> None:
        self.state = "stopping"
        try:
            self.stop_scenario()
        except Exception as exc:
            self.state = "error"
            self.error_detail = str(exc)
            raise
        self.state = "idle"
        self.running_scenario = None
        self.started_at = None
        self.error_detail = None
