"""Control-plane round driver: times rounds, tracks node liveness, merges
per-round HTTP response bodies into SQLite. Never calls a node's
/knowledge endpoint itself -- nodes fetch peer knowledge directly from
each other, so there is no central aggregator of knowledge (that logic
stays entirely in node_runner.py). Transport (concurrent HTTP fan-out,
health polling) is injected as plain callables so this class is
unit-testable with fakes -- see docker/coordinator/main.py for the real
asyncio/httpx wiring.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src.energy import sqlite_store

MAX_ACTIVITY_LOG = 300


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# post_all(node_ids, path, body) -> {node_id: response_json | None}
# None means that node's request failed, timed out, or returned non-200.
PostAll = Callable[[list, str, dict], dict]
HealthCheck = Callable[[str], bool]


@dataclass
class CoordinatorRunner:
    expected_nodes: list
    node_base_urls: dict
    num_rounds: int
    round_timeout_s: float
    db_path: str
    post_all: PostAll
    health_check: HealthCheck
    # Where to write a small completion marker JSON once run_all_rounds
    # finishes -- None (the default, and what every existing test uses)
    # skips this entirely.
    status_path: str | None = None
    activity_log: list = field(default_factory=list, init=False)

    def _log(self, message: str) -> None:
        # Pure in-memory append, no I/O -- negligible cost, and irrelevant to
        # energy accounting anyway since the coordinator itself is never
        # inside a ComputeEnergyTracker scope.
        self.activity_log.append({"ts": time.time(), "message": message})
        del self.activity_log[:-MAX_ACTIVITY_LOG]

    def wait_until_all_online(self, sleep_fn=time.sleep, poll_interval_s: float = 1.0) -> None:
        self._log(f"waiting for {self.expected_nodes} to come online...")
        while not all(self.health_check(n) for n in self.expected_nodes):
            sleep_fn(poll_interval_s)
        self._log("all nodes online")

    def run_round(self, round_idx: int) -> list:
        self._log(f"round {round_idx}: dispatching /round/start to {self.expected_nodes}")
        start_results = self.post_all(self.expected_nodes, "/round/start", {"round_idx": round_idx})
        active = sorted(n for n, r in start_results.items() if r is not None)
        self._log(f"round {round_idx}: {len(active)}/{len(self.expected_nodes)} node(s) active after /round/start")
        for node_id in active:
            r = start_results[node_id]
            # r["size_bytes"] is the size of ONE serialized knowledge payload.
            # In the HTTP pull model every OTHER active peer independently
            # fetches that same payload via its own GET /knowledge/{round_idx},
            # so the bytes actually transmitted off this node are
            # size_bytes * (active_n - 1). This mirrors src/federated/mesh.py's
            # RoundLog.total_bytes_exchanged convention exactly (same
            # "broadcast to every other active peer" multiplier), so the
            # Docker/HTTP figure stays directly comparable to the in-process
            # one. Both are upper bounds: they assume every active peer
            # fetches exactly once and none drop out mid-gather.
            sqlite_store.upsert_row(
                self.db_path,
                node_id,
                round_idx,
                _now_iso(),
                energy_kwh=r["energy_kwh"],
                duration_s=r["duration_s"],
                energy_method=r["energy_method"],
                knowledge_bytes_sent=r["size_bytes"] * max(0, len(active) - 1),
                active=1,
                baseline_crop_accuracy=r["baseline_crop_accuracy"],
                baseline_disease_accuracy=r["baseline_disease_accuracy"],
                baseline_energy_kwh=r["baseline_energy_kwh"],
                baseline_duration_s=r["baseline_duration_s"],
            )

        gather_body = {
            "round_idx": round_idx,
            "active_nodes": active,
            "peer_bases": {n: self.node_base_urls[n] for n in active},
        }
        self._log(f"round {round_idx}: dispatching /round/gather to {active}")
        gather_results = self.post_all(active, "/round/gather", gather_body)
        self._log(f"round {round_idx}: /round/gather complete")
        for node_id in active:
            r = gather_results.get(node_id)
            if r is None:
                continue
            sqlite_store.upsert_row(
                self.db_path,
                node_id,
                round_idx,
                _now_iso(),
                crop_accuracy=r["crop_accuracy"],
                disease_accuracy=r["disease_accuracy"],
            )
        return active

    def run_all_rounds(self) -> None:
        for round_idx in range(self.num_rounds):
            self.run_round(round_idx)
        self._log(f"all {self.num_rounds} round(s) complete")
        if self.status_path is not None:
            Path(self.status_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.status_path).write_text(
                json.dumps(
                    {
                        "all_rounds_complete": True,
                        "num_rounds": self.num_rounds,
                        "completed_at": _now_iso(),
                    },
                    indent=2,
                )
            )
