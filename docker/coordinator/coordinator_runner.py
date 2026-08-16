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

import time
from dataclasses import dataclass
from typing import Callable

from src.energy import sqlite_store


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

    def wait_until_all_online(self, sleep_fn=time.sleep, poll_interval_s: float = 1.0) -> None:
        while not all(self.health_check(n) for n in self.expected_nodes):
            sleep_fn(poll_interval_s)

    def run_round(self, round_idx: int) -> list:
        start_results = self.post_all(self.expected_nodes, "/round/start", {"round_idx": round_idx})
        active = sorted(n for n, r in start_results.items() if r is not None)
        for node_id in active:
            r = start_results[node_id]
            sqlite_store.upsert_row(
                self.db_path,
                node_id,
                round_idx,
                _now_iso(),
                energy_kwh=r["energy_kwh"],
                duration_s=r["duration_s"],
                energy_method=r["energy_method"],
                knowledge_bytes_sent=r["size_bytes"],
                active=1,
            )

        gather_body = {
            "round_idx": round_idx,
            "active_nodes": active,
            "peer_bases": {n: self.node_base_urls[n] for n in active},
        }
        gather_results = self.post_all(active, "/round/gather", gather_body)
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
