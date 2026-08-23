"""Control-plane round driver: times rounds and tracks node liveness only --
it owns no metrics data of its own. Never calls a node's /knowledge endpoint
itself -- nodes fetch peer knowledge directly from each other, so there is
no central aggregator of knowledge (that logic stays entirely in
node_runner.py). It also never persists round_metrics: each node already
writes its own energy/accuracy/communication numbers straight into its own
db (see node_runner.py) as part of answering /round/start and
/round/gather, so a second, coordinator-owned copy would be redundant data
-- and, worse, would look like a central store the mesh doesn't actually
have. The dashboard reads every node's own db directly and combines them at
render time (see docker/dashboard/data.py's merge_round_rows). Transport
(concurrent HTTP fan-out, health polling) is injected as plain callables so
this class is unit-testable with fakes -- see docker/coordinator/main.py
for the real asyncio/httpx wiring.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

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
    post_all: PostAll
    health_check: HealthCheck
    # Where to write a small completion marker JSON once run_all_rounds
    # finishes -- None (the default, and what every existing test uses)
    # skips this entirely.
    status_path: str | None = None
    log_file: str | None = None
    activity_log: list = field(default_factory=list, init=False)
    _log_dir_ready: bool = field(default=False, init=False)

    def _log(self, message: str) -> None:
        entry = {"ts": time.time(), "message": message}
        self.activity_log.append(entry)
        del self.activity_log[:-MAX_ACTIVITY_LOG]
        if self.log_file:
            if not self._log_dir_ready:
                Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
                self._log_dir_ready = True
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")

    def wait_until_all_online(self, sleep_fn=time.sleep, poll_interval_s: float = 1.0) -> None:
        self._log(f"waiting for {self.expected_nodes} to come online...")
        while not all(self.health_check(n) for n in self.expected_nodes):
            sleep_fn(poll_interval_s)
        self._log("all nodes online")

    def run_round(self, round_idx: int) -> list:
        # This is purely a scheduling fan-out: it decides WHEN nodes run
        # /round/start and /round/gather and WHICH nodes are active, but it
        # never reads the response bodies for anything beyond that liveness
        # check -- the energy/accuracy/communication numbers inside them are
        # already being written by each node into its own db (see
        # node_runner.py's handle_round_start/handle_round_gather). Storing
        # them again here would just be a redundant, coordinator-owned copy.
        self._log(f"round {round_idx}: dispatching /round/start to {self.expected_nodes}")
        start_results = self.post_all(self.expected_nodes, "/round/start", {"round_idx": round_idx})
        active = sorted(n for n, r in start_results.items() if r is not None)
        self._log(f"round {round_idx}: {len(active)}/{len(self.expected_nodes)} node(s) active after /round/start")

        gather_body = {
            "round_idx": round_idx,
            "active_nodes": active,
            "peer_bases": {n: self.node_base_urls[n] for n in active},
        }
        self._log(f"round {round_idx}: dispatching /round/gather to {active}")
        self.post_all(active, "/round/gather", gather_body)
        self._log(f"round {round_idx}: /round/gather complete")
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
