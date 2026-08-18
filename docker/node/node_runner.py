"""HTTP-driven wrapper around a single Node, reusing src/federated/node.py
and src/federated/aggregation.py unmodified. Round-handling logic takes its
peer-fetch mechanism as an injected callable so it's unit-testable with a
fake fetcher -- no real HTTP server, no real concurrency, needed in tests.
See docker/node/main.py for the real FastAPI + httpx wiring.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src.energy import sqlite_store
from src.energy.tracker import ComputeEnergyTracker
from src.federated.aggregation import aggregate_logits, aggregate_prototypes
from src.federated.node import KnowledgePayload, Node

from knowledge_codec import decode_knowledge, encode_knowledge

MAX_ACTIVITY_LOG = 300
# Minimum wall-clock gap between two recorded progress-heartbeat lines from
# the SAME training phase, so a fast batch loop doesn't spam the log. This
# throttle is what keeps the added overhead negligible relative to the
# minutes-long training it's reporting on -- the dashboard only needs an
# update every so often, not every batch.
PROGRESS_LOG_INTERVAL_S = 10.0


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# fetch_all_knowledge(peer_ids, peer_bases, round_idx) -> {peer_id: bytes | None}
# None means that peer's fetch failed, timed out, or (checked by the caller)
# returned a mismatched round.
FetchAllKnowledge = Callable[[list, dict, int], dict]


@dataclass
class NodeRunner:
    node_id: str
    node: Node
    shadow_node: Node
    probe_loader: object
    tracker: ComputeEnergyTracker
    db_path: str
    fetch_all_knowledge: FetchAllKnowledge
    aggregation_method: str = "trimmed_mean"
    trim_fraction: float = 0.2
    krum_neighbors: int = 2
    local_epochs: int = 2
    distill_epochs: int = 1
    lr: float = 0.001
    distill_lr: float = 0.0005
    proto_weight: float = 0.5
    kd_weight: float = 0.5
    temperature: float = 2.0
    log_file: str | None = None
    _last_round_idx: int | None = field(default=None, init=False)
    _last_knowledge_bytes: bytes | None = field(default=None, init=False)
    activity_log: list = field(default_factory=list, init=False)
    _last_progress_ts: dict = field(default_factory=dict, init=False)
    _log_dir_ready: bool = field(default=False, init=False)

    def _log(self, stage: str, message: str) -> None:
        entry = {"ts": time.time(), "stage": stage, "message": message}
        self.activity_log.append(entry)
        del self.activity_log[:-MAX_ACTIVITY_LOG]
        if self.log_file:
            if not self._log_dir_ready:
                Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
                self._log_dir_ready = True
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")

    def _make_progress_cb(self, phase_key: str, stage: str | None = None) -> Callable[[dict], None]:
        """Builds a throttled progress_cb for Node.local_train/distill: only
        actually logs once every PROGRESS_LOG_INTERVAL_S per phase_key, so
        the added overhead stays negligible no matter how fast the batch
        loop runs. `stage` overrides the logged stage label (info["phase"]
        is always "local_train" whether it's the mesh arm or the shadow
        baseline arm calling it, so the baseline arm passes "baseline_local_train"
        to keep the two visually distinct in the dashboard's log panel).
        """

        def _cb(info: dict) -> None:
            now = time.time()
            last = self._last_progress_ts.get(phase_key, 0.0)
            if now - last < PROGRESS_LOG_INTERVAL_S:
                return
            self._last_progress_ts[phase_key] = now
            self._log(
                stage or info["phase"],
                f"epoch {info['epoch']}/{info['epochs']} "
                f"batch {info['batch']}/{info['num_batches']} loss={info['loss']:.4f}",
            )

        return _cb

    def handle_round_start(self, round_idx: int) -> dict:
        self._log("round_start", f"round {round_idx}: received, starting local_train")
        # Shadow (local_only) arm runs fully BEFORE the mesh arm, sequentially
        # on the same CPU -- never concurrently. Running them concurrently
        # would have both contend for the same cores, inflating both
        # ComputeEnergyTracker.track() durations unpredictably and corrupting
        # the energy figures this run's Axis A evidence depends on. This
        # mirrors src/scenarios/harness.py's run_scenario() ordering exactly
        # (baseline arm fully computed, then the mesh arm), so the two are
        # honestly comparable under the same round_kwargs.
        with self.tracker.track(f"{self.node_id}_baseline_round_{round_idx}") as baseline_energy_record:
            self.shadow_node.local_train(
                self.local_epochs,
                self.lr,
                progress_cb=self._make_progress_cb(
                    f"round_{round_idx}_baseline_local_train", stage="baseline_local_train"
                ),
            )
            baseline_eval = self.shadow_node.evaluate()
        self._log(
            "round_start",
            f"round {round_idx}: local_only baseline done, "
            f"crop_acc={baseline_eval['crop_accuracy']:.4f} disease_acc={baseline_eval['disease_accuracy']:.4f}",
        )

        # compute_knowledge runs INSIDE the tracked scope: it is a full
        # forward pass over the probe set, a real compute cost belonging to
        # this round. The in-process pipeline (src/train.py) tracks the whole
        # round, so keeping it outside would understate energy here relative
        # to that baseline -- especially under the proxy_wall_power fallback,
        # which is linear in tracked wall time.
        with self.tracker.track(f"{self.node_id}_round_{round_idx}") as energy_record:
            self.node.local_train(
                self.local_epochs, self.lr, progress_cb=self._make_progress_cb(f"round_{round_idx}_local_train")
            )
            self._log("round_start", f"round {round_idx}: local_train done, computing knowledge")
            knowledge = self.node.compute_knowledge(self.probe_loader)
        data = encode_knowledge(round_idx, knowledge)
        self._last_round_idx = round_idx
        self._last_knowledge_bytes = data

        sqlite_store.upsert_row(
            self.db_path,
            self.node_id,
            round_idx,
            _now_iso(),
            energy_kwh=energy_record["energy_kwh"],
            duration_s=energy_record["duration_s"],
            energy_method=energy_record["method"],
            knowledge_bytes_sent=len(data),
            baseline_crop_accuracy=baseline_eval["crop_accuracy"],
            baseline_disease_accuracy=baseline_eval["disease_accuracy"],
            baseline_energy_kwh=baseline_energy_record["energy_kwh"],
            baseline_duration_s=baseline_energy_record["duration_s"],
        )
        self._log(
            "round_start",
            f"round {round_idx}: knowledge ready ({len(data)} bytes), responding to coordinator",
        )
        return {
            "round_idx": round_idx,
            "size_bytes": len(data),
            "energy_kwh": energy_record["energy_kwh"],
            "duration_s": energy_record["duration_s"],
            "energy_method": energy_record["method"],
            "baseline_crop_accuracy": baseline_eval["crop_accuracy"],
            "baseline_disease_accuracy": baseline_eval["disease_accuracy"],
            "baseline_energy_kwh": baseline_energy_record["energy_kwh"],
            "baseline_duration_s": baseline_energy_record["duration_s"],
        }

    def get_knowledge_bytes(self, round_idx: int) -> bytes | None:
        if self._last_round_idx != round_idx:
            return None
        return self._last_knowledge_bytes

    def handle_round_gather(self, round_idx: int, active_nodes: list, peer_bases: dict) -> dict:
        self._log("round_gather", f"round {round_idx}: received, fetching knowledge from {len(active_nodes) - 1} peer(s)")
        peer_ids = [n for n in active_nodes if n != self.node_id]
        fetched = self.fetch_all_knowledge(peer_ids, peer_bases, round_idx)

        peers: list[KnowledgePayload] = []
        for peer_id in peer_ids:
            data = fetched.get(peer_id)
            if data is None:
                continue
            peer_round_idx, knowledge = decode_knowledge(data)
            if peer_round_idx != round_idx:
                continue  # defensive: peer returned data for a different round
            peers.append(knowledge)
        self._log("round_gather", f"round {round_idx}: got {len(peers)}/{len(peer_ids)} peer payload(s)")

        if peers:
            consensus_prototypes = aggregate_prototypes(
                [p.prototypes for p in peers],
                method=self.aggregation_method,
                trim_fraction=self.trim_fraction,
                krum_neighbors=self.krum_neighbors,
            )
            consensus_crop_logits = aggregate_logits(
                [p.crop_logits for p in peers],
                method=self.aggregation_method,
                trim_fraction=self.trim_fraction,
                krum_neighbors=self.krum_neighbors,
            )
            consensus_disease_logits = aggregate_logits(
                [p.disease_logits for p in peers],
                method=self.aggregation_method,
                trim_fraction=self.trim_fraction,
                krum_neighbors=self.krum_neighbors,
            )
            # ENERGY SCOPE CAVEAT -- read before comparing a Docker-mesh
            # energy_kwh figure to an in-process one:
            # this round's reported energy_kwh (written by handle_round_start)
            # covers local_train + compute_knowledge ONLY. The distill() call
            # below, and the evaluate() call after it, are NOT included in
            # energy_kwh. The in-process pipeline (src/train.py, `with
            # tracker.track(...): mesh.run_round(...)`) tracks the ENTIRE
            # round -- train + knowledge + distill + evaluate -- so a
            # Docker-mesh energy_kwh is a strict UNDER-count relative to an
            # in-process one for the same work.
            # Why not just add a second tracked block here? Because
            # sqlite_store.upsert_row's ON CONFLICT DO UPDATE SET would
            # OVERWRITE (not add to) the energy_kwh already written for this
            # (node_id, round_idx) in handle_round_start, silently discarding
            # the local_train + compute_knowledge figure entirely. Closing
            # this gap properly needs accumulating upsert semantics (or a
            # per-phase energy column), which is a schema change.
            self._log("round_gather", f"round {round_idx}: distilling towards peer consensus")
            self.node.distill(
                consensus_prototypes,
                consensus_crop_logits,
                consensus_disease_logits,
                self.probe_loader,
                epochs=self.distill_epochs,
                lr=self.distill_lr,
                proto_weight=self.proto_weight,
                kd_weight=self.kd_weight,
                temperature=self.temperature,
                progress_cb=self._make_progress_cb(f"round_{round_idx}_distill"),
            )

        self._log("round_gather", f"round {round_idx}: evaluating")
        eval_result = self.node.evaluate()
        sqlite_store.upsert_row(
            self.db_path,
            self.node_id,
            round_idx,
            _now_iso(),
            crop_accuracy=eval_result["crop_accuracy"],
            disease_accuracy=eval_result["disease_accuracy"],
            active=1,
        )
        self._log(
            "round_gather",
            f"round {round_idx}: done, crop_acc={eval_result['crop_accuracy']:.4f} "
            f"disease_acc={eval_result['disease_accuracy']:.4f}",
        )
        return {"round_idx": round_idx, **eval_result}
