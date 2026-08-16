"""HTTP-driven wrapper around a single Node, reusing src/federated/node.py
and src/federated/aggregation.py unmodified. Round-handling logic takes its
peer-fetch mechanism as an injected callable so it's unit-testable with a
fake fetcher -- no real HTTP server, no real concurrency, needed in tests.
See docker/node/main.py for the real FastAPI + httpx wiring.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from src.energy import sqlite_store
from src.energy.tracker import ComputeEnergyTracker
from src.federated.aggregation import aggregate_logits, aggregate_prototypes
from src.federated.node import KnowledgePayload, Node

from knowledge_codec import decode_knowledge, encode_knowledge


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
    _last_round_idx: int | None = field(default=None, init=False)
    _last_knowledge_bytes: bytes | None = field(default=None, init=False)

    def handle_round_start(self, round_idx: int) -> dict:
        # compute_knowledge runs INSIDE the tracked scope: it is a full
        # forward pass over the probe set, a real compute cost belonging to
        # this round. The in-process pipeline (src/train.py) tracks the whole
        # round, so keeping it outside would understate energy here relative
        # to that baseline -- especially under the proxy_wall_power fallback,
        # which is linear in tracked wall time.
        with self.tracker.track(f"{self.node_id}_round_{round_idx}") as energy_record:
            self.node.local_train(self.local_epochs, self.lr)
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
        )
        return {
            "round_idx": round_idx,
            "size_bytes": len(data),
            "energy_kwh": energy_record["energy_kwh"],
            "duration_s": energy_record["duration_s"],
            "energy_method": energy_record["method"],
        }

    def get_knowledge_bytes(self, round_idx: int) -> bytes | None:
        if self._last_round_idx != round_idx:
            return None
        return self._last_knowledge_bytes

    def handle_round_gather(self, round_idx: int, active_nodes: list, peer_bases: dict) -> dict:
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
            )

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
        return {"round_idx": round_idx, **eval_result}
