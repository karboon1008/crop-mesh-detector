"""Control groups for the continual mesh: FedAvg and D-PSGD on the SAME stream.

WeightMesh inherits ContinualMesh.run_batch, so local training, pre-exchange
evaluation, the EMA, post-exchange evaluation and every record/energy phase are the
literal same code. Only the exchange step (ContinualMesh._exchange) is replaced:
full model weights (the FP32 state_dict, BatchNorm buffers included) travel instead
of prototypes + probe logits, and there is no distillation.

  fedavg  star topology. An uploader sends its weights to a server that keeps every
          node's latest published weights; a retriever downloads their
          sample-count-weighted average.
  dpsgd   ring topology (each node has two neighbours). An uploader pushes its weights
          to both neighbours; a retriever mixes its own current weights with its
          neighbours' latest published ones (same weighting rule as fedavg). A fully
          connected D-PSGD computes exactly the FedAvg average, so it is not run
          separately; its bytes are N*(N-1)*W per exchange.

  local   no exchange at all. A node the EMA rule would send to retrieve instead takes the extra supervised
          epoch(s) our distillation step also contains (Node.distill with no prototypes and no probe logits
          is exactly that). It isolates how much of the continual mesh's gain is knowledge transfer and how
          much is simply more local training.

gated=False is the textbook protocol: every node uploads and retrieves every batch.
gated=True applies the continual mesh's own EMA rules (decide_roles) to decide who
uploads and who retrieves, so the payload is the only thing that differs from ours.

Bytes count every network transmission once: FedAvg = an upload to the server plus a
download from it; D-PSGD = one push per neighbour (counted at the sender).
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.federated.continual import LEARNER, TEACHER, BatchLog, ContinualMesh


def state_bytes(state: dict) -> int:
    return sum(v.numel() * v.element_size() for v in state.values())


def ring_neighbors(index: int, n: int) -> list[int]:
    """The other nodes adjacent to `index` on a ring of `n` (none for n=1, one for n=2)."""
    return sorted({(index - 1) % n, (index + 1) % n} - {index})


def weighted_average(states: list[dict], weights: list[float]) -> dict:
    total = float(sum(weights))
    merged = {}
    for key, ref in states[0].items():
        if ref.dtype.is_floating_point:
            merged[key] = sum(sd[key].float() * (w / total) for sd, w in zip(states, weights)).to(ref.dtype)
        else:
            merged[key] = ref.clone()  # integer buffers are counters, not weights
    return merged


class _NoStore:
    """train.py records mesh.store.entries(); a weight mesh has no knowledge database."""

    def entries(self) -> dict:
        return {}


class WeightMesh(ContinualMesh):
    def __init__(self, nodes, tracker, method: str, gated: bool, store_path: Path, reset: bool, **mesh_kwargs):
        if method not in ("fedavg", "dpsgd", "local"):
            raise ValueError(f"unknown weight-exchange method: {method}")
        super().__init__(nodes, _NoStore(), tracker, **mesh_kwargs)
        self.method, self.gated, self.store_path = method, gated or method == "local", Path(store_path)
        self.model_bytes = state_bytes(nodes[0].model.state_dict())
        # node_id -> (batch it was published in, num_train then, weights on CPU): latest publication per node
        self.published: dict[str, tuple[int, int, dict]] = {}
        if self.store_path.exists() and not reset:
            self.published = torch.load(self.store_path, map_location="cpu")

    def _decide_roles(self, batch_idx: int, ema: float, prev_ema: float | None) -> list[str]:
        return super()._decide_roles(batch_idx, ema, prev_ema) if self.gated else [TEACHER, LEARNER]

    def _exchange(self, log: BatchLog, present: list, batch_idx: int, probe_loader, distill: dict) -> None:
        if self.method == "local":
            return self._extra_local_epochs(log, present, distill)
        index = {node.node_id: i for i, node in enumerate(self.nodes)}

        # uploaders publish a snapshot; every snapshot is taken before any node adopts a merged model
        for node in present:
            record = log.per_node[node.node_id]
            if TEACHER not in record.roles:
                continue
            with self._track(record, "upload"):
                snapshot = {k: v.detach().cpu().clone() for k, v in node.model.state_dict().items()}
                self.published[node.node_id] = (batch_idx, record.num_train, snapshot)
            record.uploaded = True
            fan_out = 1 if self.method == "fedavg" else len(ring_neighbors(index[node.node_id], len(self.nodes)))
            record.bytes_uploaded = fan_out * self.model_bytes

        for node in present:
            record = log.per_node[node.node_id]
            if LEARNER not in record.roles:
                continue
            if self.method == "fedavg":
                sources = dict(self.published)
            else:
                names = {self.nodes[j].node_id for j in ring_neighbors(index[node.node_id], len(self.nodes))}
                sources = {nid: pub for nid, pub in self.published.items() if nid in names}
            if not sources:
                continue  # nobody has published anything this node can reach yet
            record.peers_used = {nid: pub[0] for nid, pub in sources.items()}
            states = [pub[2] for pub in sources.values()]
            weights = [pub[1] for pub in sources.values()]
            if self.method == "dpsgd":  # mix with the node's own CURRENT weights, as decentralised averaging does
                states.append({k: v.detach().cpu() for k, v in node.model.state_dict().items()})
                weights.append(record.num_train)
            with self._track(record, "exchange"):
                merged = weighted_average(states, weights)
                node.model.load_state_dict({k: v.to(node.device) for k, v in merged.items()})
            record.bytes_downloaded = self.model_bytes if self.method == "fedavg" else 0
            record.distilled = True  # "adopted the exchanged model": what train.py's reports count as an exchange

        torch.save(self.published, self.store_path)

    def _extra_local_epochs(self, log: BatchLog, present: list, distill: dict) -> None:
        for node in present:
            record = log.per_node[node.node_id]
            if LEARNER not in record.roles:
                continue
            with self._track(record, "exchange"):
                node.distill(
                    {}, None, torch.zeros(len(node.crop_classes), dtype=torch.bool),
                    None, torch.zeros(len(node.disease_classes), dtype=torch.bool), None,
                    epochs=distill["distill_epochs"], lr=distill["distill_lr"], proto_weight=0.0,
                    kd_weight=0.0, crop_kd_weight=0.0, temperature=distill["temperature"],
                )
            record.distilled = True  # took the extra epoch(s); nothing was sent or received
