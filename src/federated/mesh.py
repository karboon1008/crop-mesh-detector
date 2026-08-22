"""Orchestrates the decentralised mesh: every node trains locally, then
broadcasts only its KnowledgePayload (prototypes + probe logits) to its
peers. Each node aggregates what it *received* with a Byzantine-robust
rule and distils its own model towards that peer consensus. There is no
central aggregator and no node ever receives another node's images,
labels, gradients, or weights.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
from src.federated.aggregation import aggregate_prototypes, aggregate_masked_logits
from src.federated.node import KnowledgePayload, Node


def _avg_accuracy(eval_: dict) -> float:
    return (eval_["crop_accuracy"] + eval_["disease_accuracy"]) / 2


def _scale_kd_weight(base_kd_weight: float, node_avg: float, peer_avg: float, min_scale: float, max_scale: float) -> float:
    """A node already ahead of its peers' pre-round average gets pulled
    less toward their consensus (it has more to lose than gain); a node
    behind that average gets pulled harder toward it (capped at
    max_scale to avoid an unstable overcorrection), instead of every
    node absorbing the same flat kd_weight regardless of whether
    distillation is likely to help or hurt it this round.
    """
    if node_avg <= 1e-6:
        return base_kd_weight
    scale = min(max(peer_avg / node_avg, min_scale), max_scale)
    return base_kd_weight * scale


@dataclass
class RoundLog:
    round_idx: int
    per_node_train_loss: dict[str, float] = field(default_factory=dict)
    pre_distill_eval: dict[str, dict[str, float | dict]] = field(default_factory=dict)
    # node_id -> {"kd_loss", "sup_loss", "proto_loss"}
    per_node_distill_loss: dict[str, dict[str, float]] = field(default_factory=dict)
    per_node_eval: dict[str, dict[str, float | dict]] = field(default_factory=dict)
    per_node_kd_weight: dict[str, float] = field(default_factory=dict)
    per_node_crop_kd_weight: dict[str, float] = field(default_factory=dict)
    total_bytes_exchanged: int = 0
    active_nodes: list[str] = field(default_factory=list)


class MeshSimulator:
    def __init__(
        self,
        nodes: list[Node],
        probe_loader: DataLoader,
        aggregation_method: str,
        trim_fraction: float,
        krum_neighbors: int,
        adaptive_kd_weight: bool = False,
        adaptive_kd_min_scale: float = 0.3,
        adaptive_kd_max_scale: float = 1.5,
    ):
        self.nodes = nodes
        self.probe_loader = probe_loader
        self.aggregation_method = aggregation_method
        self.trim_fraction = trim_fraction
        self.krum_neighbors = krum_neighbors
        self.adaptive_kd_weight = adaptive_kd_weight
        self.adaptive_kd_min_scale = adaptive_kd_min_scale
        self.adaptive_kd_max_scale = adaptive_kd_max_scale

    def run_round(
        self,
        round_idx: int,
        local_epochs: int | dict[str, int],
        distill_epochs: int,
        lr: float,
        distill_lr: float,
        proto_weight: float,
        kd_weight: float,
        temperature: float,
        crop_kd_weight: float | None = None,
    ) -> RoundLog:
        # crop_kd_weight defaults to kd_weight so existing callers that only
        # tune one flat weight keep behaving the same; pass it explicitly to
        # balance crop- vs. disease-head peer consensus independently.
        base_crop_kd_weight = kd_weight if crop_kd_weight is None else crop_kd_weight
        log = RoundLog(round_idx=round_idx)
        log.active_nodes = [node.node_id for node in self.nodes if node.active]

        # 1) local supervised training, private data never leaves this loop.
        # Disconnected nodes keep training locally — they drift, but are
        # not frozen. local_epochs may be a single int shared by every node,
        # or a {node_id: epochs} map (e.g. to give data-poor nodes more
        # epochs so they see a comparable number of gradient steps).
        for node in self.nodes:
            node_epochs = local_epochs[node.node_id] if isinstance(local_epochs, dict) else local_epochs
            log.per_node_train_loss[node.node_id] = node.local_train(node_epochs, lr)

        # 1b) snapshot every node's metrics right here, before any peer
        # knowledge is applied, so the pre- vs. post-distill comparison
        # isolates what distillation itself changed this round.
        for node in self.nodes:
            log.pre_distill_eval[node.node_id] = node.evaluate()
        pre_avg = {nid: _avg_accuracy(ev) for nid, ev in log.pre_distill_eval.items()}

        # 2) each ACTIVE node computes its small, non-invertible knowledge
        # payload. A disconnected node's payload never enters the pool.
        payloads: dict[str, KnowledgePayload] = {
            node.node_id: node.compute_knowledge(self.probe_loader)
            for node in self.nodes
            if node.active
        }

        # simulate a fully-connected broadcast among the currently-connected
        # nodes only: every payload is sent to every other ACTIVE peer once
        # (an upper bound — a real gossip relay with partial connectivity
        # would use less bandwidth than this).
        active_n = len(payloads)
        log.total_bytes_exchanged = sum(p.size_bytes() for p in payloads.values()) * max(0, active_n - 1)

        # 3) each ACTIVE node aggregates what it received from PEERS
        # (excluding its own payload) with a robust rule, then distils
        # towards it. A disconnected node receives nothing and is skipped.
        for node in self.nodes:
            if not node.active:
                continue
            peer_ids = [nid for nid in payloads if nid != node.node_id]
            peer_payloads = [payloads[nid] for nid in peer_ids]
            if not peer_payloads:
                continue  # single-node mesh: nothing to reconcile

            node_kd_weight = kd_weight
            node_crop_kd_weight = base_crop_kd_weight
            if self.adaptive_kd_weight:
                peer_avg = sum(pre_avg[nid] for nid in peer_ids) / len(peer_ids)
                node_kd_weight = _scale_kd_weight(
                    kd_weight, pre_avg[node.node_id], peer_avg,
                    self.adaptive_kd_min_scale, self.adaptive_kd_max_scale,
                )
                node_crop_kd_weight = _scale_kd_weight(
                    base_crop_kd_weight, pre_avg[node.node_id], peer_avg,
                    self.adaptive_kd_min_scale, self.adaptive_kd_max_scale,
                )
            log.per_node_kd_weight[node.node_id] = node_kd_weight
            log.per_node_crop_kd_weight[node.node_id] = node_crop_kd_weight

            consensus_prototypes = aggregate_prototypes(
                [p.prototypes for p in peer_payloads],
                method=self.aggregation_method,
                trim_fraction=self.trim_fraction,
                krum_neighbors=self.krum_neighbors,
            )
            consensus_crop_logits, crop_known_mask = aggregate_masked_logits(
                [p.crop_logits for p in peer_payloads],
                [p.known_crop_classes for p in peer_payloads],
                method=self.aggregation_method,
                trim_fraction=self.trim_fraction,
                krum_neighbors=self.krum_neighbors,
            )
            consensus_disease_logits, disease_known_mask = aggregate_masked_logits(
                [p.disease_logits for p in peer_payloads],
                [p.known_disease_classes for p in peer_payloads],
                method=self.aggregation_method,
                trim_fraction=self.trim_fraction,
                krum_neighbors=self.krum_neighbors,
            )

            log.per_node_distill_loss[node.node_id] = node.distill(
                consensus_prototypes,
                consensus_crop_logits,
                crop_known_mask,
                consensus_disease_logits,
                disease_known_mask,
                self.probe_loader,
                epochs=distill_epochs,
                lr=distill_lr,
                proto_weight=proto_weight,
                kd_weight=node_kd_weight,
                crop_kd_weight=node_crop_kd_weight,
                temperature=temperature,
            )

        # 4) evaluate every node after this round's exchange
        for node in self.nodes:
            log.per_node_eval[node.node_id] = node.evaluate()

        return log

    def run(self, num_rounds: int, **round_kwargs) -> list[RoundLog]:
        return [self.run_round(r, **round_kwargs) for r in range(num_rounds)]
