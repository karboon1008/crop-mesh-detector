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
from src.federated.aggregation import aggregate_prototypes, aggregate_logits
from src.federated.node import KnowledgePayload, Node


@dataclass
class RoundLog:
    round_idx: int
    per_node_train_loss: dict[str, float] = field(default_factory=dict)
    per_node_distill_loss: dict[str, float] = field(default_factory=dict)
    per_node_eval: dict[str, dict[str, float]] = field(default_factory=dict)
    total_bytes_exchanged: int = 0


class MeshSimulator:
    def __init__(
        self,
        nodes: list[Node],
        probe_loader: DataLoader,
        aggregation_method: str,
        trim_fraction: float,
        krum_neighbors: int,
    ):
        self.nodes = nodes
        self.probe_loader = probe_loader
        self.aggregation_method = aggregation_method
        self.trim_fraction = trim_fraction
        self.krum_neighbors = krum_neighbors

    def run_round(
        self,
        round_idx: int,
        local_epochs: int,
        distill_epochs: int,
        lr: float,
        distill_lr: float,
        proto_weight: float,
        kd_weight: float,
        temperature: float,
    ) -> RoundLog:
        log = RoundLog(round_idx=round_idx)

        # 1) local supervised training, private data never leaves this loop
        for node in self.nodes:
            log.per_node_train_loss[node.node_id] = node.local_train(local_epochs, lr)

        # 2) each node computes its small, non-invertible knowledge payload
        payloads: dict[str, KnowledgePayload] = {
            node.node_id: node.compute_knowledge(self.probe_loader) for node in self.nodes
        }

        # simulate a fully-connected broadcast: every payload is sent to
        # every OTHER peer once (an upper bound — a real gossip relay with
        # partial connectivity would use less bandwidth than this).
        n = len(self.nodes)
        log.total_bytes_exchanged = sum(p.size_bytes() for p in payloads.values()) * (n - 1)

        # 3) each node aggregates what it received from PEERS (excluding
        #    its own payload) with a robust rule, then distils towards it
        for node in self.nodes:
            peer_payloads = [p for nid, p in payloads.items() if nid != node.node_id]
            if not peer_payloads:
                continue  # single-node mesh: nothing to reconcile

            consensus_prototypes = aggregate_prototypes(
                [p.prototypes for p in peer_payloads],
                method=self.aggregation_method,
                trim_fraction=self.trim_fraction,
                krum_neighbors=self.krum_neighbors,
            )
            consensus_crop_logits = aggregate_logits(
                [p.crop_logits for p in peer_payloads],
                method=self.aggregation_method,
                trim_fraction=self.trim_fraction,
                krum_neighbors=self.krum_neighbors,
            )
            consensus_disease_logits = aggregate_logits(
                [p.disease_logits for p in peer_payloads],
                method=self.aggregation_method,
                trim_fraction=self.trim_fraction,
                krum_neighbors=self.krum_neighbors,
            )

            log.per_node_distill_loss[node.node_id] = node.distill(
                consensus_prototypes,
                consensus_crop_logits,
                consensus_disease_logits,
                self.probe_loader,
                epochs=distill_epochs,
                lr=distill_lr,
                proto_weight=proto_weight,
                kd_weight=kd_weight,
                temperature=temperature,
            )

        # 4) evaluate every node after this round's exchange
        for node in self.nodes:
            log.per_node_eval[node.node_id] = node.evaluate()

        return log

    def run(self, num_rounds: int, **round_kwargs) -> list[RoundLog]:
        return [self.run_round(r, **round_kwargs) for r in range(num_rounds)]
