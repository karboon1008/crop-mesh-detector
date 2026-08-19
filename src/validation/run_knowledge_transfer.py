"""Stage 2: 1-5 rounds of knowledge-transfer distillation on top of the
3 per-node checkpoints stage 1 (run_corn_pipeline.py) produces, plus a
budget-aligned local-only control arm -- see
docs/superpowers/specs/2026-08-19-corn-disease-knowledge-transfer-design.md.

    python -m src.validation.run_knowledge_transfer --rounds 2
"""

from __future__ import annotations

from contextlib import nullcontext

from torch.utils.data import DataLoader

from src.energy.tracker import ComputeEnergyTracker
from src.federated.aggregation import aggregate_logits, aggregate_prototypes
from src.federated.node import Node

MODEL_NAME = "mobilenet_v3_small"


def run_kt_round(
    nodes: dict[str, Node],
    control_nodes: dict[str, Node],
    probe_loader: DataLoader,
    aggregation_method: str,
    trim_fraction: float,
    krum_neighbors: int,
    distill_epochs: int,
    distill_lr: float,
    proto_weight: float,
    kd_weight: float,
    temperature: float,
    tracker: ComputeEnergyTracker | None = None,
) -> dict:
    """One knowledge-transfer round: `nodes` distill toward their peers'
    consensus (no separate local_train call); `control_nodes` run the
    same-budget local_train with no exchange at all, for the Appendix
    A.1 fairness-aligned comparison. Both dicts are mutated in place
    (each Node's .model is updated); nothing is reloaded from disk here.
    """
    payloads = {node_id: node.compute_knowledge(probe_loader) for node_id, node in nodes.items()}
    per_node_bytes_sent = {node_id: payload.size_bytes() for node_id, payload in payloads.items()}
    total_bytes_exchanged = sum(per_node_bytes_sent.values()) * max(0, len(nodes) - 1)

    per_node_distill_loss: dict[str, dict[str, float]] = {}
    for node_id, node in nodes.items():
        peer_payloads = [p for nid, p in payloads.items() if nid != node_id]
        if not peer_payloads:
            continue  # single-node mesh: nothing to reconcile (mirrors mesh.py's run_round)
        consensus_prototypes = aggregate_prototypes(
            [p.prototypes for p in peer_payloads],
            method=aggregation_method,
            trim_fraction=trim_fraction,
            krum_neighbors=krum_neighbors,
        )
        consensus_crop_logits = aggregate_logits(
            [p.crop_logits for p in peer_payloads],
            method=aggregation_method,
            trim_fraction=trim_fraction,
            krum_neighbors=krum_neighbors,
        )
        consensus_disease_logits = aggregate_logits(
            [p.disease_logits for p in peer_payloads],
            method=aggregation_method,
            trim_fraction=trim_fraction,
            krum_neighbors=krum_neighbors,
        )
        ctx = tracker.track(f"{node_id}_kt_distill") if tracker is not None else nullcontext()
        with ctx:
            per_node_distill_loss[node_id] = node.distill(
                consensus_prototypes,
                consensus_crop_logits,
                consensus_disease_logits,
                probe_loader,
                epochs=distill_epochs,
                lr=distill_lr,
                proto_weight=proto_weight,
                kd_weight=kd_weight,
                temperature=temperature,
            )

    for node_id, node in control_nodes.items():
        ctx = tracker.track(f"{node_id}_local_only_control") if tracker is not None else nullcontext()
        with ctx:
            node.local_train(epochs=distill_epochs, lr=distill_lr)

    return {
        "per_node_distill_loss": per_node_distill_loss,
        "per_node_bytes_sent": per_node_bytes_sent,
        "total_bytes_exchanged": total_bytes_exchanged,
    }
