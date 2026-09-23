"""Continual knowledge mesh: nodes see their private data as a stream of
batches and exchange knowledge through ONE shared database
(src/federated/knowledge_store.py) instead of a fixed number of
all-to-all rounds.

Per batch, every node:

  1. local training       private train batch -> backbone -> feature ->
                          linear heads -> logits -> cross-entropy -> backward
  2. pre-distill eval     on this batch's private test set
  3. decide its role      batch 0: every node is a teacher AND a learner.
                          later batches, from an EMA of the pre-distill
                          accuracy:
                            teacher  if EMA rose vs. the previous batch
                            learner  if EMA < ema_threshold (default 0.8)
                          (a node can be both, or neither — then it just
                          keeps its locally trained model)
  4. teachers             extract prototypes (this batch's private train
                          set: mean feature per class) + probe logits (this
                          batch's probe set: logits per image) and upload
                          them, replacing their previous entry
  5. learners             retrieve every OTHER node's latest entry,
                          aggregate prototypes and probe logits with
                          trimmed mean / Krum (own knowledge excluded), and
                          distil towards that consensus. Probe logits only
                          count from entries uploaded THIS batch (older
                          entries' logits are for an older batch's probe
                          images); prototypes count from every entry
  6. post-distill eval    same private test set as step 2
  7. record               post vs. pre (did distillation help?), bytes
                          uploaded/downloaded, compute energy per phase

All teachers upload before any learner retrieves, so a learner always sees
this batch's freshest knowledge plus the latest entry of every node that
didn't upload this time. Models carry over from batch to batch.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader

from src.energy.tracker import ComputeEnergyTracker
from src.evaluate import scalar_metrics
from src.federated.aggregation import aggregate_masked_logits, aggregate_prototypes
from src.federated.knowledge_store import KnowledgeStore
from src.federated.node import Node

TEACHER = "teacher"
LEARNER = "learner"


def update_ema(prev_ema: float | None, value: float, alpha: float) -> float:
    """EMA_t = alpha * value + (1 - alpha) * EMA_{t-1}; the first value seeds it."""
    if prev_ema is None:
        return value
    return alpha * value + (1 - alpha) * prev_ema


def decide_roles(batch_idx: int, ema: float, prev_ema: float | None, threshold: float) -> list[str]:
    if batch_idx == 0 or prev_ema is None:
        return [TEACHER, LEARNER]
    roles = []
    if ema > prev_ema:
        roles.append(TEACHER)
    if ema < threshold:
        roles.append(LEARNER)
    return roles


@dataclass
class NodeBatchRecord:
    node_id: str
    batch_idx: int
    num_train: int
    num_test: int
    local_epochs: int
    train_loss: float
    pre_distill_eval: dict
    ema: float
    prev_ema: float | None
    roles: list[str]
    uploaded: bool = False
    distilled: bool = False
    peers_used: dict[str, int] = field(default_factory=dict)  # peer -> batch its entry was uploaded in
    logit_peers: list[str] = field(default_factory=list)  # peers whose probe logits were fresh (uploaded this batch)
    probe_images_used: int = 0  # probe images KD ran on (0 = no fresh peer logits, prototypes only)
    distill_loss: dict[str, float] = field(default_factory=dict)
    post_distill_eval: dict = field(default_factory=dict)
    distill_gain: dict[str, float] = field(default_factory=dict)  # post - pre, per scalar metric
    improved: bool = False
    bytes_uploaded: int = 0
    bytes_downloaded: int = 0
    energy_kwh: dict[str, float] = field(default_factory=dict)  # phase -> kWh
    duration_s: dict[str, float] = field(default_factory=dict)  # phase -> seconds

    @property
    def total_energy_kwh(self) -> float:
        return sum(self.energy_kwh.values())

    @property
    def total_bytes(self) -> int:
        return self.bytes_uploaded + self.bytes_downloaded


@dataclass
class BatchLog:
    batch_idx: int
    per_node: dict[str, NodeBatchRecord] = field(default_factory=dict)
    # nodes that received too little of this batch to train + evaluate; they
    # sit it out, keeping their model, EMA, and database entry unchanged
    absent: list[str] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(r.total_bytes for r in self.per_node.values())

    @property
    def total_energy_kwh(self) -> float:
        return sum(r.total_energy_kwh for r in self.per_node.values())


class ContinualMesh:
    def __init__(
        self,
        nodes: list[Node],
        store: KnowledgeStore,
        tracker: ComputeEnergyTracker,
        aggregation_method: str = "trimmed_mean",
        trim_fraction: float = 0.2,
        krum_neighbors: int = 2,
        ema_alpha: float = 0.3,
        ema_threshold: float = 0.8,
        ema_metric: str = "pair_accuracy",
        label_prefix: str = "",
    ):
        self.nodes = nodes
        self.store = store
        self.tracker = tracker
        self.aggregation_method = aggregation_method
        self.trim_fraction = trim_fraction
        self.krum_neighbors = krum_neighbors
        self.ema_alpha = ema_alpha
        self.ema_threshold = ema_threshold
        self.ema_metric = ema_metric
        self.label_prefix = label_prefix
        self.ema: dict[str, float | None] = {node.node_id: None for node in nodes}

    @contextmanager
    def _track(self, record: NodeBatchRecord, phase: str):
        """Tracks one phase of one node's batch and adds its energy/time to `record`."""
        label = f"{self.label_prefix}batch{record.batch_idx}_{record.node_id}_{phase}"
        with self.tracker.track(label) as block:
            yield
        record.energy_kwh[phase] = record.energy_kwh.get(phase, 0.0) + block.get("energy_kwh", 0.0)
        record.duration_s[phase] = record.duration_s.get(phase, 0.0) + block.get("duration_s", 0.0)

    def _metric(self, eval_: dict) -> float:
        return float(eval_[self.ema_metric])

    def run_batch(
        self,
        batch_idx: int,
        batch_loaders: dict[str, tuple],
        probe_loader: DataLoader,
        local_epochs: dict[str, int],
        lr: float,
        distill_epochs: int,
        distill_lr: float,
        proto_weight: float,
        kd_weight: float,
        crop_kd_weight: float | None,
        temperature: float,
    ) -> BatchLog:
        """`batch_loaders`: node_id -> (train_loader, test_loader,
        crop_class_weights, disease_class_weights) for this batch. Nodes
        missing from it received no usable data this batch and sit it out.

        `probe_loader`: an unshuffled loader over THIS batch's probe set.
        """
        crop_kd_weight = kd_weight if crop_kd_weight is None else crop_kd_weight
        log = BatchLog(batch_idx=batch_idx)
        present = [node for node in self.nodes if node.node_id in batch_loaders]
        log.absent = [node.node_id for node in self.nodes if node.node_id not in batch_loaders]

        # 1) + 2) local training on this batch, then pre-distill evaluation
        for node in present:
            train_loader, test_loader, crop_weights, disease_weights = batch_loaders[node.node_id]
            node.train_loader, node.test_loader = train_loader, test_loader
            node.crop_class_weights = crop_weights.to(node.device) if crop_weights is not None else None
            node.disease_class_weights = disease_weights.to(node.device) if disease_weights is not None else None

            record = NodeBatchRecord(
                node_id=node.node_id, batch_idx=batch_idx,
                num_train=len(train_loader.dataset), num_test=len(test_loader.dataset),
                local_epochs=local_epochs[node.node_id], train_loss=0.0,
                pre_distill_eval={}, ema=0.0, prev_ema=self.ema[node.node_id], roles=[],
            )
            with self._track(record, "local_train"):
                record.train_loss = node.local_train(local_epochs[node.node_id], lr)
            with self._track(record, "evaluate"):
                record.pre_distill_eval = node.evaluate()

            # 3) role from the EMA of pre-distill performance
            record.ema = update_ema(record.prev_ema, self._metric(record.pre_distill_eval), self.ema_alpha)
            self.ema[node.node_id] = record.ema
            record.roles = decide_roles(batch_idx, record.ema, record.prev_ema, self.ema_threshold)
            log.per_node[node.node_id] = record

        # 4) teachers extract knowledge (probe logits on this batch's probe
        # set) and upload it, replacing their old entry
        for node in present:
            record = log.per_node[node.node_id]
            if TEACHER not in record.roles:
                continue
            with self._track(record, "knowledge_extraction"):
                payload = node.compute_knowledge(probe_loader)
            record.bytes_uploaded = self.store.upload(node.node_id, batch_idx, payload)
            record.uploaded = True

        # 5) learners retrieve peers' latest knowledge, aggregate (self
        # excluded), and distil towards it
        for node in present:
            record = log.per_node[node.node_id]
            if LEARNER not in record.roles:
                continue
            peers = self.store.fetch_peers(node.node_id, batch_idx, logits_batch=batch_idx)
            if not peers:
                continue  # nobody has uploaded anything yet
            record.peers_used = {peer: peer_batch for peer, (peer_batch, _) in peers.items()}
            record.bytes_downloaded = sum(payload.size_bytes() for _, payload in peers.values())
            peer_payloads = [payload for _, payload in peers.values()]
            # only entries uploaded this batch carry logits for this batch's probe
            logit_payloads = [p for p in peer_payloads if p.crop_logits.shape[0] > 0]
            record.logit_peers = sorted(peer for peer, (_, p) in peers.items() if p.crop_logits.shape[0] > 0)
            record.probe_images_used = len(probe_loader.dataset) if logit_payloads else 0

            with self._track(record, "distill"):
                consensus_prototypes = aggregate_prototypes(
                    [p.prototypes for p in peer_payloads], method=self.aggregation_method,
                    trim_fraction=self.trim_fraction, krum_neighbors=self.krum_neighbors,
                )
                if logit_payloads:
                    consensus_crop_logits, crop_known_mask = aggregate_masked_logits(
                        [p.crop_logits for p in logit_payloads], [p.known_crop_classes for p in logit_payloads],
                        method=self.aggregation_method, trim_fraction=self.trim_fraction,
                        krum_neighbors=self.krum_neighbors,
                    )
                    consensus_disease_logits, disease_known_mask = aggregate_masked_logits(
                        [p.disease_logits for p in logit_payloads], [p.known_disease_classes for p in logit_payloads],
                        method=self.aggregation_method, trim_fraction=self.trim_fraction,
                        krum_neighbors=self.krum_neighbors,
                    )
                else:
                    # no teacher uploaded this batch: prototype alignment only, no KD
                    ref = peer_payloads[0]
                    consensus_crop_logits = consensus_disease_logits = None
                    crop_known_mask = torch.zeros(ref.crop_logits.shape[1], dtype=torch.bool)
                    disease_known_mask = torch.zeros(ref.disease_logits.shape[1], dtype=torch.bool)
                record.distill_loss = node.distill(
                    consensus_prototypes, consensus_crop_logits, crop_known_mask,
                    consensus_disease_logits, disease_known_mask, probe_loader if logit_payloads else None,
                    epochs=distill_epochs, lr=distill_lr, proto_weight=proto_weight,
                    kd_weight=kd_weight, crop_kd_weight=crop_kd_weight, temperature=temperature,
                )
            record.distilled = True

        # 6) + 7) post-distill evaluation on the same test set, and the comparison
        for node in present:
            record = log.per_node[node.node_id]
            if record.distilled:
                with self._track(record, "evaluate"):
                    record.post_distill_eval = node.evaluate()
            else:
                record.post_distill_eval = record.pre_distill_eval
            pre, post = scalar_metrics(record.pre_distill_eval), scalar_metrics(record.post_distill_eval)
            record.distill_gain = {m: post[m] - pre[m] for m in pre}
            record.improved = record.distilled and self._metric(record.post_distill_eval) > self._metric(
                record.pre_distill_eval
            )
        return log
