"""One farm's node agent: runs that farm's rounds of the continual mesh on
its own edge device (a Jetson, a mini-PC, or a Raspberry Pi 5 for the
smaller models), talking only to the co-op's knowledge server.

Each round:

  1. read this farm's inbox manifest (its new photos) and the round's
     announcement from the server (the public probe images, which farms
     are expected)
  2. local training (labelled + pseudo-labelled photos), pre-distill
     evaluation, EMA -> role                       [ContinualMesh.local_phase]
  3. upload: probe logits always; prototypes too if a teacher
                                                   [ContinualMesh.upload_phase]
  4. a learner waits until the other expected farms have refreshed their
     logits for this round (or edge.round_timeout_s passes), then
     retrieves, aggregates and distils             [ContinualMesh.learn_phase]
  5. post-distill evaluation                       [ContinualMesh.finish_phase]
  6. save the model, optimizer and EMA locally, append the round's record,
     and (edge.export_onnx) export the updated model to ONNX for the
     in-field camera app

Photos, labels and weights never leave the device — only the payload in
step 3 does. A farm whose inbox is missing or too small for a round sits it
out, keeping its EMA, exactly like the single-process simulation.

    python -m src.edge.node_agent --node-id node_0 --server http://coop:8000 --follow

Everything the agent keeps lives in <edge.dir>/nodes/<node_id>/:
state.pt (model + optimizer), agent.json (EMA, rounds done), rounds.json
(one record per round, same fields as src/train.py's batch_logs.json),
model.onnx, and emissions.csv when CodeCarbon is on.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data.splits import NodeBatchLoaders
from src.edge.client import KnowledgeClient
from src.edge.data import LabelSpace, build_node_loaders, build_probe_loader, inbox_path, read_manifest
from src.energy.tracker import ComputeEnergyTracker
from src.federated.continual import LEARNER, BatchLog, ContinualMesh, NodeBatchRecord
from src.federated.node import Node
from src.models.factory import build_model


@dataclass
class RoundContext:
    """One round in progress on this farm."""

    batch_idx: int
    expected: list[str]  # farms the server expects this round (this one included)
    loaders: NodeBatchLoaders
    probe_loader: DataLoader
    record: NodeBatchRecord | None = None


class NodeAgent:
    def __init__(
        self, cfg, node_id: str, client: KnowledgeClient, edge_dir: str | Path,
        arch: str | None = None, device: str | None = None,
    ):
        self.cfg = cfg
        self.node_id = node_id
        self.client = client
        self.edge_dir = Path(edge_dir)
        self.arch = arch or cfg.get("edge.architecture", None) or cfg.get("models.architectures", ["mobilenet_v3_small"])[0]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = cfg.get("training.lr", 0.001)
        self.dir = self.edge_dir / "nodes" / node_id
        self.dir.mkdir(parents=True, exist_ok=True)

        client.register(node_id)
        self.labels = LabelSpace.from_dict(client.get_classes())
        self.node = self._build_node()
        self.agent_state = self._read_json("agent.json", {"arch": self.arch, "completed_rounds": [], "ema": None})
        if self.agent_state.get("arch", self.arch) != self.arch:
            raise SystemExit(
                f"{self.dir} holds a {self.agent_state['arch']} model, not {self.arch} — "
                f"use a different --edge-dir or delete it"
            )
        if (self.dir / "state.pt").exists():
            self.node.load_state(torch.load(self.dir / "state.pt", map_location=self.device), self.lr)

        tracker = ComputeEnergyTracker(
            enabled=cfg.get("energy.track_with_codecarbon", True),
            output_dir=self.dir,
            country_iso_code=cfg.get("energy.country_iso_code", "GBR"),
            fallback_power_watts=cfg.get("energy.fallback_power_watts", 15.0),
        )
        # a mesh of one: this farm's own node, with the knowledge server as its store
        self.mesh = ContinualMesh(
            [self.node], client, tracker,
            aggregation_method=cfg.get("federated.aggregation", "trimmed_mean"),
            trim_fraction=cfg.get("federated.trim_fraction", 0.2),
            krum_neighbors=cfg.get("federated.krum_neighbors", 2),
            ema_alpha=cfg.get("continual.ema_alpha", 0.3),
            ema_threshold=cfg.get("continual.ema_threshold", 0.8),
            ema_metric=cfg.get("continual.ema_metric", "pair_accuracy"),
        )
        self.mesh.ema[node_id] = self.agent_state["ema"]

    def _build_node(self) -> Node:
        cfg, labels = self.cfg, self.labels
        model = build_model(
            self.arch, len(labels.crop_classes), len(labels.disease_classes),
            pretrained=cfg.get("models.pretrained", True),
            freeze_low_layers_=cfg.get("training.freeze_low_layers_in_mesh", False),
        )
        return Node(
            self.node_id, model, train_loader=None, test_loader=None, device=self.device,
            crop_classes=labels.crop_classes, disease_classes=labels.disease_classes,
            crop_loss_weight=cfg.get("training.crop_loss_weight", 1.0),
            disease_loss_weight=cfg.get("training.disease_loss_weight", 1.0),
            loss_type=cfg.get("training.loss_type", "cross_entropy"),
            focal_gamma=cfg.get("training.focal_gamma", 2.0),
            pair_class_names=labels.pair_class_names, class_to_crop_disease=labels.class_to_crop_disease,
            pseudo_threshold=cfg.get("continual.pseudo_label_threshold", 0.95),
            unlabeled_weight=cfg.get("continual.unlabeled_weight", 1.0),
            pseudo_prototype_check=cfg.get("continual.pseudo_label_prototype_check", True),
        )

    def _read_json(self, name: str, default):
        path = self.dir / name
        return json.loads(path.read_text()) if path.exists() else default

    def _write_json(self, name: str, data) -> None:
        tmp = self.dir / f"{name}.tmp"
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self.dir / name)

    @property
    def completed_rounds(self) -> list[int]:
        return self.agent_state["completed_rounds"]

    def pending_rounds(self) -> list[int]:
        """Rounds the server has announced that this farm hasn't done yet."""
        done = set(self.completed_rounds)
        return [b for b in self.client.rounds() if b not in done]

    def _local_epochs(self, batch_idx: int) -> int:
        base = self.cfg.get(
            f"training.local_epochs_per_round_overrides.{self.arch}", self.cfg.get("training.local_epochs_per_round", 2)
        )
        # a farm can't see the others' data sizes, so batch 0 gets a flat
        # first_batch_local_epochs rather than the simulation's median scaling
        return self.cfg.get("continual.first_batch_local_epochs", base) if batch_idx == 0 else base

    # ---- the round, step by step (tests and run_round drive these)
    def prepare(self, batch_idx: int) -> RoundContext | None:
        """Loaders for this round, or None when this farm sits it out."""
        announcement = self.client.get_round(batch_idx)
        manifest_path = inbox_path(self.edge_dir, self.node_id, batch_idx)
        loaders = (
            build_node_loaders(self.cfg, read_manifest(manifest_path), self.labels)
            if manifest_path.exists() and self.node_id in announcement["nodes"] else None
        )
        if loaders is None:
            return None
        return RoundContext(
            batch_idx, announcement["nodes"], loaders,
            build_probe_loader(self.cfg, announcement["probe"], self.labels),
        )

    def train_and_upload(self, ctx: RoundContext) -> NodeBatchRecord:
        ctx.record = self.mesh.local_phase(
            self.node, ctx.batch_idx, ctx.loaders, self._local_epochs(ctx.batch_idx), self.lr
        )
        self.mesh.upload_phase(self.node, ctx.record, ctx.probe_loader)
        return ctx.record

    def learn_and_finish(self, ctx: RoundContext, wait: bool = True) -> NodeBatchRecord:
        record = ctx.record
        if LEARNER in record.roles and wait:
            others = [n for n in ctx.expected if n != self.node_id]
            self.client.wait_for_uploads(
                ctx.batch_idx, others,
                timeout_s=self.cfg.get("edge.round_timeout_s", 1800),
                poll_s=self.cfg.get("edge.poll_interval_s", 10),
            )
        self.mesh.learn_phase(
            self.node, record, ctx.loaders, ctx.probe_loader,
            distill_epochs=self.cfg.get("training.distill_epochs_per_round", 1),
            distill_lr=self.cfg.get("training.distill_lr", 0.0005),
            proto_weight=self.cfg.get("training.proto_weight", 0.5),
            kd_weight=self.cfg.get("training.kd_weight", 0.5),
            crop_kd_weight=self.cfg.get("training.crop_kd_weight", None),
            temperature=self.cfg.get("training.kd_temperature", 2.0),
        )
        self.mesh.finish_phase(self.node, record, ctx.loaders)
        return record

    def save_round(self, batch_idx: int, record: NodeBatchRecord | None) -> None:
        """Persists everything after a round, so a reboot never loses one."""
        from src.train import batch_log_to_json

        if record is not None:
            torch.save(self.node.state(), self.dir / "state.pt")
            if self.cfg.get("edge.export_onnx", False):
                self.export_onnx()
        log = BatchLog(batch_idx=batch_idx)
        if record is None:
            log.absent = [self.node_id]
        else:
            log.per_node[self.node_id] = record
        rounds = self._read_json("rounds.json", [])
        rounds.append(batch_log_to_json(log))
        self._write_json("rounds.json", rounds)
        self.agent_state["completed_rounds"] = sorted(set(self.completed_rounds) | {batch_idx})
        self.agent_state["ema"] = self.mesh.ema[self.node_id]
        self._write_json("agent.json", self.agent_state)

    def export_onnx(self) -> Path:
        """The updated model for the camera app (apps/, src/infer.py)."""
        from src.validation.export_onnx import export_onnx

        path = self.dir / "model.onnx"
        export_onnx(copy.deepcopy(self.node.model).cpu(), self.labels.image_size, path)
        return path

    def run_round(self, batch_idx: int) -> NodeBatchRecord | None:
        ctx = self.prepare(batch_idx)
        if ctx is None:
            print(f"[{self.node_id}] round {batch_idx}: no usable photos — sitting it out")
            self.save_round(batch_idx, None)
            return None
        record = self.train_and_upload(ctx)
        print(
            f"[{self.node_id}] round {batch_idx}: {record.num_train} labelled / {record.num_unlabeled} unlabelled / "
            f"{record.num_test} test, EMA={record.ema:.3f}, roles={'+'.join(record.roles) or 'idle'}"
        )
        self.learn_and_finish(ctx)
        metric = self.mesh.ema_metric
        print(
            f"[{self.node_id}] round {batch_idx}: {metric} pre={record.pre_distill_eval[metric]:.3f} "
            f"post={record.post_distill_eval[metric]:.3f}, peers={sorted(record.peers_used)}, "
            f"{record.total_bytes} bytes, {record.total_energy_kwh:.6f} kWh"
        )
        if record.num_early_warning:
            print(
                f"[{self.node_id}] round {batch_idx}: early warning ({record.num_early_warning} photos): "
                f"disease_accuracy pre={record.pre_early_warning_eval['disease_accuracy']:.3f} "
                f"post={record.post_early_warning_eval['disease_accuracy']:.3f}"
            )
        self.save_round(batch_idx, record)
        return record

    def run_pending(self) -> list[int]:
        done = []
        for b in self.pending_rounds():
            self.run_round(b)
            done.append(b)
        return done

    def follow(self) -> None:
        """Runs every announced round as it appears, forever."""
        poll = self.cfg.get("edge.poll_interval_s", 10)
        while True:
            if not self.run_pending():
                time.sleep(poll)


def main():
    parser = argparse.ArgumentParser(description="Run one farm's rounds of the continual mesh")
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: repo root)")
    parser.add_argument("--node-id", required=True, help="This farm's id, e.g. node_0")
    parser.add_argument("--server", default=None, help="Knowledge server URL (default: edge.server_url)")
    parser.add_argument("--edge-dir", default=None, help="Inbox + local state directory (default: edge.dir)")
    parser.add_argument("--arch", default=None, help="Model architecture (default: edge.architecture)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--round", type=int, default=None, help="Run just this round")
    mode.add_argument("--follow", action="store_true", help="Keep running new rounds as they are announced")
    args = parser.parse_args()
    cfg = Config.load(args.config)

    client = KnowledgeClient(args.server or cfg.get("edge.server_url", "http://localhost:8000"))
    agent = NodeAgent(cfg, args.node_id, client, args.edge_dir or cfg.get("edge.dir", "edge_run"), arch=args.arch)
    if args.round is not None:
        agent.run_round(args.round)
    elif args.follow:
        agent.follow()
    else:
        done = agent.run_pending()
        print(f"[{args.node_id}] ran rounds {done}" if done else f"[{args.node_id}] no new rounds")


if __name__ == "__main__":
    main()
