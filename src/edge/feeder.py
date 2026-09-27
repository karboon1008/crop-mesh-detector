"""Simulation only: stands in for the farms' cameras. Each trigger draws the
next stratified batch from PlantVillage with the same stream as
`python -m src.train` (src/data/stream.py), then

  - writes every farm's inbox manifest (its train / unlabelled / test photo
    paths, see src/edge/data.py), filling in `hidden_class` for unlabelled
    photos so pseudo-label accuracy can be measured
  - publishes the co-op's label space to the knowledge server (first time)
  - announces the round: its probe image paths and the farms with enough
    photos to take part (>= 2 labelled train, >= 1 test)

On a real deployment the manifests are written by each farm's own capture
app and the co-op announces rounds on its own schedule; nothing else changes.

    python -m src.edge.feeder --server http://localhost:8000              # batch 0
    python -m src.edge.feeder --server http://localhost:8000 --next-batch # each later batch
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from src.config import Config
from src.data.plantvillage import load_full_dataset
from src.data.stream import DataStream
from src.edge.client import KnowledgeClient
from src.edge.data import LabelSpace, inbox_path


def label_space(dataset) -> LabelSpace:
    return LabelSpace(
        crop_classes=list(dataset.labels.crop_classes),
        disease_classes=list(dataset.labels.disease_classes),
        name_to_crop_disease={
            name: tuple(dataset.labels.class_to_crop_disease[i]) for i, name in enumerate(dataset.base.classes)
        },
        image_size=dataset.image_size,
    )


def _entries(dataset, indices: list[int], key: str) -> list[dict]:
    out = []
    for idx in indices:
        path, target = dataset.base.samples[idx]
        out.append({"path": str(Path(path).resolve()), key: dataset.base.classes[target]})
    return out


def publish_batch(dataset, batch: dict, edge_dir: str | Path, client: KnowledgeClient) -> list[str]:
    """Writes the batch's inbox manifests and announces it; returns the farms
    expected to take part.
    """
    b = batch["batch_idx"]
    expected = []
    for node_id, split in batch["nodes"].items():
        manifest = {
            "batch_idx": b,
            "node_id": node_id,
            "train": _entries(dataset, split["train_idx"], "class"),
            "unlabeled": _entries(dataset, split.get("unlabeled_idx", []), "hidden_class"),
            "test": _entries(dataset, split["test_idx"], "class"),
        }
        path = inbox_path(edge_dir, node_id, b)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest))
        if len(split["train_idx"]) >= 2 and split["test_idx"]:
            expected.append(node_id)
    probe = [e["path"] for e in _entries(dataset, batch["probe_idx"], "class")]
    client.announce_round(b, probe, expected)
    return expected


def trigger(cfg, client: KnowledgeClient, edge_dir: str | Path, next_batch: bool, dataset=None) -> dict:
    """Batch 0 on a fresh stream, or (next_batch) the next one."""
    edge_dir = Path(edge_dir)
    dataset = dataset or load_full_dataset(cfg.get("data.root"), cfg.get("data.image_size", 224))
    stream_path = edge_dir / "stream.json"
    seed = cfg.get("data.seed", 42)
    probe_fraction = cfg.get("data.probe_set_fraction", 0.05)
    test_fraction = cfg.get("data.test_fraction", 0.15)

    client.put_classes(label_space(dataset).to_dict())
    if not stream_path.exists():
        if next_batch:
            raise SystemExit("No stream yet — run the feeder without --next-batch first (batch 0).")
        if client.rounds():
            raise SystemExit(
                "The knowledge server already has rounds from another run — restart it with --reset."
            )
        stream = DataStream.create(cfg, dataset, stream_path)
        size, labeled = cfg.get("continual.first_batch_size", 20000), cfg.get("continual.first_batch_labeled_fraction", 1.0)
    else:
        if not next_batch:
            raise SystemExit("The stream already has batch 0 — pass --next-batch for the next one (or --reset).")
        stream = DataStream.load(cfg, dataset, stream_path)
        size, labeled = cfg.get("continual.next_batch_size", 3000), cfg.get("continual.labeled_fraction", 0.1)
    batch = stream.next_batch(dataset, size, probe_fraction, test_fraction, seed, labeled_fraction=labeled)
    expected = publish_batch(dataset, batch, edge_dir, client)
    print(
        f"Round {batch['batch_idx']}: {batch['size']} images, probe {len(batch['probe_idx'])}, "
        f"{batch['labeled_fraction']:.0%} of train labelled, farms expected: {expected}; "
        f"{len(stream.remaining())} images left"
    )
    return batch


def reset(edge_dir: str | Path) -> None:
    """Forgets the stream, every inbox and every farm's local state (the
    simulation keeps them all under one edge_dir)."""
    edge_dir = Path(edge_dir)
    (edge_dir / "stream.json").unlink(missing_ok=True)
    shutil.rmtree(edge_dir / "inbox", ignore_errors=True)
    shutil.rmtree(edge_dir / "nodes", ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Simulated farm cameras: publish the next batch of photos")
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: repo root)")
    parser.add_argument("--server", default=None, help="Knowledge server URL (default: edge.server_url)")
    parser.add_argument("--edge-dir", default=None, help="Where inboxes and the stream live (default: edge.dir)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--next-batch", action="store_true", help="Publish the next batch")
    group.add_argument(
        "--reset", action="store_true",
        help="Start again from batch 0 (also restart the server with --reset)",
    )
    args = parser.parse_args()
    cfg = Config.load(args.config)
    edge_dir = args.edge_dir or cfg.get("edge.dir", "edge_run")
    if args.reset:
        reset(edge_dir)
    client = KnowledgeClient(args.server or cfg.get("edge.server_url", "http://localhost:8000"))
    trigger(cfg, client, edge_dir, next_batch=args.next_batch)


if __name__ == "__main__":
    main()
