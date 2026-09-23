"""The continual data stream: PlantVillage images arriving at the nodes
batch by batch, one batch per trigger.

Set up once (the first `python -m src.train`):
  - carve the 5% global probe set (fixed for every batch)
  - decide which node each remaining image belongs to (the non-IID
    partition — which farm would photograph it)

Then each batch, only when it's triggered:
  - draw a stratified sample of `size` images from the images NOT used by
    any earlier batch (continual.first_batch_size for batch 0, then
    continual.next_batch_size per `python -m src.train --next-batch`)
  - hand every sampled image to the node that owns it
  - each node splits what it received into its private train/test

Everything is saved to stream.json after every change, so the next trigger
(a separate process, possibly days later) continues from exactly where the
last one stopped, and every architecture sees the same stream.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.data.splits import BatchSplit, carve_probe_and_partition, split_node_arrival, stratified_sample


def stream_manifest(cfg, dataset) -> dict:
    """What has to be unchanged for a saved stream's indices to still point
    at the same images and partition.
    """
    return {
        "num_images": len(dataset),
        "classes": list(dataset.base.classes),
        "seed": cfg.get("data.seed", 42),
        "num_nodes": cfg.get("data.num_nodes", 6),
        "non_iid_strategy": cfg.get("data.non_iid_strategy", "dirichlet"),
        "dirichlet_alpha": cfg.get("data.dirichlet_alpha", 0.5),
        "probe_set_fraction": cfg.get("data.probe_set_fraction", 0.05),
    }


class DataStream:
    def __init__(self, path: Path, manifest: dict, probe_idx: list[int], node_shards: list[list[int]], batches: list[dict]):
        self.path = Path(path)
        self.manifest = manifest
        self.probe_idx = probe_idx
        self.node_shards = node_shards
        self.batches = batches
        self._owner = {idx: node_i for node_i, shard in enumerate(node_shards) for idx in shard}

    @classmethod
    def create(cls, cfg, dataset, path: Path) -> "DataStream":
        probe_idx, node_shards = carve_probe_and_partition(cfg, dataset)
        stream = cls(path, stream_manifest(cfg, dataset), probe_idx, node_shards, [])
        stream.save()
        return stream

    @classmethod
    def load(cls, cfg, dataset, path: Path) -> "DataStream":
        data = json.loads(Path(path).read_text())
        expected = stream_manifest(cfg, dataset)
        mismatches = [k for k in expected if data["manifest"].get(k) != expected[k]]
        if mismatches:
            raise ValueError(
                f"The saved stream at {path} was built from a different dataset/config "
                f"(changed: {mismatches}) — its image indices would point at the wrong images. "
                f"Restore the original setup, or start over with `python -m src.train --reset`."
            )
        return cls(path, data["manifest"], data["probe_idx"], data["node_shards"], data["batches"])

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "manifest": self.manifest,
            "probe_idx": self.probe_idx,
            "node_shards": self.node_shards,
            "batches": self.batches,
        }))

    @property
    def num_nodes(self) -> int:
        return len(self.node_shards)

    @property
    def num_batches(self) -> int:
        return len(self.batches)

    def remaining(self) -> list[int]:
        used = {idx for batch in self.batches for split in batch["nodes"].values()
                for idx in split["train_idx"] + split["test_idx"]}
        return sorted(idx for shard in self.node_shards for idx in shard if idx not in used)

    def next_batch(self, dataset, size: int, test_fraction: float, seed: int) -> dict:
        """Draws, routes, and splits the next batch, appends it, and saves."""
        pool = self.remaining()
        if not pool:
            raise ValueError("Every PlantVillage image has already been used — the stream is exhausted.")
        batch_idx = self.num_batches
        sample = stratified_sample(dataset, pool, size, seed=seed + 1000 * batch_idx)
        arrivals: dict[int, list[int]] = {}
        for idx in sample:
            arrivals.setdefault(self._owner[idx], []).append(idx)
        nodes = {}
        for node_i in range(self.num_nodes):
            split = split_node_arrival(dataset, arrivals.get(node_i, []), test_fraction, seed=seed + batch_idx)
            nodes[f"node_{node_i}"] = {"train_idx": split.train_idx, "test_idx": split.test_idx}
        batch = {"batch_idx": batch_idx, "requested_size": size, "size": len(sample), "nodes": nodes}
        self.batches.append(batch)
        self.save()
        return batch

    def node_split(self, batch_idx: int, node_id: str) -> BatchSplit:
        split = self.batches[batch_idx]["nodes"][node_id]
        return BatchSplit(split["train_idx"], split["test_idx"])
