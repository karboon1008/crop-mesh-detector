"""The continual data stream: PlantVillage images arriving at the nodes
batch by batch, one batch per trigger.

Set up once (the first `python -m src.train`):
  - decide which node each image belongs to (the non-IID partition —
    which farm would photograph it)

Then each batch, only when it's triggered:
  - draw a stratified sample of `size` images from the images NOT used by
    any earlier batch (continual.first_batch_size for batch 0, then
    continual.next_batch_size per `python -m src.train --next-batch`)
  - carve a stratified data.probe_set_fraction (5%) of THIS batch into the
    public probe set (e.g. 150 of a 3,000-image batch)
  - hand every other image of the batch to the node that owns it
  - each node splits what it received into its private train/test

The probe set is NOT cumulative: each batch uses only its own probe slice.
Probe logits therefore only line up with the batch they were computed on,
so a learner only takes logits from entries uploaded in the same batch
(see src/federated/continual.py).

Everything is saved to stream.json after every change, so the next trigger
(a separate process, possibly days later) continues from exactly where the
last one stopped, and every architecture sees the same stream.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.data.plantvillage import partition_nodes
from src.data.splits import BatchSplit, split_node_arrival, stratified_sample


def stream_manifest(cfg, dataset) -> dict:
    """What has to be unchanged for a saved stream's indices to still point
    at the same images and partition.
    """
    strategy = cfg.get("data.non_iid_strategy", "dirichlet")
    return {
        "num_images": len(dataset),
        "classes": list(dataset.base.classes),
        "extra_sources": cfg.get("data.extra_sources", None),
        "seed": cfg.get("data.seed", 42),
        "num_nodes": cfg.get("data.num_nodes", 6),
        "non_iid_strategy": strategy,
        # only the crop-assigning strategies read manual_node_crops
        "node_crops": (
            cfg.get("data.manual_node_crops", None) if strategy in ("manual", "dirichlet_by_crop") else None
        ),
        "dirichlet_alpha": cfg.get("data.dirichlet_alpha", 0.5),
        "probe_set_fraction": cfg.get("data.probe_set_fraction", 0.05),
    }


class DataStream:
    def __init__(self, path: Path, manifest: dict, node_shards: list[list[int]], batches: list[dict]):
        self.path = Path(path)
        self.manifest = manifest
        self.node_shards = node_shards
        self.batches = batches
        self._owner = {idx: node_i for node_i, shard in enumerate(node_shards) for idx in shard}

    @classmethod
    def create(cls, cfg, dataset, path: Path) -> "DataStream":
        node_shards = partition_nodes(
            dataset,
            list(range(len(dataset))),
            cfg.get("data.num_nodes", 6),
            cfg.get("data.non_iid_strategy", "dirichlet"),
            cfg.get("data.dirichlet_alpha", 0.5),
            cfg.get("data.seed", 42),
            manual_node_crops=cfg.get("data.manual_node_crops", None),
        )
        stream = cls(path, stream_manifest(cfg, dataset), node_shards, [])
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
        return cls(path, data["manifest"], data["node_shards"], data["batches"])

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "manifest": self.manifest,
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
        used = {idx for batch in self.batches for idx in batch["probe_idx"]}
        used |= {idx for batch in self.batches for split in batch["nodes"].values()
                 for idx in split["train_idx"] + split["test_idx"]}
        return sorted(idx for shard in self.node_shards for idx in shard if idx not in used)

    def next_batch(self, dataset, size: int, probe_fraction: float, test_fraction: float, seed: int) -> dict:
        """Draws the next batch, carves its probe slice, routes the rest to
        the owning nodes, lets each node split it, appends it, and saves.
        """
        pool = self.remaining()
        if not pool:
            raise ValueError("Every PlantVillage image has already been used — the stream is exhausted.")
        batch_idx = self.num_batches
        sample = stratified_sample(dataset, pool, size, seed=seed + 1000 * batch_idx)
        probe_idx = stratified_sample(dataset, sample, round(len(sample) * probe_fraction), seed=seed + 1000 * batch_idx + 1)
        probe_set = set(probe_idx)
        arrivals: dict[int, list[int]] = {}
        for idx in (i for i in sample if i not in probe_set):
            arrivals.setdefault(self._owner[idx], []).append(idx)
        nodes = {}
        for node_i in range(self.num_nodes):
            split = split_node_arrival(dataset, arrivals.get(node_i, []), test_fraction, seed=seed + batch_idx)
            nodes[f"node_{node_i}"] = {"train_idx": split.train_idx, "test_idx": split.test_idx}
        batch = {
            "batch_idx": batch_idx, "requested_size": size, "size": len(sample),
            "probe_idx": probe_idx, "nodes": nodes,
        }
        self.batches.append(batch)
        self.save()
        return batch

    def node_split(self, batch_idx: int, node_id: str) -> BatchSplit:
        split = self.batches[batch_idx]["nodes"][node_id]
        return BatchSplit(split["train_idx"], split["test_idx"])
