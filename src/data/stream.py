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
  - each node splits what it received into its private train/test, and
    (continual.labeled_fraction < 1) only a share of its train images keep
    their labels — the rest arrive unlabelled and are pseudo-labelled

Early-warning test (data.early_warning.enabled): when the stream is set up,
a "threat" class for a node is a disease on a crop that node grows which it
never receives but some other node does (data.non_iid_strategy
"farm_crops" makes these on purpose). A share of every threat class's images
is held out of every shard — never trained on, never in a probe — and each
node is tested on the held-out images of its own threat classes before and
after distillation: did the other farms teach it a disease on its own crop
before that disease reached it?

The probe set is NOT cumulative: each batch uses only its own probe slice.
Probe logits therefore only line up with the batch they were computed on,
so every node refreshes its logits each batch and a learner only takes
logits computed in the same batch (see src/federated/continual.py).

Everything is saved to stream.json after every change, so the next trigger
(a separate process, possibly days later) continues from exactly where the
last one stopped, and every architecture sees the same stream.
"""

from __future__ import annotations

import json
from pathlib import Path

import random

from src.data.plantvillage import farm_crop_assignment, partition_nodes
from src.data.splits import BatchSplit, split_node_arrival, stratified_sample


def stream_manifest(cfg, dataset) -> dict:
    """What has to be unchanged for a saved stream's indices to still point
    at the same images and partition.
    """
    manifest = {
        "num_images": len(dataset),
        "classes": list(dataset.base.classes),
        "seed": cfg.get("data.seed", 42),
        "num_nodes": cfg.get("data.num_nodes", 6),
        "non_iid_strategy": cfg.get("data.non_iid_strategy", "dirichlet"),
        "dirichlet_alpha": cfg.get("data.dirichlet_alpha", 0.5),
        "probe_set_fraction": cfg.get("data.probe_set_fraction", 0.05),
    }
    # only recorded when used, so streams saved before these existed still load
    if manifest["non_iid_strategy"] == "farm_crops":
        manifest["farm_crops"] = {
            "growers_per_crop": cfg.get("data.farm_crops.growers_per_crop", 2),
            "disease_spread": cfg.get("data.farm_crops.disease_spread", 0.5),
        }
    if manifest["non_iid_strategy"] == "manual_classes":
        manifest["manual_node_classes"] = cfg.get("data.manual_node_classes", None)
    if cfg.get("data.early_warning.enabled", False):
        manifest["early_warning_holdout_fraction"] = cfg.get("data.early_warning.holdout_fraction", 0.1)
    return manifest


def _crop_of(dataset, cls: int) -> str:
    return dataset.labels.crop_classes[dataset.labels.class_to_crop_disease[cls][0]]


def threat_classes(dataset, node_shards: list[list[int]], node_crops: list[list[str]]) -> list[list[int]]:
    """Per node: classes on a crop it grows that it never receives but some
    other node does (so a peer could teach it).
    """
    targets = dataset.targets
    owned = [{targets[i] for i in shard} for shard in node_shards]
    all_classes = set().union(*owned) if owned else set()
    return [
        sorted(c for c in all_classes
               if _crop_of(dataset, c) in node_crops[n] and c not in owned[n]
               and any(c in owned[m] for m in range(len(node_shards)) if m != n))
        for n in range(len(node_shards))
    ]


def carve_early_warning(
    dataset, node_shards: list[list[int]], node_crops: list[list[str]], fraction: float, seed: int,
) -> tuple[list[list[int]], dict[int, list[int]]]:
    """Holds `fraction` (at least 1 image) of every threat class out of the
    shards; returns (shards without them, {class: held-out indices}).
    """
    targets = dataset.targets
    threats = sorted(set().union(*map(set, threat_classes(dataset, node_shards, node_crops))))
    rng = random.Random(seed + 7)
    holdout: dict[int, list[int]] = {}
    for cls in threats:
        cls_indices = sorted(i for shard in node_shards for i in shard if targets[i] == cls)
        rng.shuffle(cls_indices)
        holdout[cls] = cls_indices[:max(1, round(len(cls_indices) * fraction))]
    held = {i for indices in holdout.values() for i in indices}
    return [[i for i in shard if i not in held] for shard in node_shards], holdout


class DataStream:
    def __init__(
        self, path: Path, manifest: dict, node_shards: list[list[int]], batches: list[dict],
        node_crops: list[list[str]] | None = None, early_warning: dict[int, list[int]] | None = None,
    ):
        self.path = Path(path)
        self.manifest = manifest
        self.node_shards = node_shards
        self.batches = batches
        self.node_crops = node_crops or [[] for _ in node_shards]
        # class -> held-out image indices, for the early-warning test only
        self.early_warning = early_warning or {}
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
            farm_crops=cfg.get("data.farm_crops", None),
            manual_node_classes=cfg.get("data.manual_node_classes", None),
        )
        num_nodes, seed = cfg.get("data.num_nodes", 6), cfg.get("data.seed", 42)
        if cfg.get("data.non_iid_strategy", "dirichlet") == "farm_crops":
            node_crops = farm_crop_assignment(
                dataset.labels.crop_classes, num_nodes, cfg.get("data.farm_crops.growers_per_crop", 2), seed,
            )
        else:  # a node grows whatever crops its shard contains
            node_crops = [sorted({_crop_of(dataset, dataset.targets[i]) for i in shard}) for shard in node_shards]
        early_warning = {}
        if cfg.get("data.early_warning.enabled", False):
            node_shards, early_warning = carve_early_warning(
                dataset, node_shards, node_crops, cfg.get("data.early_warning.holdout_fraction", 0.1), seed,
            )
        stream = cls(path, stream_manifest(cfg, dataset), node_shards, [], node_crops, early_warning)
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
        return cls(
            path, data["manifest"], data["node_shards"], data["batches"], data.get("node_crops"),
            {int(c): idx for c, idx in data.get("early_warning", {}).items()},
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "manifest": self.manifest,
            "node_shards": self.node_shards,
            "batches": self.batches,
            "node_crops": self.node_crops,
            "early_warning": {str(c): idx for c, idx in self.early_warning.items()},
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
                 for idx in split["train_idx"] + split["test_idx"] + split.get("unlabeled_idx", [])}
        return sorted(idx for shard in self.node_shards for idx in shard if idx not in used)

    def next_batch(
        self, dataset, size: int, probe_fraction: float, test_fraction: float, seed: int,
        labeled_fraction: float = 1.0,
    ) -> dict:
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
            split = split_node_arrival(
                dataset, arrivals.get(node_i, []), test_fraction, seed=seed + batch_idx,
                labeled_fraction=labeled_fraction,
            )
            nodes[f"node_{node_i}"] = {
                "train_idx": split.train_idx, "test_idx": split.test_idx, "unlabeled_idx": split.unlabeled_idx,
            }
        batch = {
            "batch_idx": batch_idx, "requested_size": size, "size": len(sample),
            "labeled_fraction": labeled_fraction, "probe_idx": probe_idx, "nodes": nodes,
        }
        self.batches.append(batch)
        self.save()
        return batch

    def early_warning_classes(self, dataset, node_id: str) -> list[int]:
        """The node's threat classes that have held-out images."""
        node_i = int(node_id.rsplit("_", 1)[-1])
        return [c for c in threat_classes(dataset, self.node_shards, self.node_crops)[node_i] if c in self.early_warning]

    def early_warning_idx(self, dataset, node_id: str, max_per_class: int | None = None) -> list[int]:
        """The node's early-warning test images: up to `max_per_class`
        held-out images of each of its threat classes (the same every batch).
        """
        return [i for c in self.early_warning_classes(dataset, node_id) for i in self.early_warning[c][:max_per_class]]

    def node_split(self, batch_idx: int, node_id: str) -> BatchSplit:
        split = self.batches[batch_idx]["nodes"][node_id]
        return BatchSplit(split["train_idx"], split["test_idx"], split.get("unlabeled_idx", []))
