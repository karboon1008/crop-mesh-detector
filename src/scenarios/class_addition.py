"""Scenario: a crop that no node at one farm currently grows starts
appearing there partway through the run (e.g. crop rotation).
Demonstrates that the mesh helps that node learn the new class faster
than training on it alone (the baseline), because peers who already know
the crop contribute it to the shared knowledge consensus.

Two injection mechanisms exist, because which one is physically coherent
depends on how the mesh's shards were partitioned:

- `target_withheld` — every `source_crop` sample in the target node's own
  shard is withheld before round 0 and spliced back in at
  `inject_round`. The target has genuinely never seen the crop until then,
  and when it does, the images are its own. This is the mechanism used
  under an overlapping partition (`dirichlet`), where the target node
  would otherwise already hold examples of every crop and there would be
  no new class to add.
- `peer_reserve` — a `reserve_fraction` slice of the *source* node's
  `source_crop` samples is carved out before round 0 and spliced into the
  target at `inject_round`. Used under a disjoint partition (`manual`,
  `by_crop`), where the target owns none of that crop and so has nothing
  of its own to withhold.

`scenarios.class_addition.injection_source: auto` (the default) picks
`target_withheld` when the target owns any of the crop and
`peer_reserve` otherwise; either can be forced explicitly. Whichever is
used is recorded in the report's `provenance` block.

Run: python -m src.scenarios.class_addition [--config path] [--arch name]
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

from src.config import Config
from src.data.multi_source import load_dataset
from src.data.plantvillage import make_subset, train_test_split_indices
from src.data.splits import build_dataloaders
from src.federated.mesh import MeshSimulator
from src.scenarios.harness import (
    ScenarioEvent,
    build_energy_accounting,
    build_node_set,
    build_provenance,
    require_target_node,
    run_scenario,
    write_scenario_report,
)

INJECTION_SOURCES = {"auto", "target_withheld", "peer_reserve"}


def indices_of_crop(dataset, indices, source_crop: str) -> list[int]:
    """The subset of `indices` whose crop label is `source_crop`."""
    crop_idx = dataset.labels.crop_classes.index(source_crop)
    return [
        idx for idx in indices
        if dataset.labels.class_to_crop_disease[dataset.targets[idx]][0] == crop_idx
    ]


def crop_counts_per_node(dataset, node_loaders, source_crop: str) -> dict[str, int]:
    """How many `source_crop` training samples each node's shard holds —
    the ownership map every strategy choice can be read off, without
    depending on `data.manual_node_crops` (which only exists under the
    "manual" partition strategy).
    """
    return {
        f"node_{i}": len(indices_of_crop(dataset, train_loader.dataset.indices, source_crop))
        for i, (train_loader, _) in enumerate(node_loaders)
    }


def carve_reserve_pool(dataset, source_train_indices, source_crop, reserve_fraction, seed, test_fraction):
    """Removes a `reserve_fraction` slice of `source_crop` samples from
    `source_train_indices`, returning (remaining_source_indices,
    reserve_train_indices, reserve_test_indices). The reserve is a subset
    of what the source node already owned — nothing is duplicated across
    nodes.
    """
    matching = indices_of_crop(dataset, source_train_indices, source_crop)
    if not matching:
        raise ValueError(
            f"carve_reserve_pool: source_crop '{source_crop}' has zero matching training "
            f"samples in the source node's train split — cannot carve a reserve pool"
        )
    rng = random.Random(seed)
    shuffled = matching.copy()
    rng.shuffle(shuffled)
    n_reserve = max(1, int(len(shuffled) * reserve_fraction))
    reserve = set(shuffled[:n_reserve])
    remaining_source = [idx for idx in source_train_indices if idx not in reserve]
    reserve_train_idx, reserve_test_idx = train_test_split_indices(dataset, list(reserve), test_fraction, seed)
    return remaining_source, reserve_train_idx, reserve_test_idx


def withhold_crop_from_node(dataset, train_indices, test_indices, source_crop: str):
    """Strips every `source_crop` sample out of one node's train and test
    splits, returning (remaining_train, remaining_test, withheld_train,
    withheld_test). The withheld samples are what `inject_round` splices
    back in, so the node genuinely has neither training signal nor test
    coverage for that crop until the injection happens.
    """
    withheld_train = set(indices_of_crop(dataset, train_indices, source_crop))
    withheld_test = set(indices_of_crop(dataset, test_indices, source_crop))
    remaining_train = [idx for idx in train_indices if idx not in withheld_train]
    remaining_test = [idx for idx in test_indices if idx not in withheld_test]
    return remaining_train, remaining_test, sorted(withheld_train), sorted(withheld_test)


def find_source_node(crop_counts: dict[str, int], source_crop: str, exclude_node: str | None = None) -> str:
    """The peer holding the most `source_crop` training samples — the node
    whose existing knowledge of the crop is what the mesh has to teach the
    target. `exclude_node` keeps the target itself out of the running, so
    the scenario always has a genuine teacher.
    """
    candidates = {node_id: n for node_id, n in crop_counts.items() if n > 0 and node_id != exclude_node}
    if not candidates:
        raise ValueError(
            f"source_crop '{source_crop}' is not held by any node other than '{exclude_node}' under the "
            f"current partition (per-node counts: {crop_counts}) — the mesh would have nothing to teach, "
            f"so pick a different source_crop or a different target_node"
        )
    return max(candidates, key=lambda node_id: candidates[node_id])


def make_class_addition_hook(
    target_node: str, source_crop: str, inject_round: int, inject_train_idx, inject_test_idx, dataset,
    batch_size: int, mechanism: str = "peer_reserve",
):
    def hook(round_idx, nodes, mesh):
        if round_idx != inject_round:
            return []
        target = next(n for n in nodes if n.node_id == target_node)
        target.train_loader = DataLoader(
            ConcatDataset([target.train_loader.dataset, make_subset(dataset, inject_train_idx, train=True)]),
            batch_size=batch_size, shuffle=True,
        )
        target.test_loader = DataLoader(
            ConcatDataset([target.test_loader.dataset, make_subset(dataset, inject_test_idx)]),
            batch_size=batch_size, shuffle=False,
        )
        if mesh is not None:
            return [ScenarioEvent(
                round_idx, "class_added", target_node,
                {
                    "crop": source_crop,
                    "mechanism": mechanism,
                    "n_injected_train": len(inject_train_idx),
                    "n_injected_test": len(inject_test_idx),
                    "n_injected": len(inject_train_idx) + len(inject_test_idx),
                },
            )]
        return []

    return hook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--arch", default=None)
    args = parser.parse_args()
    cfg = Config.load(args.config)

    scfg = cfg.get("scenarios.class_addition")
    if scfg is None:
        raise ValueError("config.yaml is missing a scenarios.class_addition section")
    target_node = scfg["target_node"]
    source_crop = scfg["source_crop"]
    inject_round = scfg["inject_round"]
    reserve_fraction = scfg["reserve_fraction"]
    injection_source = scfg.get("injection_source", "auto")
    num_rounds = cfg.get("scenarios.rounds", 6)
    seed = cfg.get("data.seed", 42)

    if not (0 <= inject_round < num_rounds):
        raise ValueError(f"scenarios.class_addition.inject_round ({inject_round}) must be in [0, {num_rounds})")
    if injection_source not in INJECTION_SOURCES:
        raise ValueError(
            f"scenarios.class_addition.injection_source '{injection_source}' is not supported; "
            f"must be one of {sorted(INJECTION_SOURCES)}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(cfg.get("output.dir", "outputs"))
    dataset = load_dataset(cfg)
    if source_crop not in dataset.labels.crop_classes:
        raise ValueError(f"source_crop '{source_crop}' is not a known crop: {dataset.labels.crop_classes}")

    probe_loader, node_loaders, _crop_class_weights, _disease_class_weights = build_dataloaders(cfg, dataset)
    arch = args.arch or cfg.get("models.architectures", ["mobilenet_v3_small"])[0]
    batch_size = cfg.get("training.batch_size", 32)

    require_target_node(target_node, node_loaders)

    crop_counts = crop_counts_per_node(dataset, node_loaders, source_crop)
    source_node = find_source_node(crop_counts, source_crop, exclude_node=target_node)
    mechanism = injection_source
    if mechanism == "auto":
        mechanism = "target_withheld" if crop_counts.get(target_node, 0) > 0 else "peer_reserve"
    print(f"'{source_crop}' train samples per node: {crop_counts}")
    print(f"Injection mechanism: {mechanism} (teacher peer: {source_node})")

    if mechanism == "target_withheld":
        target_idx = int(target_node.rsplit("_", 1)[-1])
        target_train_loader, target_test_loader = node_loaders[target_idx]
        remaining_train, remaining_test, inject_train_idx, inject_test_idx = withhold_crop_from_node(
            dataset, target_train_loader.dataset.indices, target_test_loader.dataset.indices, source_crop,
        )
        if not inject_train_idx:
            raise ValueError(
                f"injection_source='target_withheld' but target_node '{target_node}' holds no "
                f"'{source_crop}' training samples to withhold (per-node counts: {crop_counts}) — "
                f"use injection_source='peer_reserve' for a disjoint partition"
            )
        node_loaders[target_idx] = (
            DataLoader(make_subset(dataset, remaining_train, train=True), batch_size=batch_size, shuffle=True),
            DataLoader(make_subset(dataset, remaining_test), batch_size=batch_size, shuffle=False),
        )
    else:
        require_target_node(source_node, node_loaders)
        source_idx = int(source_node.rsplit("_", 1)[-1])
        source_train_loader, source_test_loader = node_loaders[source_idx]
        remaining_source, inject_train_idx, inject_test_idx = carve_reserve_pool(
            dataset, source_train_loader.dataset.indices, source_crop, reserve_fraction, seed,
            cfg.get("data.test_fraction", 0.15),
        )
        node_loaders[source_idx] = (
            DataLoader(make_subset(dataset, remaining_source, train=True), batch_size=batch_size, shuffle=True),
            source_test_loader,
        )

    baseline_nodes = build_node_set(
        cfg, arch, node_loaders, dataset.labels.crop_classes, dataset.labels.disease_classes, device,
        pair_class_names=dataset.base.classes, class_to_crop_disease=dataset.labels.class_to_crop_disease,
    )
    mesh_nodes = build_node_set(
        cfg, arch, node_loaders, dataset.labels.crop_classes, dataset.labels.disease_classes, device,
        pair_class_names=dataset.base.classes, class_to_crop_disease=dataset.labels.class_to_crop_disease,
    )
    mesh = MeshSimulator(
        mesh_nodes,
        probe_loader,
        aggregation_method=cfg.get("federated.aggregation", "trimmed_mean"),
        trim_fraction=cfg.get("federated.trim_fraction", 0.2),
        krum_neighbors=cfg.get("federated.krum_neighbors", 2),
        combined_training=cfg.get("training.combined_training", False),
    )

    tracker, comm_estimator = build_energy_accounting(cfg, output_dir)
    hook = make_class_addition_hook(
        target_node, source_crop, inject_round, inject_train_idx, inject_test_idx, dataset, batch_size,
        mechanism=mechanism,
    )
    round_kwargs = {
        "local_epochs": cfg.get("training.local_epochs_per_round", 2),
        "distill_epochs": cfg.get("training.distill_epochs_per_round", 1),
        "lr": cfg.get("training.lr", 0.001),
        "distill_lr": cfg.get("training.distill_lr", 0.0005),
        "proto_weight": cfg.get("training.proto_weight", 0.5),
        "kd_weight": cfg.get("training.kd_weight", 0.5),
        "crop_kd_weight": cfg.get("training.crop_kd_weight", None),
        "temperature": cfg.get("training.kd_temperature", 2.0),
    }
    records = run_scenario(
        baseline_nodes, mesh, num_rounds, hook, round_kwargs,
        tracker=tracker, comm_estimator=comm_estimator,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = write_scenario_report(
        output_dir, "class_addition", target_node,
        disruption_start_round=inject_round, disruption_end_round=inject_round,
        config_snapshot=scfg, records=records, save_plots=cfg.get("output.save_plots", True),
        provenance=build_provenance(
            cfg, arch, num_rounds, node_loaders, tracker,
            injection_mechanism=mechanism, teacher_peer=source_node,
            source_crop_counts_per_node=crop_counts,
            n_injected_train=len(inject_train_idx), n_injected_test=len(inject_test_idx),
        ),
    )
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
