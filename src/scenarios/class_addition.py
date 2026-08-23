"""Scenario: a crop that's currently only grown elsewhere in the mesh
starts appearing at one node partway through the run (e.g. crop
rotation). Demonstrates that the mesh helps that node learn the new class
faster than training on it alone (the baseline), because peers who
already know the crop contribute it to the shared knowledge consensus.

Run: python -m src.scenarios.class_addition [--config path] [--arch name]
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

from src.config import Config
from src.data.merged import load_merged_dataset
from src.data.plantvillage import make_subset, train_test_split_indices
from src.federated.mesh import MeshSimulator
from src.scenarios.harness import (
    ScenarioEvent,
    build_node_set,
    require_target_node,
    run_scenario,
    write_scenario_report,
)
from src.train import build_dataloaders


def carve_reserve_pool(dataset, source_train_indices, source_crop, reserve_fraction, seed, test_fraction):
    """Removes a `reserve_fraction` slice of `source_crop` samples from
    `source_train_indices`, returning (remaining_source_indices,
    reserve_train_indices, reserve_test_indices). The reserve is a subset
    of what the source node already owned — nothing is duplicated across
    nodes.
    """
    crop_idx = dataset.labels.crop_classes.index(source_crop)
    matching = [
        idx for idx in source_train_indices
        if dataset.labels.class_to_crop_disease[dataset.targets[idx]][0] == crop_idx
    ]
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


def _node_crop_counts(dataset, node_loaders, source_crop: str) -> dict[str, int]:
    crop_idx = dataset.labels.crop_classes.index(source_crop)
    counts = {}
    for i, (train_loader, _) in enumerate(node_loaders):
        indices = train_loader.dataset.indices
        counts[f"node_{i}"] = sum(
            1 for idx in indices
            if dataset.labels.class_to_crop_disease[dataset.targets[idx]][0] == crop_idx
        )
    return counts


def find_source_and_target_nodes(counts: dict[str, int]) -> tuple[str, str]:
    """Picks the DONOR node (most local examples of the crop, to carve the
    injected reserve pool from) and the TARGET node (fewest, to receive
    them) purely from each node's actual local data -- no dependency on
    data.manual_node_crops, so this works whether non_iid_strategy is
    "manual" (where the target's count is genuinely zero) or "dirichlet"
    (where every node has some non-zero exposure, so "fewest" is the
    closest analogue to "hasn't really grown this crop yet").
    """
    source_node = max(counts, key=counts.get)
    target_node = min(counts, key=counts.get)
    return source_node, target_node


def make_class_addition_hook(
    target_node: str, source_crop: str, inject_round: int, reserve_train_idx, reserve_test_idx, dataset, batch_size: int
):
    def hook(round_idx, nodes, mesh):
        if round_idx != inject_round:
            return []
        target = next(n for n in nodes if n.node_id == target_node)
        target.train_loader = DataLoader(
            ConcatDataset([target.train_loader.dataset, make_subset(dataset, reserve_train_idx, train=True)]),
            batch_size=batch_size, shuffle=True,
        )
        target.test_loader = DataLoader(
            ConcatDataset([target.test_loader.dataset, make_subset(dataset, reserve_test_idx)]),
            batch_size=batch_size, shuffle=False,
        )
        if mesh is not None:
            return [ScenarioEvent(
                round_idx, "class_added", target_node,
                {"crop": source_crop, "n_injected": len(reserve_train_idx) + len(reserve_test_idx)},
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
    source_crop = scfg["source_crop"]
    inject_round = scfg["inject_round"]
    reserve_fraction = scfg["reserve_fraction"]
    num_rounds = cfg.get("scenarios.rounds", 6)
    seed = cfg.get("data.seed", 42)

    if not (0 <= inject_round < num_rounds):
        raise ValueError(f"scenarios.class_addition.inject_round ({inject_round}) must be in [0, {num_rounds})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(cfg.get("output.dir", "outputs"))
    dataset = load_merged_dataset(
        cfg.get("data.root"), cfg.get("data.plantdoc_root"), cfg.get("data.image_size", 224), cfg.get("data.seed", 42)
    )
    if source_crop not in dataset.labels.crop_classes:
        raise ValueError(f"source_crop '{source_crop}' is not a known crop: {dataset.labels.crop_classes}")

    probe_loader, _global_test_loader, node_loaders, _crop_class_weights, _disease_class_weights = build_dataloaders(cfg, dataset)
    arch = args.arch or cfg.get("models.architectures", ["mobilenet_v3_small"])[0]
    batch_size = cfg.get("training.batch_size", 32)

    # Under non_iid_strategy == "manual", data.manual_node_crops assigns a
    # crop to exactly one node, so that node's count for every other crop
    # is exactly 0 and this always reproduces the same source/target pair
    # the config would have named explicitly. Under "dirichlet", no node
    # owns a crop exclusively, so we pick the donor (most local examples)
    # and target (fewest) purely from the actual per-node data instead of
    # trusting a fixed config-file assignment that no longer holds.
    counts = _node_crop_counts(dataset, node_loaders, source_crop)
    source_node, target_node = find_source_and_target_nodes(counts)
    print(
        f"class_addition: source_crop='{source_crop}' per-node local counts={counts} "
        f"-> donor={source_node} ({counts[source_node]} examples), target={target_node} ({counts[target_node]} examples)"
    )
    if source_node == target_node:
        raise ValueError(
            f"source_crop '{source_crop}': every node has the same local count ({counts[source_node]}) — "
            "no node stands out as a donor vs. a target, pick a different crop"
        )

    require_target_node(target_node, node_loaders)
    require_target_node(source_node, node_loaders)

    source_idx = int(source_node.rsplit("_", 1)[-1])
    source_train_loader, source_test_loader = node_loaders[source_idx]
    remaining_source, reserve_train_idx, reserve_test_idx = carve_reserve_pool(
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
    )

    hook = make_class_addition_hook(
        target_node, source_crop, inject_round, reserve_train_idx, reserve_test_idx, dataset, batch_size
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
    records = run_scenario(baseline_nodes, mesh, num_rounds, hook, round_kwargs)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = write_scenario_report(
        output_dir, "class_addition", target_node,
        disruption_start_round=inject_round, disruption_end_round=inject_round,
        config_snapshot=scfg, records=records, save_plots=cfg.get("output.save_plots", True),
    )
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
