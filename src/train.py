"""Entry point: for every configured architecture, trains an
aligned-budget local-only baseline and the decentralised mesh (prototype
+ probe-logit exchange only — no raw images, no weights) on PlantVillage,
then reports the collaboration gain and a compute + communication
sustainability accounting.

Run all architectures in config.yaml's models.architectures list in one go:
    python -m src.train --config config.yaml

Or restrict a single run to one architecture (e.g. to split a full sweep
across several shorter Colab sessions to stay under free-tier GPU usage
limits) — results merge into any existing outputs/results_summary.json
rather than overwriting it, so running each architecture separately still
produces one combined comparison at the end:
    python -m src.train --config config.yaml --arch mobilenet_v3_small
    python -m src.train --config config.yaml --arch efficientnet_lite0
    python -m src.train --config config.yaml --arch mobilevit_xxs

Pass --fresh to start a new sweep instead of merging into a previous one.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from src.config import Config
from src.data.mixing import EvalView, MixedDomainBatchSampler
from src.data.plantdoc import load_plantdoc_dataset
from src.data.plantwild import load_plantwild_dataset
from src.data.plantvillage import (
    carve_global_test_set,
    carve_public_probe_set,
    compute_crop_class_weights,
    compute_crop_class_weights_from_pairs,
    compute_disease_class_weights,
    compute_disease_class_weights_from_pairs,
    load_full_dataset,
    make_subset,
    partition_nodes,
    stratified_carve_by_labels,
    train_test_split_indices,
)
from src.energy.tracker import (
    CommunicationCostEstimator,
    ComputeEnergyTracker,
    sweep_totals_from_emissions_csv,
    write_sustainability_report,
)
from src.evaluate import compute_collaboration_gain
from src.federated.mesh import MeshSimulator
from src.federated.node import Node
from src.models.factory import build_model, count_parameters, model_size_mb
from src.reporting import build_per_class_rows, build_round_log_rows, plot_training_curves, write_csv


_NAMED_DATASET_LOADERS = {
    "plantdoc": ("data.plantdoc_root", load_plantdoc_dataset),
    "plantwild": ("data.plantwild_root", load_plantwild_dataset),
}


def build_dataloaders(cfg: Config, dataset):
    """Dispatches on `data.node_datasets`: unset (default) reproduces
    today's behaviour — non-IID PlantVillage partition + PlantDoc mixed
    into every node. Set means node-exclusive mode: each node trains only
    on the one dataset named for it (see `_build_node_exclusive_dataloaders`).
    """
    node_datasets = cfg.get("data.node_datasets")
    if node_datasets:
        return _build_node_exclusive_dataloaders(cfg, dataset, node_datasets)
    return _build_mixed_dataloaders(cfg, dataset)


def _node_indices_for_dataset(node_datasets: dict[str, str], name: str) -> list[int]:
    """Node indices (e.g. [1] for node_1) assigned to the named dataset, in
    ascending node order.
    """
    return sorted(
        int(str(node_key).rsplit("_", 1)[-1])
        for node_key, assigned in node_datasets.items()
        if assigned == name
    )


def _build_node_exclusive_dataloaders(cfg: Config, dataset, node_datasets: dict[str, str]):
    """Each node trains EXCLUSIVELY on one full named dataset
    ("plantvillage" | "plantdoc" | "plantwild") instead of a partitioned
    PlantVillage shard — no data ever crosses from one node's dataset to
    another's. The probe set and global-test set are carved from ALL
    assigned datasets and unioned, since they're the shared anchor every
    node is evaluated against regardless of what it trained on — a
    PlantVillage-only anchor would only ever test the PlantVillage node's
    home turf. 
    
    Each dataset's own global-test slice is also kept as a
    separate loader (see extra_global_test_loaders below, surfaced as
    "global_plantvillage"/"global_plantdoc"/"global_plantwild") so the
    pooled "global" number — dominated by PlantVillage's much larger sample
    count even after stratified carving — doesn't hide a node generalizing
    poorly to one of the smaller domains.
    """
    valid_names = {"plantvillage", "plantdoc", "plantwild"}
    unknown = sorted(set(node_datasets.values()) - valid_names)
    if unknown:
        raise ValueError(f"data.node_datasets has unknown dataset name(s) {unknown}; must be one of {sorted(valid_names)}")
    expected_keys = {f"node_{i}" for i in range(len(node_datasets))}
    if set(node_datasets.keys()) != expected_keys:
        raise ValueError(
            f"data.node_datasets keys must be exactly {sorted(expected_keys)}, got {sorted(node_datasets.keys())}"
        )

    batch_size = cfg.get("training.batch_size", 32)
    seed = cfg.get("data.seed", 42)
    test_fraction = cfg.get("data.test_fraction", 0.15)
    probe_fraction = cfg.get("data.probe_set_fraction", 0.05)
    global_test_fraction = cfg.get("data.global_test_fraction", 0.05)
    large_class_threshold = cfg.get("data.probe_set_large_class_threshold", 200)
    min_samples_small_class = cfg.get("data.probe_set_min_samples_small_class", 8)
    max_fraction_small_class = cfg.get("data.probe_set_max_fraction_small_class", 0.2)
    crop_balanced = cfg.get("training.crop_class_balanced", False)
    disease_balanced = cfg.get("training.disease_class_balanced", True)

    crop_classes = dataset.labels.crop_classes
    disease_classes = dataset.labels.disease_classes

    # Load every non-PlantVillage dataset referenced in node_datasets, once.
    loaded = {"plantvillage": dataset}
    for name, (root_key, loader_fn) in _NAMED_DATASET_LOADERS.items():
        if name not in node_datasets.values():
            continue
        root = cfg.get(root_key)
        ds = loader_fn(
            root, crop_classes, disease_classes, dataset.labels.class_to_crop_disease,
            cfg.get("data.image_size", 160),
        )
        if ds is None:
            raise ValueError(
                f"data.node_datasets assigns '{name}' to a node, but {root_key} ('{root}') "
                f"is unset or doesn't exist — in node-exclusive mode {name} is that node's "
                f"ONLY data source, so it must already be downloaded."
            )
        loaded[name] = ds
        print(f"Node-exclusive mode: loaded {name} ({len(ds)} images) for {_node_indices_for_dataset(node_datasets, name)}")

    # Carve each dataset's own probe + global-test slice, stratified by its
    # own label space, then union all of them into one shared probe/
    # global-test loader. Each dataset's own global-test subset is also kept
    # separately (see extra_global_test_loaders below) — PlantVillage
    # heavily outnumbers PlantDoc/PlantWild even after each contributes its
    # own stratified slice, so a pooled "global" accuracy alone could hide a
    # node generalizing poorly to a smaller domain behind PlantVillage's
    # much larger sample count.
    probe_subsets = []
    global_test_subsets = []
    global_test_subset_by_name: dict[str, object] = {}
    remaining_by_name: dict[str, list[int]] = {}

    pv_probe_idx, pv_rest = carve_public_probe_set(
        dataset, probe_fraction, seed, large_class_threshold, min_samples_small_class, max_fraction_small_class,
    )
    pv_global_test_idx, pv_rest = carve_global_test_set(
        dataset, pv_rest, global_test_fraction, seed, large_class_threshold, min_samples_small_class, max_fraction_small_class,
    )
    pv_global_test_subset = make_subset(dataset, pv_global_test_idx)
    probe_subsets.append(make_subset(dataset, pv_probe_idx))
    global_test_subsets.append(pv_global_test_subset)
    global_test_subset_by_name["plantvillage"] = pv_global_test_subset
    remaining_by_name["plantvillage"] = pv_rest

    for name in ("plantdoc", "plantwild"):
        if name not in loaded:
            continue
        ds = loaded[name]
        all_idx = list(range(len(ds)))
        probe_idx, rest = stratified_carve_by_labels(
            all_idx, ds.raw_labels, probe_fraction, seed,
            large_class_threshold, min_samples_small_class, max_fraction_small_class,
        )
        rest_labels = [ds.raw_labels[i] for i in rest]
        global_test_idx, rest = stratified_carve_by_labels(
            rest, rest_labels, global_test_fraction, seed,
            large_class_threshold, min_samples_small_class, max_fraction_small_class,
        )
        global_test_subset = EvalView(ds, global_test_idx)
        probe_subsets.append(EvalView(ds, probe_idx))
        global_test_subsets.append(global_test_subset)
        global_test_subset_by_name[name] = global_test_subset
        remaining_by_name[name] = rest

    probe_loader = DataLoader(ConcatDataset(probe_subsets), batch_size=batch_size, shuffle=False)
    global_test_loader = DataLoader(ConcatDataset(global_test_subsets), batch_size=batch_size, shuffle=False)
    extra_global_test_loaders = {
        name: DataLoader(subset, batch_size=batch_size, shuffle=False)
        for name, subset in global_test_subset_by_name.items()
    }

    num_nodes = len(node_datasets)
    node_loaders: list = [None] * num_nodes
    crop_class_weights: list = [None] * num_nodes
    disease_class_weights: list = [None] * num_nodes

    # PlantVillage-assigned node(s): reuse partition_nodes on whatever's left
    # after PlantVillage's own probe/global-test carve — with exactly one
    # PlantVillage-assigned node (the target config), this degenerates
    # cleanly to "all remaining PlantVillage data for that node".
    pv_node_ids = _node_indices_for_dataset(node_datasets, "plantvillage")
    if pv_node_ids:
        pv_shards = partition_nodes(
            dataset, remaining_by_name["plantvillage"], len(pv_node_ids),
            cfg.get("data.non_iid_strategy", "by_crop"), cfg.get("data.dirichlet_alpha", 0.3), seed,
            manual_node_crops=cfg.get("data.manual_node_crops", None),
        )
        for shard, node_id in zip(pv_shards, pv_node_ids):
            train_idx, test_idx = train_test_split_indices(shard, test_fraction, seed)
            train_loader = DataLoader(make_subset(dataset, train_idx, train=True), batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(make_subset(dataset, test_idx), batch_size=batch_size, shuffle=False)
            node_loaders[node_id] = (train_loader, test_loader)
            if crop_balanced:
                crop_class_weights[node_id] = compute_crop_class_weights(dataset, train_idx)
            if disease_balanced:
                disease_class_weights[node_id] = compute_disease_class_weights(dataset, train_idx)

    # PlantDoc/PlantWild-assigned node: exactly one node per dataset is
    # supported (no cross-node splitting logic for these smaller sources —
    # not needed for the node-exclusive experiment this mode is built for).
    for name in ("plantdoc", "plantwild"):
        node_ids = _node_indices_for_dataset(node_datasets, name)
        if not node_ids:
            continue
        if len(node_ids) > 1:
            raise ValueError(
                f"data.node_datasets assigns '{name}' to more than one node "
                f"({[f'node_{i}' for i in node_ids]}) — only one node per plantdoc/plantwild "
                f"dataset is supported"
            )
        node_id = node_ids[0]
        ds = loaded[name]
        rest = remaining_by_name[name]
        train_idx, test_idx = train_test_split_indices(rest, test_fraction, seed)
        train_loader = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(EvalView(ds, test_idx), batch_size=batch_size, shuffle=False)
        node_loaders[node_id] = (train_loader, test_loader)
        pairs = ds.pairs
        train_pairs = [pairs[i] for i in train_idx]
        if crop_balanced:
            crop_class_weights[node_id] = compute_crop_class_weights_from_pairs(train_pairs, len(crop_classes))
        if disease_balanced:
            disease_class_weights[node_id] = compute_disease_class_weights_from_pairs(train_pairs, len(disease_classes))

    return probe_loader, global_test_loader, node_loaders, crop_class_weights, disease_class_weights, extra_global_test_loaders


def _build_mixed_dataloaders(cfg: Config, dataset):
    """PlantVillage non-IID-partitioned across nodes, PlantDoc (if
    configured) mixed into every node's training batches. The probe set and
    global-test set are a stratified union of BOTH datasets whenever PlantDoc
    is in play — not PlantVillage alone — since PlantDoc is genuinely part
    of what every node is trained on, so it should be part of what every
    node is evaluated against too. To keep that union from silently hiding
    per-domain performance behind one pooled number (PlantVillage's ~54k
    images would swamp PlantDoc's ~2.5k in a blended accuracy), the
    PlantVillage-only and PlantDoc-only slices are ALSO kept as separate
    loaders in `extra_global_test_loaders`, surfaced as "global_plantvillage"
    / "global_plantdoc" alongside the pooled "global" (see run_baseline/
    run_mesh and src/evaluate.py's generic secondary-metric handling).
    """
    batch_size = cfg.get("training.batch_size", 32)
    seed = cfg.get("data.seed", 42)
    probe_fraction = cfg.get("data.probe_set_fraction", 0.05)
    global_test_fraction = cfg.get("data.global_test_fraction", 0.05)
    large_class_threshold = cfg.get("data.probe_set_large_class_threshold", 200)
    min_samples_small_class = cfg.get("data.probe_set_min_samples_small_class", 8)
    max_fraction_small_class = cfg.get("data.probe_set_max_fraction_small_class", 0.2)

    plantdoc_dataset = load_plantdoc_dataset(
        cfg.get("data.plantdoc_root"),
        dataset.labels.crop_classes,
        dataset.labels.disease_classes,
        dataset.labels.class_to_crop_disease,
        cfg.get("data.image_size", 160),
    )
    plantdoc_train_dataset = None
    plantdoc_train_labels = None
    plantdoc_probe_subset = None
    plantdoc_global_test_subset = None
    extra_global_test_loaders: dict[str, DataLoader] = {}
    if plantdoc_dataset is not None:
        # Same two-step carve as PlantVillage below (probe first, then
        # global-test from what's left), stratified by PlantDoc's own raw
        # folder label, BEFORE any of it becomes eligible for
        # training-mixing — otherwise these evaluation sets would just be
        # measuring memorization of the model's own training images.
        plantdoc_probe_idx, plantdoc_rest = stratified_carve_by_labels(
            list(range(len(plantdoc_dataset))), plantdoc_dataset.raw_labels, probe_fraction, seed,
            large_class_threshold, min_samples_small_class, max_fraction_small_class,
        )
        plantdoc_rest_labels = [plantdoc_dataset.raw_labels[i] for i in plantdoc_rest]
        plantdoc_global_test_idx, plantdoc_train_idx = stratified_carve_by_labels(
            plantdoc_rest, plantdoc_rest_labels, global_test_fraction, seed,
            large_class_threshold, min_samples_small_class, max_fraction_small_class,
        )
        plantdoc_probe_subset = EvalView(plantdoc_dataset, plantdoc_probe_idx)
        plantdoc_global_test_subset = EvalView(plantdoc_dataset, plantdoc_global_test_idx)
        extra_global_test_loaders["plantdoc"] = DataLoader(
            plantdoc_global_test_subset, batch_size=batch_size, shuffle=False,
        )
        plantdoc_train_dataset = Subset(plantdoc_dataset, plantdoc_train_idx)
        plantdoc_train_labels = [plantdoc_dataset.raw_labels[i] for i in plantdoc_train_idx]
        print(
            f"Loaded PlantDoc: {len(plantdoc_dataset)} images across "
            f"{len(set(plantdoc_dataset.raw_labels))} classes — {len(plantdoc_probe_idx)} in the shared probe "
            f"set, {len(plantdoc_global_test_idx)} in the shared global-test set, {len(plantdoc_train_idx)} "
            f"mixed at {cfg.get('data.plantvillage_batch_fraction', 0.85):.0%} PlantVillage / "
            f"{1 - cfg.get('data.plantvillage_batch_fraction', 0.85):.0%} PlantDoc per training batch"
        )

    probe_idx, remaining_idx = carve_public_probe_set(
        dataset, probe_fraction, seed, large_class_threshold, min_samples_small_class, max_fraction_small_class,
    )
    # measure of whether a node's model actually generalizes.
    global_test_idx, remaining_idx = carve_global_test_set(
        dataset, remaining_idx, global_test_fraction, seed,
        large_class_threshold, min_samples_small_class, max_fraction_small_class,
    )
    shards = partition_nodes(
        dataset,
        remaining_idx,
        cfg.get("data.num_nodes", 3),
        cfg.get("data.non_iid_strategy", "by_crop"),
        cfg.get("data.dirichlet_alpha", 0.3),
        seed,
        manual_node_crops=cfg.get("data.manual_node_crops", None),
    )
    pv_probe_subset = make_subset(dataset, probe_idx)
    pv_global_test_subset = make_subset(dataset, global_test_idx)
    if plantdoc_dataset is not None:
        extra_global_test_loaders["plantvillage"] = DataLoader(
            pv_global_test_subset, batch_size=batch_size, shuffle=False,
        )
        probe_loader = DataLoader(
            ConcatDataset([pv_probe_subset, plantdoc_probe_subset]), batch_size=batch_size, shuffle=False,
        )
        global_test_loader = DataLoader(
            ConcatDataset([pv_global_test_subset, plantdoc_global_test_subset]), batch_size=batch_size, shuffle=False,
        )
    else:
        probe_loader = DataLoader(pv_probe_subset, batch_size=batch_size, shuffle=False)
        global_test_loader = DataLoader(pv_global_test_subset, batch_size=batch_size, shuffle=False)

    node_loaders = []
    crop_class_weights = []
    disease_class_weights = []
    for node_id, shard in enumerate(shards):
        train_idx, test_idx = train_test_split_indices(
            shard, cfg.get("data.test_fraction", 0.15), cfg.get("data.seed", 42)
        )
        train_subset = make_subset(dataset, train_idx, train=True)
        if plantdoc_train_dataset is not None:
            train_loader = DataLoader(
                ConcatDataset([train_subset, plantdoc_train_dataset]),
                batch_sampler=MixedDomainBatchSampler(
                    primary_len=len(train_subset),
                    secondary_len=len(plantdoc_train_dataset),
                    secondary_labels=plantdoc_train_labels,
                    batch_size=batch_size,
                    primary_fraction=cfg.get("data.plantvillage_batch_fraction", 0.85),
                    seed=seed + node_id,
                ),
            )
        else:
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(make_subset(dataset, test_idx), batch_size=batch_size, shuffle=False)
        node_loaders.append((train_loader, test_loader))
        crop_class_weights.append(
            compute_crop_class_weights(dataset, train_idx)
            if cfg.get("training.crop_class_balanced", False)
            else None
        )
        disease_class_weights.append(
            compute_disease_class_weights(dataset, train_idx)
            if cfg.get("training.disease_class_balanced", True)
            else None
        )
    return probe_loader, global_test_loader, node_loaders, crop_class_weights, disease_class_weights, extra_global_test_loaders


def run_baseline(
    cfg, arch, node_loaders, global_test_loader, crop_classes, disease_classes, tracker, device,
    crop_class_weights=None, disease_class_weights=None, extra_global_test_loaders=None,
):
    """Local-only training, no exchange at all — the comparison point
    the collaboration gain is measured against.
    """
    crop_class_weights = crop_class_weights or [None] * len(node_loaders)
    disease_class_weights = disease_class_weights or [None] * len(node_loaders)
    extra_global_test_loaders = extra_global_test_loaders or {}
    evals = {}
    for i, (train_loader, test_loader) in enumerate(node_loaders):
        node_id = f"node_{i}"
        model = build_model(
            arch, len(crop_classes), len(disease_classes), pretrained=cfg.get("models.pretrained", True)
        )
        node = Node(
            node_id, model, train_loader, test_loader, device=device,
            crop_classes=crop_classes, disease_classes=disease_classes,
            crop_loss_weight=cfg.get("training.crop_loss_weight", 1.0),
            disease_loss_weight=cfg.get("training.disease_loss_weight", 1.0),
            crop_class_weights=crop_class_weights[i],
            disease_class_weights=disease_class_weights[i],
            loss_type=cfg.get("training.loss_type", "cross_entropy"),
            focal_gamma=cfg.get("training.focal_gamma", 2.0),
        )
        with tracker.track(f"{arch}_baseline_{node_id}"):
            node.local_train(cfg.get("training.baseline_epochs", 10), cfg.get("training.lr", 0.001))
        # top-level metrics: this node's own (skewed) local test split.
        # "global": the shared global_test_loader — in mixed mode with
        # PlantDoc configured, a stratified union of PlantVillage + PlantDoc
        # (see _build_mixed_dataloaders); PlantVillage-only otherwise.
        node_eval = node.evaluate()
        node_eval["global"] = node.evaluate(global_test_loader)
        # per-source breakdown (e.g. "global_plantdoc", "global_plantvillage")
        # so the pooled "global" number above never hides one domain's
        # performance behind another's much larger sample count — this is
        # the number that actually answers whether PlantDoc mixing fixed the
        # domain-shift problem it was added for.
        for name, loader in extra_global_test_loaders.items():
            node_eval[f"global_{name}"] = node.evaluate(loader)
        evals[node_id] = node_eval
    return evals


def run_mesh(
    cfg, arch, node_loaders, probe_loader, global_test_loader, crop_classes, disease_classes, tracker, device,
    output_dir, crop_class_weights=None, disease_class_weights=None, extra_global_test_loaders=None,
):
    crop_class_weights = crop_class_weights or [None] * len(node_loaders)
    extra_global_test_loaders = extra_global_test_loaders or {}
    disease_class_weights = disease_class_weights or [None] * len(node_loaders)
    nodes = []
    for i, (train_loader, test_loader) in enumerate(node_loaders):
        model = build_model(
            arch, len(crop_classes), len(disease_classes), pretrained=cfg.get("models.pretrained", True)
        )
        nodes.append(Node(
            f"node_{i}", model, train_loader, test_loader, device=device,
            crop_classes=crop_classes, disease_classes=disease_classes,
            crop_loss_weight=cfg.get("training.crop_loss_weight", 1.0),
            disease_loss_weight=cfg.get("training.disease_loss_weight", 1.0),
            crop_class_weights=crop_class_weights[i],
            disease_class_weights=disease_class_weights[i],
            loss_type=cfg.get("training.loss_type", "cross_entropy"),
            focal_gamma=cfg.get("training.focal_gamma", 2.0),
        ))

    mesh = MeshSimulator(
        nodes,
        probe_loader,
        aggregation_method=cfg.get("federated.aggregation", "trimmed_mean"),
        trim_fraction=cfg.get("federated.trim_fraction", 0.2),
        krum_neighbors=cfg.get("federated.krum_neighbors", 2),
    )

    total_bytes = 0
    round_logs = []
    for r in range(cfg.get("training.rounds", 5)):
        with tracker.track(f"{arch}_mesh_round_{r}"):
            local_epochs = cfg.get(
                f"training.local_epochs_per_round_overrides.{arch}",
                cfg.get("training.local_epochs_per_round", 2),
            )
            round_log = mesh.run_round(
                r,
                local_epochs=local_epochs,
                distill_epochs=cfg.get("training.distill_epochs_per_round", 1),
                lr=cfg.get("training.lr", 0.001),
                distill_lr=cfg.get("training.distill_lr", 0.0005),
                proto_weight=cfg.get("training.proto_weight", 0.5),
                kd_weight=cfg.get("training.kd_weight", 0.5),
                temperature=cfg.get("training.kd_temperature", 2.0),
            )
        total_bytes += round_log.total_bytes_exchanged
        round_logs.append(round_log)
        print(f"  round {r}: {round_log.total_bytes_exchanged} bytes exchanged")

    round_log_dicts = [dataclasses.asdict(rl) for rl in round_logs]
    trend_rows = build_round_log_rows(round_log_dicts)
    per_class_rows = build_per_class_rows(
        round_log_dicts, [("pre", "pre_distill_eval"), ("post", "per_node_eval")]
    )
    (output_dir / f"round_logs_{arch}.json").write_text(
        json.dumps({"rounds": round_log_dicts, "trend": trend_rows, "per_class_trend": per_class_rows}, indent=2)
    )
    if cfg.get("output.save_plots", True):
        write_csv(trend_rows, output_dir / f"trend_{arch}.csv")
        write_csv(per_class_rows, output_dir / f"trend_{arch}_per_class.csv")
        plot_training_curves(
            trend_rows, output_dir / "plots" / f"{arch}_mesh_training_curves.png",
            f"{arch}: mesh training curves",
        )

    final_evals = {}
    for node in nodes:
        node_eval = node.evaluate()
        node_eval["global"] = node.evaluate(global_test_loader)
        for name, loader in extra_global_test_loaders.items():
            node_eval[f"global_{name}"] = node.evaluate(loader)
        final_evals[node.node_id] = node_eval

    checkpoint_dir = output_dir / "checkpoints" / arch
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for node in nodes:
        torch.save(node.model.state_dict(), checkpoint_dir / f"{node.node_id}.pt")

    return final_evals, total_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: repo root)")
    parser.add_argument(
        "--arch",
        default=None,
        help="Restrict this run to a single architecture, overriding config.yaml's models.architectures list",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Discard any existing results_summary.json / run_state.json instead of merging into them",
    )
    args = parser.parse_args()
    cfg = Config.load(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(cfg.get("output.dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_full_dataset(cfg.get("data.root"), cfg.get("data.image_size", 160))
    num_crop = len(dataset.labels.crop_classes)
    num_disease = len(dataset.labels.disease_classes)
    print(f"Loaded {len(dataset)} images, {num_crop} crop classes, {num_disease} disease classes.")

    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    (checkpoints_dir / "classes.json").write_text(
        json.dumps(
            {
                "crop_classes": dataset.labels.crop_classes,
                "disease_classes": dataset.labels.disease_classes,
                "image_size": cfg.get("data.image_size", 160),
            },
            indent=2,
        )
    )

    probe_loader, global_test_loader, node_loaders, crop_class_weights, disease_class_weights, extra_global_test_loaders = build_dataloaders(cfg, dataset)

    tracker = ComputeEnergyTracker(
        enabled=cfg.get("energy.track_with_codecarbon", True),
        output_dir=output_dir,
        country_iso_code=cfg.get("energy.country_iso_code", "GBR"),
        fallback_power_watts=cfg.get("energy.fallback_power_watts", 15.0),
    )
    comm_estimator = CommunicationCostEstimator(
        cfg.get("energy.radio_energy_j_per_byte", {}),
        cfg.get("energy.grid_carbon_intensity_gco2_per_kwh", 125),
    )

    results_summary_path = output_dir / "results_summary.json"
    run_state_path = output_dir / "run_state.json"
    if args.fresh:
        results_summary_path.unlink(missing_ok=True)
        run_state_path.unlink(missing_ok=True)

    all_results = json.loads(results_summary_path.read_text()) if results_summary_path.exists() else {}
    run_state = (
        json.loads(run_state_path.read_text())
        if run_state_path.exists()
        else {"total_compute_energy_kwh": 0.0, "total_duration_s": 0.0, "num_tracked_blocks": 0, "total_bytes_exchanged": 0}
    )
    grand_total_bytes = 0

    architectures = [args.arch] if args.arch else cfg.get("models.architectures", [])
    for arch in architectures:
        print(f"\n=== Architecture: {arch} ===")
        print("-- baseline (local-only) --")
        baseline_evals = run_baseline(
            cfg, arch, node_loaders, global_test_loader,
            dataset.labels.crop_classes, dataset.labels.disease_classes, tracker, device,
            crop_class_weights=crop_class_weights, disease_class_weights=disease_class_weights,
            extra_global_test_loaders=extra_global_test_loaders,
        )
        print("-- mesh (prototype + logit exchange) --")
        mesh_evals, total_bytes = run_mesh(
            cfg, arch, node_loaders, probe_loader, global_test_loader,
            dataset.labels.crop_classes, dataset.labels.disease_classes, tracker, device, output_dir,
            crop_class_weights=crop_class_weights, disease_class_weights=disease_class_weights,
            extra_global_test_loaders=extra_global_test_loaders,
        )
        grand_total_bytes += total_bytes
        gain = compute_collaboration_gain(mesh_evals, baseline_evals)

        sample_model = build_model(arch, num_crop, num_disease, pretrained=False)
        arch_result = {
            "architecture": arch,
            "params": count_parameters(sample_model),
            "model_size_mb": model_size_mb(sample_model),
            "baseline_eval": baseline_evals,
            "mesh_eval": mesh_evals,
            "collaboration_gain": gain,
            "total_bytes_exchanged": total_bytes,
        }
        all_results[arch] = arch_result
        (output_dir / f"results_{arch}.json").write_text(json.dumps(arch_result, indent=2))
        print(f"macro_gain: {gain['macro_gain']}")
        print(f"worst_node_gain: {gain['worst_node_gain']}")
        # Prints whichever "global"/"global_<source>" gain keys are present
        # (see compute_collaboration_gain's generic secondary-metric scan) —
        # e.g. just "global" in node-exclusive mode, or "global",
        # "global_plantvillage", "global_plantdoc" together in mixed mode
        # with PlantDoc configured.
        for key in sorted(gain):
            if key.startswith("global") and key.endswith("_macro_gain"):
                worst_key = key.replace("_macro_gain", "_worst_node_gain")
                print(f"{key}: {gain[key]}")
                print(f"{worst_key}: {gain[worst_key]}")

    results_summary_path.write_text(json.dumps(all_results, indent=2))

    # Accumulate this run's compute energy/bytes on top of any prior separate
    # run(s) (e.g. one Colab session per architecture), so the sustainability
    # report reflects the whole sweep rather than just the architecture(s)
    # trained in this particular invocation.
    this_run_compute = tracker.summary()
    run_state["total_compute_energy_kwh"] += this_run_compute["total_compute_energy_kwh"]
    run_state["total_duration_s"] += this_run_compute["total_duration_s"]
    run_state["num_tracked_blocks"] += this_run_compute["num_tracked_blocks"]
    run_state["total_bytes_exchanged"] += grand_total_bytes
    run_state_path.write_text(json.dumps(run_state, indent=2))

    comm_estimate = comm_estimator.estimate_all_radios(run_state["total_bytes_exchanged"])
    overall_gain = {arch: res["collaboration_gain"]["macro_gain"] for arch, res in all_results.items()}
    write_sustainability_report(
        output_dir / "sustainability_report",
        {k: v for k, v in run_state.items() if k != "total_bytes_exchanged"},
        comm_estimate,
        overall_gain,
        cfg.get("energy.grid_carbon_intensity_gco2_per_kwh", 125),
        emissions_csv_totals=sweep_totals_from_emissions_csv(output_dir),
    )
    print(f"\nDone. Results and sustainability report written to {output_dir}/ "
          f"(now covering {len(all_results)} architecture(s): {list(all_results.keys())})")


if __name__ == "__main__":
    main()
