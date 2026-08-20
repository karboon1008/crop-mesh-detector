"""Entry point: for every configured architecture, trains an
aligned-budget local-only baseline and the decentralised mesh (prototype
+ probe-logit exchange only — no raw images, no weights) on the merged
PlantVillage+PlantDoc project dataset (see src/data/merged.py), then
reports the collaboration gain and a compute + communication
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
import statistics
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data.merged import load_merged_dataset
from src.data.plantvillage import (
    carve_global_test_set,
    carve_public_probe_set,
    compute_crop_class_weights,
    compute_disease_class_weights,
    make_subset,
    partition_nodes,
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


def scale_epochs_by_node_size(base_epochs: int, node_loaders, max_epochs: int) -> list[int]:
    """Per-node epoch count so a node with a much smaller local shard still
    sees roughly as many gradient steps per training call as a
    "typical"-sized node (steps per epoch == len(train_loader), so a
    smaller shard otherwise means both fewer steps per epoch AND the same
    epoch count as everyone else). Scales epochs UP relative to the median
    node's batch count — never down, so typical/large nodes keep
    `base_epochs` — and caps the result at `max_epochs` so a very small
    node (e.g. ~300 images) doesn't get scaled into severe overfitting or
    a blown-out runtime.
    """
    batches_per_node = [len(train_loader) for train_loader, _ in node_loaders]
    reference = statistics.median(batches_per_node)
    return [
        min(max_epochs, round(base_epochs * max(1.0, reference / batches)))
        for batches in batches_per_node
    ]


def build_dataloaders(cfg: Config, dataset):
    """`dataset` is the merged PlantVillage+PlantDoc pool (see
    src/data/merged.py) — domain mixing is already baked into it 1:1 per
    class, so no separate PlantDoc loading/mixing happens here; every split
    below is just a stratified carve/partition over one dataset.
    """
    probe_idx, remaining_idx = carve_public_probe_set(
        dataset,
        cfg.get("data.probe_set_fraction", 0.05),
        cfg.get("data.seed", 42),
        large_class_threshold=cfg.get("data.probe_set_large_class_threshold", 200),
        min_samples_small_class=cfg.get("data.probe_set_min_samples_small_class", 8),
        max_fraction_small_class=cfg.get("data.probe_set_max_fraction_small_class", 0.2),
    )
    # measure of whether a node's model actually generalizes.
    global_test_idx, remaining_idx = carve_global_test_set(
        dataset,
        remaining_idx,
        cfg.get("data.global_test_fraction", 0.05),
        cfg.get("data.seed", 42),
        large_class_threshold=cfg.get("data.probe_set_large_class_threshold", 200),
        min_samples_small_class=cfg.get("data.probe_set_min_samples_small_class", 8),
        max_fraction_small_class=cfg.get("data.probe_set_max_fraction_small_class", 0.2),
    )
    shards = partition_nodes(
        dataset,
        remaining_idx,
        cfg.get("data.num_nodes", 3),
        cfg.get("data.non_iid_strategy", "by_crop"),
        cfg.get("data.dirichlet_alpha", 0.3),
        cfg.get("data.seed", 42),
        manual_node_crops=cfg.get("data.manual_node_crops", None),
    )
    batch_size = cfg.get("training.batch_size", 32)
    probe_loader = DataLoader(make_subset(dataset, probe_idx), batch_size=batch_size, shuffle=False)
    global_test_loader = DataLoader(make_subset(dataset, global_test_idx), batch_size=batch_size, shuffle=False)

    node_loaders = []
    crop_class_weights = []
    disease_class_weights = []
    for shard in shards:
        train_idx, test_idx = train_test_split_indices(
            dataset, shard, cfg.get("data.test_fraction", 0.15), cfg.get("data.seed", 42)
        )
        train_subset = make_subset(dataset, train_idx, train=True)
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
    return probe_loader, global_test_loader, node_loaders, crop_class_weights, disease_class_weights


def dataset_manifest(cfg: Config, dataset, node_loaders) -> dict:
    """Snapshot of the config/data choices that determine node shard
    identity — written by src/train_local.py (stage 1) and re-derived by
    src/train_mesh.py (stage 2) to confirm stage 2 is loading checkpoints
    for the SAME node partition stage 1 actually trained, before it warm-
    starts from them. A drifted config.yaml between the two runs (e.g. a
    different seed or num_nodes) would otherwise silently pair a
    checkpoint with the wrong node's data.
    """
    return {
        "seed": cfg.get("data.seed", 42),
        "num_nodes": cfg.get("data.num_nodes", 3),
        "non_iid_strategy": cfg.get("data.non_iid_strategy", "by_crop"),
        "manual_node_crops": cfg.get("data.manual_node_crops", None),
        "crop_classes": dataset.labels.crop_classes,
        "disease_classes": dataset.labels.disease_classes,
        "node_shard_sizes": [
            [len(train_loader.dataset), len(test_loader.dataset)] for train_loader, test_loader in node_loaders
        ],
    }


def check_manifest_match(expected: dict, actual: dict, expected_path) -> None:
    mismatches = [
        f"  {key}: stage 1 had {expected[key]!r}, stage 2 computed {actual[key]!r}"
        for key in expected
        if expected.get(key) != actual.get(key)
    ]
    if mismatches:
        raise ValueError(
            f"Stage 2's dataset/config does not match stage 1's manifest at {expected_path} — "
            f"checkpoints would be paired with the wrong node's data:\n" + "\n".join(mismatches)
        )


def run_baseline(
    cfg, arch, node_loaders, global_test_loader, crop_classes, disease_classes, tracker, device,
    crop_class_weights=None, disease_class_weights=None, epochs_per_node=None, checkpoint_dir=None,
    pair_class_names=None, class_to_crop_disease=None, resume=False,
):
    """Local-only training, no exchange at all — the comparison point
    the collaboration gain is measured against. If `checkpoint_dir` is
    given, each node's trained model is saved there as `<node_id>.pt` —
    used by src/train_mesh.py (stage 2) to warm-start from these exact
    weights instead of a fresh random init. If `resume` is also True and
    a node's checkpoint already exists there, that node is loaded and
    re-evaluated instead of retrained — so a partial run (e.g. an HPC job
    that hit its walltime after node 8 of 13) can pick back up rather than
    retraining every node from scratch.
    """
    crop_class_weights = crop_class_weights or [None] * len(node_loaders)
    disease_class_weights = disease_class_weights or [None] * len(node_loaders)
    epochs_per_node = epochs_per_node or [cfg.get("training.baseline_epochs", 10)] * len(node_loaders)
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
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
            pair_class_names=pair_class_names, class_to_crop_disease=class_to_crop_disease,
        )
        ckpt_path = checkpoint_dir / f"{node_id}.pt" if checkpoint_dir is not None else None
        if resume and ckpt_path is not None and ckpt_path.exists():
            print(f"  {node_id}: checkpoint already exists at {ckpt_path}, skipping retraining (resume)")
            node.model.load_state_dict(torch.load(ckpt_path, map_location=device))
        else:
            with tracker.track(f"{arch}_baseline_{node_id}"):
                node.local_train(
                    epochs_per_node[i], cfg.get("training.lr", 0.001),
                    val_loader=test_loader, patience=cfg.get("training.early_stopping_patience", 5),
                    weight_decay=cfg.get("training.weight_decay", 0.0),
                )
            if ckpt_path is not None:
                torch.save(node.model.state_dict(), ckpt_path)
        # top-level metrics: this node's own (skewed) local test split. Note
        # this is the SAME split early stopping just validated against, so
        # this number is a little optimistic — global_test_loader below is
        # the untouched, unbiased generalization check.
        # "global": the untouched global_test_loader — the only number
        # that reflects whether this node can classify the full catalogue.
        node_eval = node.evaluate()
        node_eval["global"] = node.evaluate(global_test_loader)
        evals[node_id] = node_eval
    return evals


def run_mesh(
    cfg, arch, node_loaders, probe_loader, global_test_loader, crop_classes, disease_classes, tracker, device,
    output_dir, crop_class_weights=None, disease_class_weights=None, local_epochs_per_node=None,
    warm_start_checkpoint_dir=None, pair_class_names=None, class_to_crop_disease=None,
):
    """If `warm_start_checkpoint_dir` is given, each node's model is loaded
    from `<warm_start_checkpoint_dir>/<node_id>.pt` (e.g. the checkpoints
    src/train_local.py's run_baseline call wrote) instead of a fresh random
    init — stage 2 of the two-stage pipeline continuing from stage 1's
    already-trained local models. Returns (final_evals, total_bytes,
    pre_round_evals) — pre_round_evals is each node's eval() right after
    construction/warm-start, before round 0 does anything, so callers can
    measure "did the rounds that followed actually help".
    """
    crop_class_weights = crop_class_weights or [None] * len(node_loaders)
    disease_class_weights = disease_class_weights or [None] * len(node_loaders)
    default_local_epochs = cfg.get(
        f"training.local_epochs_per_round_overrides.{arch}", cfg.get("training.local_epochs_per_round", 2)
    )
    local_epochs_per_node = local_epochs_per_node or [default_local_epochs] * len(node_loaders)
    warm_start_checkpoint_dir = Path(warm_start_checkpoint_dir) if warm_start_checkpoint_dir else None
    nodes = []
    for i, (train_loader, test_loader) in enumerate(node_loaders):
        node_id = f"node_{i}"
        model = build_model(
            arch, len(crop_classes), len(disease_classes), pretrained=cfg.get("models.pretrained", True)
        )
        if warm_start_checkpoint_dir is not None:
            ckpt_path = warm_start_checkpoint_dir / f"{node_id}.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"Missing stage-1 checkpoint for {node_id} at {ckpt_path} — run "
                    f"'python -m src.train_local --config ...' for this architecture first."
                )
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        nodes.append(Node(
            node_id, model, train_loader, test_loader, device=device,
            crop_classes=crop_classes, disease_classes=disease_classes,
            crop_loss_weight=cfg.get("training.crop_loss_weight", 1.0),
            disease_loss_weight=cfg.get("training.disease_loss_weight", 1.0),
            crop_class_weights=crop_class_weights[i],
            disease_class_weights=disease_class_weights[i],
            loss_type=cfg.get("training.loss_type", "cross_entropy"),
            focal_gamma=cfg.get("training.focal_gamma", 2.0),
            pair_class_names=pair_class_names, class_to_crop_disease=class_to_crop_disease,
        ))

    pre_round_evals = {}
    for node in nodes:
        node_eval = node.evaluate()
        node_eval["global"] = node.evaluate(global_test_loader)
        pre_round_evals[node.node_id] = node_eval

    mesh = MeshSimulator(
        nodes,
        probe_loader,
        aggregation_method=cfg.get("federated.aggregation", "trimmed_mean"),
        trim_fraction=cfg.get("federated.trim_fraction", 0.2),
        krum_neighbors=cfg.get("federated.krum_neighbors", 2),
        adaptive_kd_weight=cfg.get("training.adaptive_kd_weight", False),
        adaptive_kd_min_scale=cfg.get("training.adaptive_kd_min_scale", 0.3),
        adaptive_kd_max_scale=cfg.get("training.adaptive_kd_max_scale", 1.5),
    )
    local_epochs = {node.node_id: epochs for node, epochs in zip(nodes, local_epochs_per_node)}

    total_bytes = 0
    round_logs = []
    for r in range(cfg.get("training.rounds", 5)):
        with tracker.track(f"{arch}_mesh_round_{r}"):
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
        final_evals[node.node_id] = node_eval

    checkpoint_dir = output_dir / "checkpoints" / arch
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for node in nodes:
        torch.save(node.model.state_dict(), checkpoint_dir / f"{node.node_id}.pt")

    return final_evals, total_bytes, pre_round_evals


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

    dataset = load_merged_dataset(
        cfg.get("data.root"), cfg.get("data.plantdoc_root"), cfg.get("data.image_size", 224), cfg.get("data.seed", 42)
    )
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
                "image_size": cfg.get("data.image_size", 224),
            },
            indent=2,
        )
    )

    probe_loader, global_test_loader, node_loaders, crop_class_weights, disease_class_weights = build_dataloaders(cfg, dataset)

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

    # Node shard sizes are wildly imbalanced (one-crop-per-node manual
    # split can range ~40x smallest to largest) — without this, every node
    # trains for the same number of epochs, so a small node also gets far
    # fewer total gradient steps per training call. Scale epochs up (never
    # down, capped at max_epochs_per_node) for nodes below the median size.
    max_epochs_per_node = cfg.get("training.max_epochs_per_node", 40)
    baseline_epochs_per_node = scale_epochs_by_node_size(
        cfg.get("training.baseline_epochs", 10), node_loaders, max_epochs_per_node
    )
    print(f"Per-node baseline epochs (size-scaled): {baseline_epochs_per_node}")

    architectures = [args.arch] if args.arch else cfg.get("models.architectures", [])
    for arch in architectures:
        print(f"\n=== Architecture: {arch} ===")
        base_local_epochs = cfg.get(
            f"training.local_epochs_per_round_overrides.{arch}", cfg.get("training.local_epochs_per_round", 2)
        )
        local_epochs_per_node = scale_epochs_by_node_size(base_local_epochs, node_loaders, max_epochs_per_node)
        print(f"Per-node mesh local_epochs (size-scaled): {local_epochs_per_node}")

        print("-- baseline (local-only) --")
        baseline_evals = run_baseline(
            cfg, arch, node_loaders, global_test_loader,
            dataset.labels.crop_classes, dataset.labels.disease_classes, tracker, device,
            crop_class_weights=crop_class_weights, disease_class_weights=disease_class_weights,
            epochs_per_node=baseline_epochs_per_node,
            pair_class_names=dataset.base.classes, class_to_crop_disease=dataset.labels.class_to_crop_disease,
        )
        print("-- mesh (prototype + logit exchange) --")
        mesh_evals, total_bytes, _pre_round_evals = run_mesh(
            cfg, arch, node_loaders, probe_loader, global_test_loader,
            dataset.labels.crop_classes, dataset.labels.disease_classes, tracker, device, output_dir,
            crop_class_weights=crop_class_weights, disease_class_weights=disease_class_weights,
            local_epochs_per_node=local_epochs_per_node,
            pair_class_names=dataset.base.classes, class_to_crop_disease=dataset.labels.class_to_crop_disease,
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
        if "global_macro_gain" in gain:
            print(f"global_macro_gain: {gain['global_macro_gain']}")
            print(f"global_worst_node_gain: {gain['global_worst_node_gain']}")

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
