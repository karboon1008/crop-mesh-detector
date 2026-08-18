"""Scenario: one node's local camera/lighting degrades partway through the
run (brightness shift + blur + noise applied to its images from that round
onward). Demonstrates that the mesh dampens the resulting accuracy drop
compared to a node coping with the shift on local training alone (the
baseline).

Run: python -m src.scenarios.distribution_shift [--config path] [--arch name]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from src.config import Config
from src.data.merged import load_merged_dataset
from src.federated.mesh import MeshSimulator
from src.scenarios.harness import (
    ScenarioEvent,
    build_node_set,
    require_target_node,
    run_scenario,
    write_scenario_report,
)
from src.train import build_dataloaders

ALLOWED_CORRUPTIONS = {"brightness_blur_noise"}


class CorruptedDataset(Dataset):
    """Wraps a dataset, applying a severity-scaled brightness shift +
    Gaussian blur + additive noise to the already-normalized image tensor —
    simulating a degraded camera/lighting condition. severity=0 is a no-op.
    """

    def __init__(self, base: Dataset, severity: float):
        self.base = base
        self.severity = severity

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        image, crop, disease = self.base[idx]
        if self.severity > 0:
            image = image * (1.0 + self.severity)
            kernel_size = max(3, int(self.severity * 6) | 1)
            image = TF.gaussian_blur(image, kernel_size=[kernel_size, kernel_size])
            image = image + torch.randn_like(image) * self.severity
        return image, crop, disease


def make_shift_hook(target_node: str, shift_round: int, corruption: str, severity: float, batch_size: int):
    def hook(round_idx, nodes, mesh):
        if round_idx != shift_round:
            return []
        target = next(n for n in nodes if n.node_id == target_node)
        target.train_loader = DataLoader(
            CorruptedDataset(target.train_loader.dataset, severity),
            batch_size=batch_size, shuffle=True,
        )
        target.test_loader = DataLoader(
            CorruptedDataset(target.test_loader.dataset, severity), batch_size=batch_size, shuffle=False
        )
        if mesh is not None:
            return [ScenarioEvent(round_idx, "shift_applied", target_node,
                                   {"corruption": corruption, "severity": severity})]
        return []

    return hook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--arch", default=None)
    args = parser.parse_args()
    cfg = Config.load(args.config)

    scfg = cfg.get("scenarios.distribution_shift")
    if scfg is None:
        raise ValueError("config.yaml is missing a scenarios.distribution_shift section")
    target_node = scfg["target_node"]
    shift_round = scfg["shift_round"]
    corruption = scfg["corruption"]
    severity = scfg["severity"]
    num_rounds = cfg.get("scenarios.rounds", 6)

    if not (0 <= shift_round < num_rounds):
        raise ValueError(f"scenarios.distribution_shift.shift_round ({shift_round}) must be in [0, {num_rounds})")
    if corruption not in ALLOWED_CORRUPTIONS:
        raise ValueError(
            f"scenarios.distribution_shift.corruption '{corruption}' is not supported; "
            f"must be one of {sorted(ALLOWED_CORRUPTIONS)}"
        )
    if not (severity >= 0):
        raise ValueError(f"scenarios.distribution_shift.severity ({severity}) must be >= 0")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(cfg.get("output.dir", "outputs"))
    dataset = load_merged_dataset(
        cfg.get("data.root"), cfg.get("data.plantdoc_root"), cfg.get("data.image_size", 160), cfg.get("data.seed", 42)
    )
    probe_loader, _global_test_loader, node_loaders, _crop_class_weights, _disease_class_weights = build_dataloaders(cfg, dataset)
    arch = args.arch or cfg.get("models.architectures", ["mobilenet_v3_small"])[0]
    batch_size = cfg.get("training.batch_size", 32)

    require_target_node(target_node, node_loaders)

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

    hook = make_shift_hook(target_node, shift_round, corruption, severity, batch_size)
    round_kwargs = {
        "local_epochs": cfg.get("training.local_epochs_per_round", 2),
        "distill_epochs": cfg.get("training.distill_epochs_per_round", 1),
        "lr": cfg.get("training.lr", 0.001),
        "distill_lr": cfg.get("training.distill_lr", 0.0005),
        "proto_weight": cfg.get("training.proto_weight", 0.5),
        "kd_weight": cfg.get("training.kd_weight", 0.5),
        "temperature": cfg.get("training.kd_temperature", 2.0),
    }
    records = run_scenario(baseline_nodes, mesh, num_rounds, hook, round_kwargs)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = write_scenario_report(
        output_dir, "distribution_shift", target_node,
        disruption_start_round=shift_round, disruption_end_round=shift_round,
        config_snapshot=scfg, records=records, save_plots=cfg.get("output.save_plots", True),
    )
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
