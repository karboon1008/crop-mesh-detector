"""Scenario: a node disconnects mid-run and later reconnects. Demonstrates
that the mesh tolerates a node dropping out (it keeps training locally,
just isolated) and that reconnecting lets it catch back up towards peer
consensus faster than relying on local training alone (the baseline).

Run: python -m src.scenarios.disconnection [--config path] [--arch name]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.config import Config
from src.data.plantvillage import load_full_dataset
from src.energy.tracker import CommunicationCostEstimator, ComputeEnergyTracker
from src.federated.mesh import MeshSimulator
from src.scenarios.harness import (
    ScenarioEvent,
    build_node_set,
    require_target_node,
    run_scenario,
    write_scenario_report,
)
from src.train import build_dataloaders


def make_disconnect_hook(target_node: str, disconnect_round: int, reconnect_round: int):
    def hook(round_idx, nodes, mesh):
        target = next(n for n in nodes if n.node_id == target_node)
        events = []
        if round_idx == disconnect_round:
            target.active = False
            if mesh is not None:
                events.append(ScenarioEvent(round_idx, "disconnect", target_node, {}))
        elif round_idx == reconnect_round:
            target.active = True
            if mesh is not None:
                events.append(ScenarioEvent(round_idx, "reconnect", target_node, {}))
        return events

    return hook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--arch", default=None)
    args = parser.parse_args()
    cfg = Config.load(args.config)

    scfg = cfg.get("scenarios.disconnection")
    if scfg is None:
        raise ValueError("config.yaml is missing a scenarios.disconnection section")
    target_node = scfg["target_node"]
    disconnect_round = scfg["disconnect_round"]
    reconnect_round = scfg["reconnect_round"]
    num_rounds = cfg.get("scenarios.rounds", 6)

    if not (0 <= disconnect_round < reconnect_round < num_rounds):
        raise ValueError(
            f"scenarios.disconnection needs 0 <= disconnect_round ({disconnect_round}) "
            f"< reconnect_round ({reconnect_round}) < scenarios.rounds ({num_rounds})"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(cfg.get("output.dir", "outputs"))
    dataset = load_full_dataset(cfg.get("data.root"), cfg.get("data.image_size", 160))
    num_crop = len(dataset.labels.crop_classes)
    num_disease = len(dataset.labels.disease_classes)
    probe_loader, node_loaders = build_dataloaders(cfg, dataset)
    arch = args.arch or cfg.get("models.architectures", ["mobilenet_v3_small"])[0]

    require_target_node(target_node, node_loaders)

    baseline_nodes = build_node_set(cfg, arch, node_loaders, num_crop, num_disease, device)
    mesh_nodes = build_node_set(cfg, arch, node_loaders, num_crop, num_disease, device)
    mesh = MeshSimulator(
        mesh_nodes,
        probe_loader,
        aggregation_method=cfg.get("federated.aggregation", "trimmed_mean"),
        trim_fraction=cfg.get("federated.trim_fraction", 0.2),
        krum_neighbors=cfg.get("federated.krum_neighbors", 2),
    )

    tracker = ComputeEnergyTracker(
        enabled=cfg.get("energy.track_with_codecarbon", True),
        output_dir=output_dir,
        country_iso_code=cfg.get("energy.country_iso_code", "GBR"),
    )
    comm_estimator = CommunicationCostEstimator(
        cfg.get("energy.radio_energy_j_per_byte", {}),
        cfg.get("energy.grid_carbon_intensity_gco2_per_kwh", 125),
    )

    round_kwargs = {
        "local_epochs": cfg.get("training.local_epochs_per_round", 2),
        "distill_epochs": cfg.get("training.distill_epochs_per_round", 1),
        "lr": cfg.get("training.lr", 0.001),
        "distill_lr": cfg.get("training.distill_lr", 0.0005),
        "proto_weight": cfg.get("training.proto_weight", 0.5),
        "kd_weight": cfg.get("training.kd_weight", 0.5),
        "temperature": cfg.get("training.kd_temperature", 2.0),
    }
    records = run_scenario(
        baseline_nodes, mesh, num_rounds,
        make_disconnect_hook(target_node, disconnect_round, reconnect_round),
        round_kwargs,
        tracker=tracker, comm_estimator=comm_estimator,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = write_scenario_report(
        output_dir, "disconnection", target_node,
        disruption_start_round=disconnect_round, disruption_end_round=reconnect_round,
        config_snapshot=scfg, records=records,
    )
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
