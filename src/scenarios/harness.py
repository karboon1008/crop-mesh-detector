"""Shared round-driver for the mesh scenario simulations (node
disconnection, runtime class addition, distribution shift): runs a
no-exchange baseline and the mesh side by side under the same
perturbation schedule, and writes the comparison to JSON.
"""

from __future__ import annotations

import dataclasses
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional

from src.energy.tracker import CommunicationCostEstimator, ComputeEnergyTracker
from src.evaluate import compute_collaboration_gain, scalar_metrics
from src.federated.mesh import MeshSimulator
from src.federated.node import Node
from src.models.factory import build_model
from src.reporting import build_per_class_rows, build_scenario_rows, plot_training_curves, write_csv
from src.scenarios import metrics

RECOVERY_TOLERANCE = 0.05  # accuracy points a node must be within to count as "recovered"
JOULES_PER_KWH = 3_600_000


@dataclasses.dataclass
class ScenarioEvent:
    round_idx: int
    event_type: str  # "disconnect" | "reconnect" | "class_added" | "shift_applied"
    node_id: str
    details: dict


@dataclasses.dataclass
class ScenarioRoundRecord:
    round_idx: int
    baseline_eval: dict[str, dict[str, float]]
    mesh_eval: dict[str, dict[str, float]]
    collaboration_gain: dict
    events: list[ScenarioEvent] = dataclasses.field(default_factory=list)
    active_nodes: list[str] = dataclasses.field(default_factory=list)
    total_bytes_exchanged: int = 0
    baseline_compute_energy_kwh: float = 0.0
    mesh_compute_energy_kwh: float = 0.0
    communication_energy_j: float = 0.0
    per_node_train_loss: dict[str, float] = dataclasses.field(default_factory=dict)
    # mesh side only, snapshot right after local_train, before distillation
    pre_distill_eval: dict[str, dict[str, float | dict]] = dataclasses.field(default_factory=dict)
    # node_id -> {"kd_loss", "sup_loss", "proto_loss"}
    per_node_distill_loss: dict[str, dict[str, float]] = dataclasses.field(default_factory=dict)


PerturbationHook = Callable[[int, "list[Node]", Optional[MeshSimulator]], "list[ScenarioEvent]"]


@contextmanager
def _track(tracker: ComputeEnergyTracker | None, label: str):
    """`tracker.track(label)` when a tracker was supplied, otherwise a
    context manager over an empty dict — so `run_scenario` has one code
    path whether or not energy accounting is switched on.
    """
    if tracker is None:
        yield {}
        return
    with tracker.track(label) as record:
        yield record


def node_ids_for(node_loaders) -> list[str]:
    return [f"node_{i}" for i in range(len(node_loaders))]


def require_target_node(target_node: str, node_loaders) -> None:
    ids = node_ids_for(node_loaders)
    if target_node not in ids:
        raise ValueError(f"target_node '{target_node}' is not one of {ids}")


def build_node_set(
    cfg, arch: str, node_loaders, crop_classes: list[str], disease_classes: list[str], device: str,
    pair_class_names: list[str] | None = None, class_to_crop_disease: dict[int, tuple[int, int]] | None = None,
) -> list[Node]:
    """Builds one fresh, independent Node per shard in `node_loaders` — used
    to construct both the no-exchange baseline set and the mesh set from
    the same starting shards.

    Note: each call builds independent DataLoader/model objects, even when
    given the same `node_loaders` shards twice (as `run_scenario` does for
    the baseline and mesh sets) — this is safe today only because every
    scenario perturbation hook *replaces* a node's `train_loader`/`test_loader`
    attribute wholesale rather than mutating a shared loader/dataset object
    in place; a future hook that mutates in place would need to account for
    whether the two node sets still share underlying objects.
    """
    nodes = []
    for i, (train_loader, test_loader) in enumerate(node_loaders):
        model = build_model(
            arch, len(crop_classes), len(disease_classes), pretrained=cfg.get("models.pretrained", True)
        )
        nodes.append(Node(
            f"node_{i}", model, train_loader, test_loader, device=device,
            crop_classes=crop_classes, disease_classes=disease_classes,
            pair_class_names=pair_class_names, class_to_crop_disease=class_to_crop_disease,
        ))
    return nodes


def build_energy_accounting(cfg, output_dir: Path) -> tuple[ComputeEnergyTracker, CommunicationCostEstimator]:
    """The same tracker/estimator pair `src/train.py` builds, from the same
    `energy:` section of config.yaml — so a scenario run's energy figures
    are produced by the identical accounting code as the headline run's.
    """
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
    return tracker, comm_estimator


def build_provenance(
    cfg, arch: str, num_rounds: int, node_loaders, tracker: ComputeEnergyTracker, radio: str = "wifi", **extra
) -> dict:
    """Run conditions worth recording in the report itself, because they
    are what make one scenario run comparable (or not) with another: the
    architecture, how many nodes there were and how their shards were
    skewed, and which method produced the compute-energy figures.
    """
    return {
        "architecture": arch,
        "num_nodes": len(node_loaders),
        "node_train_sizes": {f"node_{i}": len(tl.dataset) for i, (tl, _) in enumerate(node_loaders)},
        "non_iid_strategy": cfg.get("data.non_iid_strategy"),
        "dirichlet_alpha": cfg.get("data.dirichlet_alpha"),
        "seed": cfg.get("data.seed"),
        "rounds": num_rounds,
        "aggregation": cfg.get("federated.aggregation"),
        "compute_energy_method": tracker.measurement_method,
        "fallback_power_watts": cfg.get("energy.fallback_power_watts", 15.0),
        "communication_radio": radio,
        "radio_energy_j_per_byte": cfg.get(f"energy.radio_energy_j_per_byte.{radio}"),
        **extra,
    }


def run_scenario(
    baseline_nodes: list[Node],
    mesh: MeshSimulator,
    num_rounds: int,
    perturbation_hook: PerturbationHook,
    round_kwargs: dict,
    tracker: ComputeEnergyTracker | None = None,
    comm_estimator: CommunicationCostEstimator | None = None,
    radio: str = "wifi",
) -> list[ScenarioRoundRecord]:
    """Drives `num_rounds` rounds of a baseline (no-exchange) node set and a
    mesh node set side by side through the same perturbation schedule.

    When `tracker` / `comm_estimator` are supplied, each round also records
    the compute energy of both arms and the estimated transmit energy of
    that round's exchange, so the scenario reports carry the same
    three-part (compute / communication / gain-per-joule) inventory the
    main pipeline reports. Both arms' `evaluate()` calls sit *inside* their
    tracked block, so the two compute figures are measured at the same
    boundary and remain comparable. Omitting them leaves the energy fields
    at 0.0 and makes the report's derived energy metrics null rather than
    silently reporting zero as if it were measured.

    Convention: `perturbation_hook` is called once per round for the
    baseline set (third argument None) and once for the mesh set (third
    argument the MeshSimulator) — it should mutate whichever node set it's
    given, but only return a non-empty list of ScenarioEvents on the mesh
    call, so each real-world event is recorded once even though it is
    applied identically to both parallel simulations.
    """
    records: list[ScenarioRoundRecord] = []
    for round_idx in range(num_rounds):
        events = perturbation_hook(round_idx, baseline_nodes, None)
        events = events + perturbation_hook(round_idx, mesh.nodes, mesh)

        baseline_eval = {}
        baseline_compute_energy_kwh = 0.0
        for node in baseline_nodes:
            with _track(tracker, f"baseline_{node.node_id}_round_{round_idx}") as energy_record:
                node.local_train(round_kwargs["local_epochs"], round_kwargs["lr"])
                baseline_eval[node.node_id] = node.evaluate()
            baseline_compute_energy_kwh += energy_record.get("energy_kwh", 0.0)

        with _track(tracker, f"mesh_round_{round_idx}") as mesh_energy_record:
            round_log = mesh.run_round(
                round_idx,
                local_epochs=round_kwargs["local_epochs"],
                distill_epochs=round_kwargs["distill_epochs"],
                lr=round_kwargs["lr"],
                distill_lr=round_kwargs["distill_lr"],
                proto_weight=round_kwargs["proto_weight"],
                kd_weight=round_kwargs["kd_weight"],
                crop_kd_weight=round_kwargs.get("crop_kd_weight"),
                temperature=round_kwargs["temperature"],
            )
            mesh_eval = {node.node_id: node.evaluate() for node in mesh.nodes}
        mesh_compute_energy_kwh = mesh_energy_record.get("energy_kwh", 0.0)

        communication_energy_j = 0.0
        if comm_estimator is not None:
            comm = comm_estimator.estimate(round_log.total_bytes_exchanged, radio)
            communication_energy_j = comm["energy_kwh"] * JOULES_PER_KWH

        gain = compute_collaboration_gain(mesh_eval, baseline_eval)
        records.append(ScenarioRoundRecord(
            round_idx, baseline_eval, mesh_eval, gain, events,
            active_nodes=round_log.active_nodes,
            total_bytes_exchanged=round_log.total_bytes_exchanged,
            baseline_compute_energy_kwh=baseline_compute_energy_kwh,
            mesh_compute_energy_kwh=mesh_compute_energy_kwh,
            communication_energy_j=communication_energy_j,
            per_node_train_loss=round_log.per_node_train_loss,
            pre_distill_eval=round_log.pre_distill_eval,
            per_node_distill_loss=round_log.per_node_distill_loss,
        ))
        print(f"  round {round_idx}: {len(events)} event(s), macro_gain={gain['macro_gain']}")
    return records


def _recovery_round(
    records: list[ScenarioRoundRecord],
    node_id: str,
    disruption_start_round: int,
    disruption_end_round: int,
    eval_key: str,
) -> Optional[int]:
    """First round index >= disruption_end_round where `node_id`'s eval
    (from `eval_key`, "mesh_eval" or "baseline_eval") is back at or above
    (within RECOVERY_TOLERANCE below) its value from the round right before
    the disruption started, or None if it never recovers within the run.
    Recovery is one-sided: exceeding the pre-disruption value still counts
    as recovered.
    """
    pre_round = max(0, disruption_start_round - 1)
    if pre_round >= len(records):
        return None
    raw_pre_eval = getattr(records[pre_round], eval_key).get(node_id)
    if raw_pre_eval is None:
        return None
    pre_eval = scalar_metrics(raw_pre_eval)
    for record in records:
        if record.round_idx < disruption_end_round:
            continue
        current = getattr(record, eval_key).get(node_id)
        if current is None:
            continue
        current = scalar_metrics(current)
        if all(current[m] >= pre_eval[m] - RECOVERY_TOLERANCE for m in pre_eval):
            return record.round_idx
    return None


def write_scenario_report(
    output_dir: Path,
    scenario_name: str,
    target_node_id: str,
    disruption_start_round: int,
    disruption_end_round: int,
    config_snapshot: dict,
    records: list[ScenarioRoundRecord],
    save_plots: bool = True,
    provenance: dict | None = None,
) -> Path:
    """Writes outputs/scenarios/{scenario_name}.json (plus a matching .csv
    and .png when save_plots is set) and returns the .json path.

    `provenance` records the run conditions a reader would otherwise have
    to infer — architecture, node count, non-IID strategy, rounds, whether
    compute energy was measured or estimated — so a report can always be
    matched to the configuration that produced it rather than to whatever
    config.yaml happens to say later.
    """
    scenarios_dir = output_dir / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)

    round_dicts = [dataclasses.asdict(r) for r in records]
    trend_rows = build_scenario_rows(round_dicts)
    per_class_rows = build_per_class_rows(round_dicts, [("mesh", "mesh_eval"), ("baseline", "baseline_eval")])
    report = {
        "scenario": scenario_name,
        "target_node": target_node_id,
        "config": config_snapshot,
        "provenance": provenance or {},
        "disruption_start_round": disruption_start_round,
        "disruption_end_round": disruption_end_round,
        "rounds": round_dicts,
        "trend": trend_rows,
        "per_class_trend": per_class_rows,
        "summary": {
            "recovery_round_mesh": _recovery_round(
                records, target_node_id, disruption_start_round, disruption_end_round, "mesh_eval"
            ),
            "recovery_round_baseline": _recovery_round(
                records, target_node_id, disruption_start_round, disruption_end_round, "baseline_eval"
            ),
            "sustainability": {
                **metrics.energy_totals(round_dicts),
                **metrics.efficiency_metrics(round_dicts),
            },
            "adaptation": metrics.adaptation_gain(round_dicts, target_node_id, disruption_start_round),
            "retention": metrics.retention_gain(round_dicts, target_node_id, disruption_start_round),
        },
    }
    path = scenarios_dir / f"{scenario_name}.json"
    path.write_text(json.dumps(report, indent=2))

    if save_plots:
        write_csv(trend_rows, scenarios_dir / f"{scenario_name}.csv")
        write_csv(per_class_rows, scenarios_dir / f"{scenario_name}_per_class.csv")
        plot_training_curves(trend_rows, scenarios_dir / f"{scenario_name}.png", f"{scenario_name} scenario")

    return path
