"""Shared round-driver for the mesh scenario simulations (node
disconnection, runtime class addition, distribution shift): runs a
no-exchange baseline and the mesh side by side under the same
perturbation schedule, and writes the comparison to JSON.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Callable, Optional

from src.energy.tracker import CommunicationCostEstimator, ComputeEnergyTracker
from src.evaluate import compute_collaboration_gain
from src.federated.mesh import MeshSimulator
from src.federated.node import Node
from src.models.factory import build_model

RECOVERY_TOLERANCE = 0.05  # accuracy points a node must be within to count as "recovered"


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


PerturbationHook = Callable[[int, "list[Node]", Optional[MeshSimulator]], "list[ScenarioEvent]"]


def node_ids_for(node_loaders) -> list[str]:
    return [f"node_{i}" for i in range(len(node_loaders))]


def require_target_node(target_node: str, node_loaders) -> None:
    ids = node_ids_for(node_loaders)
    if target_node not in ids:
        raise ValueError(f"target_node '{target_node}' is not one of {ids}")


def build_node_set(cfg, arch: str, node_loaders, num_crop: int, num_disease: int, device: str) -> list[Node]:
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
        model = build_model(arch, num_crop, num_disease, pretrained=cfg.get("models.pretrained", True))
        nodes.append(Node(f"node_{i}", model, train_loader, test_loader, device=device))
    return nodes


def run_scenario(
    baseline_nodes: list[Node],
    mesh: MeshSimulator,
    num_rounds: int,
    perturbation_hook: PerturbationHook,
    round_kwargs: dict,
    tracker: ComputeEnergyTracker,
    comm_estimator: CommunicationCostEstimator,
    radio: str = "wifi",
) -> list[ScenarioRoundRecord]:
    """Drives `num_rounds` rounds of a baseline (no-exchange) node set and a
    mesh node set through the same perturbation schedule, tracking compute
    energy (via `tracker`, the same ComputeEnergyTracker src/train.py uses)
    and communication energy (via `comm_estimator`, fed by the mesh round's
    already-measured `total_bytes_exchanged`) for every round of both.

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
            with tracker.track(f"baseline_{node.node_id}_round_{round_idx}") as energy_record:
                node.local_train(round_kwargs["local_epochs"], round_kwargs["lr"])
            baseline_compute_energy_kwh += energy_record["energy_kwh"]
            baseline_eval[node.node_id] = node.evaluate()

        with tracker.track(f"mesh_round_{round_idx}") as mesh_energy_record:
            round_log = mesh.run_round(
                round_idx,
                local_epochs=round_kwargs["local_epochs"],
                distill_epochs=round_kwargs["distill_epochs"],
                lr=round_kwargs["lr"],
                distill_lr=round_kwargs["distill_lr"],
                proto_weight=round_kwargs["proto_weight"],
                kd_weight=round_kwargs["kd_weight"],
                temperature=round_kwargs["temperature"],
            )
        mesh_compute_energy_kwh = mesh_energy_record["energy_kwh"]
        mesh_eval = {node.node_id: node.evaluate() for node in mesh.nodes}

        comm_result = comm_estimator.estimate(round_log.total_bytes_exchanged, radio)
        communication_energy_j = comm_result["energy_kwh"] * 3_600_000

        gain = compute_collaboration_gain(mesh_eval, baseline_eval)
        records.append(ScenarioRoundRecord(
            round_idx, baseline_eval, mesh_eval, gain, events,
            active_nodes=round_log.active_nodes,
            total_bytes_exchanged=round_log.total_bytes_exchanged,
            baseline_compute_energy_kwh=baseline_compute_energy_kwh,
            mesh_compute_energy_kwh=mesh_compute_energy_kwh,
            communication_energy_j=communication_energy_j,
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
    pre_eval = getattr(records[pre_round], eval_key).get(node_id)
    if pre_eval is None:
        return None
    for record in records:
        if record.round_idx < disruption_end_round:
            continue
        current = getattr(record, eval_key).get(node_id)
        if current is None:
            continue
        if all(current[m] >= pre_eval[m] - RECOVERY_TOLERANCE for m in pre_eval):
            return record.round_idx
    return None


def _gain_per_joule(records: list[ScenarioRoundRecord]) -> Optional[float]:
    """The final round's macro collaboration gain divided by the total
    energy (compute + communication) the mesh spent across the whole run —
    Appendix A's gain_per_joule metric. Returns None when no energy was
    recorded at all, to avoid a divide-by-zero silently reporting 0.0 as if
    it were a measured (rather than absent) figure.
    """
    if not records:
        return None
    total_mesh_energy_j = (
        sum(r.mesh_compute_energy_kwh for r in records) * 3_600_000
        + sum(r.communication_energy_j for r in records)
    )
    if total_mesh_energy_j <= 0:
        return None
    final_gain = records[-1].collaboration_gain.get("macro_gain", 0.0)
    # Handle case where macro_gain is a dict (from compute_collaboration_gain)
    # by averaging its values; otherwise it's already a scalar
    if isinstance(final_gain, dict):
        values = list(final_gain.values())
        if not values:
            return None
        final_gain = sum(values) / len(values)
    return final_gain / total_mesh_energy_j


def write_scenario_report(
    output_dir: Path,
    scenario_name: str,
    target_node_id: str,
    disruption_start_round: int,
    disruption_end_round: int,
    config_snapshot: dict,
    records: list[ScenarioRoundRecord],
) -> Path:
    """Writes outputs/scenarios/{scenario_name}.json and returns its path."""
    scenarios_dir = output_dir / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "scenario": scenario_name,
        "target_node": target_node_id,
        "config": config_snapshot,
        "rounds": [dataclasses.asdict(r) for r in records],
        "summary": {
            "recovery_round_mesh": _recovery_round(
                records, target_node_id, disruption_start_round, disruption_end_round, "mesh_eval"
            ),
            "recovery_round_baseline": _recovery_round(
                records, target_node_id, disruption_start_round, disruption_end_round, "baseline_eval"
            ),
            "sustainability": {
                "total_baseline_compute_energy_kwh": sum(r.baseline_compute_energy_kwh for r in records),
                "total_mesh_compute_energy_kwh": sum(r.mesh_compute_energy_kwh for r in records),
                "total_communication_energy_j": sum(r.communication_energy_j for r in records),
                "gain_per_joule": _gain_per_joule(records),
            },
        },
    }
    path = scenarios_dir / f"{scenario_name}.json"
    path.write_text(json.dumps(report, indent=2))
    return path
