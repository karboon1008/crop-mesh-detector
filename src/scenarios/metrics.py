"""Derived scenario metrics — adaptation gain, retention gain, and the
energy/communication efficiency of the gain — computed over a scenario
report's plain-dict `rounds` list rather than over live
`ScenarioRoundRecord` objects.

Working on dicts is deliberate: it lets `harness.write_scenario_report`
compute these at the end of a run *and* lets `src.scenarios.summarise`
recompute exactly the same figures from an already-written
`outputs/scenarios/*.json`, so the tables published in the research
document can be regenerated and checked without re-running the (multi-hour,
dataset-dependent) simulation itself.

Metric definitions follow the Challenge's Appendix A suggestions
(adaptation_gain, retention_gain, communication_efficiency,
gain_per_joule) with the exact formula used stated alongside each figure,
as Appendix A.1 requires.
"""

from __future__ import annotations

from typing import Optional

JOULES_PER_KWH = 3_600_000


def scalar_eval(round_: dict, arm: str, node_id: str) -> dict[str, float]:
    """The scalar (non-nested) metrics for one node in one arm of one
    round. `arm` is "mesh_eval" or "baseline_eval". Returns {} when that
    node has no eval recorded for that round.
    """
    node_eval = round_.get(arm, {}).get(node_id)
    if not node_eval:
        return {}
    return {k: float(v) for k, v in node_eval.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def _shared_metrics(rounds: list[dict], node_id: str) -> list[str]:
    """Metric names present on both arms in every round — anything missing
    from either arm is dropped rather than compared against nothing.
    """
    common: Optional[set[str]] = None
    for round_ in rounds:
        keys = set(scalar_eval(round_, "mesh_eval", node_id)) & set(scalar_eval(round_, "baseline_eval", node_id))
        common = keys if common is None else (common & keys)
    return sorted(common or [])


def target_advantage(rounds: list[dict], node_id: str, lo: int, hi: int) -> dict[str, float]:
    """Mean per-metric (mesh − baseline) accuracy for `node_id` over rounds
    with `lo <= round_idx < hi`. Empty dict when that window holds no rounds.
    """
    window = [r for r in rounds if lo <= r["round_idx"] < hi]
    if not window:
        return {}
    metrics = _shared_metrics(window, node_id)
    return {
        m: sum(
            scalar_eval(r, "mesh_eval", node_id)[m] - scalar_eval(r, "baseline_eval", node_id)[m] for r in window
        ) / len(window)
        for m in metrics
    }


def adaptation_gain(rounds: list[dict], node_id: str, disruption_start_round: int) -> dict:
    """Appendix A's `adaptation_gain` — how far ahead of local-only training
    the disrupted node runs while it is coping with the disruption:

        adaptation_gain = mean(mesh − local_only | rounds >= disruption)

    averaged over the disrupted node's own test split, per metric.

    `pre_disruption_advantage` (the same mean over the rounds before the
    disruption) is reported alongside it so a reader can see how much of
    that lead pre-existed the disruption, and `advantage_delta` is their
    difference. The delta is deliberately *not* the headline figure: both
    arms climb towards a shared accuracy ceiling over the run, so the
    mesh's lead compresses round on round for reasons that have nothing to
    do with the disruption, which can drive the delta negative even where
    the mesh is ahead in every single round. Read `adaptation_gain` for
    "how much better did the mesh cope", and `advantage_delta` only as a
    ceiling-confounded diagnostic.
    """
    num_rounds = len(rounds)
    pre = target_advantage(rounds, node_id, 0, disruption_start_round)
    post = target_advantage(rounds, node_id, disruption_start_round, num_rounds)
    return {
        "adaptation_gain": post,
        "pre_disruption_advantage": pre,
        "advantage_delta": {m: post[m] - pre[m] for m in post if m in pre} if pre else None,
    }


def retention_gain(rounds: list[dict], node_id: str, disruption_start_round: int) -> dict:
    """How much accuracy each arm *keeps* through the disruption: the worst
    post-disruption drop below the last pre-disruption value, per arm, and
    the difference between them:

        drop_arm       = max(0, value(pre) − min(value | rounds >= disruption))
        retention_gain = drop_local_only − drop_mesh

    A positive `retention_gain` means the mesh gave up less accuracy at its
    worst moment than local-only training did — the Appendix A
    "ability to retain prior knowledge" reading of this scenario. None when
    the disruption lands on round 0 (no pre-disruption reference value).
    """
    pre_round_idx = disruption_start_round - 1
    if pre_round_idx < 0 or pre_round_idx >= len(rounds):
        return {"mesh_drop": {}, "baseline_drop": {}, "retention_gain": None}

    pre_round = rounds[pre_round_idx]
    post = [r for r in rounds if r["round_idx"] >= disruption_start_round]
    if not post:
        return {"mesh_drop": {}, "baseline_drop": {}, "retention_gain": None}

    drops = {}
    for arm in ("mesh_eval", "baseline_eval"):
        pre_eval = scalar_eval(pre_round, arm, node_id)
        drops[arm] = {
            m: max(0.0, pre_eval[m] - min(scalar_eval(r, arm, node_id).get(m, pre_eval[m]) for r in post))
            for m in pre_eval
        }
    shared = set(drops["mesh_eval"]) & set(drops["baseline_eval"])
    return {
        "mesh_drop": drops["mesh_eval"],
        "baseline_drop": drops["baseline_eval"],
        "retention_gain": {m: drops["baseline_eval"][m] - drops["mesh_eval"][m] for m in sorted(shared)},
    }


def energy_totals(rounds: list[dict]) -> dict:
    """Run totals for the three cost terms the Challenge's Appendix E
    inventory asks for, in the units each is naturally reported in.
    `additional_energy_j` is the extra energy collaboration cost over the
    local-only arm — the denominator Baseline B's "gain per additional unit
    of energy spent" question actually calls for.
    """
    baseline_kwh = sum(r.get("baseline_compute_energy_kwh", 0.0) for r in rounds)
    mesh_kwh = sum(r.get("mesh_compute_energy_kwh", 0.0) for r in rounds)
    comm_j = sum(r.get("communication_energy_j", 0.0) for r in rounds)
    return {
        "total_baseline_compute_energy_kwh": baseline_kwh,
        "total_mesh_compute_energy_kwh": mesh_kwh,
        "total_communication_energy_j": comm_j,
        "total_bytes_exchanged": sum(r.get("total_bytes_exchanged", 0) for r in rounds),
        "total_mesh_energy_j": mesh_kwh * JOULES_PER_KWH + comm_j,
        "additional_energy_j": (mesh_kwh - baseline_kwh) * JOULES_PER_KWH + comm_j,
    }


def _final_macro_gain(rounds: list[dict]) -> dict[str, float]:
    if not rounds:
        return {}
    macro_gain = rounds[-1].get("collaboration_gain", {}).get("macro_gain", {})
    return macro_gain if isinstance(macro_gain, dict) else {}


def _divide(gain: dict[str, float], denominator: float) -> Optional[dict[str, float]]:
    """None rather than 0.0 when the denominator is absent, so an
    un-instrumented run reads as "not measured" instead of "measured zero".
    """
    if not gain or denominator <= 0:
        return None
    return {m: v / denominator for m, v in gain.items()}


def efficiency_metrics(rounds: list[dict]) -> dict:
    """Appendix A's gain-per-cost metrics, all built from the final round's
    macro collaboration gain (mesh macro accuracy − local-only macro
    accuracy across every node) divided by a run-total cost.
    """
    totals = energy_totals(rounds)
    final_gain = _final_macro_gain(rounds)
    return {
        "final_round_macro_gain": final_gain,
        "gain_per_joule": _divide(final_gain, totals["total_mesh_energy_j"]),
        "gain_per_additional_joule": _divide(final_gain, totals["additional_energy_j"]),
        "gain_per_byte": _divide(final_gain, float(totals["total_bytes_exchanged"])),
        "formula": (
            "final-round macro collaboration gain (mesh macro accuracy - local-only macro accuracy, "
            "per metric, across all nodes) divided by: total mesh compute+communication energy in J "
            "(gain_per_joule); the same total minus the local-only arm's compute energy, i.e. the extra "
            "energy collaboration cost (gain_per_additional_joule, the Baseline B reading); and total "
            "bytes exchanged over the run (gain_per_byte)."
        ),
    }


def summarise_scenario(report: dict, disruption_start_round: int, tolerance: float = 0.05) -> dict:
    """The full derived-metric block for one scenario report: adaptation,
    retention, and the energy/communication efficiency of the gain.
    """
    rounds = report.get("rounds", [])
    node_id = report.get("target_node", "")
    return {
        "adaptation": adaptation_gain(rounds, node_id, disruption_start_round),
        "retention": retention_gain(rounds, node_id, disruption_start_round),
        "efficiency": efficiency_metrics(rounds),
        "energy": energy_totals(rounds),
        "recovery_tolerance": tolerance,
    }
