"""Collaboration-gain metrics: macro-average and worst-node scores for
the mesh vs. an aligned-budget local-only baseline, following the
non-IID evaluation practice the accompanying report cites (Zhu et al.;
Q. Li et al.) — a single macro-average can hide a mesh that helps
already-strong nodes while leaving a weak node behind, so both are
always reported.
"""

from __future__ import annotations


def macro_average(evals: dict[str, dict[str, float]]) -> dict[str, float]:
    if not evals:
        return {}
    metrics = next(iter(evals.values())).keys()
    return {m: sum(e[m] for e in evals.values()) / len(evals) for m in metrics}


def worst_node(evals: dict[str, dict[str, float]]) -> dict[str, float]:
    if not evals:
        return {}
    metrics = next(iter(evals.values())).keys()
    return {m: min(e[m] for e in evals.values()) for m in metrics}


def compute_collaboration_gain(
    mesh_evals: dict[str, dict[str, float]],
    baseline_evals: dict[str, dict[str, float]],
) -> dict:
    mesh_macro = macro_average(mesh_evals)
    baseline_macro = macro_average(baseline_evals)
    mesh_worst = worst_node(mesh_evals)
    baseline_worst = worst_node(baseline_evals)

    return {
        "mesh_macro": mesh_macro,
        "baseline_macro": baseline_macro,
        "macro_gain": {m: mesh_macro[m] - baseline_macro[m] for m in mesh_macro},
        "mesh_worst_node": mesh_worst,
        "baseline_worst_node": baseline_worst,
        "worst_node_gain": {m: mesh_worst[m] - baseline_worst[m] for m in mesh_worst},
        "per_node": {
            node_id: {
                "mesh": mesh_evals.get(node_id, {}),
                "baseline": baseline_evals.get(node_id, {}),
            }
            for node_id in mesh_evals
        },
    }
