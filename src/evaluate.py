"""Collaboration-gain metrics: macro-average and worst-node scores for
the mesh vs. an aligned-budget local-only baseline, following the
non-IID evaluation practice the accompanying report cites (Zhu et al.;
Q. Li et al.) — a single macro-average can hide a mesh that helps
already-strong nodes while leaving a weak node behind, so both are
always reported.
"""

from __future__ import annotations
from src.metrics import pool_confusion_matrices


def scalar_metrics(eval_: dict) -> dict[str, float]:
    return {k: v for k, v in eval_.items() if isinstance(v, (int, float))}


def pooled_metrics(evals: dict[str, dict]) -> dict:
    # "detail" (per-class confusion matrices) is only present on evals
    # produced by src.federated.node.Node.evaluate() -- callers scoring
    # exported ONNX models (e.g. src.validation.evaluate_onnx's report
    # "summary" blocks) pass a flatter dict with no "detail" key, so pooled
    # (confusion-matrix-based) metrics simply aren't available for those
    # and are skipped rather than raising.
    if not evals or "detail" not in next(iter(evals.values())):
        return {}
    heads = next(iter(evals.values()))["detail"].keys()
    return {
        head: pool_confusion_matrices([e["detail"][head]["confusion_matrix"] for e in evals.values()])
        for head in heads
    }


def pooled_gain(mesh_pooled: dict, baseline_pooled: dict) -> dict:
    if not mesh_pooled or not baseline_pooled:
        return {}
    gain = {}
    for head, mesh_head in mesh_pooled.items():
        baseline_head = baseline_pooled[head]
        gain[head] = {
            "accuracy": mesh_head["accuracy"] - baseline_head["accuracy"],
            "macro_precision": mesh_head["macro_precision"] - baseline_head["macro_precision"],
            "macro_recall": mesh_head["macro_recall"] - baseline_head["macro_recall"],
            "macro_f1": mesh_head["macro_f1"] - baseline_head["macro_f1"],
            "per_class": {
                name: {
                    metric: mesh_head["per_class"][name][metric] - baseline_head["per_class"][name][metric]
                    for metric in ("precision", "recall", "f1")
                }
                for name in mesh_head["per_class"]
            },
        }
    return gain


def macro_average(evals: dict[str, dict[str, float]]) -> dict[str, float]:
    if not evals:
        return {}
    scalar_evals = {k: scalar_metrics(v) for k, v in evals.items()}
    metrics = next(iter(scalar_evals.values())).keys()
    return {m: sum(e[m] for e in scalar_evals.values()) / len(scalar_evals) for m in metrics}


def worst_node(evals: dict[str, dict[str, float]]) -> dict[str, float]:
    if not evals:
        return {}
    scalar_evals = {k: scalar_metrics(v) for k, v in evals.items()}
    metrics = next(iter(scalar_evals.values())).keys()
    return {m: min(e[m] for e in scalar_evals.values()) for m in metrics}


def extract_global(evals: dict[str, dict]) -> dict[str, dict]:
    return {node_id: e["global"] for node_id, e in evals.items() if "global" in e}


def compute_collaboration_gain(
    mesh_evals: dict[str, dict[str, float]],
    baseline_evals: dict[str, dict[str, float]],
) -> dict:
    mesh_macro = macro_average(mesh_evals)
    baseline_macro = macro_average(baseline_evals)
    mesh_worst = worst_node(mesh_evals)
    baseline_worst = worst_node(baseline_evals)
    mesh_pooled = pooled_metrics(mesh_evals)
    baseline_pooled = pooled_metrics(baseline_evals)

    result = {
        "mesh_macro": mesh_macro,
        "baseline_macro": baseline_macro,
        "macro_gain": {m: mesh_macro[m] - baseline_macro[m] for m in mesh_macro},
        "mesh_worst_node": mesh_worst,
        "baseline_worst_node": baseline_worst,
        "worst_node_gain": {m: mesh_worst[m] - baseline_worst[m] for m in mesh_worst},
        "mesh_pooled": mesh_pooled,
        "baseline_pooled": baseline_pooled,
        "pooled_gain": pooled_gain(mesh_pooled, baseline_pooled),
        "per_node": {
            node_id: {
                "mesh": mesh_evals.get(node_id, {}),
                "baseline": baseline_evals.get(node_id, {}),
            }
            for node_id in mesh_evals
        },
    }

    # Same gain math, but on the global_test_loader eval instead of each
    # node's own skewed local split — this is the number that actually
    # answers "did mesh consensus (prototypes + soft logits, no raw data
    # exchanged) give nodes real knowledge of classes their non-IID shard
    # barely covered, more than local-only training would have?"
    mesh_global = extract_global(mesh_evals)
    baseline_global = extract_global(baseline_evals)
    if mesh_global and baseline_global:
        global_mesh_macro = macro_average(mesh_global)
        global_baseline_macro = macro_average(baseline_global)
        global_mesh_worst = worst_node(mesh_global)
        global_baseline_worst = worst_node(baseline_global)
        global_mesh_pooled = pooled_metrics(mesh_global)
        global_baseline_pooled = pooled_metrics(baseline_global)
        result.update(
            global_mesh_macro=global_mesh_macro,
            global_baseline_macro=global_baseline_macro,
            global_macro_gain={m: global_mesh_macro[m] - global_baseline_macro[m] for m in global_mesh_macro},
            global_mesh_worst_node=global_mesh_worst,
            global_baseline_worst_node=global_baseline_worst,
            global_worst_node_gain={m: global_mesh_worst[m] - global_baseline_worst[m] for m in global_mesh_worst},
            global_mesh_pooled=global_mesh_pooled,
            global_baseline_pooled=global_baseline_pooled,
            global_pooled_gain=pooled_gain(global_mesh_pooled, global_baseline_pooled),
        )
    return result
