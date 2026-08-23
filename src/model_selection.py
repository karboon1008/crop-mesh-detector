"""Shared logic for picking which trained checkpoint to use outside of
training itself — batch inference (`src/predict.py`) and the Raspberry Pi
export step (`scripts/export_for_pi.py`) both need to answer the same
question: which (architecture, node) scored best in a completed run?
"""

from __future__ import annotations

import json
from pathlib import Path


def _score(eval_: dict) -> float:
    metrics = eval_.get("global", eval_)
    return (metrics["crop_accuracy"] + metrics["disease_accuracy"]) / 2


def list_architectures(results_summary_path: Path) -> list[str]:
    """Every architecture present in a results_summary.json so far — useful
    when a full sweep was split across several separate `python -m src.train
    --arch ...` runs and you want to act on all of them at once.
    """
    results = json.loads(Path(results_summary_path).read_text())
    return list(results.keys())


def pick_best_node_for_arch(results_summary_path: Path, arch: str, eval_key: str = "mesh_eval") -> tuple[str, float]:
    """Returns the (node_id, score) with the highest average of
    crop_accuracy/disease_accuracy within one specific architecture.

    `eval_key` selects which per-node eval dict to score: "mesh_eval" for a
    completed `python -m src.train` (stage 2) run, or "local_eval" for a
    stage-1-only run (`python -m src.train_local`, see outputs/stage1_local/).
    """
    results = json.loads(Path(results_summary_path).read_text())
    if arch not in results:
        raise ValueError(f"Architecture '{arch}' not found in {results_summary_path}")
    best_score, best_node = -1.0, None
    for node_id, eval_ in results[arch][eval_key].items():
        score = _score(eval_)
        if score > best_score:
            best_score, best_node = score, node_id
    if best_node is None:
        raise ValueError(f"No {eval_key} entries found for architecture '{arch}' in {results_summary_path}")
    return best_node, best_score


def pick_best_arch_node(results_summary_path: Path, eval_key: str = "mesh_eval") -> tuple[str, str]:
    """Returns the (arch, node_id) with the highest average of
    crop_accuracy/disease_accuracy across every architecture and node in a
    results_summary.json produced by `python -m src.train`.

    `eval_key` selects which per-node eval dict to score: "mesh_eval" for a
    completed `python -m src.train` (stage 2) run, or "local_eval" for a
    stage-1-only run (`python -m src.train_local`, see outputs/stage1_local/).
    """
    results = json.loads(Path(results_summary_path).read_text())
    best_score, best_arch, best_node = -1.0, None, None
    for arch, arch_result in results.items():
        for node_id, eval_ in arch_result[eval_key].items():
            score = _score(eval_)
            if score > best_score:
                best_score, best_arch, best_node = score, arch, node_id
    if best_arch is None:
        raise ValueError(f"No {eval_key} entries found in {results_summary_path}")
    print(f"Auto-selected arch={best_arch} node={best_node} (avg test accuracy {best_score:.4f})")
    return best_arch, best_node
