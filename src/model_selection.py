"""Shared logic for picking which trained checkpoint to use outside of
training itself — batch inference (`src/predict.py`) and the Raspberry Pi
export step (`scripts/export_for_pi.py`) both need to answer the same
question: which (architecture, node) scored best in a completed run?
"""

from __future__ import annotations

import json
from pathlib import Path


def _score(eval_: dict) -> float:
    return (eval_["crop_accuracy"] + eval_["disease_accuracy"]) / 2


def list_architectures(results_summary_path: Path) -> list[str]:
    """Every architecture present in a results_summary.json so far — useful
    when a full sweep was split across several separate `python -m src.train
    --arch ...` runs and you want to act on all of them at once.
    """
    results = json.loads(Path(results_summary_path).read_text())
    return list(results.keys())


def pick_best_node_for_arch(results_summary_path: Path, arch: str) -> tuple[str, float]:
    """Returns the (node_id, score) with the highest average of
    crop_accuracy/disease_accuracy within one specific architecture.
    """
    results = json.loads(Path(results_summary_path).read_text())
    if arch not in results:
        raise ValueError(f"Architecture '{arch}' not found in {results_summary_path}")
    best_score, best_node = -1.0, None
    for node_id, eval_ in results[arch]["mesh_eval"].items():
        score = _score(eval_)
        if score > best_score:
            best_score, best_node = score, node_id
    if best_node is None:
        raise ValueError(f"No mesh_eval entries found for architecture '{arch}' in {results_summary_path}")
    return best_node, best_score


def pick_best_arch_node(results_summary_path: Path) -> tuple[str, str]:
    """Returns the (arch, node_id) with the highest average of
    crop_accuracy/disease_accuracy across every architecture and node in a
    results_summary.json produced by `python -m src.train`.
    """
    results = json.loads(Path(results_summary_path).read_text())
    best_score, best_arch, best_node = -1.0, None, None
    for arch, arch_result in results.items():
        for node_id, eval_ in arch_result["mesh_eval"].items():
            score = _score(eval_)
            if score > best_score:
                best_score, best_arch, best_node = score, arch, node_id
    if best_arch is None:
        raise ValueError(f"No mesh_eval entries found in {results_summary_path}")
    print(f"Auto-selected arch={best_arch} node={best_node} (avg test accuracy {best_score:.4f})")
    return best_arch, best_node
