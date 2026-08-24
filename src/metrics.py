"""Per-head classification metrics beyond plain accuracy.
"""

from __future__ import annotations

import numpy as np


def _top_confusions(cm: np.ndarray, labels: list[str]) -> list[dict]:
    entries = [
        {"true": true_name, "predicted": pred_name, "count": int(cm[i, j])}
        for i, true_name in enumerate(labels)
        for j, pred_name in enumerate(labels)
        if i != j and cm[i, j] > 0
    ]
    entries.sort(key=lambda e: e["count"], reverse=True)
    return entries


def head_metrics(y_true: list[int], y_pred: list[int], class_names: list[str]) -> dict:
    # scikit-learn is imported here rather than at module scope so that the
    # reporting/summarising side of the codebase — which only re-reads
    # already-computed metrics out of JSON — stays importable without it.
    # That is what lets `python -m src.scenarios.summarise` regenerate the
    # published tables on a machine with neither sklearn nor torch installed.
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    labels = list(range(len(class_names)))
    per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    present = support > 0
    if present.any():
        macro_precision = float(per_class_precision[present].mean())
        macro_recall = float(per_class_recall[present].mean())
        macro_f1 = float(per_class_f1[present].mean())
    else:
        macro_precision = macro_recall = macro_f1 = 0.0

    per_class = {
        name: {
            "precision": float(per_class_precision[i]),
            "recall": float(per_class_recall[i]),
            "f1": float(per_class_f1[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(class_names)
    }
    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "num_classes_present": int(present.sum()),
        "num_classes_total": len(class_names),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": class_names, "matrix": cm.tolist(), "top_confusions": _top_confusions(cm, class_names)
        },
    }


def pool_confusion_matrices(confusion_matrices: list[dict]) -> dict:
    labels = confusion_matrices[0]["labels"]
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for entry in confusion_matrices:
        cm += np.array(entry["matrix"])

    row_totals = cm.sum(axis=1)
    col_totals = cm.sum(axis=0)
    per_class = {}
    for i, name in enumerate(labels):
        support = int(row_totals[i])
        true_positive = int(cm[i, i])
        precision = true_positive / col_totals[i] if col_totals[i] > 0 else 0.0
        recall = true_positive / support if support > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    total = int(cm.sum())
    present = [c for c in per_class.values() if c["support"] > 0]
    macro_precision = sum(c["precision"] for c in present) / len(present) if present else 0.0
    macro_recall = sum(c["recall"] for c in present) / len(present) if present else 0.0
    macro_f1 = sum(c["f1"] for c in present) / len(present) if present else 0.0
    return {
        "accuracy": float(np.trace(cm) / total) if total > 0 else 0.0,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "num_classes_present": len(present),
        "num_classes_total": len(per_class),
        "per_class": per_class,
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist(), "top_confusions": _top_confusions(cm, labels)},
    }
