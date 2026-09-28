"""Detection metrics for the single-class leaf detector: average precision
at an IoU threshold (VOC-style, all-point interpolation), plus the
precision and recall at the detector's operating score threshold.

With one class, mAP@0.5 is just AP@0.5 for "leaf". The COCO-style
mAP@[0.5:0.95] is reported too, as the mean AP over IoU thresholds
0.50, 0.55, ..., 0.95.
"""

from __future__ import annotations

import torch
from torchvision.ops import box_iou


def match_detections(
    pred_boxes: torch.Tensor, pred_scores: torch.Tensor, gt_boxes: torch.Tensor, iou_threshold: float
) -> torch.Tensor:
    """Greedy VOC matching for one image: predictions in descending score
    order each claim the unclaimed ground-truth box they overlap most, if
    that overlap reaches `iou_threshold`. Returns a bool true-positive flag
    per prediction, in the same order as `pred_boxes`.
    """
    is_tp = torch.zeros(len(pred_boxes), dtype=torch.bool)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return is_tp
    ious = box_iou(pred_boxes, gt_boxes)
    claimed = torch.zeros(len(gt_boxes), dtype=torch.bool)
    for i in torch.argsort(pred_scores, descending=True).tolist():
        candidate_ious = ious[i].clone()
        candidate_ious[claimed] = -1.0
        best_iou, best_gt = candidate_ious.max(dim=0)
        if best_iou.item() >= iou_threshold:
            is_tp[i] = True
            claimed[best_gt] = True
    return is_tp


def average_precision(predictions: list[dict], targets: list[dict], iou_threshold: float = 0.5) -> float:
    """predictions[i] = {"boxes": (P, 4), "scores": (P,)} and
    targets[i] = {"boxes": (G, 4)} for image i. Returns AP in [0, 1]:
    the area under the precision-recall curve with precision made
    monotonically non-increasing (VOC2010+ all-point interpolation).
    """
    total_gt = sum(len(t["boxes"]) for t in targets)
    if total_gt == 0:
        return 0.0

    all_scores, all_tp = [], []
    for pred, target in zip(predictions, targets):
        all_scores.append(pred["scores"].float().cpu())
        all_tp.append(match_detections(pred["boxes"].cpu(), pred["scores"].cpu(), target["boxes"].cpu(), iou_threshold))
    if not all_scores or sum(len(s) for s in all_scores) == 0:
        return 0.0
    scores = torch.cat(all_scores)
    tp = torch.cat(all_tp).float()[torch.argsort(scores, descending=True)]

    cum_tp = torch.cumsum(tp, dim=0)
    cum_fp = torch.cumsum(1 - tp, dim=0)
    recall = cum_tp / total_gt
    precision = cum_tp / (cum_tp + cum_fp)

    recall = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    precision = torch.cat([torch.tensor([1.0]), precision, torch.tensor([0.0])])
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = torch.maximum(precision[i], precision[i + 1])
    return float(torch.sum((recall[1:] - recall[:-1]) * precision[1:]))


def detection_report(predictions: list[dict], targets: list[dict], score_threshold: float = 0.5) -> dict:
    """AP@0.5, AP@0.75, mAP@[0.5:0.95], and precision/recall of the
    predictions kept at `score_threshold` (what inference actually uses).
    """
    iou_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    aps = {t: average_precision(predictions, targets, t) for t in iou_thresholds}

    kept_tp = kept = 0
    for pred, target in zip(predictions, targets):
        keep = pred["scores"] >= score_threshold
        tp = match_detections(pred["boxes"][keep].cpu(), pred["scores"][keep].cpu(), target["boxes"].cpu(), 0.5)
        kept_tp += int(tp.sum())
        kept += int(keep.sum())
    total_gt = sum(len(t["boxes"]) for t in targets)

    return {
        "map_50": round(aps[0.5], 4),
        "map_75": round(aps[0.75], 4),
        "map_50_95": round(sum(aps.values()) / len(aps), 4),
        "precision_at_threshold": round(kept_tp / kept, 4) if kept else 0.0,
        "recall_at_threshold": round(kept_tp / total_gt, 4) if total_gt else 0.0,
        "score_threshold": score_threshold,
        "num_images": len(targets),
        "num_gt_boxes": total_gt,
    }
