"""Segmentation evaluation metrics (all computed on raw logits)."""

import torch


def compute_metrics(logits: torch.Tensor,
                    targets: torch.Tensor,
                    threshold: float = 0.5) -> dict[str, float]:
    """Return a dict of segmentation metrics.

    Parameters
    ----------
    logits  : raw model output  (B, 1, H, W)
    targets : ground-truth mask (B, 1, H, W), values ∈ {0, 1}
    threshold : decision threshold applied to sigmoid probabilities

    Returns
    -------
    dict with keys: pixel_acc, precision, recall, f1, iou, dice
    """
    probs   = torch.sigmoid(logits)
    preds   = (probs > threshold).float()
    targets = targets.float()

    TP = (preds * targets).sum()
    TN = ((1 - preds) * (1 - targets)).sum()
    FP = (preds * (1 - targets)).sum()
    FN = ((1 - preds) * targets).sum()

    eps = 1e-6

    pixel_acc = (TP + TN) / (TP + TN + FP + FN + eps)
    precision = TP / (TP + FP + eps)
    recall    = TP / (TP + FN + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    iou       = TP / (TP + FP + FN + eps)          # Jaccard index
    dice      = (2 * TP) / (2 * TP + FP + FN + eps)

    return {
        'pixel_acc': pixel_acc.item(),
        'precision': precision.item(),
        'recall':    recall.item(),
        'f1':        f1.item(),
        'iou':       iou.item(),
        'dice':      dice.item(),
    }


def aggregate_metrics(metric_list: list[dict]) -> dict[str, float]:
    """Average a list of per-batch metric dicts."""
    keys = metric_list[0].keys()
    return {k: sum(m[k] for m in metric_list) / len(metric_list) for k in keys}
