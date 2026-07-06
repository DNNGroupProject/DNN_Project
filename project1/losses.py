"""Loss functions for binary segmentation."""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Soft Dice loss computed on raw logits."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs   = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        inter   = (probs * targets).sum()
        dice    = (2.0 * inter + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """Weighted sum of Binary Cross-Entropy and Dice loss.

    Both are computed on raw logits.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.bce         = nn.BCEWithLogitsLoss()
        self.dice        = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (self.bce_weight  * self.bce(logits, targets) +
                self.dice_weight * self.dice(logits, targets))
