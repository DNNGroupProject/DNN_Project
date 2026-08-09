"""The U-Net baseline model, lifted out of `unet_baseline_colab.ipynb` (Step 4).

Same code, just importable. Until now the architecture only existed inside a
notebook cell, which meant nothing outside Colab could rebuild it -- including
`augmentation_ablation.py` here and the `adapters/unet_torch.py` that Person 4
needs in order to load this baseline's checkpoint
(see `unet_baseline_reconciliation.md`).

Don't "improve" anything in here. The committed numbers in `test_metrics.txt`
(31.04 M params, 109.48 GFLOPs, test Dice 0.8563) came from exactly this
definition, and the notebook is a structural clone of Kalana's SegFormer run
so that the two are comparable. A change here quietly invalidates both.

    python -c "from unet_model import UNet; print(sum(p.numel() for p in UNet().parameters()))"
    31037698
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv3x3 -> BN -> ReLU) x 2 — the standard U-Net block."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Vanilla U-Net: 4 encoder stages, a 2x-width bottleneck, transposed-conv
    decoder with skip connections, 1x1 head.

    The head emits **2 channels, not 1**, so F.cross_entropy and argmax stay
    identical to the SegFormer baseline instead of switching to sigmoid + BCE.
    """

    def __init__(self, in_channels=3, out_channels=2, features=(64, 128, 256, 512)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        c = in_channels
        for f in features:
            self.downs.append(DoubleConv(c, f))
            c = f

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder: upsample, concat skip, then DoubleConv
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(f * 2, f))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i // 2]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat((skip, x), dim=1)
            x = self.ups[i + 1](x)

        return self.final_conv(x)


def build_model(device="cpu", features=(64, 128, 256, 512)):
    """The notebook's build_model, with the device passed in instead of read
    from a global. `features` is a knob so the ablation can run a narrower
    U-Net on CPU; leave it at the default to get the committed baseline."""
    model = UNet(in_channels=3, out_channels=2, features=tuple(features))
    return model.to(device)


def dice_iou_score(pred_mask, true_mask, eps=1e-6):
    """pred_mask, true_mask: bool tensors, same shape. Dice/IoU for the
    'forest' (positive) class."""
    intersection = (pred_mask & true_mask).sum().float()
    pred_sum = pred_mask.sum().float()
    true_sum = true_mask.sum().float()
    union = pred_sum + true_sum - intersection

    dice = (2 * intersection + eps) / (pred_sum + true_sum + eps)
    iou = (intersection + eps) / (union + eps)
    return dice.item(), iou.item()
