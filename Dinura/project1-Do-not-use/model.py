"""U-Net baseline for binary semantic segmentation.

Architecture  (Ronneberger et al., 2015 – adapted with BatchNorm):
  Encoder  : 4 × DoubleConv  + MaxPool  (features: 64→128→256→512)
  Bottleneck: DoubleConv (512→1024)
  Decoder  : 4 × ConvTranspose2d + skip-cat + DoubleConv
  Head     : 1×1 Conv → raw logit (1 channel, no sigmoid)

Input  : (B, 3, 256, 256)
Output : (B, 1, 256, 256)  raw logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Building blocks ──────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """(Conv2d → BN → ReLU) × 2."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ─── U-Net ────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """Vanilla U-Net with BatchNorm (binary segmentation head)."""

    def __init__(self,
                 in_channels:  int        = 3,
                 out_channels: int        = 1,
                 features:     list[int]  = None):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        self.pool       = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder    = nn.ModuleList()
        self.decoder_up = nn.ModuleList()   # transposed convolutions
        self.decoder_dc = nn.ModuleList()   # double-conv after concat

        # ── Encoder ───────────────────────────────────────────────────────────
        ch = in_channels
        for f in features:
            self.encoder.append(DoubleConv(ch, f))
            ch = f

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # ── Decoder ───────────────────────────────────────────────────────────
        for f in reversed(features):
            self.decoder_up.append(
                nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
            )
            self.decoder_dc.append(DoubleConv(f * 2, f))

        # ── Output head ───────────────────────────────────────────────────────
        self.head = nn.Conv2d(features[0], out_channels, kernel_size=1)

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []

        # Encoding path
        for enc in self.encoder:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skips.reverse()          # align with decoder depth

        # Decoding path
        for up, dc, skip in zip(self.decoder_up, self.decoder_dc, skips):
            x = up(x)
            # Guard against off-by-one from odd spatial dims
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear',
                                  align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = dc(x)

        return self.head(x)


# ─── Utility ──────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(in_channels=3, out_channels=1, features=None) -> UNet:
    from config import FEATURES, IN_CHANNELS, OUT_CHANNELS
    feat = features if features is not None else FEATURES
    model = UNet(in_channels  = in_channels  or IN_CHANNELS,
                 out_channels = out_channels or OUT_CHANNELS,
                 features     = feat)
    n = count_parameters(model)
    print(f'U-Net baseline  |  trainable parameters: {n:,}')
    return model
