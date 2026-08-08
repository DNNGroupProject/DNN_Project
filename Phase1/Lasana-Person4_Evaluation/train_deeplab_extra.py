"""
Train DeepLabV3+ (MobileNetV3-Large) extra baseline for Person 4 evaluation.

CPU-friendly smoke defaults; override with env vars for a fuller run:

    DEEPLAB_MAX_SAMPLES=1200
    DEEPLAB_EPOCHS=8
    DEEPLAB_BATCH=4

Usage
-----
    cd Phase1/Lasana-Person4_Evaluation
    python train_deeplab_extra.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import config
from adapters.data import load_pairs, split_dataset
from adapters.deeplab_model import build_deeplabv3, forest_prob_from_logits
from metrics import ConfusionCounts, binarize, metrics_from_counts


def _to_tensor_images(images: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(images).permute(0, 3, 1, 2).float()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


def dice_loss_from_probs(probs: torch.Tensor, targets: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    # probs/targets: (B,H,W)
    p = probs.reshape(probs.size(0), -1)
    t = targets.reshape(targets.size(0), -1)
    inter = (p * t).sum(dim=1)
    return (1 - (2 * inter + eps) / (p.sum(dim=1) + t.sum(dim=1) + eps)).mean()


@torch.no_grad()
def eval_split(model, images, masks, device, batch_size=4) -> dict:
    model.eval()
    counts = ConfusionCounts()
    for i in range(0, len(images), batch_size):
        xb = _to_tensor_images(images[i : i + batch_size]).to(device)
        yb = torch.from_numpy(masks[i : i + batch_size]).float().to(device)
        out = model(xb)["out"]
        if out.shape[-2:] != (config.IMG_SIZE, config.IMG_SIZE):
            out = F.interpolate(out, size=(config.IMG_SIZE, config.IMG_SIZE), mode="bilinear", align_corners=False)
        probs = forest_prob_from_logits(out).cpu().numpy()
        preds = binarize(probs, 0.5)
        counts.update(preds, masks[i : i + batch_size])
    return metrics_from_counts(counts)


def main():
    seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)

    max_samples = int(os.environ.get("DEEPLAB_MAX_SAMPLES", "400"))
    epochs = int(os.environ.get("DEEPLAB_EPOCHS", "5"))
    batch_size = int(os.environ.get("DEEPLAB_BATCH", "2"))
    lr = float(os.environ.get("DEEPLAB_LR", "1e-4"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} | max_samples={max_samples} | epochs={epochs} | batch={batch_size}")

    images, masks = load_pairs(max_samples=max_samples)
    splits = split_dataset(images, masks)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]

    train_x = _to_tensor_images(X_train)
    train_y = torch.from_numpy(y_train).long()  # class indices 0/1 for CE — but we use soft BCE+Dice on forest channel
    # Use float masks for BCE
    train_y_f = torch.from_numpy(y_train).float()
    loader = DataLoader(
        TensorDataset(train_x, train_y_f),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,  # BatchNorm in ASPP fails on batch size 1
    )

    model = build_deeplabv3(num_classes=2, pretrained_backbone=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_iou = -1.0
    best_path = config.DEEPLAB_CKPT
    last_path = config.ROOT / "checkpoints" / "deeplabv3_mobilenet_last.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)["out"]
            if out.shape[-2:] != (config.IMG_SIZE, config.IMG_SIZE):
                out = F.interpolate(
                    out, size=(config.IMG_SIZE, config.IMG_SIZE), mode="bilinear", align_corners=False
                )
            # CE on 2-class logits
            ce = F.cross_entropy(out, yb.long())
            probs = forest_prob_from_logits(out)
            dsc = dice_loss_from_probs(probs, yb)
            loss = 0.5 * ce + 0.5 * dsc
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        val = eval_split(model, X_val, y_val, device, batch_size=batch_size)
        print(
            f"epoch {epoch}/{epochs}  loss={np.mean(losses):.4f}  "
            f"val_dice={val['dice']:.4f}  val_iou={val['iou']:.4f}"
        )

        ckpt = {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "val_dice": val["dice"],
            "val_iou": val["iou"],
            "variant": "deeplabv3_mobilenet",
            "seed": seed,
            "max_samples": max_samples,
        }
        torch.save(ckpt, last_path)
        if val["iou"] > best_iou:
            best_iou = val["iou"]
            torch.save(ckpt, best_path)
            print(f"  saved best -> {best_path} (iou={best_iou:.4f})")

    print("Done.")
    print(f"Best checkpoint: {best_path}")
    print("Evaluate with:  python evaluate.py --model deeplab")


if __name__ == "__main__":
    main()
