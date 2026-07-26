"""
Train ViT + Concept-Guided XAI forest segmentation.

Loss (paper arXiv:2101.03919 adapted to segmentation):
    L = L_A + λ (L_u + L_m)
where
    L_A = BCE+Dice segmentation (+ aux concept-class CE)
    L_u = concept uniqueness
    L_m = mapping consistency (semantic + counter-image)
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
from dataset import ForestConceptDataset, list_pairs
from model import ViTConceptSeg, total_loss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tp = fp = fn = tn = 0.0
    total_loss_sum = 0.0
    n = 0
    for images, masks, z, _ in loader:
        images = images.to(device)
        masks = masks.to(device)
        z = z.to(device)
        # counter batch = shuffle within batch (paper)
        perm = torch.randperm(images.size(0), device=device)
        out = model(images)
        out_c = model(images[perm])
        losses = total_loss(out, masks, z, out_c, z[perm], model)
        total_loss_sum += float(losses["loss"]) * images.size(0)
        n += images.size(0)

        pred = (torch.sigmoid(out["seg_logits"]) > 0.5).float()
        tp += (pred * masks).sum().item()
        fp += (pred * (1 - masks)).sum().item()
        fn += ((1 - pred) * masks).sum().item()
        tn += ((1 - pred) * (1 - masks)).sum().item()

    eps = 1e-6
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    return {
        "loss": total_loss_sum / max(n, 1),
        "iou": iou,
        "dice": dice,
        "acc": acc,
    }


def main():
    set_seed(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("IMG_DIR:", cfg.IMG_DIR)
    assert cfg.IMG_DIR.is_dir() and cfg.MASK_DIR.is_dir(), (
        "Forest dataset not found. Expected under Lasana/dataset/..."
    )

    pairs = list_pairs(str(cfg.IMG_DIR), str(cfg.MASK_DIR), cfg.MAX_SAMPLES)
    print(f"Pairs: {len(pairs)}")
    assert len(pairs) > 10

    idx = list(range(len(pairs)))
    tr_idx, te_idx = train_test_split(idx, test_size=0.10, random_state=cfg.SEED)
    val_frac = cfg.VAL_RATIO / (cfg.TRAIN_RATIO + cfg.VAL_RATIO)
    tr_idx, va_idx = train_test_split(tr_idx, test_size=val_frac, random_state=cfg.SEED)

    train_ds = ForestConceptDataset([pairs[i] for i in tr_idx], cfg.IMG_SIZE, augment=True)
    val_ds = ForestConceptDataset([pairs[i] for i in va_idx], cfg.IMG_SIZE, augment=False)
    test_ds = ForestConceptDataset([pairs[i] for i in te_idx], cfg.IMG_SIZE, augment=False)
    print(f"Train {len(train_ds)} | Val {len(val_ds)} | Test {len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)

    model = ViTConceptSeg().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params/1e6:.2f}M")
    print("Concepts:", cfg.CONCEPT_NAMES)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.NUM_EPOCHS, eta_min=1e-6)

    best_iou = -1.0
    best_path = cfg.CHECKPOINT_DIR / "vit_concept_seg_best.pt"
    history = []

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        model.train()
        running = {"loss": 0.0, "L_A": 0.0, "L_u": 0.0, "L_m": 0.0}
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.NUM_EPOCHS}", leave=False)
        for images, masks, z, _ in pbar:
            images = images.to(device)
            masks = masks.to(device)
            z = z.to(device)
            perm = torch.randperm(images.size(0), device=device)

            out = model(images)
            out_c = model(images[perm])
            losses = total_loss(out, masks, z, out_c, z[perm], model)

            opt.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            for k in running:
                running[k] += float(losses[k] if k != "loss" else losses["loss"]) * images.size(0)
            pbar.set_postfix(loss=float(losses["loss"]))

        sched.step()
        n_tr = len(train_ds)
        train_stats = {k: v / n_tr for k, v in running.items()}
        val_stats = evaluate(model, val_loader, device)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_stats.items()}, **{f"val_{k}": v for k, v in val_stats.items()}}
        history.append(row)
        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_stats['loss']:.4f} "
            f"(A={train_stats['L_A']:.3f} U={train_stats['L_u']:.3f} M={train_stats['L_m']:.3f}) | "
            f"val_iou={val_stats['iou']:.4f} dice={val_stats['dice']:.4f} "
            f"acc={val_stats['acc']:.4f} | {time.time()-t0:.0f}s"
        )

        if val_stats["iou"] > best_iou:
            best_iou = val_stats["iou"]
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "val_iou": best_iou,
                    "concept_names": cfg.CONCEPT_NAMES,
                    "config": {
                        "embed_dim": cfg.EMBED_DIM,
                        "depth": cfg.DEPTH,
                        "num_heads": cfg.NUM_HEADS,
                        "img_size": cfg.IMG_SIZE,
                        "patch_size": cfg.PATCH_SIZE,
                    },
                },
                best_path,
            )
            print(f"  ✓ saved best → {best_path} (val IoU={best_iou:.4f})")

    # Test
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_stats = evaluate(model, test_loader, device)
    print("\n=== Test (best checkpoint) ===")
    for k, v in test_stats.items():
        print(f"{k:8s}: {v:.4f}")

    out = {
        "test": test_stats,
        "best_val_iou": best_iou,
        "history": history,
        "concepts": cfg.CONCEPT_NAMES,
        "paper": "arXiv:2101.03919 — adapted to ViT + forest segmentation",
    }
    with open(cfg.RESULTS_DIR / "train_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Saved", cfg.RESULTS_DIR / "train_results.json")


if __name__ == "__main__":
    main()
