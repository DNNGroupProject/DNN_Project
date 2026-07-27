"""
CPU smoke-scale training for the explainability-guided SegFormer-B0
pipeline — trains BOTH the vanilla baseline (Person 2's Week 1-2
deliverable) and the SegFormer-B0 + Attention Consistency Loss variant
(Person 3's Week 3-4 "early partial training pass if time allows") on a
small subset of the shared dataset, entirely on CPU.

This mirrors the team's own convention (see Kalana-Person2/segformer_baseline.ipynb:
"verified locally on 10 images/5 epochs... this notebook just scales it up
with a real GPU"): scope here is a deliberately small, real, CPU-feasible
run to prove the whole pipeline (data -> model -> attention hooks ->
Grad-Rollout -> Attention Consistency Loss -> combined backward -> real
Dice/IoU numbers) end-to-end. See segformer_full_scale_colab.ipynb for the
notebook that scales this to the full ~5,000-image dataset / more epochs
on a GPU, matching proposal §4.1.

Usage:
    python train_segformer_smoke.py --variant vanilla
    python train_segformer_smoke.py --variant att
    python train_segformer_smoke.py --variant both   # default
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from attention_consistency import (
    AttentionConsistencyLoss,
    build_segformer,
    grad_rollout_attention_map,
)
from attention_consistency.data import load_pairs, make_splits, to_model_input
from attention_consistency.segformer_model import forest_prob

HERE = Path(__file__).resolve().parent
CKPT_DIR = HERE / "checkpoints"
RESULTS_DIR = HERE / "results"
CKPT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def dice_bce(probs: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    """L_dice, L_bce — proposal §3.3, computed at full 256x256 resolution."""
    inter = (probs * target).sum(dim=(1, 2))
    union = probs.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    l_dice = (1 - (2 * inter + eps) / (union + eps)).mean()
    l_bce = F.binary_cross_entropy(probs.clamp(eps, 1 - eps), target, reduction="mean")
    return l_dice, l_bce


@torch.no_grad()
def batch_dice_iou(probs: torch.Tensor, target: torch.Tensor, thr: float = 0.5, eps: float = 1e-6):
    pred = (probs > thr).float()
    inter = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = ((2 * inter + eps) / (union + eps)).mean().item()
    iou = ((inter + eps) / (union - inter + eps)).mean().item()
    return dice, iou


def iterate_batches(images, masks, batch_size, shuffle, seed=0):
    n = len(images)
    idx = list(range(n))
    if shuffle:
        import random

        random.Random(seed).shuffle(idx)
    for i in range(0, n, batch_size):
        sel = idx[i : i + batch_size]
        yield to_model_input(images[sel]), torch.from_numpy(masks[sel])


def run_epoch_vanilla(model, images, masks, batch_size, opt=None):
    is_train = opt is not None
    model.train() if is_train else model.eval()
    tot_loss = tot_dice = tot_iou = 0.0
    n_batches = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x, y in iterate_batches(images, masks, batch_size, shuffle=is_train):
            if is_train:
                opt.zero_grad()
            outputs = model(pixel_values=x)
            logits_up = F.interpolate(outputs.logits, size=(256, 256), mode="bilinear", align_corners=False)
            probs = forest_prob(logits_up)
            l_dice, l_bce = dice_bce(probs, y)
            loss = l_dice + 1.0 * l_bce
            if is_train:
                loss.backward()
                opt.step()
            dice, iou = batch_dice_iou(probs.detach(), y)
            tot_loss += loss.item()
            tot_dice += dice
            tot_iou += iou
            n_batches += 1
    return tot_loss / n_batches, tot_dice / n_batches, tot_iou / n_batches


def run_epoch_attention(model, images, masks, opt, att_loss_fn, lambda2, is_train):
    model.train()  # Grad-Rollout needs a live graph even in "eval" mode
    tot_loss = tot_dice = tot_iou = tot_att = 0.0
    n = len(images)
    for i in range(n):  # batch size 1 — see rollout.py's batch-size-1 constraint
        x, y = to_model_input(images[i : i + 1]), torch.from_numpy(masks[i : i + 1])
        if is_train:
            opt.zero_grad()
        attn_map, outputs = grad_rollout_attention_map(model, x, create_graph=is_train)
        logits_up = F.interpolate(outputs.logits, size=(256, 256), mode="bilinear", align_corners=False)
        probs = forest_prob(logits_up)
        l_dice, l_bce = dice_bce(probs, y)
        l_att = att_loss_fn(attn_map, y[0])
        loss = l_dice + 1.0 * l_bce + lambda2 * l_att
        if is_train:
            loss.backward()
            opt.step()
        dice, iou = batch_dice_iou(probs.detach(), y)
        tot_loss += loss.item()
        tot_dice += dice
        tot_iou += iou
        tot_att += l_att.item()
    return tot_loss / n, tot_dice / n, tot_iou / n, tot_att / n


def train_variant(variant: str, args) -> None:
    print(f"\n{'='*60}\nTraining variant: {variant}\n{'='*60}")
    splits = make_splits(args.n_train, args.n_val, args.n_test, seed=args.seed)
    train_imgs, train_masks = load_pairs(splits["train"])
    val_imgs, val_masks = load_pairs(splits["val"])
    print(f"train={len(train_imgs)} val={len(val_imgs)} (test split reserved for eval_segformer.py)")

    model = build_segformer(pretrained=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    att_loss_fn = AttentionConsistencyLoss(mode=args.att_mode, sigma=args.sigma)

    log_path = RESULTS_DIR / f"training_log_{variant}.csv"
    fieldnames = ["epoch", "train_loss", "train_dice", "train_iou", "val_loss", "val_dice", "val_iou"]
    if variant == "att":
        fieldnames += ["train_l_att", "val_l_att"]

    best_val_dice = -1.0
    rows = []
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        if variant == "vanilla":
            tr_loss, tr_dice, tr_iou = run_epoch_vanilla(model, train_imgs, train_masks, args.batch_size, opt)
            val_loss, val_dice, val_iou = run_epoch_vanilla(model, val_imgs, val_masks, args.batch_size, opt=None)
            row = dict(epoch=epoch, train_loss=tr_loss, train_dice=tr_dice, train_iou=tr_iou,
                       val_loss=val_loss, val_dice=val_dice, val_iou=val_iou)
        else:
            tr_loss, tr_dice, tr_iou, tr_att = run_epoch_attention(
                model, train_imgs, train_masks, opt, att_loss_fn, args.lambda2, is_train=True
            )
            val_loss, val_dice, val_iou, val_att = run_epoch_attention(
                model, val_imgs, val_masks, opt, att_loss_fn, args.lambda2, is_train=False
            )
            row = dict(epoch=epoch, train_loss=tr_loss, train_dice=tr_dice, train_iou=tr_iou,
                       val_loss=val_loss, val_dice=val_dice, val_iou=val_iou,
                       train_l_att=tr_att, val_l_att=val_att)
        rows.append(row)
        dt = time.time() - t0
        print(f"epoch {epoch:02d}/{args.epochs} | train_dice={tr_dice:.4f} val_dice={val_dice:.4f} "
              f"val_iou={val_iou:.4f} | {dt:.1f}s")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch, "val_dice": val_dice, "variant": variant},
                CKPT_DIR / f"segformer_b0_{variant}_best.pt",
            )

    torch.save(
        {"model_state": model.state_dict(), "epoch": args.epochs, "val_dice": val_dice, "variant": variant},
        CKPT_DIR / f"segformer_b0_{variant}_last.pt",
    )

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    summary = {
        "variant": variant,
        "n_train": len(train_imgs),
        "n_val": len(val_imgs),
        "epochs": args.epochs,
        "batch_size": args.batch_size if variant == "vanilla" else 1,
        "lr": args.lr,
        "lambda2": args.lambda2 if variant == "att" else None,
        "att_mode": args.att_mode if variant == "att" else None,
        "best_val_dice": best_val_dice,
        "final_val_dice": val_dice,
        "final_val_iou": val_iou,
        "wall_clock_s": round(time.time() - t_start, 1),
    }
    (RESULTS_DIR / f"train_summary_{variant}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {log_path}")
    print(f"Wrote checkpoints/segformer_b0_{variant}_best.pt (val_dice={best_val_dice:.4f})")
    print(json.dumps(summary, indent=2))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="both", choices=["vanilla", "att", "both"])
    p.add_argument("--n-train", type=int, default=400)
    p.add_argument("--n-val", type=int, default=60)
    p.add_argument("--n-test", type=int, default=60)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4, help="vanilla variant only; att variant is batch=1")
    p.add_argument("--lr", type=float, default=6e-5)
    p.add_argument("--lambda2", type=float, default=0.3)
    p.add_argument("--sigma", type=float, default=8.0)
    p.add_argument("--att-mode", default="mse", choices=["mse", "kl"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    variants = ["vanilla", "att"] if args.variant == "both" else [args.variant]
    for v in variants:
        train_variant(v, args)


if __name__ == "__main__":
    main()
