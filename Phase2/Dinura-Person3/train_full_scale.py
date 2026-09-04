"""Full-scale SegFormer-B0 training for Person 3's Phase 2 λ2-sweep.

Copied from Phase2/Kalana-Person2/train_full_scale.py (same train_variant /
Attention Consistency Loss path). Writes under this folder's paths.CKPT_DIR /
RESULTS_DIR — run_lambda_sweep.py points those at one sweep cell per pass.

Boundary Loss (Person 5, Week 10 stretch task) is wired in as an opt-in
`--lambda3` flag, default 0.0 — with lambda3=0.0 (the default used by every
existing sweep call, including run_lambda_sweep.py's SweepArgs shim, which
carries no lambda3/boundary_kernel attribute at all) this file's behavior
and output CSV schema are unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PERSON5_DIR = Path(__file__).resolve().parents[1] / "Dhinanjaya-Person5"
if str(PERSON5_DIR) not in sys.path:
    sys.path.insert(0, str(PERSON5_DIR))
from boundary_refinement.loss import BoundaryDiceLoss  # noqa: E402

from paths import (
    ATT_MODE,
    BATCH_SIZE,
    EPOCHS,
    LAMBDA2,
    LR,
    N_TEST,
    N_TRAIN,
    N_VAL,
    SEED,
    SIGMA,
    add_teammate_paths,
    apply_data_dirs,
    ensure_output_dirs,
)
import paths

add_teammate_paths()
apply_data_dirs()

from attention_consistency import (  # noqa: E402
    AttentionConsistencyLoss,
    build_segformer,
    grad_rollout_attention_map,
)
from attention_consistency.data import load_pairs, make_splits, to_model_input  # noqa: E402
from attention_consistency.segformer_model import forest_prob  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_input(images):
    return to_model_input(images).to(DEVICE)


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
        yield _to_input(images[sel]), torch.from_numpy(masks[sel]).to(DEVICE)


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


def run_epoch_attention(
    model, images, masks, opt, att_loss_fn, lambda2, is_train, boundary_loss_fn=None, lambda3=0.0
):
    model.train()  # Grad-Rollout needs a live graph even in "eval" mode
    tot_loss = tot_dice = tot_iou = tot_att = tot_bnd = 0.0
    n = len(images)
    for i in range(n):  # batch size 1 — see Person 3 rollout.py's batch-size-1 constraint
        x, y = _to_input(images[i : i + 1]), torch.from_numpy(masks[i : i + 1]).to(DEVICE)
        if is_train:
            opt.zero_grad()
        attn_map, outputs = grad_rollout_attention_map(model, x, create_graph=is_train)
        logits_up = F.interpolate(outputs.logits, size=(256, 256), mode="bilinear", align_corners=False)
        probs = forest_prob(logits_up)
        l_dice, l_bce = dice_bce(probs, y)
        l_att = att_loss_fn(attn_map, y[0])
        loss = l_dice + 1.0 * l_bce + lambda2 * l_att
        if boundary_loss_fn is not None:
            l_bnd = boundary_loss_fn(probs, y)  # probs, y both (1,256,256): soft P, binary Y
            loss = loss + lambda3 * l_bnd
            tot_bnd += l_bnd.item()
        if is_train:
            loss.backward()
            opt.step()
        dice, iou = batch_dice_iou(probs.detach(), y)
        tot_loss += loss.item()
        tot_dice += dice
        tot_iou += iou
        tot_att += l_att.item()
    return tot_loss / n, tot_dice / n, tot_iou / n, tot_att / n, tot_bnd / n


def train_variant(variant: str, args) -> None:
    print(f"\n{'='*60}\nTraining variant: {variant}  device={DEVICE}\n{'='*60}")
    splits = make_splits(args.n_train, args.n_val, args.n_test, seed=args.seed)
    train_imgs, train_masks = load_pairs(splits["train"])
    val_imgs, val_masks = load_pairs(splits["val"])
    print(f"train={len(train_imgs)} val={len(val_imgs)} (test split reserved for eval_full_scale.py)")

    model = build_segformer(pretrained=True).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    att_loss_fn = AttentionConsistencyLoss(mode=args.att_mode, sigma=args.sigma)

    lambda3 = float(getattr(args, "lambda3", 0.0))
    boundary_kernel = int(getattr(args, "boundary_kernel", 3))
    boundary_loss_fn = BoundaryDiceLoss(kernel_size=boundary_kernel) if lambda3 > 0 else None

    ensure_output_dirs()
    log_path = paths.RESULTS_DIR / f"training_log_{variant}.csv"
    fieldnames = ["epoch", "train_loss", "train_dice", "train_iou", "val_loss", "val_dice", "val_iou"]
    if variant == "att":
        fieldnames += ["train_l_att", "val_l_att"]
        if boundary_loss_fn is not None:
            fieldnames += ["train_l_boundary", "val_l_boundary"]

    best_val_dice = -1.0
    rows = []
    t_start = time.time()
    val_dice = 0.0
    val_iou = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        if variant == "vanilla":
            tr_loss, tr_dice, tr_iou = run_epoch_vanilla(model, train_imgs, train_masks, args.batch_size, opt)
            val_loss, val_dice, val_iou = run_epoch_vanilla(model, val_imgs, val_masks, args.batch_size, opt=None)
            row = dict(
                epoch=epoch,
                train_loss=tr_loss,
                train_dice=tr_dice,
                train_iou=tr_iou,
                val_loss=val_loss,
                val_dice=val_dice,
                val_iou=val_iou,
            )
        else:
            tr_loss, tr_dice, tr_iou, tr_att, tr_bnd = run_epoch_attention(
                model, train_imgs, train_masks, opt, att_loss_fn, args.lambda2, is_train=True,
                boundary_loss_fn=boundary_loss_fn, lambda3=lambda3,
            )
            val_loss, val_dice, val_iou, val_att, val_bnd = run_epoch_attention(
                model, val_imgs, val_masks, opt, att_loss_fn, args.lambda2, is_train=False,
                boundary_loss_fn=boundary_loss_fn, lambda3=lambda3,
            )
            row = dict(
                epoch=epoch,
                train_loss=tr_loss,
                train_dice=tr_dice,
                train_iou=tr_iou,
                val_loss=val_loss,
                val_dice=val_dice,
                val_iou=val_iou,
                train_l_att=tr_att,
                val_l_att=val_att,
            )
            if boundary_loss_fn is not None:
                row["train_l_boundary"] = tr_bnd
                row["val_l_boundary"] = val_bnd
        rows.append(row)
        dt = time.time() - t0
        print(
            f"epoch {epoch:02d}/{args.epochs} | train_dice={tr_dice:.4f} val_dice={val_dice:.4f} "
            f"val_iou={val_iou:.4f} | {dt:.1f}s"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_dice": val_dice,
                    "variant": variant,
                },
                paths.CKPT_DIR / f"segformer_b0_{variant}_best.pt",
            )

    torch.save(
        {
            "model_state": model.state_dict(),
            "epoch": args.epochs,
            "val_dice": val_dice,
            "variant": variant,
        },
        paths.CKPT_DIR / f"segformer_b0_{variant}_last.pt",
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
        "lambda3": lambda3 if variant == "att" and boundary_loss_fn is not None else None,
        "boundary_kernel": boundary_kernel if variant == "att" and boundary_loss_fn is not None else None,
        "best_val_dice": best_val_dice,
        "final_val_dice": val_dice,
        "final_val_iou": val_iou,
        "wall_clock_s": round(time.time() - t_start, 1),
        "device": str(DEVICE),
        "img_dir": str(paths.DATA_IMG_DIR if not hasattr(args, "img_dir") else args.img_dir),
        "output_dir": str(paths.RESULTS_DIR),
    }
    (paths.RESULTS_DIR / f"train_summary_{variant}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Wrote {log_path}")
    print(f"Wrote checkpoints/segformer_b0_{variant}_best.pt (val_dice={best_val_dice:.4f})")
    print(json.dumps(summary, indent=2))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="both", choices=["vanilla", "att", "both"])
    p.add_argument("--n-train", type=int, default=N_TRAIN)
    p.add_argument("--n-val", type=int, default=N_VAL)
    p.add_argument("--n-test", type=int, default=N_TEST)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="vanilla variant only; att variant is batch=1")
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--lambda2", type=float, default=LAMBDA2)
    p.add_argument("--sigma", type=float, default=SIGMA)
    p.add_argument("--att-mode", default=ATT_MODE, choices=["mse", "kl"])
    p.add_argument(
        "--lambda3", type=float, default=0.0,
        help="Boundary Dice Loss weight (Person 5, Week 10). 0.0 (default) disables it entirely.",
    )
    p.add_argument("--boundary-kernel", type=int, default=3, help="Odd structuring-element size for L_boundary.")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--img-dir", type=Path, default=None)
    p.add_argument("--mask-dir", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    apply_data_dirs(args.img_dir or paths.DATA_IMG_DIR, args.mask_dir or paths.DATA_MASK_DIR)
    if args.variant in ("att", "both") and args.lambda3 > 0:
        tag = f"{paths.run_tag(args.lambda2, args.att_mode)}_bnd{args.lambda3:g}"
        paths.set_output_dirs(paths.OUTPUT_ROOT_CKPT / "runs" / tag, paths.OUTPUT_ROOT_RESULTS / "runs" / tag)
        print(f"Boundary Loss active (λ3={args.lambda3:g}) — writing to runs/{tag}/")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    variants = ["vanilla", "att"] if args.variant == "both" else [args.variant]
    for v in variants:
        train_variant(v, args)


if __name__ == "__main__":
    main()
