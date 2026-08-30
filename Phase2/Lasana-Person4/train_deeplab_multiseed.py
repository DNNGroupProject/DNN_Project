"""
DeepLabV3+ multi-seed training + ablation aggregation (Person 4, Weeks 7–9).

Seeds 42 / 43 / 44:
  - seed 42: evaluate the existing Phase 1 smoke checkpoint (no retrain)
  - seeds 43, 44: train fresh with the same smoke settings, save under
    Phase2/Lasana-Person4/checkpoints/

Also writes the Phase 2 ablation tables by combining:
  - single-seed U-Net / SegFormer-vanilla / SegFormer+L_att (from fold)
  - 3-seed DeepLabV3+ mean±std (this script)
  - pending Boundary Loss row

Usage
-----
    cd Phase2/Lasana-Person4
    python train_deeplab_multiseed.py
    python train_deeplab_multiseed.py --skip-train   # eval + aggregate only
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
PHASE1_P4 = PROJECT / "Phase1" / "Lasana-Person4_Evaluation"
RESULTS = ROOT / "results"
CKPT_DIR = ROOT / "checkpoints"
RESULTS.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

if str(PHASE1_P4) not in sys.path:
    sys.path.insert(0, str(PHASE1_P4))

import config  # noqa: E402  (Phase1 Person4 config)
from adapters.data import load_pairs, split_dataset  # noqa: E402
from adapters.deeplab_model import build_deeplabv3, forest_prob_from_logits  # noqa: E402
from ablation_runner import aggregate_mean_std  # noqa: E402
from metrics import ConfusionCounts, binarize, metrics_from_counts  # noqa: E402

SEEDS = [42, 43, 44]
SEED42_CKPT = PHASE1_P4 / "checkpoints" / "deeplabv3_mobilenet_best.pt"
DEEPLAB_LABEL = "DeepLabV3+ (MobileNetV3) — extra baseline"

# Smoke defaults match Phase1/train_deeplab_extra.py
MAX_SAMPLES = int(os.environ.get("DEEPLAB_MAX_SAMPLES", "400"))
EPOCHS = int(os.environ.get("DEEPLAB_EPOCHS", "5"))
BATCH_SIZE = int(os.environ.get("DEEPLAB_BATCH", "2"))
LR = float(os.environ.get("DEEPLAB_LR", "1e-4"))


def _to_tensor_images(images: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(images).permute(0, 3, 1, 2).float()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


def dice_loss_from_probs(probs: torch.Tensor, targets: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
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
        out = model(xb)["out"]
        if out.shape[-2:] != (config.IMG_SIZE, config.IMG_SIZE):
            out = F.interpolate(
                out, size=(config.IMG_SIZE, config.IMG_SIZE), mode="bilinear", align_corners=False
            )
        probs = forest_prob_from_logits(out).cpu().numpy()
        preds = binarize(probs, 0.5)
        counts.update(preds, masks[i : i + batch_size])
    return metrics_from_counts(counts)


def train_one_seed(seed: int, device: torch.device) -> Path:
    """Train DeepLabV3+ for one seed; return best-checkpoint path.

    Data split is always seed 42 (same held-out test set as Phase 1).
    Only the training RNG (weight init / shuffle) varies with ``seed``.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    images, masks = load_pairs(max_samples=MAX_SAMPLES)
    # Fixed split seed 42 — multi-seed compares training init, not data folds.
    splits = split_dataset(images, masks, seed=42)

    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]

    train_x = _to_tensor_images(X_train)
    train_y_f = torch.from_numpy(y_train).float()
    loader = DataLoader(
        TensorDataset(train_x, train_y_f),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    model = build_deeplabv3(num_classes=2, pretrained_backbone=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_iou = -1.0
    best_path = CKPT_DIR / f"deeplabv3_mobilenet_seed{seed}_best.pt"

    for epoch in range(1, EPOCHS + 1):
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
            ce = F.cross_entropy(out, yb.long())
            probs = forest_prob_from_logits(out)
            dsc = dice_loss_from_probs(probs, yb)
            loss = 0.5 * ce + 0.5 * dsc
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        val = eval_split(model, X_val, y_val, device, batch_size=BATCH_SIZE)
        print(
            f"  seed={seed} epoch {epoch}/{EPOCHS}  loss={np.mean(losses):.4f}  "
            f"val_dice={val['dice']:.4f}  val_iou={val['iou']:.4f}"
        )
        ckpt = {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "val_dice": val["dice"],
            "val_iou": val["iou"],
            "variant": "deeplabv3_mobilenet",
            "seed": seed,
            "max_samples": MAX_SAMPLES,
        }
        if val["iou"] > best_iou:
            best_iou = val["iou"]
            torch.save(ckpt, best_path)
            print(f"    saved best -> {best_path} (iou={best_iou:.4f})")

    return best_path


def evaluate_checkpoint(ckpt_path: Path, seed: int, device: torch.device) -> Dict[str, Any]:
    """Evaluate a DeepLab checkpoint on the fixed seed-42 smoke test split."""
    images, masks = load_pairs(max_samples=MAX_SAMPLES)
    splits = split_dataset(images, masks, seed=42)
    X_test, y_test = splits["test"]

    model = build_deeplabv3(num_classes=2, pretrained_backbone=False).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    metrics = eval_split(model, X_test, y_test, device, batch_size=BATCH_SIZE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    row = {
        "model": DEEPLAB_LABEL,
        "seed": seed,
        "dice": round(float(metrics["dice"]), 4),
        "iou": round(float(metrics["iou"]), 4),
        "f1": round(float(metrics["f1"]), 4),
        "precision": round(float(metrics["precision"]), 4),
        "recall": round(float(metrics["recall"]), 4),
        "pixel_acc": round(float(metrics["pixel_acc"]), 4),
        "aamo": "n/a",
        "params": n_params,
        "gflops": "n/a",
        "checkpoint": str(ckpt_path),
        "max_samples": MAX_SAMPLES,
        "status": "ok",
    }
    return row


def _ckpt_for_seed(seed: int) -> Path:
    if seed == 42:
        return SEED42_CKPT
    return CKPT_DIR / f"deeplabv3_mobilenet_seed{seed}_best.pt"


def load_folded_full_scale() -> List[Dict[str, str]]:
    path = RESULTS / "baseline_comparison_full_scale.csv"
    if not path.exists():
        from fold_full_scale_results import fold, write_tables

        write_tables(fold())
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_ablation_tables(
    deeplab_per_seed: List[Dict[str, Any]],
    folded: List[Dict[str, str]],
) -> None:
    """Combine folded single-seed rows + DeepLab multi-seed into ablation tables."""
    per_seed_rows: List[Dict[str, Any]] = []

    # Single-seed rows from the folded full-scale table (skip DeepLab — replaced below)
    for r in folded:
        model = r.get("model", "")
        if "DeepLab" in model:
            continue
        if r.get("dice") in ("-", "", None) and "Boundary" in model:
            per_seed_rows.append(
                {
                    "model": model,
                    "seed": "-",
                    "dice": "-",
                    "iou": "-",
                    "f1": "-",
                    "aamo": "pending",
                    "params": "-",
                    "gflops": "-",
                    "status": "pending_checkpoint",
                }
            )
            continue
        per_seed_rows.append(
            {
                "model": model,
                "seed": 42,
                "dice": r.get("dice", "-"),
                "iou": r.get("iou", "-"),
                "f1": r.get("f1", "-"),
                "aamo": r.get("aamo", "n/a"),
                "params": r.get("params", "n/a"),
                "gflops": r.get("gflops", "n/a"),
                "status": "single_seed_full_scale",
            }
        )

    for r in deeplab_per_seed:
        per_seed_rows.append(r)

    # Write per-seed CSV
    per_path = RESULTS / "ablation_per_seed.csv"
    fields = [
        "model",
        "seed",
        "dice",
        "iou",
        "f1",
        "precision",
        "recall",
        "pixel_acc",
        "aamo",
        "params",
        "gflops",
        "status",
        "checkpoint",
        "max_samples",
    ]
    with open(per_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in per_seed_rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"Wrote {per_path}")

    # Aggregate mean±std per model
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in per_seed_rows:
        if r.get("status") == "pending_checkpoint":
            continue
        if r.get("dice") in ("-", "", None):
            continue
        by_model.setdefault(r["model"], []).append(r)

    summary: List[Dict[str, Any]] = []
    # Preserve table order from folded rows, then DeepLab
    order = [r["model"] for r in folded]
    seen = set()
    for name in order:
        if name in seen:
            continue
        seen.add(name)
        if "Boundary" in name and name not in by_model:
            summary.append(
                {
                    "model": name,
                    "n_seeds": 0,
                    "dice": "-",
                    "iou": "-",
                    "f1": "-",
                    "aamo": "pending",
                    "params": "-",
                    "gflops": "-",
                }
            )
            continue
        if name not in by_model:
            continue
        summary.append(aggregate_mean_std(by_model[name]))

    sum_csv = RESULTS / "ablation_mean_std.csv"
    sum_md = RESULTS / "ablation_mean_std.md"
    sum_fields = ["model", "n_seeds", "dice", "iou", "f1", "aamo", "params", "gflops"]
    with open(sum_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sum_fields, extrasaction="ignore")
        w.writeheader()
        for r in summary:
            w.writerow({k: r.get(k, "") for k in sum_fields})

    lines = [
        "# Ablation results (mean ± std) — Phase 2 / Lasana-Person4",
        "",
        "Full-scale rows (U-Net, SegFormer-B0, SegFormer-B0+L_att λ2=1.0) are",
        "single-seed (seed 42) pending additional GPU-trained seeds from",
        "Person 1/2/3. DeepLabV3+ extra baseline has genuine 3-seed mean±std",
        "(seeds 42/43/44, CPU smoke: 400 samples / 5 epochs).",
        "",
        "| Model | Seeds | Dice | IoU | F1 | AAMO |",
        "|-------|-------|------|-----|----|------|",
    ]
    for r in summary:
        lines.append(
            f"| {r.get('model')} | {r.get('n_seeds')} | {r.get('dice')} | "
            f"{r.get('iou')} | {r.get('f1')} | {r.get('aamo')} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Attention-consistency config = Dinura `l2_1_mse` (λ2=1.0, MSE).",
            "- Boundary Loss row pending Person 5 integration.",
            "- DeepLabV3+ smoke numbers are not paper-scale; they demonstrate",
            "  the multi-seed aggregation pipeline Person 4 owns.",
            "- U-Net / SegFormer full-scale checkpoints live on Drive (not in git);",
            "  re-running extra seeds requires Colab GPU access from teammates.",
            "",
        ]
    )
    sum_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {sum_csv}")
    print(f"Wrote {sum_md}")

    # Also dump DeepLab per-seed JSON for easy inspection
    dl_json = RESULTS / "deeplab_multiseed.json"
    dl_json.write_text(json.dumps(deeplab_per_seed, indent=2), encoding="utf-8")
    print(f"Wrote {dl_json}")


def main() -> None:
    p = argparse.ArgumentParser(description="DeepLabV3+ multi-seed + ablation tables")
    p.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training seeds 43/44; only evaluate existing checkpoints + aggregate",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=SEEDS,
        help="Seeds to run (default: 42 43 44)",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} | max_samples={MAX_SAMPLES} | epochs={EPOCHS} | "
        f"batch={BATCH_SIZE} | seeds={args.seeds}"
    )

    deeplab_rows: List[Dict[str, Any]] = []
    for seed in args.seeds:
        ckpt = _ckpt_for_seed(seed)
        if seed != 42 and not args.skip_train:
            if not ckpt.exists():
                print(f"\n=== Training DeepLabV3+ seed={seed} ===")
                ckpt = train_one_seed(seed, device)
            else:
                print(f"\n=== Reusing existing checkpoint for seed={seed}: {ckpt} ===")
        elif seed != 42 and args.skip_train and not ckpt.exists():
            print(f"[skip] seed={seed}: no checkpoint at {ckpt}")
            continue

        if seed == 42 and not ckpt.exists():
            raise FileNotFoundError(
                f"Seed-42 DeepLab checkpoint missing: {ckpt}\n"
                "Train it first in Phase1:\n"
                "  cd Phase1/Lasana-Person4_Evaluation && python train_deeplab_extra.py"
            )

        print(f"\n=== Evaluating DeepLabV3+ seed={seed} | {ckpt} ===")
        row = evaluate_checkpoint(ckpt, seed, device)
        print(
            f"  seed={seed} test dice={row['dice']:.4f} iou={row['iou']:.4f} "
            f"f1={row['f1']:.4f}"
        )
        deeplab_rows.append(row)

    folded = load_folded_full_scale()
    write_ablation_tables(deeplab_rows, folded)
    print("\nMulti-seed DeepLab + ablation tables done.")


if __name__ == "__main__":
    main()
