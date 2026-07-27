"""
Qualitative attention-drift figures (Person 2 — Transformer Lead, Week 3-4
deliverable): "2-3 qualitative attention-map figures showing attention
drifting off-target (motivation figure for the paper)".

For each of a few test images this script shows, side by side:
  image | ground-truth forest mask | vanilla-SegFormer attention | +Attention-Consistency-Loss attention

The vanilla column is the paper's motivation evidence (attention drifting
onto roads/shadows/non-forest structure); the consistency-loss column is
the proposed method's evidence that supervising attention pulls it back
onto canopy.

Usage:
    python generate_attention_figures.py --n 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from attention_consistency import build_segformer, grad_rollout_attention_map
from attention_consistency.data import list_pairs, load_pairs, to_model_input

HERE = Path(__file__).resolve().parent
CKPT_DIR = HERE / "checkpoints"
RESULTS_DIR = HERE / "results"


def load_checkpoint(variant: str) -> torch.nn.Module:
    ckpt_path = CKPT_DIR / f"segformer_b0_{variant}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"{ckpt_path} not found — run train_segformer_smoke.py --variant {variant} first."
        )
    model = build_segformer(pretrained=False)  # architecture only; weights loaded next
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded {ckpt_path.name} (epoch={ckpt['epoch']}, val_dice={ckpt['val_dice']:.4f})")
    return model


def attention_overlay(image_01: np.ndarray, attn_map: np.ndarray) -> np.ndarray:
    """Simple red-heat overlay of the attention map on the RGB image."""
    heat = plt.get_cmap("jet")(attn_map)[..., :3]
    return np.clip(0.55 * image_01 + 0.45 * heat, 0, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=3, help="number of figures to generate")
    p.add_argument("--seed", type=int, default=123, help="different seed than training splits, held-out-ish demo picks")
    args = p.parse_args()

    vanilla = load_checkpoint("vanilla")
    att = load_checkpoint("att")

    # Sample a handful of images not necessarily controlled for train/test
    # membership — this figure is illustrative (motivation + qualitative
    # evidence), the *numeric* AAMO/Dice/IoU comparison (which does respect
    # the train/val/test split) lives in eval_segformer.py.
    pairs = list_pairs(max_samples=args.n * 4, seed=args.seed)[-args.n :]
    images, masks = load_pairs(pairs)

    out_dir = RESULTS_DIR / "attention_drift_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(len(images)):
        x = to_model_input(images[i : i + 1])
        y = masks[i]

        with torch.enable_grad():
            attn_vanilla, _ = grad_rollout_attention_map(vanilla, x)
            attn_att, _ = grad_rollout_attention_map(att, x)
        attn_vanilla = attn_vanilla.numpy()
        attn_att = attn_att.numpy()

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(images[i])
        axes[0].set_title("Image")
        axes[1].imshow(y, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Ground-truth forest mask")
        axes[2].imshow(attention_overlay(images[i], attn_vanilla))
        axes[2].set_title("Vanilla SegFormer-B0\nattention (Grad-Rollout)")
        axes[3].imshow(attention_overlay(images[i], attn_att))
        axes[3].set_title("+ Attention Consistency Loss\nattention (Grad-Rollout)")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        out_path = out_dir / f"attention_drift_{i+1:02d}.png"
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print(f"Wrote {out_path}")

    print(f"\n{len(images)} attention-drift figures written to {out_dir}")


if __name__ == "__main__":
    main()
