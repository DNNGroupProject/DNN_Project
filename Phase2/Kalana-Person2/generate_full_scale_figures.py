"""Qualitative attention-drift figures for the full-scale checkpoints.

Same layout as Person 3's `generate_attention_figures.py`. Writes under
`Phase2/Kalana-Person2/results/attention_drift_figures/` only.
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from paths import (
    CKPT_DIR,
    DATA_IMG_DIR,
    DATA_MASK_DIR,
    RESULTS_DIR,
    add_teammate_paths,
    apply_data_dirs,
    ensure_output_dirs,
)

add_teammate_paths()
apply_data_dirs()

from attention_consistency import build_segformer, grad_rollout_attention_map  # noqa: E402
from attention_consistency.data import list_pairs, load_pairs, to_model_input  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(variant: str) -> torch.nn.Module:
    ckpt_path = CKPT_DIR / f"segformer_b0_{variant}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"{ckpt_path} not found — run train_full_scale.py --variant {variant} first."
        )
    model = build_segformer(pretrained=False)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()
    print(f"Loaded {ckpt_path.name} (epoch={ckpt['epoch']}, val_dice={ckpt['val_dice']:.4f})")
    return model


def attention_overlay(image_01: np.ndarray, attn_map: np.ndarray) -> np.ndarray:
    heat = plt.get_cmap("jet")(attn_map)[..., :3]
    return np.clip(0.55 * image_01 + 0.45 * heat, 0, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=3, help="number of figures to generate")
    p.add_argument(
        "--seed",
        type=int,
        default=123,
        help="different seed than training splits, held-out-ish demo picks",
    )
    p.add_argument("--img-dir", default=None)
    p.add_argument("--mask-dir", default=None)
    args = p.parse_args()
    apply_data_dirs(args.img_dir or DATA_IMG_DIR, args.mask_dir or DATA_MASK_DIR)

    vanilla = load_checkpoint("vanilla")
    att = load_checkpoint("att")

    pairs = list_pairs(max_samples=args.n * 4, seed=args.seed)[-args.n :]
    images, masks = load_pairs(pairs)

    ensure_output_dirs()
    out_dir = RESULTS_DIR / "attention_drift_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(len(images)):
        x = to_model_input(images[i : i + 1]).to(DEVICE)
        y = masks[i]

        with torch.enable_grad():
            attn_vanilla, _ = grad_rollout_attention_map(vanilla, x)
            attn_att, _ = grad_rollout_attention_map(att, x)
        attn_vanilla = attn_vanilla.detach().cpu().numpy()
        attn_att = attn_att.detach().cpu().numpy()

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
        out_path = out_dir / f"attention_drift_{i+1:02d}_full_scale.png"
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print(f"Wrote {out_path}")

    print(f"\n{len(images)} attention-drift figures written to {out_dir}")


if __name__ == "__main__":
    main()
