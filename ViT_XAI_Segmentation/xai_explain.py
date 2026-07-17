"""
XAI explanations for ViTConceptSeg.

Produces:
  1) ViT attention rollout map (transformer-native XAI)
  2) Per-concept activation maps (paper concept layer)
  3) Concept contribution percentages to forest decision (paper §III-C)
  4) Side-by-side figure: image | GT | pred | attention | top concepts
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config as cfg
from dataset import ForestConceptDataset, list_pairs
from model import ViTConceptSeg
from sklearn.model_selection import train_test_split


def overlay(rgb, heat, alpha=0.45):
    """rgb HxWx3 [0,1], heat HxW [0,1] → overlay"""
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
    cmap = plt.cm.jet(heat)[..., :3]
    return np.clip((1 - alpha) * rgb + alpha * cmap, 0, 1)


@torch.no_grad()
def explain(n_show: int = 4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = cfg.CHECKPOINT_DIR / "vit_concept_seg_best.pt"
    assert ckpt_path.exists(), f"Train first. Missing {ckpt_path}"

    pairs = list_pairs(str(cfg.IMG_DIR), str(cfg.MASK_DIR), cfg.MAX_SAMPLES)
    idx = list(range(len(pairs)))
    _, te_idx = train_test_split(idx, test_size=0.10, random_state=cfg.SEED)
    test_ds = ForestConceptDataset([pairs[i] for i in te_idx], cfg.IMG_SIZE, augment=False)
    loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    model = ViTConceptSeg().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    out_dir = cfg.RESULTS_DIR / "explanations"
    out_dir.mkdir(exist_ok=True)

    shown = 0
    for images, masks, z, rgb in loader:
        if shown >= n_show:
            break
        images = images.to(device)
        out = model(images)
        pred = torch.sigmoid(out["seg_logits"])[0, 0].cpu().numpy()
        gt = masks[0, 0].numpy()
        rgb_np = rgb[0].numpy().transpose(1, 2, 0)

        # Attention
        attn = model.attention_rollout()
        if attn is not None:
            attn_up = F.interpolate(attn, size=(cfg.IMG_SIZE, cfg.IMG_SIZE), mode="bilinear", align_corners=False)
            attn_map = attn_up[0, 0].cpu().numpy()
        else:
            attn_map = np.zeros((cfg.IMG_SIZE, cfg.IMG_SIZE), dtype=np.float32)

        # Concept maps upsampled
        concepts = F.interpolate(
            out["concepts"], size=(cfg.IMG_SIZE, cfg.IMG_SIZE),
            mode="bilinear", align_corners=False,
        )[0].cpu().numpy()  # K,H,W
        contrib = model.concept_contributions(out["gap"])[0].cpu().numpy()
        top = np.argsort(-contrib)[:3]

        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        axes[0, 0].imshow(rgb_np)
        axes[0, 0].set_title("Image")
        axes[0, 1].imshow(gt, cmap="gray")
        axes[0, 1].set_title("Ground Truth")
        axes[0, 2].imshow(pred > 0.5, cmap="gray")
        axes[0, 2].set_title("Prediction")
        axes[0, 3].imshow(overlay(rgb_np, attn_map))
        axes[0, 3].set_title("ViT Attention (XAI)")

        for j in range(3):
            k = top[j]
            axes[1, j].imshow(overlay(rgb_np, concepts[k]))
            axes[1, j].set_title(f"{cfg.CONCEPT_NAMES[k]}\ncontrib={contrib[k]*100:.1f}%")
        # contribution bar
        axes[1, 3].barh(cfg.CONCEPT_NAMES, contrib * 100, color="steelblue")
        axes[1, 3].set_xlabel("% contribution to forest")
        axes[1, 3].set_title("Concept contributions\n(paper §III-C)")
        axes[1, 3].invert_yaxis()

        for ax in axes.ravel()[:7]:
            ax.axis("off")
        plt.tight_layout()
        path = out_dir / f"explain_{shown:02d}.png"
        plt.savefig(path, dpi=140)
        plt.close()
        print("Saved", path)

        # text explanation
        txt = out_dir / f"explain_{shown:02d}.txt"
        lines = ["Decision explanation (guided concepts):"]
        for k in top:
            lines.append(f"  - {cfg.CONCEPT_NAMES[k]}: {contrib[k]*100:.1f}%")
        lines.append(f"Pseudo-label z: {z[0].numpy().round(3).tolist()}")
        txt.write_text("\n".join(lines), encoding="utf-8")
        shown += 1

    print(f"Wrote {shown} explanations to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4)
    args = parser.parse_args()
    explain(args.n)
