"""
Team-4 post-processing for the ViT + Concept XAI model.

Separate from Lasana/postprocess — same research parts, ViT backbone.

Parts:
  A) Threshold sweep + morphology
  B) Test-Time Augmentation (TTA-4)
  C) Uncertainty (entropy) + ECE + temperature scaling
  D) Lightweight Boundary Refinement Network (LBR-Net, PyTorch)
  E) Bilateral edge-aware refine
  F) Concept-guided refine (ViT-specific: use concept maps)

Base checkpoint: ViT_XAI_Segmentation/checkpoints/vit_concept_seg_best.pt
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

# ── paths ────────────────────────────────────────────────────────────────────
PP_ROOT = Path(__file__).resolve().parent          # Lasana/ViT_XAI_Postprocess
LASANA_ROOT = PP_ROOT.parent                       # Lasana/
PROJECT = LASANA_ROOT.parent                       # DNN_Project/
VIT_ROOT = PROJECT / "ViT_XAI_Segmentation"
sys.path.insert(0, str(VIT_ROOT))

import config as vit_cfg  # noqa: E402
from dataset import ForestConceptDataset, list_pairs  # noqa: E402
from model import ViTConceptSeg  # noqa: E402

CKPT = VIT_ROOT / "checkpoints" / "vit_concept_seg_best.pt"
OUT_DIR = PP_ROOT / "results"
CKPT_DIR = PP_ROOT / "checkpoints"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

SEED = vit_cfg.SEED
IMG_SIZE = vit_cfg.IMG_SIZE
BATCH = int(os.environ.get("VIT_PP_BATCH", "4"))
# Default subset for CPU; set VIT_PP_MAX_SAMPLES=0 for full dataset
_env = os.environ.get("VIT_PP_MAX_SAMPLES", "1500")
MAX_SAMPLES = None if _env in ("0", "none", "None", "") else int(_env)
LBR_EPOCHS = int(os.environ.get("VIT_PP_LBR_EPOCHS", "10"))
LBR_TRAIN_CAP = int(os.environ.get("VIT_PP_LBR_TRAIN_CAP", "800"))


# ── metrics ──────────────────────────────────────────────────────────────────
def soft_iou(y_true, y_pred, thr=0.5, eps=1e-6):
    pred = (y_pred > thr).astype(np.float32)
    yt = y_true.astype(np.float32)
    inter = (pred * yt).sum()
    union = pred.sum() + yt.sum() - inter
    return float((inter + eps) / (union + eps))


def soft_dice(y_true, y_pred, thr=0.5, eps=1e-6):
    pred = (y_pred > thr).astype(np.float32)
    yt = y_true.astype(np.float32)
    inter = (pred * yt).sum()
    return float((2 * inter + eps) / (pred.sum() + yt.sum() + eps))


def pixel_acc(y_true, y_pred, thr=0.5):
    pred = (y_pred > thr).astype(np.float32)
    return float((pred == y_true).mean())


def scores(y_true, y_pred, thr=0.5):
    return {
        "iou": soft_iou(y_true, y_pred, thr),
        "dice": soft_dice(y_true, y_pred, thr),
        "acc": pixel_acc(y_true, y_pred, thr),
    }


def ece_score(y_true, y_prob, n_bins=15):
    y_true = y_true.reshape(-1).astype(np.float32)
    y_prob = y_prob.reshape(-1).astype(np.float32)
    pred = (y_prob >= 0.5).astype(np.float32)
    conf = np.where(pred == 1, y_prob, 1.0 - y_prob)
    correct = (pred == y_true).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf >= bins[i]) & (
            conf < bins[i + 1] if i < n_bins - 1 else conf <= bins[i + 1]
        )
        if not np.any(m):
            continue
        ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


# ── classical ops ────────────────────────────────────────────────────────────
def apply_morph(mask_bin, ksize=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    single = mask_bin.ndim == 2
    if single:
        mask_bin = mask_bin[None, ..., None]
    out = []
    for i in range(len(mask_bin)):
        m = (mask_bin[i].squeeze() * 255).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        cleaned = np.zeros_like(m)
        for lab in range(1, n):
            if stats[lab, cv2.CC_STAT_AREA] >= 32:
                cleaned[labels == lab] = 255
        out.append((cleaned > 127).astype(np.float32)[..., None])
    out = np.stack(out, 0)
    return out[0].squeeze() if single else out


def bilateral_refine(images_rgb01, probs):
    """images_rgb01: B,H,W,3 in [0,1]; probs: B,H,W,1"""
    refined = np.zeros_like(probs)
    for i in range(len(images_rgb01)):
        img = (np.clip(images_rgb01[i], 0, 1) * 255).astype(np.uint8)
        p = probs[i].squeeze().astype(np.float32)
        p8 = (np.clip(p, 0, 1) * 255).astype(np.uint8)
        sm = cv2.bilateralFilter(p8, d=5, sigmaColor=40, sigmaSpace=5)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        grad = grad / (grad.max() + 1e-6)
        p_sm = sm.astype(np.float32) / 255.0
        mix = (1 - 0.5 * grad) * p_sm + (0.5 * grad) * p
        refined[i] = mix[..., None]
    return refined


def concept_guided_refine(probs, concepts_up, forest_idx_hint=0):
    """
    ViT-specific: boost probs where dense_canopy / forest_boundary concepts are high,
    dampen where open_clearing is high.
    concepts_up: B,K,H,W (already sigmoid-normalized per map optional)
    Concept order from vit_cfg: dense, sparse, shadow, clearing, boundary
    """
    # soft activations
    c = concepts_up.copy()
    # normalize each concept map to [0,1] per image
    for i in range(c.shape[0]):
        for k in range(c.shape[1]):
            mk = c[i, k]
            c[i, k] = (mk - mk.min()) / (mk.max() - mk.min() + 1e-6)

    dense = c[:, 0:1]       # B,1,H,W
    clearing = c[:, 3:4]
    boundary = c[:, 4:5]
    p = probs.transpose(0, 3, 1, 2)  # B,1,H,W
    # residual nudge
    refined = p + 0.08 * dense + 0.05 * boundary - 0.10 * clearing
    refined = np.clip(refined, 0, 1)
    return refined.transpose(0, 2, 3, 1)  # B,H,W,1


# ── LBR-Net (PyTorch) ────────────────────────────────────────────────────────
class ResidualRefine(nn.Module):
    def forward(self, p, delta):
        eps = 1e-4
        p_clip = p.clamp(eps, 1 - eps)
        logits = torch.log(p_clip) - torch.log(1 - p_clip)
        return torch.sigmoid(logits + 0.5 * delta)


class LBRNet(nn.Module):
    """Tiny residual boundary refiner: concat(RGB, P) -> refined P."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
            nn.Conv2d(32, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )
        self.refine = ResidualRefine()

    def forward(self, x):
        # x: B,4,H,W  (RGB + P)
        p = x[:, 3:4]
        delta = self.backbone(x)
        return self.refine(p, delta)


def boundary_band(masks, width=2):
    """masks: B,H,W,1 numpy"""
    bands = np.zeros_like(masks)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    for i in range(len(masks)):
        m = (masks[i].squeeze() * 255).astype(np.uint8)
        dil = cv2.dilate(m, k)
        ero = cv2.erode(m, k)
        band = cv2.subtract(dil, ero)
        bands[i] = (band > 0).astype(np.float32)[..., None]
    return bands


def train_lbr(rgb, probs, masks, rgb_val, probs_val, masks_val, device, epochs=10):
    model = LBRNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bands = boundary_band(masks)
    w = 1.0 + 3.0 * bands

    def batch_iter(r, p, m, wt, bs=BATCH, shuffle=True):
        n = len(r)
        idx = np.arange(n)
        if shuffle:
            np.random.shuffle(idx)
        for i in range(0, n, bs):
            sl = idx[i : i + bs]
            yield r[sl], p[sl], m[sl], wt[sl]

    best_state = None
    best_iou = -1.0
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for r, p, m, wt in batch_iter(rgb, probs, masks, w):
            x = torch.from_numpy(
                np.concatenate([r.transpose(0, 3, 1, 2), p.transpose(0, 3, 1, 2)], 1)
            ).float().to(device)
            y = torch.from_numpy(m.transpose(0, 3, 1, 2)).float().to(device)
            sw = torch.from_numpy(wt.transpose(0, 3, 1, 2)).float().to(device)
            pred = model(x)
            bce = F.binary_cross_entropy(pred, y, reduction="none")
            bce = (bce * sw).mean()
            inter = (pred * y).sum()
            dice = 1 - (2 * inter + 1) / (pred.sum() + y.sum() + 1)
            loss = 0.5 * bce + 0.5 * dice
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))

        # val IoU
        model.eval()
        with torch.no_grad():
            xv = torch.from_numpy(
                np.concatenate(
                    [rgb_val.transpose(0, 3, 1, 2), probs_val.transpose(0, 3, 1, 2)], 1
                )
            ).float().to(device)
            pv = []
            for i in range(0, len(xv), BATCH):
                pv.append(model(xv[i : i + BATCH]).cpu().numpy())
            pv = np.concatenate(pv, 0).transpose(0, 2, 3, 1)
        viou = soft_iou(masks_val, pv, 0.5)
        print(f"  LBR epoch {ep}/{epochs} loss={np.mean(losses):.4f} val_iou={viou:.4f}")
        if viou > best_iou:
            best_iou = viou
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_iou


# ── ViT inference helpers ────────────────────────────────────────────────────
@torch.no_grad()
def predict_loader(model, loader, device, with_concepts=False):
    model.eval()
    probs, masks, rgbs, concepts = [], [], [], []
    t0 = time.perf_counter()
    for images, mask, z, rgb in loader:
        images = images.to(device)
        out = model(images)
        p = torch.sigmoid(out["seg_logits"]).cpu().numpy()  # B,1,H,W
        probs.append(p.transpose(0, 2, 3, 1))
        masks.append(mask.numpy().transpose(0, 2, 3, 1))
        rgbs.append(rgb.numpy().transpose(0, 2, 3, 1))
        if with_concepts:
            c = out["concepts"]
            c = F.interpolate(c, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
            concepts.append(c.cpu().numpy())
    elapsed = time.perf_counter() - t0
    probs = np.concatenate(probs, 0)
    masks = np.concatenate(masks, 0)
    rgbs = np.concatenate(rgbs, 0)
    concepts = np.concatenate(concepts, 0) if with_concepts else None
    return probs, masks, rgbs, concepts, elapsed


@torch.no_grad()
def predict_tta(model, images_tensor, device):
    """images_tensor: B,3,H,W normalized. Average identity + flips."""
    model.eval()
    imgs = images_tensor.to(device)
    acc = torch.sigmoid(model(imgs)["seg_logits"])
    # hflip
    hf = torch.flip(imgs, dims=[3])
    acc = acc + torch.flip(torch.sigmoid(model(hf)["seg_logits"]), dims=[3])
    # vflip
    vf = torch.flip(imgs, dims=[2])
    acc = acc + torch.flip(torch.sigmoid(model(vf)["seg_logits"]), dims=[2])
    # hv
    hv = torch.flip(imgs, dims=[2, 3])
    acc = acc + torch.flip(torch.sigmoid(model(hv)["seg_logits"]), dims=[2, 3])
    return (acc / 4.0).cpu().numpy().transpose(0, 2, 3, 1)


def loader_tensors(loader, device, model):
    """Collect normalized tensors for TTA (in batches)."""
    all_p = []
    t0 = time.perf_counter()
    for images, _, _, _ in loader:
        all_p.append(predict_tta(model, images, device))
    elapsed = time.perf_counter() - t0
    return np.concatenate(all_p, 0), elapsed


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== ViT post-processing research ===")
    print("Device:", device)
    print("Checkpoint:", CKPT)
    print("MAX_SAMPLES:", MAX_SAMPLES)
    assert CKPT.exists(), f"Missing {CKPT} — train ViT model first"

    pairs = list_pairs(str(vit_cfg.IMG_DIR), str(vit_cfg.MASK_DIR), MAX_SAMPLES)
    print(f"Pairs: {len(pairs)}")
    idx = list(range(len(pairs)))
    tr_idx, te_idx = train_test_split(idx, test_size=0.10, random_state=SEED)
    val_frac = vit_cfg.VAL_RATIO / (vit_cfg.TRAIN_RATIO + vit_cfg.VAL_RATIO)
    tr_idx, va_idx = train_test_split(tr_idx, test_size=val_frac, random_state=SEED)

    train_ds = ForestConceptDataset([pairs[i] for i in tr_idx], IMG_SIZE, augment=False)
    val_ds = ForestConceptDataset([pairs[i] for i in va_idx], IMG_SIZE, augment=False)
    test_ds = ForestConceptDataset([pairs[i] for i in te_idx], IMG_SIZE, augment=False)
    print(f"Train {len(train_ds)} | Val {len(val_ds)} | Test {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    model = ViTConceptSeg().to(device)
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded epoch {ckpt.get('epoch')} val_iou={ckpt.get('val_iou', float('nan')):.4f}")

    print("\nPredicting val/test...")
    p_val, y_val, rgb_val, c_val, t_val = predict_loader(
        model, val_loader, device, with_concepts=True
    )
    p_test, y_test, rgb_test, c_test, t_test = predict_loader(
        model, test_loader, device, with_concepts=True
    )
    ms_per = (t_test / len(test_ds)) * 1000
    print(f"Test inference: {ms_per:.1f} ms/image")

    rows = []
    baseline_iou = None

    def add_row(name, y_true, y_prob, thr, extra_ms=0.0, note=""):
        nonlocal baseline_iou
        sc = scores(y_true, y_prob, thr)
        ece = ece_score(y_true, y_prob)
        if baseline_iou is None and name.startswith("0_"):
            baseline_iou = sc["iou"]
        delta = 0.0 if baseline_iou is None else (sc["iou"] - baseline_iou) / baseline_iou * 100
        row = {
            "method": name,
            "thr": thr,
            "iou": round(sc["iou"], 6),
            "dice": round(sc["dice"], 6),
            "acc": round(sc["acc"], 6),
            "ece": round(ece, 6),
            "delta_iou_pct": round(delta, 3),
            "ms_per_image": round(ms_per + extra_ms, 2),
            "note": note,
        }
        rows.append(row)
        print(
            f"{name:32s} IoU={sc['iou']:.4f} Dice={sc['dice']:.4f} "
            f"Acc={sc['acc']:.4f} ECE={ece:.4f} dIoU%={delta:+.2f}"
        )
        return sc

    # ── A ────────────────────────────────────────────────────────────────────
    print("\n--- Part A: threshold + morphology ---")
    add_row("0_baseline_thr0.5", y_test, p_test, 0.5, note="ViT raw")

    best_t, best_iou = 0.5, -1.0
    sweep = {}
    for t in np.arange(0.30, 0.75, 0.05):
        sc = scores(y_val, p_val, float(t))
        sweep[f"{t:.2f}"] = sc["iou"]
        if sc["iou"] > best_iou:
            best_iou, best_t = sc["iou"], float(t)
    print(f"Best val t*={best_t:.2f} (val IoU={best_iou:.4f})")
    (OUT_DIR / "threshold_sweep.json").write_text(json.dumps(sweep, indent=2), encoding="utf-8")

    add_row("1_thr_star", y_test, p_test, best_t, note=f"t*={best_t:.2f}")
    t0 = time.perf_counter()
    morph = apply_morph((p_test > best_t).astype(np.float32))
    morph_ms = (time.perf_counter() - t0) / len(y_test) * 1000
    add_row("2_thr_star_morph", y_test, morph, 0.5, extra_ms=morph_ms, note="open+close")

    # ── B TTA ────────────────────────────────────────────────────────────────
    print("\n--- Part B: TTA-4 ---")
    p_tta, tta_elapsed = loader_tensors(test_loader, device, model)
    tta_ms = (tta_elapsed / len(y_test)) * 1000
    add_row("3_tta4", y_test, p_tta, best_t, extra_ms=tta_ms - ms_per, note="4-view avg")
    morph_tta = apply_morph((p_tta > best_t).astype(np.float32))
    add_row(
        "4_tta4_morph", y_test, morph_tta, 0.5,
        extra_ms=tta_ms - ms_per + morph_ms, note="TTA+morph",
    )

    # ── C uncertainty ────────────────────────────────────────────────────────
    print("\n--- Part C: uncertainty / ECE ---")
    eps = 1e-6
    pclip = np.clip(p_test, eps, 1 - eps)
    entropy = -(pclip * np.log(pclip) + (1 - pclip) * np.log(1 - pclip))
    np.save(OUT_DIR / "test_entropy_maps.npy", entropy.astype(np.float32))
    thr_map = np.where(entropy > np.median(entropy), min(best_t + 0.1, 0.7), best_t)
    adaptive = (p_test > thr_map).astype(np.float32)
    add_row("5_confidence_adaptive_thr", y_test, adaptive, 0.5, note="strict on high H")

    best_T, best_ece = 1.0, ece_score(y_val, p_val)
    for T in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        pc = np.clip(p_val, 1e-4, 1 - 1e-4)
        logits = np.log(pc) - np.log(1 - pc)
        pT = 1 / (1 + np.exp(-logits / T))
        e = ece_score(y_val, pT)
        if e < best_ece:
            best_ece, best_T = e, T
    pc = np.clip(p_test, 1e-4, 1 - 1e-4)
    logits = np.log(pc) - np.log(1 - pc)
    p_temp = 1 / (1 + np.exp(-logits / best_T))
    add_row("6_temperature_scaled", y_test, p_temp, best_t, note=f"T*={best_T}")

    # ── E bilateral ──────────────────────────────────────────────────────────
    print("\n--- Part E: bilateral refine ---")
    t0 = time.perf_counter()
    p_bil = bilateral_refine(rgb_test, p_test)
    bil_ms = (time.perf_counter() - t0) / len(y_test) * 1000
    add_row("7_bilateral_refine", y_test, p_bil, best_t, extra_ms=bil_ms, note="CRF-lite")

    # ── F concept-guided (ViT-specific) ──────────────────────────────────────
    print("\n--- Part F: concept-guided refine ---")
    t0 = time.perf_counter()
    p_cg = concept_guided_refine(p_test, c_test)
    cg_ms = (time.perf_counter() - t0) / len(y_test) * 1000
    add_row(
        "8_concept_guided", y_test, p_cg, best_t, extra_ms=cg_ms,
        note="dense/boundary up, clearing down",
    )
    morph_cg = apply_morph((p_cg > best_t).astype(np.float32))
    add_row("9_concept_guided_morph", y_test, morph_cg, 0.5, extra_ms=cg_ms + morph_ms)

    # ── D LBR-Net ────────────────────────────────────────────────────────────
    print("\n--- Part D: train LBR-Net ---")
    # cap train size for CPU
    n_lbr = min(len(train_ds), LBR_TRAIN_CAP)
    print(f"Predicting {n_lbr} train samples for LBR...")
    lbr_subset = Subset(train_ds, list(range(n_lbr)))
    lbr_loader = DataLoader(lbr_subset, batch_size=BATCH, shuffle=False, num_workers=0)
    p_tr, y_tr, rgb_tr, _, _ = predict_loader(model, lbr_loader, device, with_concepts=False)

    lbr, lbr_val_iou = train_lbr(
        rgb_tr, p_tr, y_tr, rgb_val, p_val, y_val, device, epochs=LBR_EPOCHS
    )
    lbr_path = CKPT_DIR / "vit_lbr_net.pt"
    torch.save({"model": lbr.state_dict(), "val_iou": lbr_val_iou}, lbr_path)
    print("Saved", lbr_path)

    lbr.eval()
    with torch.no_grad():
        xt = torch.from_numpy(
            np.concatenate(
                [rgb_test.transpose(0, 3, 1, 2), p_test.transpose(0, 3, 1, 2)], 1
            )
        ).float().to(device)
        t0 = time.perf_counter()
        outs = []
        for i in range(0, len(xt), BATCH):
            outs.append(lbr(xt[i : i + BATCH]).cpu().numpy())
        p_lbr = np.concatenate(outs, 0).transpose(0, 2, 3, 1)
        lbr_ms = (time.perf_counter() - t0) / len(y_test) * 1000

    add_row("10_lbr_net", y_test, p_lbr, best_t, extra_ms=lbr_ms, note="novel tiny refiner")

    # TTA then LBR
    with torch.no_grad():
        xtta = torch.from_numpy(
            np.concatenate(
                [rgb_test.transpose(0, 3, 1, 2), p_tta.transpose(0, 3, 1, 2)], 1
            )
        ).float().to(device)
        outs = []
        for i in range(0, len(xtta), BATCH):
            outs.append(lbr(xtta[i : i + BATCH]).cpu().numpy())
        p_lbr_tta = np.concatenate(outs, 0).transpose(0, 2, 3, 1)
    add_row(
        "11_tta4_lbr_net", y_test, p_lbr_tta, best_t,
        extra_ms=(tta_ms - ms_per) + lbr_ms, note="TTA then LBR",
    )

    # concept-guided then LBR
    with torch.no_grad():
        xcg = torch.from_numpy(
            np.concatenate(
                [rgb_test.transpose(0, 3, 1, 2), p_cg.transpose(0, 3, 1, 2)], 1
            )
        ).float().to(device)
        outs = []
        for i in range(0, len(xcg), BATCH):
            outs.append(lbr(xcg[i : i + BATCH]).cpu().numpy())
        p_lbr_cg = np.concatenate(outs, 0).transpose(0, 2, 3, 1)
    add_row(
        "12_concept_lbr", y_test, p_lbr_cg, best_t,
        extra_ms=cg_ms + lbr_ms, note="concept-guided then LBR",
    )

    # ── save ─────────────────────────────────────────────────────────────────
    csv_path = OUT_DIR / "comparison_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lines = [
        "# ViT + XAI — Post-Processing Research Results",
        "",
        f"Base model: `{CKPT.name}` (ViT Concept Seg)  ",
        f"Samples: {len(pairs)} | split train/val/test = "
        f"{len(train_ds)}/{len(val_ds)}/{len(test_ds)} | seed={SEED} | device={device}",
        "",
        f"Best validation threshold **t\\* = {best_t:.2f}** | Temperature **T\\* = {best_T}**",
        "",
        "| Method | IoU | Dice | Acc | ECE | ΔIoU% | ms/img |",
        "|--------|-----|------|-----|-----|-------|--------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['iou']:.4f} | {r['dice']:.4f} | {r['acc']:.4f} | "
            f"{r['ece']:.4f} | {r['delta_iou_pct']:+.2f} | {r['ms_per_image']:.1f} |"
        )
    lines += [
        "",
        "## Parts",
        "",
        "- **A** Threshold + morphology",
        "- **B** TTA-4",
        "- **C** Entropy / ECE / temperature / adaptive thr",
        "- **D** LBR-Net → `checkpoints/vit_lbr_net.pt`",
        "- **E** Bilateral refine",
        "- **F** Concept-guided refine (ViT-specific)",
        "",
        "## vs Lasana post-process",
        "",
        "Same Team-4 methods; this folder targets the **ViT+concept** checkpoint "
        "and adds concept-guided refinement using the paper concept maps.",
        "",
    ]
    md_path = OUT_DIR / "RESULTS.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print("\nSaved:", csv_path)
    print("Saved:", md_path)
    print("Done.")


if __name__ == "__main__":
    main()
