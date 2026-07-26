"""
Post-processing research runner (Team Member 4).

Parts:
  A) Threshold sweep + morphology
  B) Test-Time Augmentation (TTA)
  C) Uncertainty (entropy) + ECE calibration
  D) Train lightweight boundary refinement network (LBR-Net)
  E) Edge-aware bilateral refinement (CRF-style substitute; pydensecrf unavailable)

Uses Lasana U-Net checkpoint + same 1200-sample CPU split (seed=42).
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
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model, layers

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_lasana import (  # noqa: E402
    IMG_SIZE,
    MASK_THRESHOLD,
    SEED,
    UNET_FEATURES,
    build_unet,
    find_mask_path,
    load_dataset,
)

IMAGE_FOLDER = ROOT / "dataset" / "Forest Segmented" / "Forest Segmented" / "images"
MASK_FOLDER = ROOT / "dataset" / "Forest Segmented" / "Forest Segmented" / "masks"
CKPT = ROOT / "checkpoints" / "lasana_unet_best.keras"
OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SAMPLES = int(os.environ.get("LASANA_MAX_SAMPLES", "1200"))
BATCH = 4
TRAIN_RATIO, VAL_RATIO = 0.80, 0.10


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
    """Expected Calibration Error over pixels."""
    y_true = y_true.reshape(-1).astype(np.float32)
    y_prob = y_prob.reshape(-1).astype(np.float32)
    # confidence = max(p, 1-p); correctness vs predicted class
    pred = (y_prob >= 0.5).astype(np.float32)
    conf = np.where(pred == 1, y_prob, 1.0 - y_prob)
    correct = (pred == y_true).astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i + 1] if i < n_bins - 1 else conf <= bins[i + 1])
        if not np.any(m):
            continue
        ece += (m.mean()) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


# ── post-process ops ─────────────────────────────────────────────────────────
def apply_morph(mask_bin, ksize=3, mode="open_close"):
    """mask_bin: float {0,1} HxW or BxHxWx1"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    single = mask_bin.ndim == 2
    if single:
        mask_bin = mask_bin[None, ..., None]
    out = []
    for i in range(len(mask_bin)):
        m = (mask_bin[i].squeeze() * 255).astype(np.uint8)
        if mode == "open":
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        elif mode == "close":
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        else:  # open then close
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        # remove tiny components
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        cleaned = np.zeros_like(m)
        min_area = 32
        for lab in range(1, n):
            if stats[lab, cv2.CC_STAT_AREA] >= min_area:
                cleaned[labels == lab] = 255
        out.append((cleaned > 127).astype(np.float32)[..., None])
    out = np.stack(out, axis=0)
    return out[0].squeeze() if single else out


def bilateral_refine(images, probs, thr=0.5):
    """Edge-aware probability smoothing (DenseCRF-style substitute)."""
    refined = np.zeros_like(probs)
    for i in range(len(images)):
        img = (images[i] * 255).astype(np.uint8)
        p = probs[i].squeeze().astype(np.float32)
        # bilateral on prob guided by grayscale intensity proxy via joint bilateral on 3ch
        p8 = (p * 255).astype(np.uint8)
        # filter each channel of a stacked guide
        sm = cv2.bilateralFilter(p8, d=5, sigmaColor=40, sigmaSpace=5)
        # strengthen edge consistency: mix with guided version
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # simple joint: where image gradient high, trust original less smoothing
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        grad = grad / (grad.max() + 1e-6)
        p_sm = sm.astype(np.float32) / 255.0
        mix = (1 - 0.5 * grad) * p_sm + (0.5 * grad) * p
        refined[i] = mix[..., None]
    return refined


# ── TTA ──────────────────────────────────────────────────────────────────────
def predict_tta(model, images, batch=BATCH):
    """Average over identity, hflip, vflip, hvflip."""
    n = len(images)
    acc = np.zeros((n, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)

    def pred_batch(x):
        out = []
        for i in range(0, len(x), batch):
            out.append(model.predict(x[i : i + batch], verbose=0))
        return np.concatenate(out, axis=0)

    # identity
    acc += pred_batch(images)
    # hflip
    hf = images[:, :, ::-1, :]
    ph = pred_batch(hf)[:, :, ::-1, :]
    acc += ph
    # vflip
    vf = images[:, ::-1, :, :]
    pv = pred_batch(vf)[:, ::-1, :, :]
    acc += pv
    # hvflip
    hv = images[:, ::-1, ::-1, :]
    phv = pred_batch(hv)[:, ::-1, ::-1, :]
    acc += phv
    return acc / 4.0


# ── LBR-Net ──────────────────────────────────────────────────────────────────
class ResidualRefine(layers.Layer):
    """P_refined = sigmoid(logit(P) + 0.5 * delta)."""

    def call(self, inputs):
        p, delta = inputs
        eps = 1e-4
        p_clip = tf.clip_by_value(p, eps, 1.0 - eps)
        logits = tf.math.log(p_clip) - tf.math.log(1.0 - p_clip)
        return tf.sigmoid(logits + 0.5 * delta)


def build_lbr_net():
    """Tiny residual boundary refiner: concat(RGB, P) -> delta -> refined P."""
    inp = Input(shape=(IMG_SIZE, IMG_SIZE, 4), name="rgb_p")
    x = layers.SeparableConv2D(32, 3, padding="same", use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(32, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(16, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    delta = layers.Conv2D(1, 1, padding="same", activation=None)(x)
    p = layers.Lambda(lambda t: t[..., 3:4], name="prob_slice")(inp)
    out = ResidualRefine(name="residual_refine")([p, delta])
    return Model(inp, out, name="lbr_net")


def boundary_band(masks, width=2):
    """Ring around GT edges for weighted training."""
    bands = np.zeros_like(masks)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    for i in range(len(masks)):
        m = (masks[i].squeeze() * 255).astype(np.uint8)
        dil = cv2.dilate(m, k)
        ero = cv2.erode(m, k)
        band = cv2.subtract(dil, ero)
        bands[i] = (band > 0).astype(np.float32)[..., None]
    return bands


def train_lbr(X_train, y_train, p_train, X_val, y_val, p_val, epochs=8):
    model = build_lbr_net()
    # weight boundary pixels higher
    bands_tr = boundary_band(y_train)
    w_tr = 1.0 + 3.0 * bands_tr  # 1 interior, 4 on boundary

    def weighted_bce_dice(y_true, y_pred):
        # approximate: pixel-wise BCE weighted, + dice
        eps = 1e-7
        bce = -(y_true * tf.math.log(y_pred + eps) + (1 - y_true) * tf.math.log(1 - y_pred + eps))
        # weights passed via sample — use mean over spatial; Keras sample_weight works on batch
        bce = tf.reduce_mean(bce)
        inter = tf.reduce_sum(y_true * y_pred)
        dice = 1 - (2 * inter + 1) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1)
        return 0.5 * bce + 0.5 * dice

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=weighted_bce_dice)

    xtr = np.concatenate([X_train, p_train], axis=-1)
    xva = np.concatenate([X_val, p_val], axis=-1)
    # flatten sample weights to (N, H*W) not supported easily — use image-level mean band weight
    sw = w_tr.mean(axis=(1, 2, 3))

    hist = model.fit(
        xtr,
        y_train,
        sample_weight=sw,
        validation_data=(xva, y_val),
        epochs=epochs,
        batch_size=BATCH,
        verbose=1,
    )
    return model, hist


# ── predict helper ───────────────────────────────────────────────────────────
def predict_all(model, images, batch=BATCH):
    outs = []
    t0 = time.perf_counter()
    for i in range(0, len(images), batch):
        outs.append(model.predict(images[i : i + batch], verbose=0))
    elapsed = time.perf_counter() - t0
    return np.concatenate(outs, axis=0), elapsed


def main():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    print("=== Post-processing research ===")
    print("Checkpoint:", CKPT)
    assert CKPT.exists(), f"Missing {CKPT}"
    assert IMAGE_FOLDER.is_dir() and MASK_FOLDER.is_dir()

    print(f"\nLoading {MAX_SAMPLES} samples...")
    images, masks = load_dataset(
        str(IMAGE_FOLDER), str(MASK_FOLDER), IMG_SIZE, MASK_THRESHOLD, MAX_SAMPLES
    )
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, masks, test_size=0.10, random_state=SEED
    )
    val_frac = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_frac, random_state=SEED
    )
    del images, masks, X_temp, y_temp
    print(f"Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}")

    print("\nLoading U-Net...")
    unet = build_unet(features=UNET_FEATURES)
    unet.load_weights(str(CKPT))
    print("Features:", UNET_FEATURES)

    print("\nPredicting val/test (baseline forward)...")
    p_val, t_val = predict_all(unet, X_val)
    p_test, t_test = predict_all(unet, X_test)
    ms_per = (t_test / len(X_test)) * 1000
    print(f"Test inference: {ms_per:.1f} ms/image (batch={BATCH})")

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
            f"{name:28s} IoU={sc['iou']:.4f} Dice={sc['dice']:.4f} "
            f"Acc={sc['acc']:.4f} ECE={ece:.4f} ΔIoU%={delta:+.2f}"
        )
        return sc

    # ── Part A ───────────────────────────────────────────────────────────────
    print("\n--- Part A: threshold + morphology ---")
    add_row("0_baseline_thr0.5", y_test, p_test, 0.5, note="Lasana default")

    best_t, best_iou = 0.5, -1
    sweep = {}
    for t in np.arange(0.30, 0.75, 0.05):
        sc = scores(y_val, p_val, float(t))
        sweep[f"{t:.2f}"] = sc["iou"]
        if sc["iou"] > best_iou:
            best_iou, best_t = sc["iou"], float(t)
    print(f"Best val threshold t*={best_t:.2f} (val IoU={best_iou:.4f})")
    (OUT_DIR / "threshold_sweep.json").write_text(json.dumps(sweep, indent=2), encoding="utf-8")

    add_row("1_thr_star", y_test, p_test, best_t, note=f"t*={best_t:.2f} from val")

    bin_test = (p_test > best_t).astype(np.float32)
    t0 = time.perf_counter()
    morph = apply_morph(bin_test, ksize=3, mode="open_close")
    morph_ms = (time.perf_counter() - t0) / len(X_test) * 1000
    # morph returns binary — treat as prob for scoring
    add_row(
        "2_thr_star_morph",
        y_test,
        morph,
        0.5,
        extra_ms=morph_ms,
        note="open+close+rm small",
    )

    # ── Part B: TTA ──────────────────────────────────────────────────────────
    print("\n--- Part B: TTA-4 ---")
    t0 = time.perf_counter()
    p_tta = predict_tta(unet, X_test)
    tta_ms = (time.perf_counter() - t0) / len(X_test) * 1000
    add_row("3_tta4", y_test, p_tta, best_t, extra_ms=tta_ms - ms_per, note="4 views avg")

    bin_tta = (p_tta > best_t).astype(np.float32)
    morph_tta = apply_morph(bin_tta, ksize=3)
    add_row("4_tta4_morph", y_test, morph_tta, 0.5, extra_ms=tta_ms - ms_per + morph_ms, note="TTA+morph")

    # ── Part C: uncertainty + ECE ────────────────────────────────────────────
    print("\n--- Part C: uncertainty / ECE ---")
    # Binary entropy map
    eps = 1e-6
    p = np.clip(p_test, eps, 1 - eps)
    entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    np.save(OUT_DIR / "test_entropy_maps.npy", entropy.astype(np.float32))
    # Confidence-based: abstain high-entropy pixels from IoU (report coverage)
    # Adaptive: use higher threshold where entropy high
    thr_map = np.where(entropy > np.median(entropy), min(best_t + 0.1, 0.7), best_t)
    adaptive = (p_test > thr_map).astype(np.float32)
    add_row(
        "5_confidence_adaptive_thr",
        y_test,
        adaptive,
        0.5,
        note="stricter thr on high entropy",
    )
    print(f"Mean entropy: {entropy.mean():.4f} | ECE baseline already in table")

    # Temperature scaling on val logits-proxy
    # Fit T by grid search minimizing ECE on val
    best_T, best_ece = 1.0, ece_score(y_val, p_val)
    for T in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        # temperature on logit
        pc = np.clip(p_val, 1e-4, 1 - 1e-4)
        logits = np.log(pc) - np.log(1 - pc)
        pT = 1 / (1 + np.exp(-logits / T))
        e = ece_score(y_val, pT)
        if e < best_ece:
            best_ece, best_T = e, T
    pc = np.clip(p_test, 1e-4, 1 - 1e-4)
    logits = np.log(pc) - np.log(1 - pc)
    p_temp = 1 / (1 + np.exp(-logits / best_T))
    add_row(
        "6_temperature_scaled",
        y_test,
        p_temp,
        best_t,
        note=f"T*={best_T} (val ECE)",
    )

    # ── Part E: bilateral (CRF-lite) ──────────────────────────────────────────
    print("\n--- Part E: bilateral edge-aware refine ---")
    t0 = time.perf_counter()
    p_bil = bilateral_refine(X_test, p_test)
    bil_ms = (time.perf_counter() - t0) / len(X_test) * 1000
    add_row("7_bilateral_refine", y_test, p_bil, best_t, extra_ms=bil_ms, note="CRF substitute")

    # ── Part D: train LBR-Net ─────────────────────────────────────────────────
    print("\n--- Part D: train LBR-Net ---")
    p_train, _ = predict_all(unet, X_train)
    lbr, _ = train_lbr(X_train, y_train, p_train, X_val, y_val, p_val, epochs=8)
    lbr_path = ROOT / "checkpoints" / "lbr_net.keras"
    lbr.save(str(lbr_path))

    x_test_in = np.concatenate([X_test, p_test], axis=-1)
    t0 = time.perf_counter()
    p_lbr = lbr.predict(x_test_in, batch_size=BATCH, verbose=0)
    lbr_ms = (time.perf_counter() - t0) / len(X_test) * 1000
    add_row("8_lbr_net", y_test, p_lbr, best_t, extra_ms=lbr_ms, note="novel tiny refiner")

    # combo: TTA probs through LBR
    x_tta_in = np.concatenate([X_test, p_tta], axis=-1)
    p_lbr_tta = lbr.predict(x_tta_in, batch_size=BATCH, verbose=0)
    add_row(
        "9_tta4_lbr_net",
        y_test,
        p_lbr_tta,
        best_t,
        extra_ms=(tta_ms - ms_per) + lbr_ms,
        note="TTA then LBR",
    )

    # ── save table ───────────────────────────────────────────────────────────
    csv_path = OUT_DIR / "comparison_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # markdown summary
    lines = [
        "# Post-Processing Research Results",
        "",
        f"Base model: `{CKPT.name}` | samples: {MAX_SAMPLES} | "
        f"split train/val/test = {len(X_train)}/{len(X_val)}/{len(X_test)} | seed={SEED}",
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
        "## Parts completed",
        "",
        "- **A** Threshold sweep + morphology",
        "- **B** TTA-4 (+ morph)",
        "- **C** Entropy uncertainty, adaptive threshold, temperature scaling / ECE",
        "- **D** Trained LBR-Net (lightweight boundary refiner) → `checkpoints/lbr_net.keras`",
        "- **E** Bilateral edge-aware refine (DenseCRF substitute; `pydensecrf` failed to build on this machine)",
        "",
        "## Files",
        "",
        f"- `{csv_path}`",
        f"- `{OUT_DIR / 'threshold_sweep.json'}`",
        f"- `{OUT_DIR / 'test_entropy_maps.npy'}`",
        f"- `{lbr_path}`",
        "",
    ]
    md_path = OUT_DIR / "RESULTS.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print("\nSaved:", csv_path)
    print("Saved:", md_path)
    print("Done.")


if __name__ == "__main__":
    main()
