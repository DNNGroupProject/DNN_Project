"""Continue research: train LBR-Net and append results to comparison table."""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_research import (  # noqa: E402
    BATCH,
    CKPT,
    IMAGE_FOLDER,
    MASK_FOLDER,
    MAX_SAMPLES,
    OUT_DIR,
    TRAIN_RATIO,
    VAL_RATIO,
    apply_morph,
    ece_score,
    predict_all,
    predict_tta,
    scores,
    train_lbr,
)
from train_lasana import IMG_SIZE, MASK_THRESHOLD, SEED, UNET_FEATURES, build_unet, load_dataset


def main():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

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

    unet = build_unet(features=UNET_FEATURES)
    unet.load_weights(str(CKPT))

    print("Predicting train/val/test for LBR...")
    p_train, _ = predict_all(unet, X_train)
    p_val, _ = predict_all(unet, X_val)
    p_test, t_test = predict_all(unet, X_test)
    ms_per = (t_test / len(X_test)) * 1000

    best_t = 0.65  # from Part A val sweep

    print("Training LBR-Net (8 epochs)...")
    lbr, _ = train_lbr(X_train, y_train, p_train, X_val, y_val, p_val, epochs=8)
    lbr_path = ROOT / "checkpoints" / "lbr_net.keras"
    lbr.save(str(lbr_path))
    print("Saved", lbr_path)

    x_test_in = np.concatenate([X_test, p_test], axis=-1)
    t0 = time.perf_counter()
    p_lbr = lbr.predict(x_test_in, batch_size=BATCH, verbose=0)
    lbr_ms = (time.perf_counter() - t0) / len(X_test) * 1000

    print("TTA for combo...")
    t0 = time.perf_counter()
    p_tta = predict_tta(unet, X_test)
    tta_ms = (time.perf_counter() - t0) / len(X_test) * 1000
    p_lbr_tta = lbr.predict(
        np.concatenate([X_test, p_tta], axis=-1), batch_size=BATCH, verbose=0
    )

    baseline_iou = 0.737858
    rows = []

    def add(name, y_true, y_prob, thr, extra_ms=0.0, note=""):
        sc = scores(y_true, y_prob, thr)
        ece = ece_score(y_true, y_prob)
        delta = (sc["iou"] - baseline_iou) / baseline_iou * 100
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

    add("8_lbr_net", y_test, p_lbr, best_t, extra_ms=lbr_ms, note="novel tiny refiner")
    add(
        "9_tta4_lbr_net",
        y_test,
        p_lbr_tta,
        best_t,
        extra_ms=(tta_ms - ms_per) + lbr_ms,
        note="TTA then LBR",
    )
    morph_lbr = apply_morph((p_lbr > best_t).astype(np.float32), ksize=3)
    add("10_lbr_morph", y_test, morph_lbr, 0.5, extra_ms=lbr_ms, note="LBR+morph")

    csv_path = OUT_DIR / "comparison_table.csv"
    existing = []
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
        existing = [r for r in existing if not r["method"].startswith(("8_", "9_", "10_"))]

    all_rows = existing + rows
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    lines = [
        "# Post-Processing Research Results",
        "",
        f"Base model: `{CKPT.name}` | samples: {MAX_SAMPLES} | seed={SEED}",
        "",
        f"Best validation threshold **t\\* = {best_t:.2f}**",
        "",
        "| Method | IoU | Dice | Acc | ECE | ΔIoU% | ms/img |",
        "|--------|-----|------|-----|-----|-------|--------|",
    ]
    for r in all_rows:
        lines.append(
            f"| {r['method']} | {float(r['iou']):.4f} | {float(r['dice']):.4f} | "
            f"{float(r['acc']):.4f} | {float(r['ece']):.4f} | "
            f"{float(r['delta_iou_pct']):+.2f} | {float(r['ms_per_image']):.1f} |"
        )
    lines += [
        "",
        "## Parts completed",
        "",
        "- **A** Threshold sweep + morphology",
        "- **B** TTA-4 (+ morph)",
        "- **C** Entropy uncertainty, adaptive threshold, temperature scaling / ECE",
        "- **D** Trained LBR-Net → `checkpoints/lbr_net.keras`",
        "- **E** Bilateral edge-aware refine (DenseCRF substitute)",
        "",
        "## Finding (this CPU run)",
        "",
        "Classical post-process gains were small on this already-smoothed U-Net subset. "
        "LBR-Net is the learnable novel module — compare its ΔIoU% and latency in the table.",
        "",
    ]
    (OUT_DIR / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("Updated", csv_path)
    print("Done.")


if __name__ == "__main__":
    main()
