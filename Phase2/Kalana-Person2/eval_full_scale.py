"""Evaluate full-scale SegFormer checkpoints with Person 4's metrics.

Same formulas as `Phase1/Dinura-Person3/eval_segformer.py`. Does **not**
copy weights or merge rows into `Lasana-Person4_Evaluation/` — those writes
are Person 4's folder, left untouched. Results stay under this folder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from paths import (
    CKPT_DIR,
    DATA_IMG_DIR,
    DATA_MASK_DIR,
    N_TEST,
    N_TRAIN,
    N_VAL,
    RESULTS_DIR,
    SEED,
    add_teammate_paths,
    apply_data_dirs,
    ensure_output_dirs,
)

add_teammate_paths()
apply_data_dirs()

from aamo import mean_aamo  # noqa: E402
from attention_consistency import build_segformer, grad_rollout_attention_map  # noqa: E402
from attention_consistency.data import load_pairs, make_splits, to_model_input  # noqa: E402
from attention_consistency.segformer_model import forest_prob  # noqa: E402
from efficiency import count_torch_params, efficiency_row, gflops_torch, measure_fps  # noqa: E402
from evaluate import row_dict, save_prediction_grid, write_comparison_table  # noqa: E402
from metrics import ConfusionCounts, binarize, format_metrics, metrics_from_counts  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VARIANT_LABEL = {
    "vanilla": "SegFormer-B0 (no attention loss)",
    "att": "SegFormer-B0 + Attention Consistency Loss",
}


def load_checkpoint(variant: str) -> tuple:
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
    return model, ckpt


def evaluate_variant(variant: str, args) -> dict:
    print(f"\n{'='*60}\nEvaluating: {VARIANT_LABEL[variant]}  device={DEVICE}\n{'='*60}")
    model, ckpt = load_checkpoint(variant)

    splits = make_splits(args.n_train, args.n_val, args.n_test, seed=args.seed)
    images, masks = load_pairs(splits["test"])
    print(f"test set: {len(images)} images (held out from training, seed={args.seed})")

    probs_list, attn_list = [], []
    for i in range(len(images)):
        x = to_model_input(images[i : i + 1]).to(DEVICE)
        with torch.enable_grad():
            attn_map, outputs = grad_rollout_attention_map(model, x)
        logits_up = F.interpolate(outputs.logits, size=(256, 256), mode="bilinear", align_corners=False)
        probs = forest_prob(logits_up)[0].detach().cpu().numpy()
        probs_list.append(probs)
        attn_list.append(attn_map.detach().cpu().numpy())

    probs = np.stack(probs_list)
    attentions = np.stack(attn_list)

    counts = ConfusionCounts()
    preds = binarize(probs, 0.5)
    counts.update(preds, masks)
    seg = metrics_from_counts(counts)
    print(format_metrics(seg))

    aamo_val = round(mean_aamo(list(attentions), list(masks), thr=0.5)["aamo"], 4)
    print(f"AAMO = {aamo_val}")

    params = count_torch_params(model)
    gflops = gflops_torch(model, input_size=(1, 3, 256, 256))

    warm_x = to_model_input(images[0:1]).to(DEVICE)

    def _run():
        with torch.no_grad():
            model(pixel_values=warm_x)

    speed = measure_fps(_run, warmup=3, runs=10)
    eff = efficiency_row(params=params, gflops=gflops, fps=speed["fps"], ms_per_image=speed["ms_per_image"])
    print(f"params={eff['params']} | gflops={eff['gflops']} | fps={eff['fps']} | ms/img={eff['ms_per_image']}")

    row = row_dict(VARIANT_LABEL[variant], seg, eff, aamo_val)
    row["checkpoint_epoch"] = ckpt["epoch"]
    row["checkpoint_val_dice"] = round(float(ckpt["val_dice"]), 4)
    row["n_test"] = len(images)
    row["device"] = str(DEVICE)

    ensure_output_dirs()
    save_prediction_grid(images, masks, probs, RESULTS_DIR / f"prediction_grid_{variant}.png", threshold=0.5)
    (RESULTS_DIR / f"eval_{variant}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    write_comparison_table([row], RESULTS_DIR)
    _relabel_comparison_table()
    print(f"Wrote {RESULTS_DIR / f'eval_{variant}.json'}  (Lasana-Person4 left untouched)")
    return row


def _relabel_comparison_table() -> None:
    md = RESULTS_DIR / "baseline_comparison.md"
    if not md.exists():
        return
    text = md.read_text(encoding="utf-8")
    text = text.replace(
        "# Baseline comparison (Person 4)",
        "# Full-scale SegFormer (Phase 2 / Kalana-Person2)\n\n"
        "Person 4 formulas (`metrics.py` / `aamo.py` / `efficiency.py`). "
        "Smoke-scale rows were not copied here.",
    )
    md.write_text(text, encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="both", choices=["vanilla", "att", "both"])
    p.add_argument("--n-train", type=int, default=N_TRAIN)
    p.add_argument("--n-val", type=int, default=N_VAL)
    p.add_argument("--n-test", type=int, default=N_TEST)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--img-dir", type=Path, default=DATA_IMG_DIR)
    p.add_argument("--mask-dir", type=Path, default=DATA_MASK_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    apply_data_dirs(args.img_dir, args.mask_dir)
    variants = ["vanilla", "att"] if args.variant == "both" else [args.variant]
    rows = [evaluate_variant(v, args) for v in variants]
    print("\nDone. Rows written under Phase2/Kalana-Person2/results/:")
    for r in rows:
        print(" ", r["model"], "-> dice", r["dice"], "iou", r["iou"], "aamo", r["aamo"])


if __name__ == "__main__":
    main()
