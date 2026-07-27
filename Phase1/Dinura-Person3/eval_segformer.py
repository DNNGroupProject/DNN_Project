"""
Evaluate the two trained SegFormer-B0 checkpoints (vanilla, +Attention
Consistency Loss) on the held-out test split, using Person 4 (Lasana)'s
own metrics/AAMO/efficiency modules — so the numbers are directly
comparable to the U-Net row Lasana already produced, using the exact
same formulas.

This is Person 4's "fill in the pending SegFormer rows" deliverable,
run here because it needs the checkpoint + attention hooks this folder
just built. Writes results both under Dinura-Person3/results/ and
(merged, additive) into Lasana-Person4_Evaluation/results/, and drops
matching checkpoints into Lasana-Person4_Evaluation/checkpoints/ so
`python evaluate.py --model segformer[-att]` works for real from Person 4's
own folder afterwards (see adapters/segformer.py, added alongside).

Usage:
    python eval_segformer.py --variant vanilla
    python eval_segformer.py --variant att
    python eval_segformer.py --variant both   # default
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PHASE1_DIR = HERE.parent
LASANA_EVAL_DIR = PHASE1_DIR / "Lasana-Person4_Evaluation"
sys.path.insert(0, str(LASANA_EVAL_DIR))  # for aamo.py, metrics.py, efficiency.py, evaluate.py

from aamo import mean_aamo  # noqa: E402
from metrics import ConfusionCounts, binarize, format_metrics, metrics_from_counts  # noqa: E402
from efficiency import count_torch_params, efficiency_row, gflops_torch, measure_fps  # noqa: E402
from evaluate import row_dict, save_prediction_grid, write_comparison_table  # noqa: E402

from attention_consistency import build_segformer, grad_rollout_attention_map
from attention_consistency.data import load_pairs, make_splits, to_model_input
from attention_consistency.segformer_model import forest_prob

CKPT_DIR = HERE / "checkpoints"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

VARIANT_LABEL = {
    "vanilla": "SegFormer-B0 (no attention loss)",
    "att": "SegFormer-B0 + Attention Consistency Loss",
}
# Matches config.py's expected filenames in Lasana-Person4_Evaluation, so
# evaluate.py --model segformer[-att] resolves to real weights.
LASANA_CKPT_NAME = {
    "vanilla": "segformer_b0_vanilla.pt",
    "att": "segformer_b0_att_consistency.pt",
}


def load_checkpoint(variant: str) -> torch.nn.Module:
    ckpt_path = CKPT_DIR / f"segformer_b0_{variant}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"{ckpt_path} not found — run train_segformer_smoke.py --variant {variant} first."
        )
    model = build_segformer(pretrained=False)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def evaluate_variant(variant: str, args) -> dict:
    print(f"\n{'='*60}\nEvaluating: {VARIANT_LABEL[variant]}\n{'='*60}")
    model, ckpt = load_checkpoint(variant)

    # Same seed/sizes as train_segformer_smoke.py defaults -> identical,
    # never-seen-during-training test split.
    splits = make_splits(args.n_train, args.n_val, args.n_test, seed=args.seed)
    images, masks = load_pairs(splits["test"])
    print(f"test set: {len(images)} images (held out from training, seed={args.seed})")

    probs_list, attn_list = [], []
    for i in range(len(images)):
        x = to_model_input(images[i : i + 1])
        with torch.enable_grad():
            attn_map, outputs = grad_rollout_attention_map(model, x)
        logits_up = F.interpolate(outputs.logits, size=(256, 256), mode="bilinear", align_corners=False)
        probs = forest_prob(logits_up)[0].detach().numpy()
        probs_list.append(probs)
        attn_list.append(attn_map.numpy())

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

    warm_x = to_model_input(images[0:1])

    def _run():
        with torch.no_grad():
            model(pixel_values=warm_x)

    speed = measure_fps(_run, warmup=3, runs=10)
    eff = efficiency_row(params=params, gflops=gflops, fps=speed["fps"], ms_per_image=speed["ms_per_image"])
    print(f"params={eff['params']} | gflops={eff['gflops']} | fps={eff['fps']} | ms/img={eff['ms_per_image']}")

    row = row_dict(VARIANT_LABEL[variant], seg, eff, aamo_val)
    row["checkpoint_epoch"] = ckpt["epoch"]
    row["checkpoint_val_dice"] = round(float(ckpt["val_dice"]), 4)

    # Person-3-local copy of results.
    save_prediction_grid(images, masks, probs, RESULTS_DIR / f"prediction_grid_{variant}.png", threshold=0.5)
    (RESULTS_DIR / f"eval_{variant}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    write_comparison_table([row], RESULTS_DIR)

    # Fill Lasana's actual pending row + drop a real checkpoint so
    # evaluate.py --model segformer[-att] works from Person 4's own folder.
    write_comparison_table([row], LASANA_EVAL_DIR / "results")
    (LASANA_EVAL_DIR / "checkpoints").mkdir(exist_ok=True)
    dst_ckpt = LASANA_EVAL_DIR / "checkpoints" / LASANA_CKPT_NAME[variant]
    shutil.copy2(CKPT_DIR / f"segformer_b0_{variant}_best.pt", dst_ckpt)
    print(f"Copied checkpoint -> {dst_ckpt}")

    return row


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="both", choices=["vanilla", "att", "both"])
    p.add_argument("--n-train", type=int, default=400)
    p.add_argument("--n-val", type=int, default=60)
    p.add_argument("--n-test", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    variants = ["vanilla", "att"] if args.variant == "both" else [args.variant]
    rows = [evaluate_variant(v, args) for v in variants]
    print("\nDone. Rows written:")
    for r in rows:
        print(" ", r["model"], "-> dice", r["dice"], "iou", r["iou"], "aamo", r["aamo"])


if __name__ == "__main__":
    main()
