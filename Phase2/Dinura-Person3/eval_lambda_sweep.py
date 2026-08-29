"""Evaluate every completed λ2 / att-mode sweep cell (Person 3, Phase 2).

Reads checkpoints under checkpoints/runs/l2_*/ and writes eval_att.json into
the matching results/runs/l2_*/ folder, then rebuilds the sweep table.

Usage:
    python eval_lambda_sweep.py
    python eval_lambda_sweep.py --lambda2 0.1 0.5 --att-mode mse
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import paths
from paths import (
    ATT_MODE,
    DEFAULT_LAMBDA2_SWEEP,
    N_TEST,
    N_TRAIN,
    N_VAL,
    SEED,
    apply_data_dirs,
    use_run_dirs,
)
import eval_full_scale as E
from aggregate_sweep import write_sweep_table


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Person 3 λ2 sweep cells")
    p.add_argument("--lambda2", type=float, nargs="+", default=list(DEFAULT_LAMBDA2_SWEEP))
    p.add_argument("--att-mode", default=ATT_MODE, choices=["mse", "kl"])
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--n-train", type=int, default=N_TRAIN)
    p.add_argument("--n-val", type=int, default=N_VAL)
    p.add_argument("--n-test", type=int, default=N_TEST)
    p.add_argument("--img-dir", type=Path, default=None)
    p.add_argument("--mask-dir", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    apply_data_dirs(args.img_dir or paths.DATA_IMG_DIR, args.mask_dir or paths.DATA_MASK_DIR)

    class EvalArgs:
        n_train, n_val, n_test = args.n_train, args.n_val, args.n_test
        seed = args.seed

    rows = []
    for lam in args.lambda2:
        tag = use_run_dirs(lam, args.att_mode)
        best = paths.CKPT_DIR / "segformer_b0_att_best.pt"
        if not best.exists():
            print(f"SKIP eval {tag}: no checkpoint at {best}")
            continue
        print(f"\n>>> Evaluating sweep cell {tag}")
        row = E.evaluate_variant("att", EvalArgs())
        row["lambda2"] = float(lam)
        row["att_mode"] = args.att_mode
        row["run_tag"] = tag
        row["model"] = f"SegFormer-B0 + Att (λ2={lam:g}, {args.att_mode})"
        (paths.RESULTS_DIR / "eval_att.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows.append(row)

    write_sweep_table()
    print("\nEval pass done.")
    for r in rows:
        print(f"  {r['model']}: dice={r['dice']} iou={r['iou']} aamo={r['aamo']}")


if __name__ == "__main__":
    main()
