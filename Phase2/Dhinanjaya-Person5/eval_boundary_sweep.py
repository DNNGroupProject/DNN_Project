"""Evaluate every completed Boundary Loss λ3 sweep cell (Person 5, Phase 2).

Reads checkpoints under Dinura-Person3/checkpoints/runs/l2_1_mse_bnd<λ3>/
and writes eval_att.json into the matching results folder, then rebuilds
the boundary-sweep table (see aggregate_boundary_sweep.py). Mirrors
Phase2/Dinura-Person3/eval_lambda_sweep.py's pattern.

Usage:
    python eval_boundary_sweep.py
    python eval_boundary_sweep.py --lambda3 0.1 0.5
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_boundary_sweep import (  # noqa: E402
    DEFAULT_LAMBDA3_SWEEP,
    WINNER_ATT_MODE,
    WINNER_LAMBDA2,
    PERSON3_DIR,
    use_boundary_run_dirs,
)

if str(PERSON3_DIR) not in sys.path:
    sys.path.insert(0, str(PERSON3_DIR))

import argparse  # noqa: E402
import paths  # noqa: E402
from paths import N_TEST, N_TRAIN, N_VAL, SEED, apply_data_dirs  # noqa: E402
import eval_full_scale as E  # noqa: E402

from aggregate_boundary_sweep import write_boundary_sweep_table  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Person 5 Boundary Loss λ3 sweep cells")
    p.add_argument("--lambda3", type=float, nargs="+", default=list(DEFAULT_LAMBDA3_SWEEP))
    p.add_argument("--lambda2", type=float, default=WINNER_LAMBDA2)
    p.add_argument("--att-mode", default=WINNER_ATT_MODE, choices=["mse", "kl"])
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
    for lam3 in args.lambda3:
        tag = use_boundary_run_dirs(args.lambda2, args.att_mode, lam3)
        best = paths.CKPT_DIR / "segformer_b0_att_best.pt"
        if not best.exists():
            print(f"SKIP eval {tag}: no checkpoint at {best}")
            continue
        print(f"\n>>> Evaluating boundary sweep cell {tag}")
        row = E.evaluate_variant("att", EvalArgs())
        row["lambda2"] = float(args.lambda2)
        row["att_mode"] = args.att_mode
        row["lambda3"] = float(lam3)
        row["run_tag"] = tag
        row["model"] = f"SegFormer-B0 + Att + Boundary (λ2={args.lambda2:g}, λ3={lam3:g}, {args.att_mode})"
        (paths.RESULTS_DIR / "eval_att.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows.append(row)

    write_boundary_sweep_table()
    print("\nEval pass done.")
    for r in rows:
        print(f"  {r['model']}: dice={r['dice']} iou={r['iou']} aamo={r['aamo']}")


if __name__ == "__main__":
    main()
