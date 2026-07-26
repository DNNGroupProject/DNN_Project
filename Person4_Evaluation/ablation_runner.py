"""
Ablation study runner (Person 4, Weeks 7–9).

Proposal configs (Table 2):
  1. U-Net (CNN baseline)
  2. SegFormer-B0 (no attention loss)
  3. SegFormer-B0 + Attention Consistency Loss
  4. SegFormer-B0 + Attention Consistency + Boundary Loss (stretch)

Multi-seed: report mean ± std when multiple seed result JSONs exist.

Usage
-----
    python ablation_runner.py
    python ablation_runner.py --seeds 42 43 44
    python ablation_runner.py --only unet
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import config


CONFIGS = [
    {"key": "unet", "model": "unet", "label": "U-Net (CNN baseline)"},
    {
        "key": "segformer_vanilla",
        "model": "segformer",
        "label": "SegFormer-B0 (no attention loss)",
    },
    {
        "key": "segformer_att",
        "model": "segformer-att",
        "label": "SegFormer-B0 + Attention Consistency Loss",
    },
    {
        "key": "segformer_boundary",
        "model": "segformer-boundary",
        "label": "SegFormer-B0 + Attention Consistency + Boundary Loss",
    },
]


def run_one(model: str, seed: int, max_samples: Optional[int]) -> Optional[Dict[str, Any]]:
    """Invoke evaluate.evaluate_one programmatically."""
    from evaluate import evaluate_one
    from adapters.segformer_stub import SegFormerNotReadyError

    class Args:
        pass

    args = Args()
    args.model = model
    args.checkpoint = None
    args.split = "test"
    args.threshold = config.DEFAULT_THRESHOLD
    args.aamo_thr = config.AAMO_THRESHOLD
    args.batch_size = 4
    args.max_samples = max_samples
    args.attention_npy = None
    args.no_grid = True

    # Seed is recorded for future multi-seed training; current U-Net ckpt is single-seed.
    try:
        row = evaluate_one(args)
        row["seed"] = seed
        return row
    except SegFormerNotReadyError as e:
        print(f"[skip] {model}: {e}")
        return None
    except FileNotFoundError as e:
        print(f"[skip] {model}: {e}")
        return None


def aggregate_mean_std(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    numeric = ["dice", "iou", "f1", "precision", "recall", "pixel_acc"]
    out: Dict[str, Any] = {"model": rows[0]["model"], "n_seeds": len(rows)}
    for k in numeric:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[k]))
            except (TypeError, ValueError, KeyError):
                pass
        if vals:
            out[k] = f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"
        else:
            out[k] = "n/a"
    # AAMO
    avals = []
    for r in rows:
        try:
            if r.get("aamo") not in (None, "n/a", ""):
                avals.append(float(r["aamo"]))
        except (TypeError, ValueError):
            pass
    out["aamo"] = (
        f"{np.mean(avals):.4f} ± {np.std(avals):.4f}" if avals else "n/a"
    )
    out["params"] = rows[0].get("params", "n/a")
    out["gflops"] = rows[0].get("gflops", "n/a")
    return out


def write_ablation_tables(per_seed: List[Dict[str, Any]], summary: List[Dict[str, Any]]):
    out_dir = config.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "ablation_per_seed.csv"
    if per_seed:
        fields = list(per_seed[0].keys())
        with open(raw_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(per_seed)
        print(f"Wrote {raw_path}")

    sum_path = out_dir / "ablation_mean_std.csv"
    md_path = out_dir / "ablation_mean_std.md"
    if summary:
        fields = [
            "model",
            "n_seeds",
            "dice",
            "iou",
            "f1",
            "aamo",
            "params",
            "gflops",
        ]
        with open(sum_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in summary:
                w.writerow({k: r.get(k, "") for k in fields})

        lines = [
            "# Ablation results (mean ± std)",
            "",
            "| Model | Seeds | Dice | IoU | F1 | AAMO |",
            "|-------|-------|------|-----|----|------|",
        ]
        for r in summary:
            lines.append(
                f"| {r.get('model')} | {r.get('n_seeds')} | {r.get('dice')} | "
                f"{r.get('iou')} | {r.get('f1')} | {r.get('aamo')} |"
            )
        lines.append("")
        lines.append(
            "SegFormer rows appear once Person 2/3 checkpoints are wired. "
            "Multi-seed mean±std requires separately trained checkpoints per seed."
        )
        md_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Wrote {sum_path}")
        print(f"Wrote {md_path}")


def main():
    p = argparse.ArgumentParser(description="Person 4 ablation runner")
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument(
        "--only",
        default=None,
        help="Run only one config key: unet | segformer_vanilla | segformer_att | segformer_boundary",
    )
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()

    configs = CONFIGS
    if args.only:
        configs = [c for c in CONFIGS if c["key"] == args.only]
        if not configs:
            raise SystemExit(f"Unknown --only {args.only}")

    per_seed: List[Dict[str, Any]] = []
    by_model: Dict[str, List[Dict[str, Any]]] = {}

    for cfg in configs:
        for seed in args.seeds:
            print(f"\n=== {cfg['label']} | seed={seed} ===")
            row = run_one(cfg["model"], seed, args.max_samples)
            if row is None:
                # Placeholder row for proposal table
                placeholder = {
                    "model": cfg["label"],
                    "seed": seed,
                    "dice": "-",
                    "iou": "-",
                    "f1": "-",
                    "aamo": "n/a" if "U-Net" in cfg["label"] else "pending",
                    "params": "n/a",
                    "gflops": "n/a",
                    "status": "pending_checkpoint",
                }
                per_seed.append(placeholder)
                continue
            per_seed.append(row)
            by_model.setdefault(row["model"], []).append(row)

    summary = [aggregate_mean_std(rs) for rs in by_model.values()]
    # Also list pending configs with empty summary note
    for cfg in configs:
        if cfg["label"] not in by_model:
            summary.append(
                {
                    "model": cfg["label"],
                    "n_seeds": 0,
                    "dice": "-",
                    "iou": "-",
                    "f1": "-",
                    "aamo": "n/a" if "U-Net" in cfg["label"] else "pending",
                    "params": "n/a",
                    "gflops": "n/a",
                }
            )

    write_ablation_tables(per_seed, summary)
    print("\nAblation scaffold done.")


if __name__ == "__main__":
    main()
