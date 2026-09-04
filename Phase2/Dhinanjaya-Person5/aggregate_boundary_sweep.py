"""Aggregate Boundary Loss λ3-sweep results and pick the winning config
(Person 5, Phase 2 Week 10, proposal §3.4).

Scans Dinura-Person3's results/runs/l2_*_bnd*/ (where run_boundary_sweep.py
/ train_full_scale.py's --lambda3 write, since they reuse Dinura's paths.py
output roots — see this folder's README) for train_summary_att.json +
eval_att.json, and writes the summary into **this** folder (Person 5's own
results/, per CONTRIBUTING.md's "write to your own folder" convention —
mirrors Phase2/Lasana-Person4/fold_full_scale_results.py reading teammate
folders but writing its own table):
  results/boundary_sweep_comparison.{csv,md}
  results/boundary_winning_config.json

Selection rule differs from Dinura's λ2 sweep (aggregate_sweep.py, which is
AAMO-first — λ2/L_att targets attention interpretability). L_boundary
targets segmentation/boundary *quality*, not attention faithfulness, so:
  1. Highest test Dice.
  2. Break ties with highest test IoU.
AAMO is still recorded per cell for reference — L_att's weight/mode is
unchanged across the λ3 sweep, so a cell's AAMO should stay close to the
λ2=1.0 winner's (0.7476); a large drop would flag L_boundary fighting
L_att and is worth a second look before trusting that cell's Dice/IoU win.

Usage:
    python aggregate_boundary_sweep.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent


def _find_person3_dir() -> Path:
    """Same layout-detection as run_boundary_sweep.py's _find_person3_dir."""
    for candidate in (HERE.parent / "Dinura-Person3", HERE):
        if (candidate / "paths.py").is_file() and (candidate / "train_full_scale.py").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find Dinura-Person3's paths.py/train_full_scale.py. "
        f"Expected {HERE.parent / 'Dinura-Person3'} (repo) or {HERE} (Colab bundle)."
    )


PERSON3_DIR = _find_person3_dir()
if str(PERSON3_DIR) not in sys.path:
    sys.path.insert(0, str(PERSON3_DIR))

from paths import OUTPUT_ROOT_RESULTS as DINURA_RESULTS  # noqa: E402

OWN_RESULTS = HERE / "results"

TAG_RE = re.compile(r"^l2_(.+?)_(mse|kl)_bnd(.+)$")


def _parse_tag(tag: str) -> Optional[Dict[str, Any]]:
    m = TAG_RE.match(tag)
    if not m:
        return None
    return {"run_tag": tag, "lambda2": float(m.group(1)), "att_mode": m.group(2), "lambda3": float(m.group(3))}


def collect_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    runs_dir = DINURA_RESULTS / "runs"
    if not runs_dir.is_dir():
        return rows
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        meta = _parse_tag(run_dir.name)
        if meta is None:
            continue
        train_path = run_dir / "train_summary_att.json"
        eval_path = run_dir / "eval_att.json"
        train = json.loads(train_path.read_text(encoding="utf-8")) if train_path.exists() else {}
        ev = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else {}

        final_val_l_boundary = None
        log_path = run_dir / "training_log_att.csv"
        if log_path.exists():
            with open(log_path, newline="", encoding="utf-8") as f:
                log_rows = list(csv.DictReader(f))
            if log_rows and "val_l_boundary" in log_rows[-1]:
                final_val_l_boundary = float(log_rows[-1]["val_l_boundary"])

        rows.append(
            {
                "run_tag": meta["run_tag"],
                "lambda2": meta["lambda2"],
                "att_mode": meta["att_mode"],
                "lambda3": meta["lambda3"],
                "best_val_dice": train.get("best_val_dice", ""),
                "final_val_dice": train.get("final_val_dice", ""),
                "final_val_iou": train.get("final_val_iou", ""),
                "final_val_l_boundary": final_val_l_boundary if final_val_l_boundary is not None else "",
                "wall_clock_s": train.get("wall_clock_s", ""),
                "test_dice": ev.get("dice", ""),
                "test_iou": ev.get("iou", ""),
                "test_aamo": ev.get("aamo", ""),
                "checkpoint_epoch": ev.get("checkpoint_epoch", ""),
                "status": "complete" if ev else ("trained" if train else "empty"),
            }
        )
    return rows


def pick_winner(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    scored = [r for r in rows if r.get("test_dice") != ""]
    if not scored:
        scored = [r for r in rows if r.get("best_val_dice") != ""]
        if not scored:
            return None

        def key_val(r):
            return (-float(r["best_val_dice"]), float(r["lambda3"]))

        return sorted(scored, key=key_val)[0]

    def key_test(r):
        return (-float(r["test_dice"]), -float(r["test_iou"] or 0), float(r["lambda3"]))

    return sorted(scored, key=key_test)[0]


def _fmt(v) -> str:
    if v == "" or v is None:
        return "—"
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)


def write_boundary_sweep_table() -> Dict[str, Any]:
    OWN_RESULTS.mkdir(parents=True, exist_ok=True)
    rows = collect_rows()
    fields = [
        "run_tag", "lambda2", "att_mode", "lambda3", "best_val_dice", "final_val_dice",
        "final_val_iou", "final_val_l_boundary", "test_dice", "test_iou", "test_aamo",
        "checkpoint_epoch", "wall_clock_s", "status",
    ]
    csv_path = OWN_RESULTS / "boundary_sweep_comparison.csv"
    md_path = OWN_RESULTS / "boundary_sweep_comparison.md"
    win_path = OWN_RESULTS / "boundary_winning_config.json"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    winner = pick_winner(rows)
    lines = [
        "# Boundary Loss λ3 sweep (Person 5, Phase 2 Week 10)",
        "",
        "Fixed at Dinura's λ2-sweep winner (λ2=1.0, MSE, `l2_1_mse`). Full-scale "
        "attention variant only. Split 3576/766/766 seed 42 (same held-out test "
        "set as every other full-scale row). Raw per-cell artifacts live under "
        "`Phase2/Dinura-Person3/{checkpoints,results}/runs/l2_1_mse_bnd<λ3>/` "
        "(train_full_scale.py's `--lambda3` output routing) — this table just "
        "summarizes them.",
        "",
        "**Selection rule (differs from Dinura's λ2 sweep):** highest test Dice, "
        "then highest test IoU. L_boundary targets segmentation/boundary quality, "
        "not attention faithfulness — AAMO is recorded for reference only and "
        "should stay close to the λ2=1.0 winner's AAMO (0.7476); a large drop "
        "would flag L_boundary fighting L_att.",
        "",
        "| run | λ3 | best val Dice | test Dice | test IoU | test AAMO | status |",
        "|-----|----|--------------|-----------|----------|-----------|--------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['run_tag']} | {r['lambda3']} | {_fmt(r['best_val_dice'])} | "
            f"{_fmt(r['test_dice'])} | {_fmt(r['test_iou'])} | {_fmt(r['test_aamo'])} | {r['status']} |"
        )
    lines.append("")
    if winner:
        lines.append(
            f"**Current winner:** `{winner['run_tag']}` (λ3={winner['lambda3']}) — "
            f"test Dice={_fmt(winner.get('test_dice'))}, "
            f"test IoU={_fmt(winner.get('test_iou'))}, "
            f"test AAMO={_fmt(winner.get('test_aamo'))}."
        )
        lines.append("")
        lines.append(
            "Checkpoint path (once trained): "
            f"`Phase2/Dinura-Person3/checkpoints/runs/{winner['run_tag']}/segformer_b0_att_best.pt`"
        )
    else:
        lines.append("**No winner yet** — no completed sweep cells found.")
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "winner": winner,
        "n_cells": len(rows),
        "n_complete": sum(1 for r in rows if r["status"] == "complete"),
        "selection_rule": "max test Dice, then max test IoU (boundary quality, not AAMO)",
    }
    win_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {win_path}")
    return payload


def main():
    payload = write_boundary_sweep_table()
    w = payload.get("winner")
    if w:
        print(f"Winner: {w['run_tag']}  dice={_fmt(w.get('test_dice'))}  iou={_fmt(w.get('test_iou'))}")
    else:
        print("No winner yet.")


if __name__ == "__main__":
    main()
