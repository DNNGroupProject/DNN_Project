"""Aggregate λ2-sweep results and pick the winning config (Person 3, Phase 2).

Scans results/runs/l2_*/ for train_summary_att.json + eval_att.json, writes:
  results/sweep_comparison.{csv,md}
  results/winning_config.json

Selection rule:
  1. Highest test AAMO (paper's attention-faithfulness metric).
  2. Break ties with highest test Dice.
  3. Prefer MSE over KL on ties.

Usage:
    python aggregate_sweep.py
"""
from __future__ import annotations

import csv
import json
import re
from typing import Any, Dict, List, Optional

from paths import OUTPUT_ROOT_RESULTS, ensure_output_dirs

TAG_RE = re.compile(r"^l2_(.+?)_(mse|kl)$")


def _parse_tag(tag: str) -> Optional[Dict[str, Any]]:
    m = TAG_RE.match(tag)
    if not m:
        return None
    return {"run_tag": tag, "lambda2": float(m.group(1)), "att_mode": m.group(2)}


def collect_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    runs_dir = OUTPUT_ROOT_RESULTS / "runs"
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

        final_val_l_att = None
        log_path = run_dir / "training_log_att.csv"
        if log_path.exists():
            with open(log_path, newline="", encoding="utf-8") as f:
                log_rows = list(csv.DictReader(f))
            if log_rows and "val_l_att" in log_rows[-1]:
                final_val_l_att = float(log_rows[-1]["val_l_att"])

        rows.append(
            {
                "run_tag": meta["run_tag"],
                "lambda2": meta["lambda2"],
                "att_mode": meta["att_mode"],
                "best_val_dice": train.get("best_val_dice", ""),
                "final_val_dice": train.get("final_val_dice", ""),
                "final_val_iou": train.get("final_val_iou", ""),
                "final_val_l_att": final_val_l_att if final_val_l_att is not None else "",
                "wall_clock_s": train.get("wall_clock_s", ""),
                "test_dice": ev.get("dice", ""),
                "test_iou": ev.get("iou", ""),
                "test_aamo": ev.get("aamo", ""),
                "checkpoint_epoch": ev.get("checkpoint_epoch", ""),
                "source": train.get("source", "dinura_sweep"),
                "status": "complete" if ev else ("trained" if train else "empty"),
            }
        )
    return rows


def pick_winner(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    scored = [r for r in rows if r.get("test_aamo") != "" and r.get("test_dice") != ""]
    if not scored:
        scored = [r for r in rows if r.get("best_val_dice") != ""]
        if not scored:
            return None

        def key_val(r):
            mode_rank = 0 if r["att_mode"] == "mse" else 1
            return (-float(r["best_val_dice"]), mode_rank, float(r["lambda2"]))

        return sorted(scored, key=key_val)[0]

    def key_test(r):
        mode_rank = 0 if r["att_mode"] == "mse" else 1
        return (-float(r["test_aamo"]), -float(r["test_dice"]), mode_rank, float(r["lambda2"]))

    return sorted(scored, key=key_test)[0]


def _fmt(v) -> str:
    if v == "" or v is None:
        return "—"
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)


def write_sweep_table() -> Dict[str, Any]:
    ensure_output_dirs()
    rows = collect_rows()
    fields = [
        "run_tag", "lambda2", "att_mode", "best_val_dice", "final_val_dice",
        "final_val_iou", "final_val_l_att", "test_dice", "test_iou", "test_aamo",
        "checkpoint_epoch", "wall_clock_s", "source", "status",
    ]
    csv_path = OUTPUT_ROOT_RESULTS / "sweep_comparison.csv"
    md_path = OUTPUT_ROOT_RESULTS / "sweep_comparison.md"
    win_path = OUTPUT_ROOT_RESULTS / "winning_config.json"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    winner = pick_winner(rows)
    lines = [
        "# λ2 / att-mode sweep (Phase 2 / Dinura-Person3)",
        "",
        "Full-scale attention-variant only. Split 3576/766/766 seed 42 "
        "(same held-out test set as Chanupa U-Net + Kalana default-λ2 run).",
        "",
        "**Selection rule:** highest test AAMO, then highest test Dice; "
        "MSE preferred over KL on ties. Cells without test eval fall back "
        "to best val Dice.",
        "",
        "| run | λ2 | mode | best val Dice | test Dice | test IoU | test AAMO | status |",
        "|-----|----|------|---------------|-----------|----------|-----------|--------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['run_tag']} | {r['lambda2']} | {r['att_mode']} | "
            f"{_fmt(r['best_val_dice'])} | {_fmt(r['test_dice'])} | "
            f"{_fmt(r['test_iou'])} | {_fmt(r['test_aamo'])} | {r['status']} |"
        )
    lines.append("")
    if winner:
        lines.append(
            f"**Current winner:** `{winner['run_tag']}` "
            f"(λ2={winner['lambda2']}, {winner['att_mode']}) — "
            f"test AAMO={_fmt(winner.get('test_aamo'))}, "
            f"test Dice={_fmt(winner.get('test_dice'))}, "
            f"best val Dice={_fmt(winner.get('best_val_dice'))}."
        )
        lines.append("")
        lines.append(
            "Checkpoint path (once trained): "
            f"`checkpoints/runs/{winner['run_tag']}/segformer_b0_att_best.pt`"
        )
        if winner.get("source") == "kalana_default_lambda2":
            lines.append("")
            lines.append(
                "This cell was seeded from Kalana's default-λ2=0.3 MSE full-scale "
                "run (`Phase2/Kalana-Person2/results/`). The `.pt` still lives on "
                "his Drive (`MyDrive/segformer_full_scale_outputs/checkpoints/`); "
                "copy it into this run folder before Lasana / Boundary Refinement "
                "integration if needed."
            )
    else:
        lines.append("**No winner yet** — no completed sweep cells found.")
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "winner": winner,
        "n_cells": len(rows),
        "n_complete": sum(1 for r in rows if r["status"] == "complete"),
        "selection_rule": "max test AAMO, then max test Dice, MSE preferred on ties",
    }
    win_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {win_path}")
    return payload


def main():
    payload = write_sweep_table()
    w = payload.get("winner")
    if w:
        print(
            f"Winner: {w['run_tag']}  aamo={_fmt(w.get('test_aamo'))}  "
            f"dice={_fmt(w.get('test_dice'))}"
        )
    else:
        print("No winner yet.")


if __name__ == "__main__":
    main()
