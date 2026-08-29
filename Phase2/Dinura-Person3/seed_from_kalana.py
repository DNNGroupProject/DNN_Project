"""Seed the λ2=0.3 / MSE sweep cell from Kalana's finished full-scale run.

Copies train summary, training log, and eval JSON into
results/runs/l2_0.3_mse/ (not the .pt — too large / on Drive).

Usage:
    python seed_from_kalana.py
"""
from __future__ import annotations

import json
import shutil

from paths import KALANA_PHASE2_DIR, run_results_dir, run_tag

LAMBDA2 = 0.3
ATT_MODE = "mse"
FILES = ("train_summary_att.json", "training_log_att.csv", "eval_att.json")


def main() -> None:
    src = KALANA_PHASE2_DIR / "results"
    dst = run_results_dir(LAMBDA2, ATT_MODE)
    dst.mkdir(parents=True, exist_ok=True)

    missing = [n for n in FILES if not (src / n).exists()]
    if missing:
        raise SystemExit(
            f"Kalana results missing under {src}: {missing}. "
            "Need Phase2/Kalana-Person2/results/ from his full-scale run."
        )

    for name in FILES:
        shutil.copy2(src / name, dst / name)

    summary_path = dst / "train_summary_att.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["source"] = "kalana_default_lambda2"
    summary["seeded_from"] = str(src.as_posix())
    summary["run_tag"] = run_tag(LAMBDA2, ATT_MODE)
    summary["note"] = (
        "Metrics only. Checkpoint remains on Kalana's Drive at "
        "MyDrive/segformer_full_scale_outputs/checkpoints/segformer_b0_att_best.pt"
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    eval_path = dst / "eval_att.json"
    ev = json.loads(eval_path.read_text(encoding="utf-8"))
    ev["lambda2"] = LAMBDA2
    ev["att_mode"] = ATT_MODE
    ev["run_tag"] = run_tag(LAMBDA2, ATT_MODE)
    ev["source"] = "kalana_default_lambda2"
    ev["model"] = f"SegFormer-B0 + Att (λ2={LAMBDA2:g}, {ATT_MODE})"
    eval_path.write_text(json.dumps(ev, indent=2), encoding="utf-8")

    print(f"Seeded {dst} from {src}")
    print("Next: python aggregate_sweep.py")


if __name__ == "__main__":
    main()
