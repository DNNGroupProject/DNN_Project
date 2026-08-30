"""
Fold full-scale Table-1 rows into Phase 2 Person 4 results.

Sources (no model loading — table joins only):
  - U-Net:          Phase1/Lasana-Person4_Evaluation/results/baseline_comparison.csv
  - SegFormer-B0:   Phase2/Kalana-Person2/results/baseline_comparison.csv (vanilla)
  - SegFormer+Att:  Phase2/Dinura-Person3/results/runs/l2_1_mse/  (sweep winner λ2=1.0 MSE)
  - Boundary Loss:  pending (Dhinanjaya still integrating)
  - DeepLabV3+:     Phase1/Lasana-Person4_Evaluation/results/baseline_comparison.csv (extra)

Writes:
  results/baseline_comparison_full_scale.csv
  results/baseline_comparison_full_scale.md
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent  # DNN_Project
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

PHASE1_P4 = PROJECT / "Phase1" / "Lasana-Person4_Evaluation" / "results"
KALANA = PROJECT / "Phase2" / "Kalana-Person2" / "results"
DINURA_WIN = PROJECT / "Phase2" / "Dinura-Person3" / "results" / "runs" / "l2_1_mse"
WINNING_CFG = PROJECT / "Phase2" / "Dinura-Person3" / "results" / "winning_config.json"

FIELDS = [
    "model",
    "dice",
    "iou",
    "f1",
    "precision",
    "recall",
    "pixel_acc",
    "aamo",
    "params",
    "gflops",
    "fps",
    "ms_per_image",
    "source",
    "n_seeds",
    "notes",
]


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _find_row(rows: List[Dict[str, str]], substr: str) -> Optional[Dict[str, str]]:
    for r in rows:
        if substr in (r.get("model") or ""):
            return r
    return None


def _row(
    model: str,
    src: Optional[Dict[str, str]],
    *,
    source: str,
    n_seeds: int = 1,
    notes: str = "",
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {k: "" for k in FIELDS}
    out["model"] = model
    out["source"] = source
    out["n_seeds"] = n_seeds
    out["notes"] = notes
    if src:
        for k in FIELDS:
            if k in ("model", "source", "n_seeds", "notes"):
                continue
            if k in src and src[k] not in (None, ""):
                out[k] = src[k]
    if overrides:
        out.update(overrides)
    return out


def fold() -> List[Dict[str, Any]]:
    p4 = _read_csv_rows(PHASE1_P4 / "baseline_comparison.csv")
    kalana = _read_csv_rows(KALANA / "baseline_comparison.csv")
    dinura = _read_csv_rows(DINURA_WIN / "baseline_comparison.csv")

    winner_meta = {}
    if WINNING_CFG.exists():
        winner_meta = json.loads(WINNING_CFG.read_text(encoding="utf-8")).get("winner", {})

    unet = _find_row(p4, "U-Net")
    vanilla = _find_row(kalana, "no attention")
    att = _find_row(dinura, "Attention Consistency")
    deeplab = _find_row(p4, "DeepLab")

    att_notes = (
        f"Dinura sweep winner {winner_meta.get('run_tag', 'l2_1_mse')} "
        f"(λ2={winner_meta.get('lambda2', 1.0)}, {winner_meta.get('att_mode', 'mse')}); "
        "selection: max test AAMO then max Dice. "
        "Supersedes Kalana default-λ2=0.3 attention row."
    )

    rows = [
        _row(
            "U-Net (CNN baseline)",
            unet,
            source="Phase1/Lasana-Person4_Evaluation (Chanupa PyTorch ckpt)",
            notes="Full-scale 3576/766/766 seed 42; dataset-wide Dice/IoU.",
        ),
        _row(
            "SegFormer-B0 (no attention loss)",
            vanilla,
            source="Phase2/Kalana-Person2 (full-scale Colab)",
            notes="Full-scale 3576/766/766 seed 42; Person 4 metrics/aamo formulas.",
        ),
        _row(
            "SegFormer-B0 + Attention Consistency Loss (λ2=1.0 MSE)",
            att,
            source="Phase2/Dinura-Person3/results/runs/l2_1_mse",
            notes=att_notes,
        ),
        _row(
            "SegFormer-B0 + Attention Consistency + Boundary Loss",
            None,
            source="pending",
            n_seeds=0,
            notes="Blocked until Dhinanjaya wires Boundary Refinement into training.",
            overrides={
                "dice": "-",
                "iou": "-",
                "f1": "-",
                "aamo": "pending",
                "params": "-",
                "gflops": "-",
                "fps": "-",
            },
        ),
        _row(
            "DeepLabV3+ (MobileNetV3) — extra baseline",
            deeplab,
            source="Phase1/Lasana-Person4_Evaluation (CPU smoke)",
            notes=(
                "CPU smoke (400 samples / 5 epochs). Multi-seed mean±std for this "
                "row is produced by train_deeplab_multiseed.py (seeds 42/43/44)."
            ),
        ),
    ]
    return rows


def write_tables(rows: List[Dict[str, Any]]) -> None:
    csv_path = RESULTS / "baseline_comparison_full_scale.csv"
    md_path = RESULTS / "baseline_comparison_full_scale.md"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    lines = [
        "# Full-scale baseline comparison (Phase 2 / Lasana-Person4)",
        "",
        "Folded from Chanupa (U-Net), Kalana (SegFormer-B0 vanilla), Dinura",
        "(`l2_1_mse` attention winner), and Person 4's DeepLabV3+ extra baseline.",
        "All full-scale rows share the 3576/766/766 seed-42 held-out test set.",
        "",
        "| Model | Dice | IoU | F1 | AAMO | Params | GFLOPs | FPS | Source |",
        "|-------|------|-----|----|------|--------|--------|-----|--------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r.get('dice', '-')} | {r.get('iou', '-')} | "
            f"{r.get('f1', '-')} | {r.get('aamo', '-')} | {r.get('params', '-')} | "
            f"{r.get('gflops', '-')} | {r.get('fps', '-')} | {r.get('source', '')} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Attention-consistency row uses Dinura's sweep winner "
            "(`λ2=1.0`, MSE, run tag `l2_1_mse`: Dice 0.8577 / IoU 0.7508 / "
            "AAMO 0.7476), not Kalana's default-λ2=0.3 attention run.",
            "- Boundary Loss row stays pending until Person 5 finishes integration.",
            "- DeepLabV3+ is CPU smoke-scale; see `ablation_mean_std.md` for the "
            "multi-seed (42/43/44) mean±std of that extra baseline.",
            "- Phase 1 `Lasana-Person4_Evaluation/results/` is left untouched "
            "(frozen short-paper snapshot per CONTRIBUTING.md).",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def main() -> None:
    rows = fold()
    write_tables(rows)


if __name__ == "__main__":
    main()
