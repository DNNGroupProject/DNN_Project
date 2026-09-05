"""
Fold full-scale Table-1 rows into Phase 2 Person 4 results.

Sources (no model loading — table joins only):
  - U-Net:          Phase1/Lasana-Person4_Evaluation/results/baseline_comparison.csv
  - SegFormer-B0:   Phase2/Kalana-Person2/results/baseline_comparison.csv (vanilla)
  - SegFormer+Att:  Phase2/Dinura-Person3/results/runs/l2_1_mse/  (sweep winner λ2=1.0 MSE)
  - Boundary Loss:  Phase2/Dhinanjaya-Person5/results/baseline_comparison.csv (λ2=1.0, λ3=0.2 sweep winner)
  - DeepLabV3+:     results/deeplab_multiseed.json seed-42 entry (single eval path)

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
DEEPLAB_MULTI = RESULTS / "deeplab_multiseed.json"

BOUNDARY = PROJECT / "Phase2" / "Dhinanjaya-Person5" / "results" / "baseline_comparison.csv"

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


def _deeplab_seed42_row() -> Dict[str, Any]:
    """Single evaluation path: seed-42 entry from train_deeplab_multiseed.py."""
    if not DEEPLAB_MULTI.exists():
        raise FileNotFoundError(
            f"Missing {DEEPLAB_MULTI}\n"
            "Run: python train_deeplab_multiseed.py --skip-train"
        )
    entries = json.loads(DEEPLAB_MULTI.read_text(encoding="utf-8"))
    for e in entries:
        if int(e.get("seed", -1)) == 42:
            return e
    raise ValueError(f"No seed=42 entry in {DEEPLAB_MULTI}")


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
    boundary = _read_csv_rows(BOUNDARY)

    winner_meta = {}
    if WINNING_CFG.exists():
        winner_meta = json.loads(WINNING_CFG.read_text(encoding="utf-8")).get("winner", {})

    unet = _find_row(p4, "U-Net")
    vanilla = _find_row(kalana, "no attention")
    att = _find_row(dinura, "Attention Consistency")
    bound = _find_row(boundary, "Boundary")
    dl42 = _deeplab_seed42_row()

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
            bound,
            source="Phase2/Dhinanjaya-Person5/results/baseline_comparison.csv "
            "(run l2_1_mse_bnd0.2)",
            notes=(
                "λ3-sweep winner (λ2=1.0 MSE, λ3=0.2, boundary_kernel=3); "
                "selection: max test Dice then max test IoU. AAMO 0.6218 vs "
                "0.7476 for the λ3=0 attention row — see "
                "Phase2/Dhinanjaya-Person5/results/baseline_comparison.md."
            ),
        ),
        _row(
            "DeepLabV3+ (MobileNetV3) — extra baseline",
            None,
            source="Phase2/Lasana-Person4/train_deeplab_multiseed.py (seed 42, 400-sample smoke)",
            notes=(
                "CPU smoke (400 samples / 5 epochs), seed-42 eval from "
                "deeplab_multiseed.json. Multi-seed mean±std in ablation_mean_std.md."
            ),
            overrides={
                "dice": dl42["dice"],
                "iou": dl42["iou"],
                "f1": dl42["f1"],
                "precision": dl42.get("precision", ""),
                "recall": dl42.get("recall", ""),
                "pixel_acc": dl42.get("pixel_acc", ""),
                "aamo": dl42.get("aamo", "n/a"),
                "params": dl42.get("params", ""),
                "gflops": dl42.get("gflops", "n/a"),
            },
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
        "U-Net / SegFormer / L_att rows share the 3576/766/766 seed-42 test set;",
        "the DeepLabV3+ row is a 400-sample CPU-smoke subset evaluated by",
        "`train_deeplab_multiseed.py`.",
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
            "- Boundary Loss row is the λ3-sweep winner (λ2=1.0 MSE, λ3=0.2) "
            "from Phase2/Dhinanjaya-Person5/results/; note its AAMO (0.6218) "
            "drops vs. the plain attention row's 0.7476.",
            "- DeepLabV3+ Dice/IoU come from `deeplab_multiseed.json` seed 42 "
            "(same path as the multi-seed ablation); see `ablation_mean_std.md` "
            "for seeds 42/43/44 mean±std.",
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
