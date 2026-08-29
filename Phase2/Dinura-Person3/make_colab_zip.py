"""Build Phase2/Dinura-Person3/lambda_sweep.zip for Colab.

Includes sweep code, Person 3 attention_consistency, Person 4 metrics,
Kalana's λ2=0.3 result files (for seeding), and the 5,108-pair dataset.

    python Phase2/Dinura-Person3/make_colab_zip.py
"""
from __future__ import annotations

import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ZIP_PATH = HERE / "lambda_sweep.zip"
ROOT = Path("lambda_sweep")

ATTN = REPO / "Phase1" / "Dinura-Person3" / "attention_consistency"
P4 = REPO / "Phase1" / "Lasana-Person4_Evaluation"
DATA = REPO / "Phase1" / "Kalana-Person2"
KALANA_RESULTS = REPO / "Phase2" / "Kalana-Person2" / "results"

CODE_FILES = [
    "paths.py",
    "train_full_scale.py",
    "eval_full_scale.py",
    "generate_full_scale_figures.py",
    "run_lambda_sweep.py",
    "eval_lambda_sweep.py",
    "aggregate_sweep.py",
    "seed_from_kalana.py",
    "lambda_sweep_colab.ipynb",
]
P4_FILES = ["aamo.py", "metrics.py", "efficiency.py"]
KALANA_SEED_FILES = [
    "train_summary_att.json",
    "training_log_att.csv",
    "eval_att.json",
]


def add_file(zf: zipfile.ZipFile, src: Path, arc: Path) -> None:
    zf.write(src, arc.as_posix())


def add_dir(zf: zipfile.ZipFile, src: Path, arc_prefix: Path) -> None:
    files = [p for p in src.iterdir() if p.is_file()]
    for i, path in enumerate(files, 1):
        add_file(zf, path, arc_prefix / path.name)
        if i % 1000 == 0 or i == len(files):
            print(f"  {arc_prefix.as_posix()}: {i}/{len(files)}")


def main() -> None:
    img = DATA / "images"
    mask = DATA / "masks"
    if not img.is_dir() or not mask.is_dir():
        raise SystemExit(f"Dataset missing under {DATA}")
    for name in CODE_FILES:
        if not (HERE / name).exists():
            raise SystemExit(f"Missing {name}")

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    print("Writing", ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in CODE_FILES:
            add_file(z, HERE / name, ROOT / name)
        add_file(z, HERE / "tests" / "test_sweep_helpers.py", ROOT / "tests" / "test_sweep_helpers.py")
        for src in sorted(ATTN.glob("*.py")):
            add_file(z, src, ROOT / "vendor" / "attention_consistency" / src.name)
        for name in P4_FILES:
            add_file(z, P4 / name, ROOT / "vendor" / name)
        for name in KALANA_SEED_FILES:
            src = KALANA_RESULTS / name
            if src.exists():
                add_file(z, src, ROOT / "vendor" / "kalana_phase2" / "results" / name)
        z.writestr(
            (ROOT / "README.txt").as_posix(),
            "Dinura Phase 2 λ2 sweep.\n"
            "1. Upload as MyDrive/lambda_sweep.zip\n"
            "2. Open lambda_sweep_colab.ipynb (GPU) and Run all\n"
            "3. Outputs -> MyDrive/lambda_sweep_outputs/\n",
        )
        z.writestr((ROOT / "results" / ".keep").as_posix(), "")
        z.writestr((ROOT / "checkpoints" / ".keep").as_posix(), "")
        print("Adding images…")
        add_dir(z, img, ROOT / "data" / "images")
        print("Adding masks…")
        add_dir(z, mask, ROOT / "data" / "masks")

    mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    print(f"Wrote {ZIP_PATH} ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
