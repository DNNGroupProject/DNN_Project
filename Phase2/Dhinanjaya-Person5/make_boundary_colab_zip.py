"""Build Phase2/Dhinanjaya-Person5/boundary_sweep.zip for Colab.

Mirrors Phase2/Dinura-Person3/make_colab_zip.py's flat-bundle pattern
(train_full_scale.py etc. sit at the bundle root, teammate code under
vendor/, dataset under data/) but for the Boundary Loss λ3 sweep: adds
this folder's boundary_refinement/ package + sweep scripts alongside
Dinura's training/eval code, so train_full_scale.py's lazy boundary
import (_load_boundary_dice_loss_cls) finds boundary_refinement/ as a
sibling of itself in the bundle (see that function's docstring).

    python Phase2/Dhinanjaya-Person5/make_boundary_colab_zip.py
"""
from __future__ import annotations

import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PERSON3 = HERE.parent / "Dinura-Person3"
REPO = HERE.parents[1]
ZIP_PATH = HERE / "boundary_sweep.zip"
ROOT = Path("boundary_sweep")

ATTN = REPO / "Phase1" / "Dinura-Person3" / "attention_consistency"
P4 = REPO / "Phase1" / "Lasana-Person4_Evaluation"
DATA = REPO / "Phase1" / "Kalana-Person2"

# Dinura's training/eval code + paths.py (owns the checkpoints/results roots).
PERSON3_FILES = [
    "paths.py",
    "train_full_scale.py",
    "eval_full_scale.py",
]
P4_FILES = ["aamo.py", "metrics.py", "efficiency.py"]

# This folder's own sweep code.
OWN_CODE_FILES = [
    "run_boundary_sweep.py",
    "eval_boundary_sweep.py",
    "aggregate_boundary_sweep.py",
    "boundary_sweep_colab.ipynb",
]
BOUNDARY_REFINEMENT_FILES = ["__init__.py", "boundary_ops.py", "loss.py"]


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
    for name in PERSON3_FILES:
        if not (PERSON3 / name).exists():
            raise SystemExit(f"Missing Dinura-Person3/{name}")
    for name in OWN_CODE_FILES:
        if not (HERE / name).exists():
            raise SystemExit(f"Missing {name} (build boundary_sweep_colab.ipynb first?)")
    for name in BOUNDARY_REFINEMENT_FILES:
        if not (HERE / "boundary_refinement" / name).exists():
            raise SystemExit(f"Missing boundary_refinement/{name}")

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    print("Writing", ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in PERSON3_FILES:
            add_file(z, PERSON3 / name, ROOT / name)
        for name in OWN_CODE_FILES:
            add_file(z, HERE / name, ROOT / name)
        for name in BOUNDARY_REFINEMENT_FILES:
            add_file(z, HERE / "boundary_refinement" / name, ROOT / "boundary_refinement" / name)
        add_file(z, HERE / "tests" / "test_boundary_sweep_helpers.py", ROOT / "tests" / "test_boundary_sweep_helpers.py")
        add_file(z, HERE / "tests" / "test_boundary_refinement.py", ROOT / "tests" / "test_boundary_refinement.py")
        for src in sorted(ATTN.glob("*.py")):
            add_file(z, src, ROOT / "vendor" / "attention_consistency" / src.name)
        for name in P4_FILES:
            add_file(z, P4 / name, ROOT / "vendor" / name)
        z.writestr(
            (ROOT / "README.txt").as_posix(),
            "Person 5 Boundary Loss lambda3 sweep.\n"
            "1. Upload as MyDrive/boundary_sweep.zip\n"
            "2. Open boundary_sweep_colab.ipynb (GPU) and Run all\n"
            "3. Outputs -> MyDrive/boundary_sweep_outputs/\n"
            "Fixed at Dinura's lambda2-sweep winner (l2_1_mse: lambda2=1.0, MSE) -- "
            "this sweep only varies lambda3 (Boundary Dice Loss weight).\n",
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
