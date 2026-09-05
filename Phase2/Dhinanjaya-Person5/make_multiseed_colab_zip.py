"""Build Phase2/Dhinanjaya-Person5/multiseed_train_colab.zip -- one shared
bundle for distributing the multi-seed + MSE-vs-KL GPU work across the team.

Bundles Dinura-Person3's (unmodified) train_full_scale.py/eval_full_scale.py/
paths.py -- already supports everything needed via CLI flags (--variant,
--seed, --lambda2, --att-mode, --lambda3, --boundary-kernel), no code changes
needed -- plus this folder's boundary_refinement/ package (for lambda3>0
runs), vendored attention_consistency + Person 4 metrics, and the full
5,108-image dataset. No checkpoint is bundled: every job trains fresh.

Every teammate uploads this SAME zip; multiseed_train_colab.ipynb's one
config cell (VARIANT/SEED/LAMBDA2/ATT_MODE/LAMBDA3/OUTPUT_TAG) picks which
of the 7 assigned jobs that person's Colab session runs -- see
multiseed_kickoff.md for the assignment table.

    python Phase2/Dhinanjaya-Person5/make_multiseed_colab_zip.py
"""
from __future__ import annotations

import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PERSON3 = HERE.parent / "Dinura-Person3"
REPO = HERE.parents[1]
ZIP_PATH = HERE / "multiseed_train_colab.zip"
ROOT = Path("multiseed_train")

ATTN = REPO / "Phase1" / "Dinura-Person3" / "attention_consistency"
P4 = REPO / "Phase1" / "Lasana-Person4_Evaluation"
DATA = REPO / "Phase1" / "Kalana-Person2"

PERSON3_FILES = [
    "paths.py",
    "train_full_scale.py",
    "eval_full_scale.py",
]
P4_FILES = ["aamo.py", "metrics.py", "efficiency.py"]
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
    for name in BOUNDARY_REFINEMENT_FILES:
        if not (HERE / "boundary_refinement" / name).exists():
            raise SystemExit(f"Missing boundary_refinement/{name}")
    nb = HERE / "multiseed_train_colab.ipynb"
    if not nb.exists():
        raise SystemExit(f"Missing {nb.name} (build it first)")

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    print("Writing", ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in PERSON3_FILES:
            add_file(z, PERSON3 / name, ROOT / name)
        for name in BOUNDARY_REFINEMENT_FILES:
            add_file(z, HERE / "boundary_refinement" / name, ROOT / "boundary_refinement" / name)
        add_file(z, nb, ROOT / nb.name)
        for src in sorted(ATTN.glob("*.py")):
            add_file(z, src, ROOT / "vendor" / "attention_consistency" / src.name)
        for name in P4_FILES:
            add_file(z, P4 / name, ROOT / "vendor" / name)
        z.writestr(
            (ROOT / "README.txt").as_posix(),
            "Shared multi-seed / MSE-vs-KL training bundle -- see\n"
            "Phase2/Dhinanjaya-Person5/multiseed_kickoff.md for your assigned job.\n"
            "1. Upload as MyDrive/multiseed_train_colab.zip\n"
            "2. Open multiseed_train_colab.ipynb, edit the ONE config cell to your\n"
            "   assigned job's values, Run all.\n"
            "3. Send back MyDrive/multiseed_outputs_<OUTPUT_TAG>/ when done.\n",
        )
        print("Adding images…")
        add_dir(z, img, ROOT / "data" / "images")
        print("Adding masks…")
        add_dir(z, mask, ROOT / "data" / "masks")

    mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    print(f"Wrote {ZIP_PATH} ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
