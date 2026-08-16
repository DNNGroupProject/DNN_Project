"""Build Phase2/Kalana-Person2/segformer_full_scale.zip for Colab.

Contains only what the GPU run needs: this folder's code, Person 3's
attention_consistency package, Person 4's metric modules, and the 5,108
image/mask pairs. Not the rest of the repo.

Run from anywhere:
    python Phase2/Kalana-Person2/make_colab_zip.py
"""
from __future__ import annotations

import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ZIP_PATH = HERE / "segformer_full_scale.zip"
ROOT = Path("segformer_full_scale")

ATTN = REPO / "Phase1" / "Dinura-Person3" / "attention_consistency"
P4 = REPO / "Phase1" / "Lasana-Person4_Evaluation"
DATA = REPO / "Phase1" / "Kalana-Person2"

CODE_FILES = [
    "paths.py",
    "train_full_scale.py",
    "eval_full_scale.py",
    "generate_full_scale_figures.py",
    "segformer_full_scale_colab.ipynb",
]
P4_FILES = ["aamo.py", "metrics.py", "efficiency.py"]
README_TXT = (
    "Colab bundle for the full-scale SegFormer run (Person 2).\n\n"
    "1. Upload this zip to Google Drive as MyDrive/segformer_full_scale.zip\n"
    "2. Open segformer_full_scale_colab.ipynb in Colab (GPU runtime) and Run all\n"
    "3. Checkpoints/results land on Drive under MyDrive/segformer_full_scale_outputs/\n"
)


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

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    print("Writing", ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in CODE_FILES:
            add_file(z, HERE / name, ROOT / name)
        add_file(z, HERE / "tests" / "test_split_identity.py", ROOT / "tests" / "test_split_identity.py")
        for src in sorted(ATTN.glob("*.py")):
            add_file(z, src, ROOT / "vendor" / "attention_consistency" / src.name)
        for name in P4_FILES:
            add_file(z, P4 / name, ROOT / "vendor" / name)
        z.writestr((ROOT / "README.txt").as_posix(), README_TXT)
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
