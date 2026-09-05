"""Build Phase2/Dhinanjaya-Person5/vanilla_train_colab.zip for Colab.

One-zip-upload version of Kalana-Person2's full-scale training pipeline:
bundles his (unmodified) train_full_scale.py / eval_full_scale.py /
generate_full_scale_figures.py / paths.py + vendored attention_consistency
+ Person 4 metrics + the full 5,108-image dataset, PLUS our own already-local
lambda2=1.0 attention checkpoint (checkpoints/runs/l2_1_mse/segformer_b0_att_best.pt)
baked directly into the bundle's checkpoints/ folder.

vanilla_train_colab.ipynb (also in this folder) auto-copies that bundled
checkpoint into the Drive-persistent output dir before training starts, so
Kalana's own per-variant "skip if checkpoint already exists" guard skips
attention (already have a properly-tuned one) and only spends GPU time on
vanilla -- no separate manual checkpoint upload needed, one zip does it all.

    python Phase2/Dhinanjaya-Person5/make_vanilla_train_colab_zip.py
"""
from __future__ import annotations

import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PERSON2 = HERE.parent / "Kalana-Person2"
PERSON3 = HERE.parent / "Dinura-Person3"
REPO = HERE.parents[1]
ZIP_PATH = HERE / "vanilla_train_colab.zip"
ROOT = Path("segformer_full_scale")

ATTN = REPO / "Phase1" / "Dinura-Person3" / "attention_consistency"
P4 = REPO / "Phase1" / "Lasana-Person4_Evaluation"
DATA = REPO / "Phase1" / "Kalana-Person2"
ATT_CKPT = PERSON3 / "checkpoints" / "runs" / "l2_1_mse" / "segformer_b0_att_best.pt"

PERSON2_FILES = [
    "paths.py",
    "train_full_scale.py",
    "eval_full_scale.py",
    "generate_full_scale_figures.py",
]
P4_FILES = ["aamo.py", "metrics.py", "efficiency.py"]


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
    for name in PERSON2_FILES:
        if not (PERSON2 / name).exists():
            raise SystemExit(f"Missing Kalana-Person2/{name}")
    if not ATT_CKPT.exists():
        raise SystemExit(f"Missing {ATT_CKPT}")

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    print("Writing", ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in PERSON2_FILES:
            add_file(z, PERSON2 / name, ROOT / name)
        add_file(
            z,
            PERSON2 / "tests" / "test_split_identity.py",
            ROOT / "tests" / "test_split_identity.py",
        )
        add_file(z, HERE / "vanilla_train_colab.ipynb", ROOT / "vanilla_train_colab.ipynb")
        for src in sorted(ATTN.glob("*.py")):
            add_file(z, src, ROOT / "vendor" / "attention_consistency" / src.name)
        for name in P4_FILES:
            add_file(z, P4 / name, ROOT / "vendor" / name)
        add_file(z, ATT_CKPT, ROOT / "checkpoints" / "segformer_b0_att_best.pt")
        z.writestr(
            (ROOT / "README.txt").as_posix(),
            "One-zip full-scale vanilla training + Figure 2 regeneration.\n"
            "1. Upload as MyDrive/vanilla_train_colab.zip\n"
            "2. Open vanilla_train_colab.ipynb (GPU runtime) and Run all\n"
            "3. It trains ONLY vanilla (~2h) -- attention is pre-seeded from\n"
            "   this zip's bundled lambda2=1.0 checkpoint, so it's skipped.\n"
            "4. Downloads attention_drift_*_full_scale.png + segformer_b0_vanilla_best.pt\n"
            "   automatically at the end.\n",
        )
        print("Adding images…")
        add_dir(z, img, ROOT / "data" / "images")
        print("Adding masks…")
        add_dir(z, mask, ROOT / "data" / "masks")

    mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    print(f"Wrote {ZIP_PATH} ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
