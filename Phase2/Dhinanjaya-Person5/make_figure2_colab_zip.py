"""Build Phase2/Dhinanjaya-Person5/figure2_colab.zip for Colab.

Regenerates Figure 2 (qualitative attention-drift comparison) using the real
lambda2=1.0 attention checkpoint instead of the stale lambda2=0.3 one the
paper's caption has been flagging as "map pending". Mirrors
Phase2/Dinura-Person3/make_colab_zip.py's flat-bundle pattern
(generate_full_scale_figures.py sits at the bundle root, teammate code
under vendor/, dataset under data/) but only bundles what that one script
needs -- no training code, no P4 metrics (generate_full_scale_figures.py
doesn't import them).

The lambda2=1.0 "att" checkpoint (checkpoints/runs/l2_1_mse/segformer_b0_att_best.pt)
is bundled directly since it's already local on this machine. The vanilla
full-scale checkpoint (Kalana's) is NOT bundled -- it was never fetched to
this machine, only exists on Drive -- so the companion notebook prompts an
upload for it before running.

    python Phase2/Dhinanjaya-Person5/make_figure2_colab_zip.py
"""
from __future__ import annotations

import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PERSON3 = HERE.parent / "Dinura-Person3"
REPO = HERE.parents[1]
ZIP_PATH = HERE / "figure2_colab.zip"
ROOT = Path("figure2")

ATTN = REPO / "Phase1" / "Dinura-Person3" / "attention_consistency"
DATA = REPO / "Phase1" / "Kalana-Person2"
ATT_CKPT = PERSON3 / "checkpoints" / "runs" / "l2_1_mse" / "segformer_b0_att_best.pt"

PERSON3_FILES = [
    "paths.py",
    "generate_full_scale_figures.py",
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
    for name in PERSON3_FILES:
        if not (PERSON3 / name).exists():
            raise SystemExit(f"Missing Dinura-Person3/{name}")
    if not ATT_CKPT.exists():
        raise SystemExit(f"Missing {ATT_CKPT}")

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    print("Writing", ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in PERSON3_FILES:
            add_file(z, PERSON3 / name, ROOT / name)
        for src in sorted(ATTN.glob("*.py")):
            add_file(z, src, ROOT / "vendor" / "attention_consistency" / src.name)
        add_file(z, ATT_CKPT, ROOT / "checkpoints" / "segformer_b0_att_best.pt")
        z.writestr(
            (ROOT / "README.txt").as_posix(),
            "Figure 2 regeneration (real lambda2=1.0 attention checkpoint).\n"
            "1. Upload as MyDrive/figure2_colab.zip\n"
            "2. Open figure2_colab.ipynb and Run all (CPU is fine, GPU optional)\n"
            "3. When prompted, upload segformer_b0_vanilla_best.pt (Kalana's full-scale\n"
            "   vanilla SegFormer-B0 checkpoint) -- NOT bundled here, only ever lived\n"
            "   on Drive, never fetched to the machine that built this zip.\n"
            "4. Download the resulting attention_drift_01_full_scale.png.\n"
            "segformer_b0_att_best.pt (the lambda2=1.0 winner, l2_1_mse) IS already\n"
            "bundled at checkpoints/ -- no need to re-upload it.\n",
        )
        print("Adding images…")
        add_dir(z, img, ROOT / "data" / "images")
        print("Adding masks…")
        add_dir(z, mask, ROOT / "data" / "masks")

    mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    print(f"Wrote {ZIP_PATH} ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
