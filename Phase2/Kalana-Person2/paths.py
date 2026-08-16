"""Path helpers for the Phase 2 full-scale SegFormer run.

Works in two layouts:

- **repo** — this file lives at `Phase2/Kalana-Person2/paths.py` and teammate
  code / dataset stay in `Phase1/`.
- **Colab zip** — this file sits next to `vendor/attention_consistency/` and
  `data/{images,masks}/` inside the unzipped bundle.

All train/eval/figure writes go under CKPT_DIR / RESULTS_DIR. On Colab those
should be pointed at Drive so a disconnected runtime does not eat the run.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

N_TRAIN, N_VAL, N_TEST = 3576, 766, 766
SEED = 42
EPOCHS = 20
BATCH_SIZE = 16
LR = 6e-5
LAMBDA2 = 0.3
SIGMA = 8.0
ATT_MODE = "mse"


def _layout() -> dict:
    vendor_pkg = HERE / "vendor" / "attention_consistency"
    if vendor_pkg.is_dir():
        return {
            "mode": "bundle",
            "repo_root": HERE,
            "person3": HERE / "vendor",
            "person4": HERE / "vendor",
            "chanupa": HERE / "vendor" / "chanupa",
            "data_img": HERE / "data" / "images",
            "data_mask": HERE / "data" / "masks",
        }
    repo = HERE.parents[1]
    return {
        "mode": "repo",
        "repo_root": repo,
        "person3": repo / "Phase1" / "Dinura-Person3",
        "person4": repo / "Phase1" / "Lasana-Person4_Evaluation",
        "chanupa": repo / "Phase1" / "Chanupa-Person1",
        "data_img": repo / "Phase1" / "Kalana-Person2" / "images",
        "data_mask": repo / "Phase1" / "Kalana-Person2" / "masks",
    }


_L = _layout()
LAYOUT_MODE = _L["mode"]
REPO_ROOT = _L["repo_root"]
PERSON3_DIR = _L["person3"]
PERSON4_DIR = _L["person4"]
CHANUPA_DIR = _L["chanupa"]
DATA_DIR = _L["data_img"].parent
DATA_IMG_DIR = _L["data_img"]
DATA_MASK_DIR = _L["data_mask"]
CKPT_DIR = HERE / "checkpoints"
RESULTS_DIR = HERE / "results"


def add_teammate_paths() -> None:
    """Import Person 3's package and Person 4's eval modules. No writes."""
    for d in (str(PERSON4_DIR), str(PERSON3_DIR)):
        if d not in sys.path:
            sys.path.insert(0, d)


def ensure_output_dirs() -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def set_output_dirs(ckpt_dir: Path | None = None, results_dir: Path | None = None) -> None:
    """Point checkpoints/results at Drive (Colab) so they survive a runtime drop."""
    global CKPT_DIR, RESULTS_DIR
    if ckpt_dir is not None:
        CKPT_DIR = Path(ckpt_dir)
    if results_dir is not None:
        RESULTS_DIR = Path(results_dir)
    ensure_output_dirs()


def apply_data_dirs(img_dir: Path | None = None, mask_dir: Path | None = None) -> None:
    """Override attention_consistency.data's dataset root."""
    global DATA_IMG_DIR, DATA_MASK_DIR, DATA_DIR
    add_teammate_paths()
    DATA_IMG_DIR = Path(img_dir) if img_dir is not None else DATA_IMG_DIR
    DATA_MASK_DIR = Path(mask_dir) if mask_dir is not None else DATA_MASK_DIR
    DATA_DIR = DATA_IMG_DIR.parent
    import attention_consistency.data as data_mod

    data_mod.IMG_DIR = DATA_IMG_DIR
    data_mod.MASK_DIR = DATA_MASK_DIR
