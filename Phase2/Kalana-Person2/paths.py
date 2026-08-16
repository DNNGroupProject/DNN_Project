"""Path helpers for the Phase 2 full-scale SegFormer run.

All train/eval/figure writes go under this folder. Teammate folders are
added to sys.path for imports only.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]  # Phase2/Kalana-Person2 -> repo root

PERSON3_DIR = REPO_ROOT / "Phase1" / "Dinura-Person3"
PERSON4_DIR = REPO_ROOT / "Phase1" / "Lasana-Person4_Evaluation"
CHANUPA_DIR = REPO_ROOT / "Phase1" / "Chanupa-Person1"
DATA_DIR = REPO_ROOT / "Phase1" / "Kalana-Person2"
DATA_IMG_DIR = DATA_DIR / "images"
DATA_MASK_DIR = DATA_DIR / "masks"

CKPT_DIR = HERE / "checkpoints"
RESULTS_DIR = HERE / "results"

N_TRAIN, N_VAL, N_TEST = 3576, 766, 766
SEED = 42
EPOCHS = 20
BATCH_SIZE = 16
LR = 6e-5
LAMBDA2 = 0.3
SIGMA = 8.0
ATT_MODE = "mse"


def add_teammate_paths() -> None:
    """Import Person 3's package and Person 4's eval modules. No writes."""
    for d in (str(PERSON4_DIR), str(PERSON3_DIR)):
        if d not in sys.path:
            sys.path.insert(0, d)


def ensure_output_dirs() -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def apply_data_dirs(img_dir: Path | None = None, mask_dir: Path | None = None) -> None:
    """Override attention_consistency.data's dataset root (needed when the
    Drive layout is not repo-root/Phase1/...). Safe no-op if the package is
    not imported yet — call after add_teammate_paths()."""
    add_teammate_paths()
    import attention_consistency.data as data_mod

    data_mod.IMG_DIR = Path(img_dir) if img_dir is not None else DATA_IMG_DIR
    data_mod.MASK_DIR = Path(mask_dir) if mask_dir is not None else DATA_MASK_DIR
