"""Path helpers for Person 3's Phase 2 λ2-sweep.

Writes under Phase2/Dinura-Person3/ only. Teammate code is import-only.
Supports repo layout and the Colab zip layout (vendor/ + data/).

OUTPUT_ROOT_CKPT / OUTPUT_ROOT_RESULTS are the sweep roots (e.g. a Drive
folder on Colab). Per-cell dirs are always `<root>/runs/l2_<λ>_<mode>/`.
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

# Kickoff-suggested sweep (proposal initial λ2=0.3 included).
DEFAULT_LAMBDA2_SWEEP = (0.1, 0.3, 0.5, 1.0)


def _layout() -> dict:
    vendor_pkg = HERE / "vendor" / "attention_consistency"
    if vendor_pkg.is_dir():
        return {
            "mode": "bundle",
            "repo_root": HERE,
            "person3": HERE / "vendor",
            "person4": HERE / "vendor",
            "data_img": HERE / "data" / "images",
            "data_mask": HERE / "data" / "masks",
            "kalana_phase2": HERE / "vendor" / "kalana_phase2",
        }
    repo = HERE.parents[1]
    return {
        "mode": "repo",
        "repo_root": repo,
        "person3": repo / "Phase1" / "Dinura-Person3",
        "person4": repo / "Phase1" / "Lasana-Person4_Evaluation",
        "data_img": repo / "Phase1" / "Kalana-Person2" / "images",
        "data_mask": repo / "Phase1" / "Kalana-Person2" / "masks",
        "kalana_phase2": repo / "Phase2" / "Kalana-Person2",
    }


_L = _layout()
LAYOUT_MODE = _L["mode"]
REPO_ROOT = _L["repo_root"]
PERSON3_DIR = _L["person3"]
PERSON4_DIR = _L["person4"]
KALANA_PHASE2_DIR = _L["kalana_phase2"]
DATA_DIR = _L["data_img"].parent
DATA_IMG_DIR = _L["data_img"]
DATA_MASK_DIR = _L["data_mask"]

OUTPUT_ROOT_CKPT = HERE / "checkpoints"
OUTPUT_ROOT_RESULTS = HERE / "results"
CKPT_DIR = OUTPUT_ROOT_CKPT
RESULTS_DIR = OUTPUT_ROOT_RESULTS


def run_tag(lambda2: float, att_mode: str) -> str:
    """Folder name for one sweep cell, e.g. l2_0.3_mse."""
    return f"l2_{float(lambda2):g}_{att_mode}"


def run_ckpt_dir(lambda2: float, att_mode: str) -> Path:
    return OUTPUT_ROOT_CKPT / "runs" / run_tag(lambda2, att_mode)


def run_results_dir(lambda2: float, att_mode: str) -> Path:
    return OUTPUT_ROOT_RESULTS / "runs" / run_tag(lambda2, att_mode)


def add_teammate_paths() -> None:
    for d in (str(PERSON4_DIR), str(PERSON3_DIR)):
        if d not in sys.path:
            sys.path.insert(0, d)


def ensure_output_dirs() -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def set_output_roots(ckpt_root: Path | None = None, results_root: Path | None = None) -> None:
    """Set sweep roots (Drive folder on Colab). Does not select a cell."""
    global OUTPUT_ROOT_CKPT, OUTPUT_ROOT_RESULTS, CKPT_DIR, RESULTS_DIR
    if ckpt_root is not None:
        OUTPUT_ROOT_CKPT = Path(ckpt_root)
    if results_root is not None:
        OUTPUT_ROOT_RESULTS = Path(results_root)
    CKPT_DIR = OUTPUT_ROOT_CKPT
    RESULTS_DIR = OUTPUT_ROOT_RESULTS
    ensure_output_dirs()


def set_output_dirs(ckpt_dir: Path | None = None, results_dir: Path | None = None) -> None:
    """Point active write dirs (usually one sweep cell)."""
    global CKPT_DIR, RESULTS_DIR
    if ckpt_dir is not None:
        CKPT_DIR = Path(ckpt_dir)
    if results_dir is not None:
        RESULTS_DIR = Path(results_dir)
    ensure_output_dirs()


def use_run_dirs(lambda2: float, att_mode: str) -> str:
    """Point CKPT_DIR / RESULTS_DIR at one cell under the current roots."""
    tag = run_tag(lambda2, att_mode)
    set_output_dirs(run_ckpt_dir(lambda2, att_mode), run_results_dir(lambda2, att_mode))
    return tag


def apply_data_dirs(img_dir: Path | None = None, mask_dir: Path | None = None) -> None:
    global DATA_IMG_DIR, DATA_MASK_DIR, DATA_DIR
    add_teammate_paths()
    DATA_IMG_DIR = Path(img_dir) if img_dir is not None else DATA_IMG_DIR
    DATA_MASK_DIR = Path(mask_dir) if mask_dir is not None else DATA_MASK_DIR
    DATA_DIR = DATA_IMG_DIR.parent
    import attention_consistency.data as data_mod

    data_mod.IMG_DIR = DATA_IMG_DIR
    data_mod.MASK_DIR = DATA_MASK_DIR
