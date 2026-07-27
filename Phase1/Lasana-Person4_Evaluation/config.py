"""Paths and defaults for Person 4 evaluation."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent

# Shared Forest Segmented dataset (same as Lasana)
DATA_DIR = PROJECT / "Lasana" / "dataset" / "Forest Segmented" / "Forest Segmented"
IMG_DIR = DATA_DIR / "images"
MASK_DIR = DATA_DIR / "masks"

# Existing U-Net checkpoint (Person 1 / Lasana)
UNET_CKPT = PROJECT / "Lasana" / "checkpoints" / "lasana_unet_best.keras"

# Placeholders — Person 2 / 3 drop checkpoints here when ready
SEGFORMER_CKPT = ROOT / "checkpoints" / "segformer_b0_vanilla.pt"
SEGFORMER_ATT_CKPT = ROOT / "checkpoints" / "segformer_b0_att_consistency.pt"
SEGFORMER_BOUND_CKPT = ROOT / "checkpoints" / "segformer_b0_att_boundary.pt"

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "checkpoints").mkdir(parents=True, exist_ok=True)

SEED = 42
IMG_SIZE = 256
MASK_THRESHOLD = 127
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
DEFAULT_THRESHOLD = 0.5
AAMO_THRESHOLD = 0.5

# Efficiency probe input
FLOPS_INPUT_SHAPE = (1, IMG_SIZE, IMG_SIZE, 3)  # Keras NHWC
TORCH_FLOPS_INPUT = (1, 3, IMG_SIZE, IMG_SIZE)  # PyTorch NCHW
FPS_WARMUP = 3
FPS_RUNS = 20
