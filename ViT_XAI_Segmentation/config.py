"""Configuration for ViT + Concept-Guided XAI forest segmentation."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent

# Dataset (same Forest Segmented data as Lasana)
DATA_DIR = PROJECT / "Lasana" / "dataset" / "Forest Segmented" / "Forest Segmented"
IMG_DIR = DATA_DIR / "images"
MASK_DIR = DATA_DIR / "masks"

CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
CHECKPOINT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Image
IMG_SIZE = 224          # ViT-friendly (divisible by patch size)
PATCH_SIZE = 16         # 224/16 = 14 → 196 tokens
MASK_THRESHOLD = 127
SEED = 42
MAX_SAMPLES = None      # set int for CPU smoke runs, e.g. 800

# Split
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10

# ViT encoder (lightweight for CPU; raise dims on GPU)
EMBED_DIM = 192
DEPTH = 6
NUM_HEADS = 3
MLP_RATIO = 4.0
DROP_RATE = 0.1
ATTN_DROP = 0.0

# Concept layer (adapted from arXiv:2101.03919)
# Forest-domain concepts (no CUB captions available → pseudo-labels)
CONCEPT_NAMES = [
    "dense_canopy",      # high forest coverage, dark green
    "sparse_trees",      # mixed forest / background
    "shadow_region",     # dark regions often at canopy edges
    "open_clearing",     # bright non-forest
    "forest_boundary",   # edge band between forest and clear
]
NUM_CONCEPTS = len(CONCEPT_NAMES)
CONCEPT_EMBED_DIM = 32

# Loss weights (paper: L = L_A + λ(L_u + L_m))
LAMBDA_CONCEPT = 0.4    # λ in the paper
ALPHA_TRIPLET = 0.5     # margin α for semantic consistency
BETA_COUNTER = 0.5      # margin β for counter loss
BCE_WEIGHT = 0.5
DICE_WEIGHT = 0.5

# Train
BATCH_SIZE = 4
NUM_EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0

DEVICE = "cuda"  # overwritten in train.py if no GPU
