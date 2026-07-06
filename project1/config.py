import os

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(ROOT_DIR, '..', 'data', 'Forest Segmented', 'Forest Segmented')
IMG_DIR        = os.path.join(DATA_DIR, 'images')
MASK_DIR       = os.path.join(DATA_DIR, 'masks')
CSV_PATH       = os.path.join(DATA_DIR, 'meta_data.csv')
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoints')
RESULTS_DIR    = os.path.join(ROOT_DIR, 'results')

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR,    exist_ok=True)

# ─── Data ─────────────────────────────────────────────────────────────────────
IMG_SIZE        = 256   # input spatial resolution (H = W)
MASK_THRESHOLD  = 127   # binarise JPEG-compressed masks (0=background, 1=forest)

# Train / Val / Test split ratios
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10
RANDOM_SEED = 42

# ─── Model ────────────────────────────────────────────────────────────────────
MODEL_NAME   = 'unet_baseline'
IN_CHANNELS  = 3
OUT_CHANNELS = 1                     # binary segmentation
FEATURES     = [64, 128, 256, 512]   # U-Net encoder channel progression

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE   = 16
NUM_EPOCHS   = 50
LR           = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 0    # keep 0 for Windows multiprocessing compatibility

# ─── Loss ─────────────────────────────────────────────────────────────────────
BCE_WEIGHT  = 0.5
DICE_WEIGHT = 0.5

# ─── LR Scheduler (CosineAnnealingLR) ─────────────────────────────────────────
LR_T_MAX   = NUM_EPOCHS
LR_ETA_MIN = 1e-6

# ─── Checkpoint ───────────────────────────────────────────────────────────────
BEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, f'{MODEL_NAME}_best.pth')
LAST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, f'{MODEL_NAME}_last.pth')
