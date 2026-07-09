"""
Forest Segmentation — U-Net Training Script
============================================
Trains a U-Net model on the "Forest Segmented" dataset to perform
binary segmentation (forest vs. non-forest).

Outputs saved to Lasana/results/:
  - training_curves.png   : loss / IoU / Dice plots
  - prediction_grid.png   : sample predictions vs ground truth
  - test_metrics.txt      : final test-set scores
  - training_log.csv      : per-epoch metrics

Checkpoints saved to Lasana/checkpoints/:
  - lasana_unet_best.keras : best val IoU checkpoint
  - lasana_unet_last.keras : last epoch checkpoint
"""

# ─── Standard library ────────────────────────────────────────────────────────
import os
import random

# ─── Third-party ─────────────────────────────────────────────────────────────
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — saves to file, no GUI window
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    CSVLogger,
)


# =============================================================================
# PATHS
# =============================================================================

# Base directory is wherever this script lives (Lasana/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_FOLDER   = os.path.join(BASE_DIR, "dataset", "Forest Segmented", "Forest Segmented", "images")
MASK_FOLDER    = os.path.join(BASE_DIR, "dataset", "Forest Segmented", "Forest Segmented", "masks")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR    = os.path.join(BASE_DIR, "results")

# Create output directories if they don't already exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR,    exist_ok=True)


# =============================================================================
# HYPERPARAMETERS
# =============================================================================

SEED           = 42       # for reproducibility across numpy, random, and TF
IMG_SIZE       = 256      # all images/masks are resized to 256×256
MASK_THRESHOLD = 127      # pixel value above this → forest (1), else background (0)
TRAIN_RATIO    = 0.80     # 80% of data used for training
VAL_RATIO      = 0.10     # 10% for validation; remaining 10% for test
LEARNING_RATE  = 1e-3     # Adam initial learning rate
BCE_WEIGHT     = 0.5      # weight for Binary Cross-Entropy in combined loss
DICE_WEIGHT    = 0.5      # weight for Dice loss in combined loss

# ── GPU / CPU auto-config ────────────────────────────────────────────────────
# If a GPU is available, use a full-scale model.
# On CPU, use a smaller model and cap samples so training is still feasible.
HAS_GPU = len(tf.config.list_physical_devices("GPU")) > 0

if HAS_GPU:
    BATCH_SIZE          = 8
    NUM_EPOCHS          = 50
    EARLY_STOP_PATIENCE = 10   # stop if val IoU doesn't improve for 10 epochs
    LR_PATIENCE         = 5    # halve LR if val IoU stalls for 5 epochs
    UNET_FEATURES       = (64, 128, 256, 512)   # full-depth U-Net
    DEFAULT_MAX_SAMPLES = None                  # use all data
else:
    BATCH_SIZE          = 4
    NUM_EPOCHS          = 15
    EARLY_STOP_PATIENCE = 5
    LR_PATIENCE         = 3
    UNET_FEATURES       = (32, 64, 128)   # shallower/narrower model for CPU
    DEFAULT_MAX_SAMPLES = 1200            # cap at 1200 image-mask pairs on CPU

# Allow overriding sample count and epoch count via environment variables
MAX_SAMPLES = os.environ.get("LASANA_MAX_SAMPLES")
MAX_SAMPLES = int(MAX_SAMPLES) if MAX_SAMPLES else DEFAULT_MAX_SAMPLES

if os.environ.get("LASANA_EPOCHS"):
    NUM_EPOCHS = int(os.environ["LASANA_EPOCHS"])

# ── Checkpoint / log paths ───────────────────────────────────────────────────
BEST_CKPT = os.path.join(CHECKPOINT_DIR, "lasana_unet_best.keras")
LAST_CKPT = os.path.join(CHECKPOINT_DIR, "lasana_unet_last.keras")
LOG_CSV   = os.path.join(RESULTS_DIR,    "training_log.csv")

# Set all random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# =============================================================================
# DATA LOADING
# =============================================================================

def find_mask_path(mask_folder, filename):
    """
    Given an image filename, find the matching mask file in mask_folder.

    Handles two naming conventions:
      - Same name as image (e.g. image.jpg → mask.jpg)
      - Satellite naming: replaces '_sat_' with '_mask_' in the stem

    Tries all common image extensions if the original extension isn't found.
    Returns the mask path if found, or None if no match exists.
    """
    stem, ext = os.path.splitext(filename)
    common_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

    # Build a list of candidate stem names to try
    candidate_stems = [stem]
    if "_sat_" in stem:
        # e.g. "tile_sat_001" → also try "tile_mask_001"
        candidate_stems.append(stem.replace("_sat_", "_mask_"))

    # Try the original extension first, then all others
    ext_order = [ext] + [e for e in common_exts if e != ext]

    for candidate_stem in candidate_stems:
        for candidate_ext in ext_order:
            candidate = os.path.join(mask_folder, candidate_stem + candidate_ext)
            if os.path.exists(candidate):
                return candidate

    return None  # no matching mask found


def load_dataset(image_folder, mask_folder, img_size=256, mask_threshold=127, max_samples=None):
    """
    Load all image-mask pairs from disk into numpy arrays.

    Steps per pair:
      1. Read image with OpenCV, convert BGR → RGB, resize, normalise to [0,1]
      2. Read mask as grayscale, resize with nearest-neighbour (preserves hard labels),
         threshold to binary float32, add channel dimension → shape (H, W, 1)

    Skips files that can't be read or have no matching mask.
    If max_samples is set, only the first N sorted files are loaded.

    Returns:
      images : float32 array of shape (N, img_size, img_size, 3)
      masks  : float32 array of shape (N, img_size, img_size, 1)
    """
    # Collect sorted list of valid image filenames
    image_files = sorted(
        f for f in os.listdir(image_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
    )

    # Optionally cap the number of samples
    if max_samples is not None:
        image_files = image_files[:max_samples]

    images, masks = [], []
    skipped = 0

    for i, file in enumerate(image_files):
        # Progress update every 500 files
        if (i + 1) % 500 == 0:
            print(f"  loading {i + 1}/{len(image_files)}...")

        image_path = os.path.join(image_folder, file)
        mask_path  = find_mask_path(mask_folder, file)

        # --- Load image ---
        image = cv2.imread(image_path)
        if image is None or mask_path is None:
            skipped += 1
            continue

        # --- Load mask ---
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            skipped += 1
            continue

        # Preprocess image: BGR→RGB, resize, normalise
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0

        # Preprocess mask: resize (nearest-neighbour keeps hard 0/1 labels), threshold, add channel dim
        mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > mask_threshold).astype(np.float32)   # → binary 0.0 or 1.0
        mask = np.expand_dims(mask, axis=-1)                 # (H, W) → (H, W, 1)

        images.append(image)
        masks.append(mask)

    images = np.array(images, dtype=np.float32)
    masks  = np.array(masks,  dtype=np.float32)

    if len(images) == 0:
        raise RuntimeError("No valid image-mask pairs loaded. Check IMAGE_FOLDER and MASK_FOLDER.")

    print(f"Loaded {len(images)} pairs | skipped {skipped}")
    print("Images:", images.shape, "| Masks:", masks.shape)
    print(f"Forest pixel ratio: {masks.mean():.4f}")
    return images, masks


# =============================================================================
# DATA AUGMENTATION & tf.data PIPELINE
# =============================================================================

def augment_pair(image, mask):
    """
    Apply identical geometric transforms to both image and mask so they stay aligned.
    Also applies photometric jitter to the image only (mask labels must not change).

    Augmentations applied:
      - Random horizontal flip
      - Random vertical flip
      - Random 90° rotation (0 / 90 / 180 / 270°)
      - Random brightness ± 0.1  (image only)
      - Random contrast  ×[0.9, 1.1]  (image only)
    """
    # Horizontal flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask  = tf.image.flip_left_right(mask)

    # Vertical flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask  = tf.image.flip_up_down(mask)

    # Rotate by k * 90 degrees (k ∈ {0, 1, 2, 3})
    k     = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    mask  = tf.image.rot90(mask,  k)

    # Photometric jitter on image only
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.clip_by_value(image, 0.0, 1.0)   # clamp to valid range

    # Re-binarise mask after geometric ops (interpolation artefacts → values near 0.5)
    mask = tf.cast(mask > 0.5, tf.float32)

    return image, mask


def make_dataset(x, y, batch_size, shuffle=False, augment=False):
    """
    Build a tf.data.Dataset from numpy arrays.

    - shuffle : randomly reorder samples each epoch (use for training only)
    - augment : apply augment_pair to each sample (use for training only)
    - batching and prefetching are always applied for efficiency
    """
    ds = tf.data.Dataset.from_tensor_slices((x, y))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(x), 1024), seed=SEED)

    if augment:
        ds = ds.map(augment_pair, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# =============================================================================
# MODEL ARCHITECTURE — U-Net
# =============================================================================

def conv_block(x, filters):
    """
    Basic convolutional block: two consecutive Conv2D → BatchNorm → ReLU layers.

    Using use_bias=False because BatchNormalization already provides a bias-like shift.
    This block is used in both the encoder and decoder paths.
    """
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    return x


def build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 3), features=None):
    """
    Build a U-Net for binary segmentation.

    Architecture:
      Encoder: repeated (conv_block → MaxPool2D) — each step doubles filters, halves spatial size
      Bottleneck: conv_block with features[-1]*2 filters (widest/deepest point)
      Decoder: repeated (UpSampling2D → skip concatenation → conv_block) — mirror of encoder

    Skip connections concatenate encoder feature maps with the corresponding decoder layer,
    allowing the network to recover fine spatial details lost during downsampling.

    Output: 1×1 Conv with sigmoid → per-pixel probability in [0, 1]

    Args:
      input_shape : (H, W, C) of input images
      features    : tuple of filter counts for each encoder stage
                    (defaults to UNET_FEATURES set by GPU/CPU detection)
    """
    if features is None:
        features = UNET_FEATURES

    inputs = Input(shape=input_shape)
    x      = inputs
    skips  = []   # stores encoder outputs for skip connections

    # ── Encoder (downsampling path) ──────────────────────────────────────────
    for f in features:
        x = conv_block(x, f)
        skips.append(x)            # save before pooling for skip connection
        x = layers.MaxPooling2D(2)(x)   # halve spatial dimensions

    # ── Bottleneck ───────────────────────────────────────────────────────────
    x = conv_block(x, features[-1] * 2)

    # ── Decoder (upsampling path) ────────────────────────────────────────────
    for f, skip in zip(reversed(features), reversed(skips)):
        x = layers.UpSampling2D(2)(x)          # double spatial dimensions
        x = layers.Concatenate()([skip, x])     # fuse with encoder skip features
        x = conv_block(x, f)

    # ── Output layer ─────────────────────────────────────────────────────────
    # 1×1 conv collapses to 1 channel; sigmoid gives per-pixel forest probability
    outputs = layers.Conv2D(1, 1, activation="sigmoid", padding="same")(x)

    return Model(inputs, outputs, name="lasana_unet")


# =============================================================================
# LOSS & METRICS
# =============================================================================

def dice_coef(y_true, y_pred, smooth=1.0):
    """
    Soft Dice coefficient — measures overlap between prediction and ground truth.

    Formula: (2 * |A ∩ B| + smooth) / (|A| + |B| + smooth)
    smooth avoids division by zero and stabilises training when masks are sparse.
    Range: 0 (no overlap) → 1 (perfect overlap).
    """
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def dice_loss(y_true, y_pred):
    """Dice loss = 1 - Dice coefficient (minimise to maximise overlap)."""
    return 1.0 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    """
    Combined BCE + Dice loss.

    BCE is good for pixel-level correctness; Dice addresses class imbalance
    (forest pixels may be a small fraction of each image).
    Weights are controlled by BCE_WEIGHT and DICE_WEIGHT (both 0.5 by default).
    """
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    return BCE_WEIGHT * bce + DICE_WEIGHT * dice_loss(y_true, y_pred)


def iou_coef(y_true, y_pred, threshold=0.5, smooth=1.0):
    """
    Intersection over Union (Jaccard index) — primary evaluation metric.

    Predictions are first thresholded to binary before computing IoU,
    so this reflects real segmentation quality rather than soft probabilities.
    Range: 0 (no overlap) → 1 (perfect overlap).
    """
    y_pred_bin = tf.cast(y_pred > threshold, tf.float32)   # binarise predictions
    y_true_f   = tf.reshape(y_true,    [-1])
    y_pred_f   = tf.reshape(y_pred_bin, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union        = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


# =============================================================================
# MAIN TRAINING ROUTINE
# =============================================================================

def main():
    # ── Environment info ─────────────────────────────────────────────────────
    print("TensorFlow:", tf.__version__)
    print("GPUs:", tf.config.list_physical_devices("GPU"))
    print(
        f"Mode: {'GPU' if HAS_GPU else 'CPU-light'} | "
        f"features={UNET_FEATURES} | batch={BATCH_SIZE} | "
        f"epochs={NUM_EPOCHS} | max_samples={MAX_SAMPLES}"
    )
    print("IMAGE_FOLDER:", IMAGE_FOLDER)
    print("MASK_FOLDER :", MASK_FOLDER)

    # Fail early with a clear message if paths are wrong
    assert os.path.isdir(IMAGE_FOLDER), f"Missing images folder: {IMAGE_FOLDER}"
    assert os.path.isdir(MASK_FOLDER),  f"Missing masks folder:  {MASK_FOLDER}"

    # ── Load data ────────────────────────────────────────────────────────────
    print("\nLoading dataset...")
    images, masks = load_dataset(IMAGE_FOLDER, MASK_FOLDER, IMG_SIZE, MASK_THRESHOLD, MAX_SAMPLES)

    # ── Train / Val / Test split ─────────────────────────────────────────────
    # Step 1: split off 10% as test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, masks, test_size=0.10, random_state=SEED
    )
    # Step 2: from remaining 90%, split off val so it equals ~10% of total
    val_fraction_of_temp = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)   # ≈ 0.111
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction_of_temp, random_state=SEED
    )

    # Free the full arrays to save RAM
    del images, masks, X_temp, y_temp
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ── Build tf.data pipelines ──────────────────────────────────────────────
    # Training: shuffle + augment; validation/test: no shuffle, no augment
    train_ds = make_dataset(X_train, y_train, BATCH_SIZE, shuffle=True,  augment=True)
    val_ds   = make_dataset(X_val,   y_val,   BATCH_SIZE, shuffle=False, augment=False)
    test_ds  = make_dataset(X_test,  y_test,  BATCH_SIZE, shuffle=False, augment=False)

    # ── Build and compile model ──────────────────────────────────────────────
    model = build_unet()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=bce_dice_loss,
        metrics=[
            "accuracy",
            dice_coef,
            iou_coef,
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    model.summary()

    # ── Callbacks ────────────────────────────────────────────────────────────
    callbacks = [
        # Save the checkpoint with the best validation IoU
        ModelCheckpoint(BEST_CKPT, monitor="val_iou_coef", mode="max", save_best_only=True, verbose=1),
        # Also save the most recent epoch (useful for resuming)
        ModelCheckpoint(LAST_CKPT, save_best_only=False, verbose=0),
        # Stop early if val IoU hasn't improved; restore weights from best epoch
        EarlyStopping(monitor="val_iou_coef", mode="max", patience=EARLY_STOP_PATIENCE,
                      restore_best_weights=True, verbose=1),
        # Halve the learning rate when val IoU plateaus
        ReduceLROnPlateau(monitor="val_iou_coef", mode="max", factor=0.5,
                          patience=LR_PATIENCE, min_lr=1e-6, verbose=1),
        # Log all metrics per epoch to a CSV file
        CSVLogger(LOG_CSV),
    ]

    # ── Train ────────────────────────────────────────────────────────────────
    print("\nTraining...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=NUM_EPOCHS,
        callbacks=callbacks,
    )

    # ── Plot training curves ─────────────────────────────────────────────────
    hist        = history.history
    epochs_range = range(1, len(hist["loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs_range, hist["loss"],         label="train")
    axes[0].plot(epochs_range, hist["val_loss"],     label="val")
    axes[0].set_title("BCE + Dice Loss")
    axes[0].legend()

    axes[1].plot(epochs_range, hist["iou_coef"],     label="train")
    axes[1].plot(epochs_range, hist["val_iou_coef"], label="val")
    axes[1].set_title("IoU")
    axes[1].legend()

    axes[2].plot(epochs_range, hist["dice_coef"],     label="train")
    axes[2].plot(epochs_range, hist["val_dice_coef"], label="val")
    axes[2].set_title("Dice")
    axes[2].legend()

    plt.tight_layout()
    curve_path = os.path.join(RESULTS_DIR, "training_curves.png")
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print("Saved:", curve_path)

    # ── Evaluate on test set ─────────────────────────────────────────────────
    # Load the best checkpoint before evaluating
    if os.path.exists(BEST_CKPT):
        model.load_weights(BEST_CKPT)
        print("Loaded best checkpoint.")

    results = model.evaluate(test_ds, return_dict=True)
    print("\n=== Test metrics ===")
    for k, v in results.items():
        print(f"{k:12s}: {v:.4f}")

    # Save metrics to text file
    metrics_path = os.path.join(RESULTS_DIR, "test_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        for k, v in results.items():
            f.write(f"{k}: {v:.6f}\n")
    print("Saved:", metrics_path)

    # ── Prediction grid ──────────────────────────────────────────────────────
    # Show up to 4 test samples: original image | ground truth | model prediction
    n_show    = min(4, len(X_test))
    preds     = model.predict(X_test[:n_show], verbose=0)
    preds_bin = (preds > 0.5).astype(np.float32)   # threshold to binary mask

    fig, axes = plt.subplots(n_show, 3, figsize=(10, 3 * n_show))
    if n_show == 1:
        axes = np.expand_dims(axes, 0)   # ensure 2-D indexing works for single row

    for i in range(n_show):
        axes[i, 0].imshow(X_test[i]);              axes[i, 0].set_title("Image");        axes[i, 0].axis("off")
        axes[i, 1].imshow(y_test[i].squeeze(), cmap="gray"); axes[i, 1].set_title("Ground Truth"); axes[i, 1].axis("off")
        axes[i, 2].imshow(preds_bin[i].squeeze(), cmap="gray"); axes[i, 2].set_title("Prediction"); axes[i, 2].axis("off")

    plt.tight_layout()
    pred_path = os.path.join(RESULTS_DIR, "prediction_grid.png")
    plt.savefig(pred_path, dpi=150)
    plt.close()
    print("Saved:", pred_path)
    print("\nDone.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
