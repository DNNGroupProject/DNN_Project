"""
==========================================================
Vanilla CNN  —  Hyperparameter Random Search
----------------------------------------------------------
Forest Cover Semantic Segmentation  |  256 × 256 px

Architecture stays strictly "vanilla":
  Sequential  Conv2D  ▶  MaxPool  ▶  UpSampling
  No skip connections, no batch normalisation.
  Dropout (decoder only) is allowed as pure regularisation.

Search space
─────────────────────────────────────────────────────────
  learning_rate  : log-uniform  [1e-4 … 5e-3]
  batch_size     : {8, 16, 32}
  base_filters   : {16, 32, 64}      (1st conv layer; scales 2× each block)
  kernel_size    : {3, 5}
  depth          : {2, 3}            (encoder conv-pool stages)
  dropout_rate   : {0.0, 0.1, 0.2, 0.3}
  optimizer      : {adam, rmsprop}
  loss_fn        : {binary_crossentropy, dice, bce_dice}

Primary metric  : val_iou   (higher = better)
─────────────────────────────────────────────────────────
Outputs
  • hyperparam_search_results.csv   — all trials ranked by val_iou
  • best_vanilla_cnn.h5             — best model weights
  • best_hyperparams.txt            — best configuration summary
==========================================================
"""

import os
import csv
import time
import random
import numpy as np
import cv2

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ──────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────

_BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(_BASE_DIR, "images")
MASK_FOLDER  = os.path.join(_BASE_DIR, "masks")
IMG_SIZE     = 256

# Number of random hyperparameter combinations to try.
# Increase for a more thorough search (costs more time).
N_TRIALS     = 20

# Hard ceiling on epochs per trial.
# EarlyStopping will cut most runs far shorter.
MAX_EPOCHS   = 30

RANDOM_SEED  = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ──────────────────────────────────────────────────────────
# Hyperparameter search space
# ──────────────────────────────────────────────────────────

PARAM_GRID = {
    # Log-spaced learning rates covering two decades
    "learning_rate": [1e-4, 3e-4, 5e-4, 1e-3, 2e-3, 5e-3],
    "batch_size":    [8, 16, 32],
    "base_filters":  [16, 32, 64],     # filters in 1st encoder stage
    "kernel_size":   [3, 5],
    "depth":         [2, 3],           # encoder stages (each halves spatial dim)
    "dropout_rate":  [0.0, 0.1, 0.2, 0.3],
    "optimizer_name":["adam", "rmsprop"],
    "loss_fn":       ["binary_crossentropy", "dice", "bce_dice"],
}


# ──────────────────────────────────────────────────────────
# Custom loss functions
# ──────────────────────────────────────────────────────────

def dice_loss(y_true, y_pred):
    """Soft Dice loss — better for imbalanced binary masks."""
    smooth = 1e-6
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1.0 - (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def bce_dice_loss(y_true, y_pred):
    """Balanced combo: drives both pixel accuracy and region overlap."""
    bce = tf.keras.losses.binary_crossentropy(
        tf.reshape(y_true, [-1]), tf.reshape(y_pred, [-1])
    )
    return bce + dice_loss(y_true, y_pred)


LOSS_MAP = {
    "binary_crossentropy": "binary_crossentropy",
    "dice":                dice_loss,
    "bce_dice":            bce_dice_loss,
}


# ──────────────────────────────────────────────────────────
# Custom IoU metric
# ──────────────────────────────────────────────────────────

class BinaryIoU(tf.keras.metrics.Metric):
    """
    Streaming binary IoU at threshold 0.5.
    Accumulates intersection and union across batches.
    """

    def __init__(self, threshold: float = 0.5, name: str = "iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold    = threshold
        self._intersection = self.add_weight(name="intersection", shape=(), initializer="zeros")
        self._union        = self.add_weight(name="union",        shape=(), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_bin = tf.cast(y_pred >= self.threshold, tf.float32)
        y_true_f   = tf.cast(y_true, tf.float32)
        intersection = tf.reduce_sum(y_true_f * y_pred_bin)
        union = (
            tf.reduce_sum(y_true_f)
            + tf.reduce_sum(y_pred_bin)
            - intersection
        )
        self._intersection.assign_add(intersection)
        self._union.assign_add(union + 1e-6)

    def result(self):
        return self._intersection / self._union

    def reset_state(self):
        self._intersection.assign(0.0)
        self._union.assign(0.0)


# ──────────────────────────────────────────────────────────
# Model builder
# ──────────────────────────────────────────────────────────

def build_vanilla_cnn(
    base_filters: int,
    kernel_size:  int,
    depth:        int,
    dropout_rate: float,
    learning_rate: float,
    optimizer_name: str,
    loss_fn,
) -> tf.keras.Model:
    """
    Build a vanilla CNN segmentation model with configurable width and depth.

    Encoder  : `depth` blocks of  Conv2D → MaxPool2D
    Bottleneck: one extra Conv2D at the deepest spatial level
    Decoder  : `depth` blocks of  UpSampling2D → Conv2D (+ optional Dropout)
    Output   : Conv2D(1, kernel=1, sigmoid)

    Filter counts double each encoder stage and halve each decoder stage,
    matching the classical encoder-decoder pattern.
    """
    ks = (kernel_size, kernel_size)
    model = Sequential(name="vanilla_cnn")

    # ── Encoder ──────────────────────────────────────────
    for stage in range(depth):
        filters = base_filters * (2 ** stage)
        if stage == 0:
            model.add(Conv2D(filters, ks, activation="relu",
                             padding="same",
                             input_shape=(IMG_SIZE, IMG_SIZE, 3)))
        else:
            model.add(Conv2D(filters, ks, activation="relu", padding="same"))
        model.add(MaxPooling2D((2, 2)))

    # ── Bottleneck ────────────────────────────────────────
    bottleneck_filters = base_filters * (2 ** depth)
    model.add(Conv2D(bottleneck_filters, ks, activation="relu", padding="same"))

    # ── Decoder ──────────────────────────────────────────
    for stage in range(depth - 1, -1, -1):
        filters = base_filters * (2 ** stage)
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(filters, ks, activation="relu", padding="same"))
        if dropout_rate > 0.0:
            model.add(Dropout(dropout_rate))

    # ── Output ────────────────────────────────────────────
    model.add(Conv2D(1, (1, 1), activation="sigmoid", padding="same"))

    # ── Compile ───────────────────────────────────────────
    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy", BinaryIoU(name="iou")],
    )
    return model


# ──────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────

def find_mask_path(mask_folder: str, filename: str):
    """Return best-matching mask path for a given image filename."""
    stem, ext = os.path.splitext(filename)
    common_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    candidate_stems = [stem]
    if "_sat_" in stem:
        candidate_stems.append(stem.replace("_sat_", "_mask_"))
    ext_order = [ext] + [e for e in common_exts if e != ext]
    for cstem in candidate_stems:
        for cext in ext_order:
            candidate = os.path.join(mask_folder, cstem + cext)
            if os.path.exists(candidate):
                return candidate
    return None


def load_dataset():
    print("Loading dataset …")
    images, masks = [], []
    for file in sorted(os.listdir(IMAGE_FOLDER)):
        img_path  = os.path.join(IMAGE_FOLDER, file)
        mask_path = find_mask_path(MASK_FOLDER, file)

        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0

        if mask_path is None:
            continue
        msk = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if msk is None:
            continue
        msk = cv2.resize(msk, (IMG_SIZE, IMG_SIZE)) / 255.0
        msk = np.expand_dims(msk, axis=-1)

        images.append(img)
        masks.append(msk)

    images = np.array(images, dtype=np.float32)
    masks  = np.array(masks,  dtype=np.float32)
    print(f"  Loaded {len(images)} image-mask pairs  |  shape {images.shape}")
    return images, masks


# ──────────────────────────────────────────────────────────
# Sampling helpers
# ──────────────────────────────────────────────────────────

def sample_params() -> dict:
    """Draw one random combination from the search space."""
    return {k: random.choice(v) for k, v in PARAM_GRID.items()}


def params_to_key(p: dict) -> str:
    return (
        f"lr={p['learning_rate']:.0e}_bs={p['batch_size']}"
        f"_bf={p['base_filters']}_ks={p['kernel_size']}"
        f"_d={p['depth']}_dr={p['dropout_rate']}"
        f"_opt={p['optimizer_name']}_loss={p['loss_fn']}"
    )


# ──────────────────────────────────────────────────────────
# Single-trial training
# ──────────────────────────────────────────────────────────

def run_trial(params: dict, X_train, y_train, X_val, y_val) -> dict:
    """Train one configuration and return val_iou + metadata."""
    # Clear previous session to free memory between trials
    tf.keras.backend.clear_session()

    loss_fn = LOSS_MAP[params["loss_fn"]]

    model = build_vanilla_cnn(
        base_filters  = params["base_filters"],
        kernel_size   = params["kernel_size"],
        depth         = params["depth"],
        dropout_rate  = params["dropout_rate"],
        learning_rate = params["learning_rate"],
        optimizer_name= params["optimizer_name"],
        loss_fn       = loss_fn,
    )

    callbacks = [
        EarlyStopping(
            monitor="val_iou", mode="max",
            patience=5, restore_best_weights=True,
            verbose=0,
        ),
        ReduceLROnPlateau(
            monitor="val_iou", mode="max",
            factor=0.5, patience=3, min_lr=1e-6,
            verbose=0,
        ),
    ]

    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=params["batch_size"],
        callbacks=callbacks,
        verbose=0,
    )
    elapsed = time.time() - t0

    # Best val_iou achieved across all epochs
    val_iou_history = history.history.get("val_iou", [0.0])
    best_val_iou    = float(max(val_iou_history))
    best_epoch      = int(np.argmax(val_iou_history)) + 1

    val_acc_history  = history.history.get("val_accuracy", [0.0])
    best_val_acc     = float(val_acc_history[np.argmax(val_iou_history)])

    val_loss_history = history.history.get("val_loss", [0.0])
    best_val_loss    = float(val_loss_history[np.argmax(val_iou_history)])

    result = {
        **params,
        "best_val_iou":  round(best_val_iou, 6),
        "best_val_acc":  round(best_val_acc, 6),
        "best_val_loss": round(best_val_loss, 6),
        "best_epoch":    best_epoch,
        "elapsed_sec":   round(elapsed, 1),
        "total_params":  model.count_params(),
    }

    return result, model


# ──────────────────────────────────────────────────────────
# Conference-grade metric computation
# ──────────────────────────────────────────────────────────

def compute_paper_metrics(model: tf.keras.Model,
                          X: np.ndarray,
                          y: np.ndarray,
                          threshold: float = 0.5) -> dict:
    """
    Compute all standard binary-segmentation metrics used in papers.

    Aggregates TP / FP / FN / TN across the entire split before computing
    ratios (macro-aggregate), consistent with most segmentation papers.

    Returns
    -------
    dict with keys:
        pixel_accuracy, iou (Jaccard), dice (F1),
        precision, recall, specificity
    """
    y_pred_prob = model.predict(X, batch_size=16, verbose=0)
    y_pred = (y_pred_prob >= threshold).astype(np.float32)
    y_true = y.astype(np.float32)

    # Flatten to 1-D for element-wise ops
    y_pred_f = y_pred.ravel()
    y_true_f = y_true.ravel()

    TP = np.sum(y_true_f * y_pred_f)
    FP = np.sum((1 - y_true_f) * y_pred_f)
    FN = np.sum(y_true_f * (1 - y_pred_f))
    TN = np.sum((1 - y_true_f) * (1 - y_pred_f))

    eps = 1e-7
    pixel_accuracy = (TP + TN) / (TP + TN + FP + FN + eps)
    iou             = TP / (TP + FP + FN + eps)
    dice            = (2.0 * TP) / (2.0 * TP + FP + FN + eps)  # == F1
    precision       = TP / (TP + FP + eps)
    recall          = TP / (TP + FN + eps)           # sensitivity
    specificity     = TN / (TN + FP + eps)

    return {
        "pixel_accuracy": float(pixel_accuracy),
        "iou":            float(iou),
        "dice_f1":        float(dice),
        "precision":      float(precision),
        "recall":         float(recall),
        "specificity":    float(specificity),
    }


# ──────────────────────────────────────────────────────────
# Main search loop
# ──────────────────────────────────────────────────────────

def main():
    print("=" * 58)
    print("  Vanilla CNN — Hyperparameter Random Search")
    print(f"  Trials : {N_TRIALS}   |   Max epochs / trial : {MAX_EPOCHS}")
    print("=" * 58)

    # ── Load data ─────────────────────────────────────────
    images, masks = load_dataset()

    if len(images) == 0:
        raise RuntimeError(
            "No image-mask pairs loaded. "
            "Check IMAGE_FOLDER / MASK_FOLDER paths."
        )

    # 80 % train+val  /  20 % held-out test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        images, masks, test_size=0.20, random_state=RANDOM_SEED
    )
    # 80/20 split of train+val  →  64 % train / 16 % val / 20 % test
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.20, random_state=RANDOM_SEED
    )

    print(f"  Train : {len(X_train)}  |  Val : {len(X_val)}  |  Test : {len(X_test)}")
    print()

    # ── Output files ──────────────────────────────────────
    csv_path        = os.path.join(_BASE_DIR, "hyperparam_search_results.csv")
    best_model_path = os.path.join(_BASE_DIR, "best_vanilla_cnn.h5")
    best_hp_path    = os.path.join(_BASE_DIR, "best_hyperparams.txt")

    csv_fieldnames = [
        "trial", "best_val_iou", "best_val_acc", "best_val_loss",
        "best_epoch", "elapsed_sec", "total_params",
        "learning_rate", "batch_size", "base_filters", "kernel_size",
        "depth", "dropout_rate", "optimizer_name", "loss_fn",
    ]

    # ── Search ────────────────────────────────────────────
    all_results  = []
    best_val_iou = -1.0
    best_model   = None
    best_params  = None

    seen_keys = set()  # avoid duplicate configurations

    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=csv_fieldnames)
        writer.writeheader()

        trial_num = 0
        attempts  = 0

        while trial_num < N_TRIALS and attempts < N_TRIALS * 5:
            attempts += 1
            params  = sample_params()
            key     = params_to_key(params)

            if key in seen_keys:
                continue
            seen_keys.add(key)

            trial_num += 1
            print(f"Trial {trial_num:>2}/{N_TRIALS}  |  {key}")

            try:
                result, model = run_trial(params, X_train, y_train, X_val, y_val)
            except Exception as exc:
                print(f"  ✗ Trial failed: {exc}")
                continue

            result["trial"] = trial_num
            all_results.append(result)

            # Write row immediately (safe against crashes)
            writer.writerow({k: result[k] for k in csv_fieldnames})
            fcsv.flush()

            status = (
                f"  val_iou={result['best_val_iou']:.4f}  "
                f"val_acc={result['best_val_acc']:.4f}  "
                f"epoch={result['best_epoch']}  "
                f"t={result['elapsed_sec']}s"
            )

            if result["best_val_iou"] > best_val_iou:
                best_val_iou = result["best_val_iou"]
                best_params  = params
                best_model   = model
                model.save(best_model_path)
                status += "  ← NEW BEST ★"

            print(status)
            print()

    # ── Final evaluation on held-out test set ─────────────
    print("=" * 58)
    print("  Evaluating best model on held-out test set …")
    print("=" * 58)

    if best_model is None:
        print("  No successful trial completed.")
        return

    # Keras built-in metrics (loss, accuracy, iou)
    keras_results = best_model.evaluate(X_test, y_test, verbose=0)
    keras_metrics = dict(zip(best_model.metrics_names, keras_results))

    # Full paper metric suite
    paper_metrics = compute_paper_metrics(best_model, X_test, y_test)
    test_metrics  = {**keras_metrics, **paper_metrics}

    print("  Test results:")
    for name, value in test_metrics.items():
        print(f"    {name:<20} {value:.6f}")

    # ── Sort and display leaderboard ──────────────────────
    all_results.sort(key=lambda x: x["best_val_iou"], reverse=True)

    print()
    print("=" * 58)
    print("  Top-5 configurations by val IoU")
    print("=" * 58)
    for rank, r in enumerate(all_results[:5], 1):
        print(
            f"  #{rank}  val_iou={r['best_val_iou']:.4f}  "
            f"lr={r['learning_rate']:.0e}  "
            f"bs={r['batch_size']}  "
            f"bf={r['base_filters']}  "
            f"ks={r['kernel_size']}  "
            f"d={r['depth']}  "
            f"dr={r['dropout_rate']}  "
            f"opt={r['optimizer_name']}  "
            f"loss={r['loss_fn']}"
        )

    # ── Conference table printout ─────────────────────────
    print()
    print("=" * 58)
    print("  CONFERENCE TABLE  —  Vanilla CNN Baseline (Test Set)")
    print("=" * 58)
    print(f"  {'Metric':<22} {'Value':>8}")
    print("  " + "-" * 32)
    print(f"  {'Pixel Accuracy':<22} {paper_metrics['pixel_accuracy']:>8.4f}")
    print(f"  {'IoU (Jaccard)':<22} {paper_metrics['iou']:>8.4f}")
    print(f"  {'Dice / F1':<22} {paper_metrics['dice_f1']:>8.4f}")
    print(f"  {'Precision':<22} {paper_metrics['precision']:>8.4f}")
    print(f"  {'Recall (Sensitivity)':<22} {paper_metrics['recall']:>8.4f}")
    print(f"  {'Specificity':<22} {paper_metrics['specificity']:>8.4f}")
    print("=" * 58)

    # ── Save best hyperparams summary ─────────────────────
    with open(best_hp_path, "w") as f:
        f.write("Vanilla CNN Baseline — Best Hyperparameters\n")
        f.write("=" * 44 + "\n\n")
        f.write("--- Hyperparameters ---\n")
        for k, v in best_params.items():
            f.write(f"{k:<20} {v}\n")
        f.write("\n--- Conference Table (Test Set) ---\n")
        f.write(f"{'Metric':<22} {'Value'}\n")
        f.write("-" * 32 + "\n")
        f.write(f"{'Pixel Accuracy':<22} {paper_metrics['pixel_accuracy']:.6f}\n")
        f.write(f"{'IoU (Jaccard)':<22} {paper_metrics['iou']:.6f}\n")
        f.write(f"{'Dice / F1':<22} {paper_metrics['dice_f1']:.6f}\n")
        f.write(f"{'Precision':<22} {paper_metrics['precision']:.6f}\n")
        f.write(f"{'Recall (Sensitivity)':<22} {paper_metrics['recall']:.6f}\n")
        f.write(f"{'Specificity':<22} {paper_metrics['specificity']:.6f}\n")
        f.write("\n--- Keras metrics ---\n")
        for name, value in keras_metrics.items():
            f.write(f"{name:<20} {value:.6f}\n")
        f.write("\n--- Search info ---\n")
        f.write(f"{'N_TRIALS':<20} {N_TRIALS}\n")
        f.write(f"{'MAX_EPOCHS':<20} {MAX_EPOCHS}\n")
        f.write(f"{'RANDOM_SEED':<20} {RANDOM_SEED}\n")
        f.write(f"{'Dataset size':<20} {len(images)}\n")
        f.write(f"{'Train / Val / Test':<20} {len(X_train)} / {len(X_val)} / {len(X_test)}\n")

    print()
    print(f"  Best val IoU      : {best_val_iou:.4f}")
    print(f"  Results CSV       : {csv_path}")
    print(f"  Best model        : {best_model_path}")
    print(f"  Best params file  : {best_hp_path}")
    print("=" * 58)


# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
