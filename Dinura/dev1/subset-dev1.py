# ── 0 · Mount Google Drive (Colab only) ──────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# ── 1 · Imports & reproducibility ────────────────────────────────────────────
import os
import shutil
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))





# ── 2 · Configuration & paths ─────────────────────────────────────────────────
# - IMG_SIZE=256            matches the Forest Segmented patches / Kalana-Dinura baselines
# - MASK_THRESHOLD=127      JPEG masks are rarely pure 0/255; mid-gray must be forced to 0/1
# - 80/10/10 split          val drives early stopping / LR; test stays untouched for the final report
# - Longer training + early stopping — U-Net needs more epochs than Kalana's 10,
#   but we stop when val IoU stalls.

BASE_DIR = os.path.abspath(os.getcwd())

# Manual override (set to your actual Drive paths):
IMAGE_FOLDER = r"/content/drive/MyDrive/DNN-Project/Kalana/images"
MASK_FOLDER  = r"/content/drive/MyDrive/DNN-Project/Kalana/masks"

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)




# ── Hyperparameters ──────────────────────────────────────────────────────────
IMG_SIZE = 256
MASK_THRESHOLD = 127
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10          # of full data; test = remainder (~0.10)
BATCH_SIZE = 8            # raise to 16 if GPU memory allows

# --- FULL-DATASET TRAINING SCHEDULE ---
# Flip USE_SUBSET to False and use this block for the real run.
# NUM_EPOCHS = 50
# LEARNING_RATE = 1e-3
# BCE_WEIGHT = 0.5
# DICE_WEIGHT = 0.5
# EARLY_STOP_PATIENCE = 10
# LR_PATIENCE = 5

# --- SUBSET / QUICK-EXPERIMENT SCHEDULE (active) ---
# Fewer epochs + shorter patience so a failed experiment doesn't burn time.
USE_SUBSET = True
SUBSET_FRACTION = 0.10    # fraction of full dataset to use while experimenting

NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
BCE_WEIGHT = 0.5
DICE_WEIGHT = 0.5
EARLY_STOP_PATIENCE = 4
LR_PATIENCE = 2

BEST_CKPT = os.path.join(CHECKPOINT_DIR, "lasana_unet_best.keras")
LAST_CKPT = os.path.join(CHECKPOINT_DIR, "lasana_unet_last.keras")
LOG_CSV = os.path.join(RESULTS_DIR, "training_log.csv")

print("IMAGE_FOLDER:", IMAGE_FOLDER)
print("MASK_FOLDER :", MASK_FOLDER)
assert IMAGE_FOLDER and MASK_FOLDER, (
    "Could not find images/masks. Set IMAGE_FOLDER and MASK_FOLDER manually."
)



# ── 3 · Copy data to local disk (faster I/O than Drive) ──────────────────────
LOCAL_IMAGE_FOLDER = os.path.join(BASE_DIR, "local_images")
LOCAL_MASK_FOLDER = os.path.join(BASE_DIR, "local_masks")
os.makedirs(LOCAL_IMAGE_FOLDER, exist_ok=True)
os.makedirs(LOCAL_MASK_FOLDER, exist_ok=True)

print(f"Copying images from {IMAGE_FOLDER} to {LOCAL_IMAGE_FOLDER}...")
for item in os.listdir(IMAGE_FOLDER):
    s = os.path.join(IMAGE_FOLDER, item)
    if os.path.isfile(s):
        shutil.copy2(s, os.path.join(LOCAL_IMAGE_FOLDER, item))

print(f"Copying masks from {MASK_FOLDER} to {LOCAL_MASK_FOLDER}...")
for item in os.listdir(MASK_FOLDER):
    s = os.path.join(MASK_FOLDER, item)
    if os.path.isfile(s):
        shutil.copy2(s, os.path.join(LOCAL_MASK_FOLDER, item))

IMAGE_FOLDER = LOCAL_IMAGE_FOLDER
MASK_FOLDER = LOCAL_MASK_FOLDER
print("Copy complete. New IMAGE_FOLDER:", IMAGE_FOLDER)
print("New MASK_FOLDER :", MASK_FOLDER)





# ── 4 · Load image–mask pairs ─────────────────────────────────────────────────
def find_mask_path(mask_folder, filename):
    """Return best matching mask path for an image filename."""
    stem, ext = os.path.splitext(filename)
    common_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    candidate_stems = [stem]
    if "_sat_" in stem:
        candidate_stems.append(stem.replace("_sat_", "_mask_"))

    ext_order = [ext] + [e for e in common_exts if e != ext]
    for candidate_stem in candidate_stems:
        for candidate_ext in ext_order:
            candidate = os.path.join(mask_folder, candidate_stem + candidate_ext)
            if os.path.exists(candidate):
                return candidate
    return None


def load_dataset_paths(image_folder, mask_folder):
    """Returns lists of image and mask paths for every valid pair on disk."""
    image_files = sorted(
        f for f in os.listdir(image_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
    )

    image_paths, mask_paths = [], []
    skipped = 0

    for file in image_files:
        image_path = os.path.join(image_folder, file)
        mask_path = find_mask_path(mask_folder, file)

        if not os.path.exists(image_path):
            skipped += 1
            continue
        if not mask_path or not os.path.exists(mask_path):
            skipped += 1
            continue

        image_paths.append(image_path)
        mask_paths.append(mask_path)

    if len(image_paths) == 0:
        raise RuntimeError("No valid image-mask pairs found. Check paths.")

    print(f"Found {len(image_paths)} image-mask pairs | skipped {skipped}")
    return image_paths, mask_paths


image_paths, mask_paths = load_dataset_paths(IMAGE_FOLDER, MASK_FOLDER)

# ── Optional subset selection (deterministic shuffle before the split below) ─
if USE_SUBSET:
    _rng = random.Random(SEED)
    _combined = list(zip(image_paths, mask_paths))
    _rng.shuffle(_combined)
    _n_subset = int(len(_combined) * SUBSET_FRACTION)
    _combined = _combined[:_n_subset]
    image_paths, mask_paths = (list(t) for t in zip(*_combined))
    print(f"Using SUBSET: {len(image_paths)} pairs ({SUBSET_FRACTION*100:.0f}% of full dataset)")
else:
    print(f"Using FULL dataset: {len(image_paths)} pairs")




# ── 5 · Data augmentation (train only) ────────────────────────────────────────
# Random horizontal/vertical flips, 90° rotations, mild brightness/contrast changes.
# Val/test are NOT augmented — metrics must reflect real data.

def augment_pair(image, mask):
    """Apply the same geometric transform to image and mask."""
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)

    k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    mask = tf.image.rot90(mask, k)

    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.clip_by_value(image, 0.0, 1.0)

    mask = tf.cast(mask > 0.5, tf.float32)
    return image, mask


def _parse_image_mask_pair(image_path, mask_path, img_size, mask_threshold):
    """Loads and preprocesses a single image-mask pair."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (img_size, img_size), method=tf.image.ResizeMethod.BILINEAR)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_image(mask, channels=1, expand_animations=False)
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    # Nearest-neighbor keeps hard 0/1 labels
    mask = tf.image.resize(mask, (img_size, img_size), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    mask = tf.cast(mask > (mask_threshold / 255.0), tf.float32)

    return image, mask


def make_dataset(image_paths, mask_paths, batch_size, img_size, mask_threshold,
                  shuffle=False, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    ds = ds.map(
        lambda img_p, mask_p: _parse_image_mask_pair(img_p, mask_p, img_size, mask_threshold),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(image_paths), 1024), seed=SEED)
    if augment:
        ds = ds.map(augment_pair, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds






# ── 6 · Train / Val / Test split ──────────────────────────────────────────────
# Train  — learn weights.
# Val    — pick best checkpoint & tune LR / early stop (never used for the final claim).
# Test   — report once at the end so you don't overfit to the published number.

X_train_paths, X_test_paths, y_train_paths, y_test_paths = train_test_split(image_paths, mask_paths, test_size=0.10, random_state=SEED)
val_fraction_of_temp = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
X_train_paths, X_val_paths, y_train_paths, y_val_paths = train_test_split(
    X_train_paths, y_train_paths, test_size=val_fraction_of_temp, random_state=SEED)

print(f"Train paths: {len(X_train_paths)} | Val paths: {len(X_val_paths)} | Test paths: {len(X_test_paths)}")


train_ds = make_dataset(X_train_paths, y_train_paths, BATCH_SIZE, IMG_SIZE, MASK_THRESHOLD,
                         shuffle=True, augment=True)
val_ds = make_dataset(X_val_paths, y_val_paths, BATCH_SIZE, IMG_SIZE, MASK_THRESHOLD,
                       shuffle=False, augment=False)
test_ds = make_dataset(X_test_paths, y_test_paths, BATCH_SIZE, IMG_SIZE, MASK_THRESHOLD,
                        shuffle=False, augment=False)

print("Datasets ready.")








# ── 7 · U-Net model ────────────────────────────────────────────────────────────
# DoubleConv (Conv->BN->ReLU x2)  — extracts features; BatchNorm stabilizes activations.
# Encoder + MaxPool               — grows channels, shrinks spatial size (context).
# Bottleneck                      — deepest abstract representation.
# Decoder + UpSampling            — rebuilds resolution.
# Skip connections (concatenate)  — inject encoder detail into decoder so forest
#                                    edges stay sharp.
# 1x1 Conv + sigmoid              — one probability per pixel (forest vs background).

def conv_block(x, filters):
    """Two 3x3 convolutions with BatchNorm + ReLU."""
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 3), features=(64, 128, 256, 512)):
    inputs = Input(shape=input_shape)
    x = inputs
    skips = []

    for f in features:
        x = conv_block(x, f)
        skips.append(x)
        x = layers.MaxPooling2D(2)(x)

    x = conv_block(x, features[-1] * 2)

    for f, skip in zip(reversed(features), reversed(skips)):
        x = layers.UpSampling2D(2)(x)
        x = layers.Concatenate()([skip, x])
        x = conv_block(x, f)

    outputs = layers.Conv2D(1, 1, activation="sigmoid", padding="same")(x)
    return Model(inputs, outputs, name="lasana_unet")


model = build_unet()
model.summary()





# ── 8 · Loss & metrics ────────────────────────────────────────────────────────
# BCE            — good per-pixel probability calibration.
# Dice loss       — 1 - Dice; rewards overlap of predicted forest with ground truth.
# Combined loss  — balances both.
# IoU / Dice     — report these; pixel accuracy alone can look high while forest
#                  regions are wrong.

def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    return BCE_WEIGHT * bce + DICE_WEIGHT * dice_loss(y_true, y_pred)


def iou_coef(y_true, y_pred, threshold=0.5, smooth=1.0):
    y_pred_bin = tf.cast(y_pred > threshold, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred_bin, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)




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
print("Compiled with BCE + Dice loss; tracking accuracy, Dice, IoU, Precision, Recall.")





# ── 9 · Train ──────────────────────────────────────────────────────────────────
# ModelCheckpoint (best val IoU) — keep the best forest-overlap model, not the last epoch.
# EarlyStopping                  — stop when val IoU stops improving.
# ReduceLROnPlateau              — lower LR when learning stalls.
# CSVLogger                      — save history for plots / reports.

callbacks = [
    ModelCheckpoint(BEST_CKPT, monitor="val_iou_coef", mode="max",
                     save_best_only=True, verbose=1),
    ModelCheckpoint(LAST_CKPT, save_best_only=False, verbose=0),
    EarlyStopping(monitor="val_iou_coef", mode="max",
                  patience=EARLY_STOP_PATIENCE, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_iou_coef", mode="max",
                       factor=0.5, patience=LR_PATIENCE, min_lr=1e-6, verbose=1),
    CSVLogger(LOG_CSV),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=NUM_EPOCHS,
    callbacks=callbacks,
)

print(f"Best checkpoint: {BEST_CKPT}")







# ── 10 · Training curves ──────────────────────────────────────────────────────
# Plot IoU/Dice, not only loss: loss can drop while overlap quality stalls.

hist = history.history
epochs_range = range(1, len(hist["loss"]) + 1)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(epochs_range, hist["loss"], label="train")
axes[0].plot(epochs_range, hist["val_loss"], label="val")
axes[0].set_title("BCE + Dice Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(epochs_range, hist["iou_coef"], label="train")
axes[1].plot(epochs_range, hist["val_iou_coef"], label="val")
axes[1].set_title("IoU")
axes[1].set_xlabel("Epoch")
axes[1].legend()

axes[2].plot(epochs_range, hist["dice_coef"], label="train")
axes[2].plot(epochs_range, hist["val_dice_coef"], label="val")
axes[2].set_title("Dice")
axes[2].set_xlabel("Epoch")
axes[2].legend()

plt.tight_layout()
curve_path = os.path.join(RESULTS_DIR, "training_curves.png")
plt.savefig(curve_path, dpi=150)
plt.show()
print("Saved:", curve_path)





# ── 11 · Test-set evaluation ───────────────────────────────────────────────────
# Load the best val-IoU weights and evaluate once on the held-out test set.

if os.path.exists(BEST_CKPT):
    model.load_weights(BEST_CKPT)
    print("Loaded best checkpoint.")

results = model.evaluate(test_ds, return_dict=True)
print("\n=== Test metrics ===")
for k, v in results.items():
    print(f"{k:12s}: {v:.4f}")





# ── 12 · Visual predictions ───────────────────────────────────────────────────
# Side-by-side: original image | ground-truth mask | prediction.
# NOTE: pulled straight from test_ds (the tf.data pipeline), since the pipeline
# no longer keeps raw X_test/y_test arrays in memory.

N_SHOW = 4
sample_images, sample_masks = next(iter(test_ds.unbatch().batch(N_SHOW)))
preds = model.predict(sample_images, verbose=0)
preds_bin = (preds > 0.5).astype(np.float32)

fig, axes = plt.subplots(N_SHOW, 3, figsize=(10, 3 * N_SHOW))
if N_SHOW == 1:
    axes = np.expand_dims(axes, 0)

for i in range(N_SHOW):
    axes[i, 0].imshow(sample_images[i].numpy())
    axes[i, 0].set_title("Image")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(sample_masks[i].numpy().squeeze(), cmap="gray")
    axes[i, 1].set_title("Ground Truth")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(preds_bin[i].squeeze(), cmap="gray")
    axes[i, 2].set_title("Prediction")
    axes[i, 2].axis("off")

plt.tight_layout()
pred_path = os.path.join(RESULTS_DIR, "prediction_grid.png")
plt.savefig(pred_path, dpi=150)
plt.show()
print("Saved:", pred_path)


# Artifacts:
#   - checkpoints/lasana_unet_best.keras
#   - results/training_log.csv
#   - results/training_curves.png
#   - results/prediction_grid.png
