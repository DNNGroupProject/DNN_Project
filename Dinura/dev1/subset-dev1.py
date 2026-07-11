from google.colab import drive
drive.mount('/content/drive')

import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import shutil
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger)

# ── Fix random seed for reproducibility ──────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.getcwd())

# Try common locations in order; first existing pair wins
_CANDIDATES = [
    (os.path.join(BASE_DIR, "dataset", "Forest Segmented", "Forest Segmented", "images"),
     os.path.join(BASE_DIR, "dataset", "Forest Segmented", "Forest Segmented", "masks")),
    (os.path.join(BASE_DIR, "images"), os.path.join(BASE_DIR, "masks")),
    (os.path.join(BASE_DIR, "..", "data", "Forest Segmented", "Forest Segmented", "images"),
     os.path.join(BASE_DIR, "..", "data", "Forest Segmented", "Forest Segmented", "masks")),
]

IMAGE_FOLDER = None
MASK_FOLDER = None
for img_c, mask_c in _CANDIDATES:
    if os.path.isdir(img_c) and os.path.isdir(mask_c):
        IMAGE_FOLDER, MASK_FOLDER = img_c, mask_c
        break

# Manual override if auto-detect fails:
IMAGE_FOLDER = r"/content/drive/MyDrive/DNN-Project/Kalana/images"
MASK_FOLDER = r"/content/drive/MyDrive/DNN-Project/Kalana/masks"

# ── Copy data to local storage for faster access ──────────────────────────
LOCAL_IMAGE_FOLDER = os.path.join(BASE_DIR, "local_images")
LOCAL_MASK_FOLDER = os.path.join(BASE_DIR, "local_masks")

os.makedirs(LOCAL_IMAGE_FOLDER, exist_ok=True)
os.makedirs(LOCAL_MASK_FOLDER, exist_ok=True)

print(f"Copying images from {IMAGE_FOLDER} to {LOCAL_IMAGE_FOLDER}...")
for item in os.listdir(IMAGE_FOLDER):
    s = os.path.join(IMAGE_FOLDER, item)
    d = os.path.join(LOCAL_IMAGE_FOLDER, item)
    if os.path.isfile(s):
        shutil.copy2(s, d)

print(f"Copying masks from {MASK_FOLDER} to {LOCAL_MASK_FOLDER}...")
for item in os.listdir(MASK_FOLDER):
    s = os.path.join(MASK_FOLDER, item)
    d = os.path.join(LOCAL_MASK_FOLDER, item)
    if os.path.isfile(s):
        shutil.copy2(s, d)

# Update paths to local copies
IMAGE_FOLDER = LOCAL_IMAGE_FOLDER
MASK_FOLDER = LOCAL_MASK_FOLDER

print("Copy complete. IMAGE_FOLDER and MASK_FOLDER updated to local paths.")
print("New IMAGE_FOLDER:", IMAGE_FOLDER)
print("New MASK_FOLDER :", MASK_FOLDER)

# ── Create output directories ──────────────────────────────────────────────
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Hyperparameters ──────────────────────────────────────────────────────────
IMG_SIZE = 256
MASK_THRESHOLD = 127
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
BATCH_SIZE = 8

# ── SUBSET / QUICK-EXPERIMENT TRAINING SCHEDULE ───────────────────────────
USE_SUBSET = True
SUBSET_FRACTION = 0.15

NUM_EPOCHS = 15
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
assert IMAGE_FOLDER and MASK_FOLDER, "Could not find images/masks. Set IMAGE_FOLDER and MASK_FOLDER manually."

# ── Helper functions for loading data ──────────────────────────────────────
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
    """Returns lists of image and mask paths."""
    image_files = sorted(
        f for f in os.listdir(image_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
    )

    image_paths, mask_paths = [], []
    skipped = 0

    for file in image_files:
        image_path = os.path.join(image_folder, file)
        mask_path = find_mask_path(mask_folder, file)

        if not os.path.exists(image_path) or not mask_path or not os.path.exists(mask_path):
            skipped += 1
            continue

        image_paths.append(image_path)
        mask_paths.append(mask_path)

    if len(image_paths) == 0:
        raise RuntimeError("No valid image-mask pairs found. Check paths.")

    print(f"Found {len(image_paths)} image-mask pairs | skipped {skipped}")
    return image_paths, mask_paths

# ── Load dataset paths ──────────────────────────────────────────────────────
image_paths, mask_paths = load_dataset_paths(IMAGE_FOLDER, MASK_FOLDER)

# ── SUBSET SELECTION ──────────────────────────────────────────────────────
if USE_SUBSET:
    _rng = random.Random(SEED)
    _combined = list(zip(image_paths, mask_paths))
    _rng.shuffle(_combined)
    _n_subset = int(len(_combined) * SUBSET_FRACTION)
    _combined = _combined[:_n_subset]
    image_paths, mask_paths = zip(*_combined)
    image_paths, mask_paths = list(image_paths), list(mask_paths)
    print(f"Using SUBSET: {len(image_paths)} pairs ({SUBSET_FRACTION*100:.0f}% of full dataset)")
else:
    print(f"Using FULL dataset: {len(image_paths)} pairs")

# ── Train/Val/Test Split ────────────────────────────────────────────────────
X_train_paths, X_test_paths, y_train_paths, y_test_paths = train_test_split(
    image_paths, mask_paths, test_size=0.10, random_state=SEED
)
val_fraction_of_temp = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
X_train_paths, X_val_paths, y_train_paths, y_val_paths = train_test_split(
    X_train_paths, y_train_paths, test_size=val_fraction_of_temp, random_state=SEED
)

print(f"Train paths: {len(X_train_paths)} | Val paths: {len(X_val_paths)} | Test paths: {len(X_test_paths)}")

# ── Data Augmentation & Dataset Pipeline ──────────────────────────────────
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
    """Loads, preprocesses, and augments a single image-mask pair."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (img_size, img_size), method=tf.image.ResizeMethod.BILINEAR)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_image(mask, channels=1, expand_animations=False)
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    mask = tf.image.resize(mask, (img_size, img_size), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    mask = tf.cast(mask > (mask_threshold / 255.0), tf.float32)

    return image, mask

def make_dataset(image_paths, mask_paths, batch_size, img_size, mask_threshold, shuffle=False, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    ds = ds.map(lambda img_p, mask_p: _parse_image_mask_pair(img_p, mask_p, img_size, mask_threshold), 
                num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(image_paths), 1024), seed=SEED)
    if augment:
        ds = ds.map(augment_pair, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(X_train_paths, y_train_paths, BATCH_SIZE, IMG_SIZE, MASK_THRESHOLD, shuffle=True, augment=True)
val_ds = make_dataset(X_val_paths, y_val_paths, BATCH_SIZE, IMG_SIZE, MASK_THRESHOLD, shuffle=False, augment=False)
test_ds = make_dataset(X_test_paths, y_test_paths, BATCH_SIZE, IMG_SIZE, MASK_THRESHOLD, shuffle=False, augment=False)

print("Datasets ready.")

# ── U-Net Model ────────────────────────────────────────────────────────────
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

# ── Loss & Metrics ────────────────────────────────────────────────────────
def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.reduce_mean(bce)
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
    metrics=["accuracy", dice_coef, iou_coef,
             tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall")],
)

print("Compiled with BCE + Dice loss; tracking accuracy, Dice, IoU, Precision, Recall.")

# ── Training ──────────────────────────────────────────────────────────────
callbacks = [
    ModelCheckpoint(BEST_CKPT, monitor="val_iou_coef", mode="max", save_best_only=True, verbose=1),
    ModelCheckpoint(LAST_CKPT, save_best_only=False, verbose=0),
    EarlyStopping(monitor="val_iou_coef", mode="max", patience=EARLY_STOP_PATIENCE, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_iou_coef", mode="max", factor=0.5, patience=LR_PATIENCE, min_lr=1e-6, verbose=1),
    CSVLogger(LOG_CSV),
]

history = model.fit(train_ds, validation_data=val_ds, epochs=NUM_EPOCHS, callbacks=callbacks)
print(f"Best checkpoint: {BEST_CKPT}")

# ── Training Curves ──────────────────────────────────────────────────────
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

# ── Test-set Evaluation ──────────────────────────────────────────────────
if os.path.exists(BEST_CKPT):
    model.load_weights(BEST_CKPT)
    print("Loaded best checkpoint.")

results = model.evaluate(test_ds, return_dict=True)
print("\n=== Test metrics ===")
for k, v in results.items():
    print(f"{k:12s}: {v:.4f}")

# ── Visual Predictions ───────────────────────────────────────────────────
n_show = min(4, len(X_test_paths))
# Need to load some test images for visualization
test_images = []
test_masks = []
for i in range(n_show):
    img = tf.io.read_file(X_test_paths[i])
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    test_images.append(img.numpy())
    
    mask = tf.io.read_file(y_test_paths[i])
    mask = tf.image.decode_image(mask, channels=1, expand_animations=False)
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    mask = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    mask = tf.cast(mask > (MASK_THRESHOLD / 255.0), tf.float32)
    test_masks.append(mask.numpy())

X_test = np.array(test_images)
y_test = np.array(test_masks)

preds = model.predict(X_test, verbose=0)
preds_bin = (preds > 0.5).astype(np.float32)

fig, axes = plt.subplots(n_show, 3, figsize=(10, 3 * n_show))
if n_show == 1:
    axes = np.expand_dims(axes, 0)

for i in range(n_show):
    axes[i, 0].imshow(X_test[i])
    axes[i, 0].set_title("Image")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(y_test[i].squeeze(), cmap="gray")
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

print("\n=== Summary ===")
print("Artifacts:")
print(f"  - {BEST_CKPT}")
print(f"  - {LOG_CSV}")
print(f"  - {curve_path}")
print(f"  - {pred_path}")