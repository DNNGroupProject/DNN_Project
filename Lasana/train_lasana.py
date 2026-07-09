"""
Train Lasana U-Net on Forest Segmented dataset and report test metrics.
"""
import os
import random
import numpy as np
import cv2
import matplotlib

matplotlib.use("Agg")
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

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(
    BASE_DIR, "dataset", "Forest Segmented", "Forest Segmented", "images"
)
MASK_FOLDER = os.path.join(
    BASE_DIR, "dataset", "Forest Segmented", "Forest Segmented", "masks"
)
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Hyperparameters ──────────────────────────────────────────────────────────
SEED = 42
IMG_SIZE = 256
MASK_THRESHOLD = 127
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
LEARNING_RATE = 1e-3
BCE_WEIGHT = 0.5
DICE_WEIGHT = 0.5

# CPU has no GPU in this environment — use a lighter U-Net + sample cap so
# training finishes in a reasonable time. Override with env vars if needed.
HAS_GPU = len(tf.config.list_physical_devices("GPU")) > 0
if HAS_GPU:
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    EARLY_STOP_PATIENCE = 10
    LR_PATIENCE = 5
    UNET_FEATURES = (64, 128, 256, 512)
    DEFAULT_MAX_SAMPLES = None
else:
    BATCH_SIZE = 4
    NUM_EPOCHS = 15
    EARLY_STOP_PATIENCE = 5
    LR_PATIENCE = 3
    UNET_FEATURES = (32, 64, 128)  # lighter for CPU
    DEFAULT_MAX_SAMPLES = 1200     # ~80/10/10 of 1200 pairs

MAX_SAMPLES = os.environ.get("LASANA_MAX_SAMPLES")
MAX_SAMPLES = int(MAX_SAMPLES) if MAX_SAMPLES else DEFAULT_MAX_SAMPLES
if os.environ.get("LASANA_EPOCHS"):
    NUM_EPOCHS = int(os.environ["LASANA_EPOCHS"])

BEST_CKPT = os.path.join(CHECKPOINT_DIR, "lasana_unet_best.keras")
LAST_CKPT = os.path.join(CHECKPOINT_DIR, "lasana_unet_last.keras")
LOG_CSV = os.path.join(RESULTS_DIR, "training_log.csv")

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def find_mask_path(mask_folder, filename):
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


def load_dataset(image_folder, mask_folder, img_size=256, mask_threshold=127, max_samples=None):
    image_files = sorted(
        f
        for f in os.listdir(image_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
    )
    if max_samples is not None:
        image_files = image_files[:max_samples]

    images, masks = [], []
    skipped = 0
    for i, file in enumerate(image_files):
        if (i + 1) % 500 == 0:
            print(f"  loading {i + 1}/{len(image_files)}...")
        image_path = os.path.join(image_folder, file)
        mask_path = find_mask_path(mask_folder, file)
        image = cv2.imread(image_path)
        if image is None or mask_path is None:
            skipped += 1
            continue
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            skipped += 1
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0

        mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > mask_threshold).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)

        images.append(image)
        masks.append(mask)

    images = np.array(images, dtype=np.float32)
    masks = np.array(masks, dtype=np.float32)
    if len(images) == 0:
        raise RuntimeError("No valid image-mask pairs loaded.")
    print(f"Loaded {len(images)} pairs | skipped {skipped}")
    print("Images:", images.shape, "| Masks:", masks.shape)
    print(f"Forest pixel ratio: {masks.mean():.4f}")
    return images, masks


def augment_pair(image, mask):
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


def make_dataset(x, y, batch_size, shuffle=False, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(x), 1024), seed=SEED)
    if augment:
        ds = ds.map(augment_pair, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 3), features=None):
    if features is None:
        features = UNET_FEATURES
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


def main():
    print("TensorFlow:", tf.__version__)
    print("GPUs:", tf.config.list_physical_devices("GPU"))
    print(f"Mode: {'GPU' if HAS_GPU else 'CPU-light'} | features={UNET_FEATURES} | "
          f"batch={BATCH_SIZE} | epochs={NUM_EPOCHS} | max_samples={MAX_SAMPLES}")
    print("IMAGE_FOLDER:", IMAGE_FOLDER)
    print("MASK_FOLDER :", MASK_FOLDER)
    assert os.path.isdir(IMAGE_FOLDER), f"Missing images: {IMAGE_FOLDER}"
    assert os.path.isdir(MASK_FOLDER), f"Missing masks: {MASK_FOLDER}"

    print("\nLoading dataset...")
    images, masks = load_dataset(
        IMAGE_FOLDER, MASK_FOLDER, IMG_SIZE, MASK_THRESHOLD, MAX_SAMPLES
    )

    X_temp, X_test, y_temp, y_test = train_test_split(
        images, masks, test_size=0.10, random_state=SEED
    )
    val_fraction_of_temp = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction_of_temp, random_state=SEED
    )
    del images, masks, X_temp, y_temp
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    train_ds = make_dataset(X_train, y_train, BATCH_SIZE, shuffle=True, augment=True)
    val_ds = make_dataset(X_val, y_val, BATCH_SIZE, shuffle=False, augment=False)
    test_ds = make_dataset(X_test, y_test, BATCH_SIZE, shuffle=False, augment=False)

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

    callbacks = [
        ModelCheckpoint(
            BEST_CKPT, monitor="val_iou_coef", mode="max", save_best_only=True, verbose=1
        ),
        ModelCheckpoint(LAST_CKPT, save_best_only=False, verbose=0),
        EarlyStopping(
            monitor="val_iou_coef",
            mode="max",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_iou_coef",
            mode="max",
            factor=0.5,
            patience=LR_PATIENCE,
            min_lr=1e-6,
            verbose=1,
        ),
        CSVLogger(LOG_CSV),
    ]

    print("\nTraining...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=NUM_EPOCHS,
        callbacks=callbacks,
    )

    # Curves
    hist = history.history
    epochs_range = range(1, len(hist["loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs_range, hist["loss"], label="train")
    axes[0].plot(epochs_range, hist["val_loss"], label="val")
    axes[0].set_title("BCE + Dice Loss")
    axes[0].legend()
    axes[1].plot(epochs_range, hist["iou_coef"], label="train")
    axes[1].plot(epochs_range, hist["val_iou_coef"], label="val")
    axes[1].set_title("IoU")
    axes[1].legend()
    axes[2].plot(epochs_range, hist["dice_coef"], label="train")
    axes[2].plot(epochs_range, hist["val_dice_coef"], label="val")
    axes[2].set_title("Dice")
    axes[2].legend()
    plt.tight_layout()
    curve_path = os.path.join(RESULTS_DIR, "training_curves.png")
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print("Saved:", curve_path)

    if os.path.exists(BEST_CKPT):
        model.load_weights(BEST_CKPT)
        print("Loaded best checkpoint.")

    results = model.evaluate(test_ds, return_dict=True)
    print("\n=== Test metrics ===")
    for k, v in results.items():
        print(f"{k:12s}: {v:.4f}")

    metrics_path = os.path.join(RESULTS_DIR, "test_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        for k, v in results.items():
            f.write(f"{k}: {v:.6f}\n")
    print("Saved:", metrics_path)

    # Prediction grid
    n_show = min(4, len(X_test))
    preds = model.predict(X_test[:n_show], verbose=0)
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
    plt.close()
    print("Saved:", pred_path)
    print("\nDone.")


if __name__ == "__main__":
    main()
