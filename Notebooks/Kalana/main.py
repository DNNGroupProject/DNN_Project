"""
==========================================================
Simple CNN Semantic Segmentation
----------------------------------------------------------

This is intentionally written as a BEGINNER BASELINE.

No U-Net
No Transfer Learning
No Data Augmentation
No Batch Normalization
No Skip Connections

The purpose is to understand how semantic segmentation works
before moving to research models.

Author: Your Name
==========================================================
"""

# ==========================================================
# Import libraries
# ==========================================================

import os
import numpy as np
import cv2

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import UpSampling2D


# ==========================================================
# Dataset locations
# ==========================================================

IMAGE_FOLDER = "images"
MASK_FOLDER = "masks"

# Image size
IMG_SIZE = 256


def find_mask_path(mask_folder, filename):
    """Return the best matching mask path for a given image filename."""
    direct_path = os.path.join(mask_folder, filename)
    if os.path.exists(direct_path):
        return direct_path

    stem = os.path.splitext(filename)[0]
    common_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

    for ext in common_exts:
        candidate = os.path.join(mask_folder, stem + ext)
        if os.path.exists(candidate):
            return candidate

    return None


# ==========================================================
# Read all image names
# ==========================================================

image_files = sorted(os.listdir(IMAGE_FOLDER))

images = []
masks = []


# ==========================================================
# Load every image and its mask
# ==========================================================

for file in image_files:

    # Image path
    image_path = os.path.join(IMAGE_FOLDER, file)

    # Mask path (same filename first, then extension fallback)
    mask_path = find_mask_path(MASK_FOLDER, file)

    # -------------------------
    # Read image
    # -------------------------
    image = cv2.imread(image_path)

    if image is None:
        print(f"[WARN] Skipping unreadable image: {image_path}")
        continue

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize image
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    # Convert pixel values to 0-1
    image = image / 255.0

    # -------------------------
    # Read mask
    # -------------------------
    if mask_path is None:
        print(f"[WARN] No matching mask found for image: {file}")
        continue

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print(f"[WARN] Skipping unreadable mask: {mask_path}")
        continue

    # Resize mask
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

    # Convert mask values to 0 or 1
    mask = mask / 255.0

    # Add channel dimension
    mask = np.expand_dims(mask, axis=-1)

    images.append(image)
    masks.append(mask)


# ==========================================================
# Convert lists into NumPy arrays
# ==========================================================

images = np.array(images, dtype=np.float32)
masks = np.array(masks, dtype=np.float32)

if len(images) == 0 or len(masks) == 0:
    raise RuntimeError(
        "No valid image-mask pairs were loaded. Check IMAGE_FOLDER/MASK_FOLDER contents."
    )


print("Images shape :", images.shape)
print("Masks shape  :", masks.shape)


# ==========================================================
# Split dataset
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    images,
    masks,
    test_size=0.2,
    random_state=42
)

print("Training Images :", len(X_train))
print("Testing Images  :", len(X_test))


# ==========================================================
# Build Simple CNN
# ==========================================================

model = Sequential()

# ---------------- Encoder ----------------

model.add(
    Conv2D(
        filters=32,
        kernel_size=(3,3),
        activation='relu',
        padding='same',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
)

model.add(MaxPooling2D((2,2)))

model.add(
    Conv2D(
        64,
        (3,3),
        activation='relu',
        padding='same'
    )
)

model.add(MaxPooling2D((2,2)))

model.add(
    Conv2D(
        128,
        (3,3),
        activation='relu',
        padding='same'
    )
)

# ---------------- Decoder ----------------

model.add(UpSampling2D((2,2)))

model.add(
    Conv2D(
        64,
        (3,3),
        activation='relu',
        padding='same'
    )
)

model.add(UpSampling2D((2,2)))

model.add(
    Conv2D(
        32,
        (3,3),
        activation='relu',
        padding='same'
    )
)

# Output layer

model.add(
    Conv2D(
        1,
        (1,1),
        activation='sigmoid',
        padding='same'
    )
)

model.summary()


# ==========================================================
# Compile model
# ==========================================================

model.compile(

    optimizer='adam',

    loss='binary_crossentropy',

    metrics=['accuracy']

)


# ==========================================================
# Train
# ==========================================================

history = model.fit(

    X_train,
    y_train,

    validation_split=0.2,

    epochs=10,

    batch_size=8

)


# ==========================================================
# Evaluate
# ==========================================================

loss, accuracy = model.evaluate(X_test, y_test)

print("Test Loss :", loss)
print("Accuracy  :", accuracy)


# ==========================================================
# Save model
# ==========================================================

model.save("baseline_segmentation_model.h5")


# ==========================================================
# Predict one image
# ==========================================================

prediction = model.predict(X_test[:1])

prediction = prediction[0]

# Convert probabilities into binary image

prediction = (prediction > 0.5).astype(np.uint8)


# ==========================================================
# Display result
# ==========================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(X_test[0])
plt.title("Original Image")

plt.subplot(1,3,2)
plt.imshow(y_test[0].squeeze(), cmap='gray')
plt.title("Ground Truth")

plt.subplot(1,3,3)
plt.imshow(prediction.squeeze(), cmap='gray')
plt.title("Prediction")

plt.show()