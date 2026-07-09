import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf

# ==========================================================
# Dataset locations
# ==========================================================

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(_BASE_DIR, "images")
MASK_FOLDER = os.path.join(_BASE_DIR, "masks")
IMG_SIZE = 256


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


# ==========================================================
# Load images and masks
# ==========================================================

images, masks = [], []

for file in sorted(os.listdir(IMAGE_FOLDER)):
    image_path = os.path.join(IMAGE_FOLDER, file)
    mask_path = find_mask_path(MASK_FOLDER, file)

    image = cv2.imread(image_path)
    if image is None:
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) / 255.0

    if mask_path is None:
        continue
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        continue
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE)) / 255.0
    mask = np.expand_dims(mask, axis=-1)

    images.append(image)
    masks.append(mask)

images = np.array(images, dtype=np.float32)
masks = np.array(masks, dtype=np.float32)

_, X_test, _, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)

# ==========================================================
# Load model and run inference
# ==========================================================

model = tf.keras.models.load_model(os.path.join(_BASE_DIR, "baseline_segmentation_model.h5"))

# Test the model on 1 input image from the test set
test_image = X_test[0]
test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension
prediction = model.predict(test_image)
prediction = (prediction > 0.5).astype(np.uint8)

# Display the result
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(X_test[0])
plt.title("Original Image")

plt.subplot(1, 3, 2)
plt.imshow(y_test[0].squeeze(), cmap='gray')
plt.title("Ground Truth")

plt.subplot(1, 3, 3)
plt.imshow(prediction[0].squeeze(), cmap='gray')
plt.title("Prediction")

plt.show()
