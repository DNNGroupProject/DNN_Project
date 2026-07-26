"""
Pseudo concept labels for forest images.

The advisor paper (arXiv:2101.03919) uses caption word-phrases (CUB/Flowers).
Forest Segmented has no text descriptions, so we derive weak concept indicators
from the RGB image + binary mask. These guide the concept layer (L_u / L_m).
"""
from __future__ import annotations

import numpy as np


def forest_pseudo_concepts(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Args:
        image_rgb: float32 HxWx3 in [0,1]
        mask: float32 HxW or HxWx1 in {0,1}  (1=forest)
    Returns:
        z: float32 vector of length 5 (presence of each concept in [0,1])
    """
    if mask.ndim == 3:
        mask = mask.squeeze(-1)
    mask = (mask > 0.5).astype(np.float32)
    h, w = mask.shape
    n = float(h * w)
    forest_ratio = float(mask.mean())

    # color stats
    gray = image_rgb.mean(axis=-1)
    # simple "greenness": G - mean(R,B)
    greenness = image_rgb[..., 1] - 0.5 * (image_rgb[..., 0] + image_rgb[..., 2])

    # boundary band via morphological XOR of dilate/erode (numpy only)
    # use max/min pooling approx with shifts
    m = mask
    dil = m.copy()
    ero = m.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.roll(np.roll(m, dy, 0), dx, 1)
            dil = np.maximum(dil, shifted)
            ero = np.minimum(ero, shifted)
    boundary = ((dil - ero) > 0).astype(np.float32)
    boundary_ratio = float(boundary.mean())

    dark = (gray < 0.35).astype(np.float32)
    bright = (gray > 0.65).astype(np.float32)
    green = (greenness > 0.05).astype(np.float32)

    # Concept presence scores in [0,1]
    dense_canopy = float(np.clip(forest_ratio * (green * mask).sum() / (mask.sum() + 1e-6), 0, 1))
    # high forest + green → dense
    dense_canopy = float(np.clip(0.5 * dense_canopy + 0.5 * max(0.0, forest_ratio - 0.55) / 0.45, 0, 1))

    sparse_trees = float(np.clip(1.0 - abs(forest_ratio - 0.45) / 0.45, 0, 1))
    # peaks when forest_ratio near mid

    shadow_region = float(np.clip((dark * mask).sum() / n * 4.0, 0, 1))

    open_clearing = float(np.clip((1.0 - forest_ratio) * (bright * (1 - mask)).sum() / (n * 0.25 + 1e-6), 0, 1))
    open_clearing = float(np.clip(open_clearing, 0, 1))

    forest_boundary = float(np.clip(boundary_ratio * 8.0, 0, 1))

    z = np.array(
        [dense_canopy, sparse_trees, shadow_region, open_clearing, forest_boundary],
        dtype=np.float32,
    )
    # soft binarize for uniqueness loss stability (keep soft scores)
    return np.clip(z, 0.0, 1.0)
