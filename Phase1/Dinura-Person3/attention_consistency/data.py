"""Dataset loading for the Dinura-Person3 SegFormer pipeline.

Reuses the dataset committed under Kalana-Person2/{images,masks} (5,108
pairs — the same "Forest Segmented"-style dataset the rest of the team
uses) rather than the Lasana-Person4_Evaluation config paths, which point
at a `Lasana/dataset/...` folder that isn't present in this checkout.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split

PERSON3_DIR = Path(__file__).resolve().parent.parent
PHASE1_DIR = PERSON3_DIR.parent
IMG_DIR = PHASE1_DIR / "Kalana-Person2" / "images"
MASK_DIR = PHASE1_DIR / "Kalana-Person2" / "masks"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _mask_path_for(image_filename: str) -> Optional[Path]:
    stem, ext = os.path.splitext(image_filename)
    candidate_stem = stem.replace("_sat_", "_mask_") if "_sat_" in stem else stem
    for e in (ext, ".jpg", ".jpeg", ".png"):
        p = MASK_DIR / f"{candidate_stem}{e}"
        if p.exists():
            return p
    return None


def list_pairs(max_samples: Optional[int] = None, seed: int = 42) -> List[Tuple[Path, Path]]:
    files = sorted(f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    rng = np.random.RandomState(seed)
    rng.shuffle(files)
    pairs = []
    for f in files:
        mp = _mask_path_for(f)
        if mp is not None:
            pairs.append((IMG_DIR / f, mp))
        if max_samples is not None and len(pairs) >= max_samples:
            break
    if not pairs:
        raise FileNotFoundError(f"No image/mask pairs found under {IMG_DIR} / {MASK_DIR}")
    return pairs


def load_pairs(
    pairs: List[Tuple[Path, Path]], img_size: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    images, masks = [], []
    for img_path, mask_path in pairs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size)).astype(np.float32) / 255.0

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)

        images.append(img)
        masks.append(mask)
    return np.stack(images), np.stack(masks)


def make_splits(
    n_train: int, n_val: int, n_test: int, seed: int = 42
) -> dict:
    """Stratified-by-shuffle split, disjoint train/val/test, same seed
    convention (42) the rest of the team uses."""
    total = n_train + n_val + n_test
    pairs = list_pairs(max_samples=total, seed=seed)
    train_pairs, rest = train_test_split(pairs, train_size=n_train, random_state=seed)
    val_pairs, test_pairs = train_test_split(
        rest, train_size=n_val, test_size=n_test, random_state=seed
    )
    return {"train": train_pairs, "val": val_pairs, "test": test_pairs}


def to_model_input(images_hw3_01: np.ndarray) -> torch.Tensor:
    """(N,H,W,3) in [0,1] -> (N,3,H,W) ImageNet-normalized tensor, matching
    the pretrained MiT-B0 encoder's expected input distribution."""
    x = (images_hw3_01 - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(x.astype(np.float32)).permute(0, 3, 1, 2).contiguous()
    return x
