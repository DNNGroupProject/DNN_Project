"""Forest dataset with pseudo concept labels for guided concept learning."""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from pseudo_concepts import forest_pseudo_concepts
import config as cfg


def find_mask_path(mask_folder: str, filename: str) -> Optional[str]:
    stem, ext = os.path.splitext(filename)
    exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    stems = [stem]
    if "_sat_" in stem:
        stems.append(stem.replace("_sat_", "_mask_"))
    order = [ext] + [e for e in exts if e != ext]
    for s in stems:
        for e in order:
            p = os.path.join(mask_folder, s + e)
            if os.path.exists(p):
                return p
    return None


def list_pairs(img_dir, mask_dir, max_samples=None) -> List[Tuple[str, str]]:
    files = sorted(
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
    )
    pairs = []
    for f in files:
        mp = find_mask_path(mask_dir, f)
        if mp is None:
            continue
        pairs.append((os.path.join(img_dir, f), mp))
        if max_samples and len(pairs) >= max_samples:
            break
    return pairs


class ForestConceptDataset(Dataset):
    def __init__(self, pairs, img_size=224, augment=False, mask_threshold=127):
        self.pairs = pairs
        self.img_size = img_size
        self.augment = augment
        self.mask_threshold = mask_threshold

    def __len__(self):
        return len(self.pairs)

    def _augment(self, img, mask):
        if np.random.rand() > 0.5:
            img = np.ascontiguousarray(img[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
        if np.random.rand() > 0.5:
            img = np.ascontiguousarray(img[::-1, :])
            mask = np.ascontiguousarray(mask[::-1, :])
        k = np.random.randint(0, 4)
        if k:
            img = np.rot90(img, k).copy()
            mask = np.rot90(mask, k).copy()
        return img, mask

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
        image = cv2.imread(ip)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0
        mask = (mask > self.mask_threshold).astype(np.float32)

        if self.augment:
            image, mask = self._augment(image, mask)

        z = forest_pseudo_concepts(image, mask)

        # ImageNet-ish normalize for ViT stability
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_n = (image - mean) / std

        image_t = torch.from_numpy(image_n.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(mask[None, ...]).float()
        z_t = torch.from_numpy(z).float()
        # keep unnormalized RGB for XAI overlays
        rgb_t = torch.from_numpy(image.transpose(2, 0, 1)).float()
        return image_t, mask_t, z_t, rgb_t
