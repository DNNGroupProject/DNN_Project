import os
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import config


# ─── ImageNet normalisation statistics ────────────────────────────────────────
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


class ForestDataset(Dataset):
    """Binary forest-segmentation dataset.

    Each sample is a (image, mask) pair where:
      image : FloatTensor (3, H, W)  – ImageNet-normalised
      mask  : FloatTensor (1, H, W)  – 0.0 = background, 1.0 = forest
    """

    def __init__(self, df, img_dir, mask_dir,
                 img_size=256, augment=False, mask_threshold=127):
        self.df              = df.reset_index(drop=True)
        self.img_dir         = img_dir
        self.mask_dir        = mask_dir
        self.img_size        = img_size
        self.augment         = augment
        self.mask_threshold  = mask_threshold

        self.img_tf = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=_MEAN, std=_STD),
        ])
        self.mask_resize = T.Resize(
            (img_size, img_size), interpolation=T.InterpolationMode.NEAREST
        )

    # ── length ────────────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.df)

    # ── single sample ─────────────────────────────────────────────────────────
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img  = Image.open(os.path.join(self.img_dir,  row['image'])).convert('RGB')
        mask = Image.open(os.path.join(self.mask_dir, row['mask'])).convert('L')

        # ── joint augmentation (training only) ────────────────────────────────
        if self.augment:
            if random.random() > 0.5:
                img  = TF.hflip(img)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                img  = TF.vflip(img)
                mask = TF.vflip(mask)
            angle = random.choice([0, 90, 180, 270])
            if angle:
                img  = TF.rotate(img,  angle)
                mask = TF.rotate(mask, angle)

        # ── image tensor ──────────────────────────────────────────────────────
        img_t = self.img_tf(img)                             # (3, H, W)

        # ── mask tensor ───────────────────────────────────────────────────────
        mask = self.mask_resize(mask)
        mask_arr = np.array(mask, dtype=np.float32)
        mask_t   = torch.from_numpy(
            (mask_arr > self.mask_threshold).astype(np.float32)
        ).unsqueeze(0)                                       # (1, H, W)

        return img_t, mask_t


# ─── Split helper ─────────────────────────────────────────────────────────────

def make_splits(csv_path, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Return (train_df, val_df, test_df) from a metadata CSV."""
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    n         = len(df)
    n_train   = int(n * train_ratio)
    n_val     = int(n * val_ratio)

    train_df  = df.iloc[:n_train]
    val_df    = df.iloc[n_train : n_train + n_val]
    test_df   = df.iloc[n_train + n_val :]

    return train_df, val_df, test_df


# ─── DataLoader factory ───────────────────────────────────────────────────────

def get_loaders():
    """Return (train_loader, val_loader, test_loader)."""
    train_df, val_df, test_df = make_splits(
        config.CSV_PATH,
        train_ratio = config.TRAIN_RATIO,
        val_ratio   = config.VAL_RATIO,
        seed        = config.RANDOM_SEED,
    )

    train_ds = ForestDataset(train_df, config.IMG_DIR, config.MASK_DIR,
                             img_size=config.IMG_SIZE, augment=True,
                             mask_threshold=config.MASK_THRESHOLD)
    val_ds   = ForestDataset(val_df,   config.IMG_DIR, config.MASK_DIR,
                             img_size=config.IMG_SIZE, augment=False,
                             mask_threshold=config.MASK_THRESHOLD)
    test_ds  = ForestDataset(test_df,  config.IMG_DIR, config.MASK_DIR,
                             img_size=config.IMG_SIZE, augment=False,
                             mask_threshold=config.MASK_THRESHOLD)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              shuffle=True,  num_workers=config.NUM_WORKERS,
                              pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=config.NUM_WORKERS,
                              pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=config.NUM_WORKERS,
                              pin_memory=False)

    print(f'Dataset split  →  train: {len(train_ds)}  |  val: {len(val_ds)}  |  test: {len(test_ds)}')
    return train_loader, val_loader, test_loader
