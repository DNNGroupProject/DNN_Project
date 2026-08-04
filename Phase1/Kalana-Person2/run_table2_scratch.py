"""
Local runner for scratch SegFormer Table 2 metrics.
Uses the same settings as segformer_baseline_scratch_colab.ipynb.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import SegformerConfig, SegformerForSemanticSegmentation

ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / "images"
MASKS_DIR = ROOT / "masks"
CKPT = ROOT / "checkpoints" / "segformer_b0_scratch.pt"
RESULTS = ROOT / "results"
CKPT.parent.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 20
LR = 6e-5
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}")

MIT_B0_CONFIG = dict(
    num_channels=3,
    num_encoder_blocks=4,
    depths=[2, 2, 2, 2],
    sr_ratios=[8, 4, 2, 1],
    hidden_sizes=[32, 64, 160, 256],
    num_attention_heads=[1, 2, 5, 8],
    patch_sizes=[7, 3, 3, 3],
    strides=[4, 2, 2, 2],
    mlp_ratios=[4, 4, 4, 4],
    decoder_hidden_size=256,
)


def mask_name_to_image_name(mask_filename: str) -> str:
    return mask_filename.replace("_mask", "_sat")


class ForestSegDataset(Dataset):
    def __init__(self, mask_filenames, images_dir, masks_dir, img_size=256):
        self.mask_filenames = list(mask_filenames)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.mask_filenames)

    def __getitem__(self, idx):
        mask_fname = self.mask_filenames[idx]
        image_fname = mask_name_to_image_name(mask_fname)
        image = Image.open(self.images_dir / image_fname).convert("RGB")
        image = image.resize((self.img_size, self.img_size))
        mask = Image.open(self.masks_dir / mask_fname).convert("L")
        mask = mask.resize((self.img_size, self.img_size), resample=Image.NEAREST)
        image = np.array(image, dtype=np.float32) / 255.0
        image = (image - self.mean) / self.std
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = (np.array(mask, dtype=np.int64) > 127).astype(np.int64)
        mask = torch.from_numpy(mask).long()
        return image, mask


def build_model():
    cfg = SegformerConfig(
        num_labels=2,
        id2label={0: "non_forest", 1: "forest"},
        label2id={"non_forest": 0, "forest": 1},
        **MIT_B0_CONFIG,
    )
    return SegformerForSemanticSegmentation(cfg).to(device)


def confusion_counts(pred_mask, true_mask):
    tp = (pred_mask & true_mask).sum().item()
    fp = (pred_mask & ~true_mask).sum().item()
    fn = (~pred_mask & true_mask).sum().item()
    tn = (~pred_mask & ~true_mask).sum().item()
    return tp, fp, fn, tn


def scores_from_counts(tp, fp, fn, eps=1e-6):
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    f1 = precision * recall * 2.0 / (precision + recall + eps)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def count_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def estimate_gflops(model):
    try:
        from thop import profile

        class _Wrap(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                return self.inner(pixel_values=x).logits

        m = model
        was = m.training
        m.eval()
        wrap = _Wrap(m).to(device)
        dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
        flops, _ = profile(wrap, inputs=(dummy,), verbose=False)
        if was:
            m.train()
        return float(flops) / 1e9
    except Exception as e:
        print("GFLOPs failed:", e)
        return None


def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = total_dice = total_iou = 0.0
    n_batches = 0
    tp = fp = fn = 0
    with torch.set_grad_enabled(is_train):
        for images, masks in tqdm(loader, leave=False):
            images, masks = images.to(device), masks.to(device)
            outputs = model(pixel_values=images)
            logits = F.interpolate(
                outputs.logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
            )
            loss = F.cross_entropy(logits, masks)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            preds = logits.argmax(dim=1).bool()
            tpc, fpc, fnc, _ = confusion_counts(preds, masks.bool())
            tp += tpc
            fp += fpc
            fn += fnc
            s = scores_from_counts(tpc, fpc, fnc)
            total_loss += loss.item()
            total_dice += s["dice"]
            total_iou += s["iou"]
            n_batches += 1
    micro = scores_from_counts(tp, fp, fn)
    return total_loss / n_batches, total_dice / n_batches, total_iou / n_batches, micro


@torch.no_grad()
def evaluate_table2(model, loader):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    tp = fp = fn = 0
    for images, masks in tqdm(loader, leave=False, desc="test"):
        images, masks = images.to(device), masks.to(device)
        outputs = model(pixel_values=images)
        logits = F.interpolate(
            outputs.logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
        )
        total_loss += F.cross_entropy(logits, masks).item()
        n_batches += 1
        preds = logits.argmax(dim=1).bool()
        tpc, fpc, fnc, _ = confusion_counts(preds, masks.bool())
        tp += tpc
        fp += fpc
        fn += fnc
    metrics = scores_from_counts(tp, fp, fn)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    mask_files = sorted(
        f for f in os.listdir(MASKS_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    rng = random.Random(SEED)
    all_files = list(mask_files)
    rng.shuffle(all_files)
    n = len(all_files)
    n_val = int(n * VAL_SPLIT)
    n_test = int(n * TEST_SPLIT)
    val_files = all_files[:n_val]
    test_files = all_files[n_val : n_val + n_test]
    train_files = all_files[n_val + n_test :]
    print(f"split train/val/test = {len(train_files)}/{len(val_files)}/{len(test_files)}")

    model = build_model()
    params = count_params(model)
    gflops = estimate_gflops(model)
    print(f"params={params/1e6:.2f}M  gflops={gflops}")

    # Save efficiency early so Table 2 Params/FLOPs are available even if train is long
    eff = {
        "model": "SegFormer-B0 (from scratch)",
        "params": params,
        "params_M": round(params / 1e6, 2),
        "gflops": None if gflops is None else round(gflops, 3),
    }
    with open(RESULTS / "table2_segformer_scratch_efficiency.json", "w", encoding="utf-8") as f:
        json.dump(eff, f, indent=2)

    train_loader = DataLoader(
        ForestSegDataset(train_files, IMAGES_DIR, MASKS_DIR),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        ForestSegDataset(val_files, IMAGES_DIR, MASKS_DIR),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        ForestSegDataset(test_files, IMAGES_DIR, MASKS_DIR),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    best_val_dice = -1.0
    t0 = time.time()

    if CKPT.exists():
        print(f"Loading existing checkpoint {CKPT}")
        model.load_state_dict(torch.load(CKPT, map_location=device))
    else:
        for epoch in range(1, EPOCHS + 1):
            tr = run_epoch(model, train_loader, optimizer)
            va = run_epoch(model, val_loader)
            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"train_dice={tr[1]:.4f} val_dice={va[1]:.4f} val_iou={va[2]:.4f} "
                f"micro_f1={va[3]['f1']:.4f} | {(time.time()-t0)/60:.1f} min"
            )
            if va[1] > best_val_dice:
                best_val_dice = va[1]
                torch.save(model.state_dict(), CKPT)
                print(f"  saved {CKPT}")

        model.load_state_dict(torch.load(CKPT, map_location=device))

    test_m = evaluate_table2(model, test_loader)
    row = {
        "model": "SegFormer-B0 (from scratch)",
        "dice": round(test_m["dice"], 4),
        "iou": round(test_m["iou"], 4),
        "f1": round(test_m["f1"], 4),
        "precision": round(test_m["precision"], 4),
        "recall": round(test_m["recall"], 4),
        "params": params,
        "params_M": round(params / 1e6, 2),
        "gflops": None if gflops is None else round(gflops, 3),
        "device": str(device),
        "wall_clock_min": round((time.time() - t0) / 60, 1),
    }
    print("\n========== TABLE 2 — SegFormer-B0 (scratch) ==========")
    for k in ("dice", "iou", "f1", "precision", "params_M", "gflops"):
        print(f"{k:12s}: {row[k]}")
    print("====================================================")

    with open(RESULTS / "table2_segformer_scratch.json", "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)
    with open(RESULTS / "table2_segformer_scratch.txt", "w", encoding="utf-8") as f:
        f.write(
            f"Model: {row['model']}\n"
            f"Dice: {row['dice']:.4f}\n"
            f"IoU: {row['iou']:.4f}\n"
            f"F1: {row['f1']:.4f}\n"
            f"Precision: {row['precision']:.4f}\n"
            f"Params: {row['params_M']:.2f}M\n"
            f"GFLOPs: {row['gflops']}\n"
        )
    print(f"Wrote {RESULTS / 'table2_segformer_scratch.json'}")


if __name__ == "__main__":
    main()
