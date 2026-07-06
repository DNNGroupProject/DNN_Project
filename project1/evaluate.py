"""Evaluate the trained U-Net baseline on the held-out test set.

Usage
-----
    python evaluate.py

Outputs (written to config.RESULTS_DIR):
    results/test_metrics.txt         – all metrics printed + saved
    results/test_predictions.png     – grid: image | ground truth | prediction
"""

import os

import torch
import numpy as np
from tqdm import tqdm

import config
from dataset import get_loaders
from losses import BCEDiceLoss
from metrics import compute_metrics, aggregate_metrics
from model import build_model

# ─── Device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _unnormalise(t: torch.Tensor) -> np.ndarray:
    """Convert a normalised (3,H,W) tensor to a uint8 HxWx3 numpy array."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img  = t.cpu().numpy().transpose(1, 2, 0)
    img  = img * std + mean
    img  = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


# ─── Evaluation loop ──────────────────────────────────────────────────────────

def evaluate(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    metric_list  = []
    sample_batch = None     # store one batch for visualisation

    with torch.no_grad():
        for i, (imgs, masks) in enumerate(tqdm(test_loader, desc='Testing',
                                               dynamic_ncols=True)):
            imgs  = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(imgs)
            loss   = criterion(logits, masks)

            running_loss += loss.item()
            metric_list.append(compute_metrics(logits, masks))

            if sample_batch is None:
                sample_batch = (imgs.cpu(), masks.cpu(), logits.cpu())

    avg_loss    = running_loss / len(test_loader)
    avg_metrics = aggregate_metrics(metric_list)
    return avg_loss, avg_metrics, sample_batch


# ─── Visualisation ────────────────────────────────────────────────────────────

def _save_predictions(sample_batch, n_samples: int = 8):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        imgs, masks, logits = sample_batch
        preds = (torch.sigmoid(logits) > 0.5).float()
        n     = min(n_samples, imgs.shape[0])

        fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
        if n == 1:
            axes = axes[np.newaxis, :]

        col_titles = ['Satellite Image', 'Ground Truth', 'Prediction']
        for j, t in enumerate(col_titles):
            axes[0, j].set_title(t, fontsize=10, fontweight='bold')

        for i in range(n):
            axes[i, 0].imshow(_unnormalise(imgs[i]))
            axes[i, 1].imshow(masks[i, 0].numpy(), cmap='Greens', vmin=0, vmax=1)
            axes[i, 2].imshow(preds[i, 0].numpy(), cmap='Greens', vmin=0, vmax=1)
            for j in range(3):
                axes[i, j].axis('off')

        fig.suptitle('U-Net Baseline – Test Set Predictions', fontsize=12)
        fig.tight_layout()
        out = os.path.join(config.RESULTS_DIR, 'test_predictions.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Predictions grid → {out}')
    except Exception as e:
        print(f'[warn] Could not save prediction grid: {e}')


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(config.BEST_CHECKPOINT):
        raise FileNotFoundError(
            f'No checkpoint found at {config.BEST_CHECKPOINT}\n'
            'Run train.py first.'
        )

    # ── Load model ────────────────────────────────────────────────────────────
    model = build_model().to(DEVICE)
    ckpt  = torch.load(config.BEST_CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])
    print(f'Loaded checkpoint from epoch {ckpt["epoch"]}  '
          f'(val IoU={ckpt["val_iou"]:.4f})')

    # ── Data ──────────────────────────────────────────────────────────────────
    _, _, test_loader = get_loaders()
    criterion         = BCEDiceLoss(config.BCE_WEIGHT, config.DICE_WEIGHT)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    test_loss, metrics, sample_batch = evaluate(model, test_loader, criterion)

    # ── Report ────────────────────────────────────────────────────────────────
    lines = [
        '=' * 50,
        'U-Net Baseline  –  Test Set Results',
        '=' * 50,
        f'Loss (BCE+Dice) : {test_loss:.6f}',
        '',
        f'IoU (Jaccard)   : {metrics["iou"]:.4f}',
        f'Dice Coefficient: {metrics["dice"]:.4f}',
        f'Pixel Accuracy  : {metrics["pixel_acc"]:.4f}',
        f'Precision       : {metrics["precision"]:.4f}',
        f'Recall          : {metrics["recall"]:.4f}',
        f'F1 Score        : {metrics["f1"]:.4f}',
        '=' * 50,
    ]
    report = '\n'.join(lines)
    print('\n' + report)

    out_txt = os.path.join(config.RESULTS_DIR, 'test_metrics.txt')
    with open(out_txt, 'w') as f:
        f.write(report + '\n')
    print(f'\nMetrics saved → {out_txt}')

    # ── Visualise ─────────────────────────────────────────────────────────────
    _save_predictions(sample_batch, n_samples=8)


if __name__ == '__main__':
    main()
