"""Training script for the U-Net baseline.

Usage
-----
    python train.py

Outputs (written to config.CHECKPOINT_DIR / config.RESULTS_DIR):
    checkpoints/unet_baseline_best.pth   – best checkpoint (highest val IoU)
    checkpoints/unet_baseline_last.pth   – last-epoch checkpoint
    results/training_log.csv            – per-epoch metrics table
    results/training_curves.png         – loss + IoU learning curves
"""

import csv
import os
import time

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import config
from dataset import get_loaders
from losses import BCEDiceLoss
from metrics import compute_metrics, aggregate_metrics
from model import build_model

# ─── Device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')


# ─── One epoch ────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, phase: str):
    """Run one training or validation epoch.

    Returns
    -------
    avg_loss : float
    avg_metrics : dict
    """
    is_train = phase == 'train'
    model.train() if is_train else model.eval()

    running_loss = 0.0
    metric_list  = []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, masks in tqdm(loader, desc=f'  {phase:5s}', leave=False,
                                dynamic_ncols=True):
            imgs  = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(imgs)
            loss   = criterion(logits, masks)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            running_loss += loss.item()
            metric_list.append(compute_metrics(logits.detach(), masks))

    avg_loss    = running_loss / len(loader)
    avg_metrics = aggregate_metrics(metric_list)
    return avg_loss, avg_metrics


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = get_loaders()

    # ── Model / Loss / Optimiser ──────────────────────────────────────────────
    model     = build_model().to(DEVICE)
    criterion = BCEDiceLoss(bce_weight=config.BCE_WEIGHT,
                            dice_weight=config.DICE_WEIGHT)
    optimizer = optim.Adam(model.parameters(),
                           lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=config.LR_T_MAX,
                                  eta_min=config.LR_ETA_MIN)

    # ── Logging ───────────────────────────────────────────────────────────────
    log_path   = os.path.join(config.RESULTS_DIR, 'training_log.csv')
    fieldnames = ['epoch', 'lr',
                  'train_loss', 'train_iou', 'train_dice', 'train_pixel_acc',
                  'val_loss',   'val_iou',   'val_dice',   'val_pixel_acc']
    log_file   = open(log_path, 'w', newline='')
    writer     = csv.DictWriter(log_file, fieldnames=fieldnames)
    writer.writeheader()

    best_val_iou = 0.0
    history      = {k: [] for k in fieldnames}

    print(f'\nTraining for {config.NUM_EPOCHS} epochs on {DEVICE}\n{"─"*60}')

    for epoch in range(1, config.NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_m = run_epoch(model, train_loader, criterion,
                                        optimizer, phase='train')
        val_loss,   val_m   = run_epoch(model, val_loader,   criterion,
                                        optimizer, phase='val')
        scheduler.step()
        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        # ── Print ─────────────────────────────────────────────────────────────
        print(f'Epoch [{epoch:3d}/{config.NUM_EPOCHS}]  '
              f'lr={lr_now:.2e}  '
              f'train_loss={train_loss:.4f}  train_iou={train_m["iou"]:.4f}  '
              f'val_loss={val_loss:.4f}  val_iou={val_m["iou"]:.4f}  '
              f'({elapsed:.1f}s)')

        # ── CSV log ───────────────────────────────────────────────────────────
        row = dict(epoch=epoch, lr=f'{lr_now:.6f}',
                   train_loss=f'{train_loss:.6f}',
                   train_iou=f'{train_m["iou"]:.6f}',
                   train_dice=f'{train_m["dice"]:.6f}',
                   train_pixel_acc=f'{train_m["pixel_acc"]:.6f}',
                   val_loss=f'{val_loss:.6f}',
                   val_iou=f'{val_m["iou"]:.6f}',
                   val_dice=f'{val_m["dice"]:.6f}',
                   val_pixel_acc=f'{val_m["pixel_acc"]:.6f}')
        writer.writerow(row)
        log_file.flush()
        for k, v in row.items():
            history[k].append(v)

        # ── Checkpoint ────────────────────────────────────────────────────────
        ckpt = dict(epoch=epoch, model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    val_iou=val_m['iou'], val_dice=val_m['dice'])

        torch.save(ckpt, config.LAST_CHECKPOINT)

        if val_m['iou'] > best_val_iou:
            best_val_iou = val_m['iou']
            torch.save(ckpt, config.BEST_CHECKPOINT)
            print(f'  ↳ New best val IoU: {best_val_iou:.4f}  (checkpoint saved)')

    log_file.close()
    print(f'\nTraining complete.  Best val IoU: {best_val_iou:.4f}')
    print(f'Log  → {log_path}')
    print(f'Best → {config.BEST_CHECKPOINT}')

    # ── Plot learning curves ──────────────────────────────────────────────────
    _plot_curves(log_path)


def _plot_curves(log_path: str):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import pandas as pd

        df  = pd.read_csv(log_path)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(df['epoch'], df['train_loss'].astype(float), label='Train')
        axes[0].plot(df['epoch'], df['val_loss'].astype(float),   label='Val')
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
        axes[0].set_title('BCEDice Loss'); axes[0].legend(); axes[0].grid(True)

        axes[1].plot(df['epoch'], df['train_iou'].astype(float), label='Train IoU')
        axes[1].plot(df['epoch'], df['val_iou'].astype(float),   label='Val IoU')
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('IoU')
        axes[1].set_title('Jaccard Index (IoU)'); axes[1].legend(); axes[1].grid(True)

        fig.suptitle('U-Net Baseline – Training Curves', fontsize=13)
        fig.tight_layout()
        out = os.path.join(config.RESULTS_DIR, 'training_curves.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Curves → {out}')
    except Exception as e:
        print(f'[warn] Could not save training curves: {e}')


if __name__ == '__main__':
    main()
