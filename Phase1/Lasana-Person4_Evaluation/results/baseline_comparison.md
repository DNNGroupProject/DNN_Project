# Baseline comparison (Person 4)

| Model | Dice | IoU | F1 | AAMO | Params | GFLOPs | FPS |
|-------|------|-----|----|------|--------|--------|-----|
| U-Net (CNN baseline) | 0.8615 | 0.7568 | 0.8615 | n/a | 31037698 | 109.48 | 1.47 |
| SegFormer-B0 (no attention loss) | 0.8017 | 0.669 | 0.8017 | 0.0144 | 3714658 | 1.692 | 12.45 |
| SegFormer-B0 + Attention Consistency Loss | 0.8191 | 0.6936 | 0.8191 | 0.432 | 3714658 | 1.692 | 14.55 |
| SegFormer-B0 + Attention Consistency + Boundary Loss | - | - | - | pending | - | - | - |
| DeepLabV3+ (MobileNetV3) — extra baseline | 0.7369 | 0.5834 | 0.7369 | n/a | 11020594 | n/a | 8.51 |

## U-Net row (2026-08-15, Person 4 adapter)

Official baseline is **Chanupa's PyTorch U-Net**, not the old Keras 1.95M checkpoint.

- Checkpoint: `Phase1/Chanupa-Person1/checkpoints/unet_baseline_best.pt`
- Adapter: `adapters/unet_torch.py` (`python evaluate.py --model unet`)
- Split: Chanupa `dataset.make_splits` — 3576 / 766 / 766, seed 42
- Inputs: `/255` then **ImageNet mean/std** (same as `dataset.py`; do not skip)
- Params **31,037,698**, GFLOPs **109.48** (matches `test_metrics.txt`)
- FPS **1.47** (~681 ms/img, CPU); re-measure on GPU for the paper if needed
- AAMO n/a (no attention)

`evaluate.py` reports **dataset-wide** Dice/IoU (accumulate TP/FP over all 766 test images): **0.8615 / 0.7568**.

Chanupa's `test_metrics.txt` is **Dice 0.8563 / IoU 0.7534** using the notebook's **mean-of-batch** `dice_iou_score` (batch 8, argmax). Same checkpoint and split; different reduction. Paper row can keep 0.8563/0.7534 if it should match the training notebook exactly.

**Ping Dhinanjaya:** paper draft still cites 1.95M / 0.8492 Keras numbers — swap to this row.

DeepLabV3+ remains a CPU smoke extra baseline. Boundary-loss SegFormer still pending.
