# Baseline comparison (Person 4)

| Model | Dice | IoU | F1 | AAMO | Params | GFLOPs | FPS |
|-------|------|-----|----|------|--------|--------|-----|
| U-Net (CNN baseline) | 0.8492 | 0.7379 | 0.8492 | n/a | 1951105 | n/a | 3.48 |
| SegFormer-B0 (no attention loss) | 0.8017 | 0.669 | 0.8017 | 0.0144 | 3714658 | 1.692 | 12.45 |
| SegFormer-B0 + Attention Consistency Loss | 0.8191 | 0.6936 | 0.8191 | 0.432 | 3714658 | 1.692 | 14.55 |
| SegFormer-B0 + Attention Consistency + Boundary Loss | - | - | - | pending | - | - | - |
| DeepLabV3+ (MobileNetV3) — extra baseline | 0.7369 | 0.5834 | 0.7369 | n/a | 11020594 | n/a | 8.51 |

Notes:
- DeepLabV3+ is the **extra baseline** (proposal “if time allows”). Smoke-trained locally (`max_samples=200`, 1–2 epochs); re-run `train_deeplab_extra.py` on GPU for paper-scale numbers.
- AAMO is n/a for U-Net / DeepLab (no attention). SegFormer AAMO uses Person 2 Grad-Rollout.
- Fill boundary-loss row once that stretch checkpoint exists.

Proposal Table 2 columns + efficiency. AAMO = n/a until attention maps exist.