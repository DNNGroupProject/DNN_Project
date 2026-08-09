# Ablation results (mean ± std)

| Model | Seeds | Dice | IoU | F1 | AAMO |
|-------|-------|------|-----|----|------|
| U-Net (CNN baseline) | 1 | 0.8492 ± 0.0000 | 0.7379 ± 0.0000 | 0.8492 ± 0.0000 | n/a |
| SegFormer-B0 (no attention loss) | 1 | 0.8017 ± 0.0000 | 0.6690 ± 0.0000 | 0.8017 ± 0.0000 | 0.0144 ± 0.0000 |
| SegFormer-B0 + Attention Consistency Loss | 1 | 0.8191 ± 0.0000 | 0.6936 ± 0.0000 | 0.8191 ± 0.0000 | 0.4320 ± 0.0000 |
| SegFormer-B0 + Attention Consistency + Boundary Loss | 0 | - | - | - | pending |
| DeepLabV3+ (MobileNetV3) — extra baseline | 1 | 0.7369 ± 0.0000 | 0.5834 ± 0.0000 | 0.7369 ± 0.0000 | n/a |

Notes:
- DeepLabV3+ is the optional extra baseline (proposal “if time allows”).
- DeepLab smoke numbers: `max_samples=200`; retrain with `train_deeplab_extra.py` for paper-scale.
- Multi-seed mean±std requires separately trained checkpoints per seed.
