# Ablation results (mean ± std)

| Model | Seeds | Dice | IoU | F1 | AAMO |
|-------|-------|------|-----|----|------|
| U-Net (CNN baseline) | 1 | 0.8492 ± 0.0000 | 0.7379 ± 0.0000 | 0.8492 ± 0.0000 | n/a |
| SegFormer-B0 (no attention loss) | 0 | - | - | - | pending |
| SegFormer-B0 + Attention Consistency Loss | 0 | - | - | - | pending |
| SegFormer-B0 + Attention Consistency + Boundary Loss | 0 | - | - | - | pending |

SegFormer rows appear once Person 2/3 checkpoints are wired.