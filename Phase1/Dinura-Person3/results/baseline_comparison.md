# Baseline comparison (Person 4)

| Model | Dice | IoU | F1 | AAMO | Params | GFLOPs | FPS |
|-------|------|-----|----|------|--------|--------|-----|
| SegFormer-B0 (no attention loss) | 0.8017 | 0.669 | 0.8017 | 0.0144 | 3714658 | 1.692 | 14.72 |
| SegFormer-B0 + Attention Consistency Loss | 0.8191 | 0.6936 | 0.8191 | 0.432 | 3714658 | 1.692 | 13.55 |

Proposal Table 2 columns + efficiency. AAMO = n/a until attention maps exist.