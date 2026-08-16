# Full-scale results — landed 2026-08-16

GPU Colab run, 5,108-pair dataset, split 3576/766/766 seed 42, 20 epochs.
Test = Chanupa U-Net's held-out 766 images. Checkpoints on Drive, not in git.

| Model | Dice | IoU | F1 | AAMO | best epoch |
|---|---|---|---|---|---|
| SegFormer-B0 (no attention loss) | 0.8743 | 0.7766 | 0.8743 | 0.0334 | 5 |
| SegFormer-B0 + Attention Consistency Loss | 0.8690 | 0.7684 | 0.8690 | 0.5752 | 3 |

Ping Dhinanjaya — these replace the smoke-scale rows (400 images / 8 epochs)
in the paper abstract, Table 1, Figure 2, and the scale caveats.
