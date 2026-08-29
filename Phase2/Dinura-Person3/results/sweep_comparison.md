# λ2 / att-mode sweep (Phase 2 / Dinura-Person3)

Full-scale attention-variant only. Split 3576/766/766 seed 42 (same held-out test set as Chanupa U-Net + Kalana default-λ2 run).

**Selection rule:** highest test AAMO, then highest test Dice; MSE preferred over KL on ties. Cells without test eval fall back to best val Dice.

| run | λ2 | mode | best val Dice | test Dice | test IoU | test AAMO | status |
|-----|----|------|---------------|-----------|----------|-----------|--------|
| l2_0.1_mse | 0.1 | mse | — | 0.8523 | 0.7425 | 0.5481 | complete |
| l2_0.3_mse | 0.3 | mse | 0.7902 | 0.8690 | 0.7684 | 0.5752 | complete |
| l2_0.5_mse | 0.5 | mse | — | 0.8644 | 0.7612 | 0.6028 | complete |
| l2_1_mse | 1.0 | mse | 0.7941 | 0.8577 | 0.7508 | 0.7476 | complete |

**Current winner:** `l2_1_mse` (λ2=1.0, mse) — test AAMO=0.7476, test Dice=0.8577, best val Dice=0.7941.

Checkpoint path (once trained): `checkpoints/runs/l2_1_mse/segformer_b0_att_best.pt`
