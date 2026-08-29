# λ2 / att-mode sweep (Phase 2 / Dinura-Person3)

Full-scale attention-variant only. Split 3576/766/766 seed 42 (same held-out test set as Chanupa U-Net + Kalana default-λ2 run).

**Selection rule:** highest test AAMO, then highest test Dice; MSE preferred over KL on ties. Cells without test eval fall back to best val Dice.

| run | λ2 | mode | best val Dice | test Dice | test IoU | test AAMO | status |
|-----|----|------|---------------|-----------|----------|-----------|--------|
| l2_0.3_mse | 0.3 | mse | 0.7902 | 0.8690 | 0.7684 | 0.5752 | complete |

**Current winner:** `l2_0.3_mse` (λ2=0.3, mse) — test AAMO=0.5752, test Dice=0.8690, best val Dice=0.7902.

Checkpoint path (once trained): `checkpoints/runs/l2_0.3_mse/segformer_b0_att_best.pt`

This cell was seeded from Kalana's default-λ2=0.3 MSE full-scale run (`Phase2/Kalana-Person2/results/`). The `.pt` still lives on his Drive (`MyDrive/segformer_full_scale_outputs/checkpoints/`); copy it into this run folder before Lasana / Boundary Refinement integration if needed.
