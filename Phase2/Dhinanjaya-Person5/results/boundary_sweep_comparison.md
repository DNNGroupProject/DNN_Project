# Boundary Loss λ3 sweep (Person 5, Phase 2 Week 10)

Fixed at Dinura's λ2-sweep winner (λ2=1.0, MSE, `l2_1_mse`). Full-scale attention variant only. Split 3576/766/766 seed 42 (same held-out test set as every other full-scale row). Raw per-cell artifacts live under `Phase2/Dinura-Person3/{checkpoints,results}/runs/l2_1_mse_bnd<λ3>/` (train_full_scale.py's `--lambda3` output routing) — this table just summarizes them.

**Selection rule (differs from Dinura's λ2 sweep):** highest test Dice, then highest test IoU. L_boundary targets segmentation/boundary quality, not attention faithfulness — AAMO is recorded for reference only and should stay close to the λ2=1.0 winner's AAMO (0.7476); a large drop would flag L_boundary fighting L_att.

| run | λ3 | best val Dice | test Dice | test IoU | test AAMO | status |
|-----|----|--------------|-----------|----------|-----------|--------|
| l2_1_mse_bnd0.1 | 0.1 | 0.7933 | 0.8558 | 0.7479 | 0.7775 | complete |
| l2_1_mse_bnd0.2 | 0.2 | 0.7924 | 0.8669 | 0.7650 | 0.6218 | complete |
| l2_1_mse_bnd0.5 | 0.5 | 0.7944 | 0.8611 | 0.7560 | 0.6805 | complete |

**Current winner:** `l2_1_mse_bnd0.2` (λ3=0.2) — test Dice=0.8669, test IoU=0.7650, test AAMO=0.6218.

Checkpoint path (once trained): `Phase2/Dinura-Person3/checkpoints/runs/l2_1_mse_bnd0.2/segformer_b0_att_best.pt`
