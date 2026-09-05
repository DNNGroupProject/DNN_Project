# Boundary Loss — Table-1 handoff row (Phase 2 / Dhinanjaya-Person5)

Single-row CSV consumed by `Phase2/Lasana-Person4/fold_full_scale_results.py`'s
Boundary Loss TODO (`Phase2/Dhinanjaya-Person5/results/baseline_comparison.csv`).
Sourced from the λ3-sweep winner — see `boundary_sweep_comparison.md` for the
full 3-cell sweep and selection rule.

| Model | Dice | IoU | F1 | AAMO | Params | GFLOPs | FPS | Config |
|-------|------|-----|----|------|--------|--------|-----|--------|
| SegFormer-B0 + Attention Consistency + Boundary Loss | 0.8669 | 0.7650 | 0.8669 | 0.6218 | 3,714,658 | 1.692 | 113.23 | λ2=1.0 (MSE), λ3=0.2, boundary_kernel=3 |

Same 3576/766/766 seed-42 held-out test set as every other full-scale row.
Checkpoint: `Phase2/Dinura-Person3/checkpoints/runs/l2_1_mse_bnd0.2/segformer_b0_att_best.pt`.

**Note on AAMO**: 0.6218 vs. the λ2=1.0 attention-only winner's 0.7476 (λ3=0) —
a real drop, not noise (λ3=0.1's AAMO is 0.7775, λ3=0.5's is 0.6805; not
monotonic). L_boundary is pulling some attention mass away from L_att's target
at λ3=0.2 while still improving Dice/IoU. Worth a sentence in the paper's
discussion/limitations if this row is used as the reported Boundary Loss
result — flagging for Dhinanjaya's own write-up, not blocking Lasana's fold-in.
