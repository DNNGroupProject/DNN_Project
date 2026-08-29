# λ2-sweep results — complete

All four MSE cells trained and evaluated on the held-out test set (n=766).
See `sweep_comparison.md` and `winning_config.json` for the full table.

| Cell | Status |
|---|---|
| `l2_0.3_mse` | **Complete** (seeded from Kalana full-scale: test Dice 0.8690, AAMO 0.5752) |
| `l2_0.1_mse` | **Complete** (test Dice 0.8523, AAMO 0.5481) |
| `l2_0.5_mse` | **Complete** (test Dice 0.8644, AAMO 0.6028) |
| `l2_1_mse` | **Complete** (test Dice 0.8577, AAMO 0.7476) |
| KL at winning λ2 | Optional, not run |

**Winner: `l2_1_mse`** (λ2=1.0, mse) — selection rule: max test AAMO, then max test Dice.

Checkpoint for λ2=0.3 still on Kalana's Drive:
`MyDrive/segformer_full_scale_outputs/checkpoints/segformer_b0_att_best.pt`
