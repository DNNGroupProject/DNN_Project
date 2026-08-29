# λ2-sweep results — in progress

| Cell | Status |
|---|---|
| `l2_0.3_mse` | **Complete** (seeded from Kalana full-scale: test Dice 0.8690, AAMO 0.5752) |
| `l2_0.1_mse` | Needs Colab GPU (`lambda_sweep_colab.ipynb` Step 3) |
| `l2_0.5_mse` | Needs Colab GPU |
| `l2_1_mse` | Needs Colab GPU |
| KL at winning λ2 | Optional, after MSE cells finish |

Current interim winner (only complete cell): **`l2_0.3_mse`**. Re-run
`python aggregate_sweep.py` after any new cell lands.

Checkpoint for λ2=0.3 still on Kalana's Drive:
`MyDrive/segformer_full_scale_outputs/checkpoints/segformer_b0_att_best.pt`
