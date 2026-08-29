# Checkpoints

Per-cell weights land under `runs/l2_<λ>_<mode>/segformer_b0_att_{best,last}.pt`.

Do **not** commit `.pt` files (GitHub 100 MiB limit). Keep them on Drive
(`MyDrive/lambda_sweep_outputs/checkpoints/`). Commit metrics under
`results/` instead.

For the seeded λ2=0.3 / MSE cell, the weights still live on Kalana's Drive:
`MyDrive/segformer_full_scale_outputs/checkpoints/segformer_b0_att_best.pt`
— copy into `runs/l2_0.3_mse/` before Boundary Refinement / Lasana handoff
if needed.
