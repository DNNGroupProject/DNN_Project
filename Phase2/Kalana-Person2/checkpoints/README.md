# Checkpoints

After the Colab GPU run this folder holds:

- `segformer_b0_vanilla_{best,last}.pt`
- `segformer_b0_att_{best,last}.pt`

Do **not** commit the `.pt` files. Full-scale SegFormer-B0 weights will pass
GitHub's 100 MiB limit (Chanupa already hit this on `unet_baseline_best.pt`).
Keep them on Drive next to this folder; commit the `results/` JSON / markdown
/ figures instead.

Dict keys match Person 2's Phase 1 handoff: `model_state`, `epoch`,
`val_dice`, `variant`.
