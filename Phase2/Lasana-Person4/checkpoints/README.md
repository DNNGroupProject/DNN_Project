# Checkpoints

DeepLabV3+ multi-seed smoke weights land here as:

- `deeplabv3_mobilenet_seed43_best.pt`
- `deeplabv3_mobilenet_seed44_best.pt`

Seed 42 reuses the Phase 1 checkpoint at
`Phase1/Lasana-Person4_Evaluation/checkpoints/deeplabv3_mobilenet_best.pt`
(not re-copied here).

Do **not** commit `.pt` files. Commit metrics under `results/` instead.
