# Full-scale baseline comparison (Phase 2 / Lasana-Person4)

Folded from Chanupa (U-Net), Kalana (SegFormer-B0 vanilla), Dinura
(`l2_1_mse` attention winner), and Person 4's DeepLabV3+ extra baseline.
U-Net / SegFormer / L_att rows share the 3576/766/766 seed-42 test set;
the DeepLabV3+ row is a 400-sample CPU-smoke subset evaluated by
`train_deeplab_multiseed.py`.

| Model | Dice | IoU | F1 | AAMO | Params | GFLOPs | FPS | Source |
|-------|------|-----|----|------|--------|--------|-----|--------|
| U-Net (CNN baseline) | 0.8615 | 0.7568 | 0.8615 | n/a | 31037698 | 109.48 | 1.47 | Phase1/Lasana-Person4_Evaluation (Chanupa PyTorch ckpt) |
| SegFormer-B0 (no attention loss) | 0.8743 | 0.7766 | 0.8743 | 0.0334 | 3714658 | 1.692 | 84.52 | Phase2/Kalana-Person2 (full-scale Colab) |
| SegFormer-B0 + Attention Consistency Loss (λ2=1.0 MSE) | 0.8577 | 0.7508 | 0.8577 | 0.7476 | 3714658 | 1.692 | 103.36 | Phase2/Dinura-Person3/results/runs/l2_1_mse |
| SegFormer-B0 + Attention Consistency + Boundary Loss | 0.8669 | 0.765 | 0.8669 | 0.6218 | 3714658 | 1.692 | 113.23 | Phase2/Dhinanjaya-Person5/results/baseline_comparison.csv (run l2_1_mse_bnd0.2) |
| DeepLabV3+ (MobileNetV3) — extra baseline | 0.7821 | 0.6422 | 0.7821 | n/a | 11020594 | n/a |  | Phase2/Lasana-Person4/train_deeplab_multiseed.py (seed 42, 400-sample smoke) |

## Notes

- Attention-consistency row uses Dinura's sweep winner (`λ2=1.0`, MSE, run tag `l2_1_mse`: Dice 0.8577 / IoU 0.7508 / AAMO 0.7476), not Kalana's default-λ2=0.3 attention run.
- Boundary Loss row is the λ3-sweep winner (λ2=1.0 MSE, λ3=0.2) from Phase2/Dhinanjaya-Person5/results/; note its AAMO (0.6218) drops vs. the plain attention row's 0.7476.
- DeepLabV3+ Dice/IoU come from `deeplab_multiseed.json` seed 42 (same path as the multi-seed ablation); see `ablation_mean_std.md` for seeds 42/43/44 mean±std.
- Phase 1 `Lasana-Person4_Evaluation/results/` is left untouched (frozen short-paper snapshot per CONTRIBUTING.md).
