# Full-scale baseline comparison (Phase 2 / Lasana-Person4)

Folded from Chanupa (U-Net), Kalana (SegFormer-B0 vanilla), Dinura
(`l2_1_mse` attention winner), and Person 4's DeepLabV3+ extra baseline.
All full-scale rows share the 3576/766/766 seed-42 held-out test set.

| Model | Dice | IoU | F1 | AAMO | Params | GFLOPs | FPS | Source |
|-------|------|-----|----|------|--------|--------|-----|--------|
| U-Net (CNN baseline) | 0.8615 | 0.7568 | 0.8615 | n/a | 31037698 | 109.48 | 1.47 | Phase1/Lasana-Person4_Evaluation (Chanupa PyTorch ckpt) |
| SegFormer-B0 (no attention loss) | 0.8743 | 0.7766 | 0.8743 | 0.0334 | 3714658 | 1.692 | 84.52 | Phase2/Kalana-Person2 (full-scale Colab) |
| SegFormer-B0 + Attention Consistency Loss (λ2=1.0 MSE) | 0.8577 | 0.7508 | 0.8577 | 0.7476 | 3714658 | 1.692 | 103.36 | Phase2/Dinura-Person3/results/runs/l2_1_mse |
| SegFormer-B0 + Attention Consistency + Boundary Loss | - | - | - | pending | - | - | - | pending |
| DeepLabV3+ (MobileNetV3) — extra baseline | 0.7369 | 0.5834 | 0.7369 | n/a | 11020594 | n/a | 8.51 | Phase1/Lasana-Person4_Evaluation (CPU smoke) |

## Notes

- Attention-consistency row uses Dinura's sweep winner (`λ2=1.0`, MSE, run tag `l2_1_mse`: Dice 0.8577 / IoU 0.7508 / AAMO 0.7476), not Kalana's default-λ2=0.3 attention run.
- Boundary Loss row stays pending until Person 5 finishes integration.
- DeepLabV3+ is CPU smoke-scale; see `ablation_mean_std.md` for the multi-seed (42/43/44) mean±std of that extra baseline.
- Phase 1 `Lasana-Person4_Evaluation/results/` is left untouched (frozen short-paper snapshot per CONTRIBUTING.md).
