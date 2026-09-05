# Ablation results (mean ± std) — Phase 2 / Lasana-Person4

Std is **sample** standard deviation (ddof=1). Single-seed rows
(U-Net, SegFormer-B0, SegFormer-B0+L_att λ2=1.0, SegFormer-B0+L_att+L_boundary)
show the bare value pending additional GPU-trained seeds from Person 1/2/3/5.
DeepLabV3+ extra baseline has genuine 3-seed mean±std (seeds 42/43/44, CPU
smoke: 400 samples / 5 epochs).

| Model | Seeds | Dice | IoU | F1 | AAMO |
|-------|-------|------|-----|----|------|
| U-Net (CNN baseline) | 1 | 0.8615 | 0.7568 | 0.8615 | n/a |
| SegFormer-B0 (no attention loss) | 1 | 0.8743 | 0.7766 | 0.8743 | 0.0334 |
| SegFormer-B0 + Attention Consistency Loss (λ2=1.0 MSE) | 1 | 0.8577 | 0.7508 | 0.8577 | 0.7476 |
| SegFormer-B0 + Attention Consistency + Boundary Loss | 1 | 0.8669 | 0.7650 | 0.8669 | 0.6218 |
| DeepLabV3+ (MobileNetV3) — extra baseline | 3 | 0.7862 ± 0.0193 | 0.6481 ± 0.0264 | 0.7862 ± 0.0193 | n/a |

## Notes

- Attention-consistency config = Dinura `l2_1_mse` (λ2=1.0, MSE).
- Boundary Loss row resolved 2026-09-05 — Person 5's λ3=0.2 sweep winner
  (single-seed, batch-size-1; see `Phase2/Dhinanjaya-Person5/results/`).
- DeepLabV3+ smoke numbers are not paper-scale; they demonstrate
  the multi-seed aggregation pipeline Person 4 owns.
- U-Net / SegFormer / Boundary full-scale checkpoints live on Drive or are
  gitignored locally (not in git); re-running extra seeds requires Colab
  GPU access from teammates.
- Teammate GPU multi-seed reporting must also use sample std (ddof=1).
