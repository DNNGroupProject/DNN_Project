# ViT + XAI — Post-Processing Research Results

Base model: `vit_concept_seg_best.pt` (ViT Concept Seg)  
Samples: 1500 | split train/val/test = 1200/150/150 | seed=42 | device=cpu

Best validation threshold **t\* = 0.70** | Temperature **T\* = 1.5**

| Method | IoU | Dice | Acc | ECE | ΔIoU% | ms/img |
|--------|-----|------|-----|-----|-------|--------|
| 0_baseline_thr0.5 | 0.7775 | 0.8748 | 0.8438 | 0.0592 | +0.00 | 19.4 |
| 1_thr_star | 0.7735 | 0.8723 | 0.8480 | 0.0592 | -0.51 | 19.4 |
| 2_thr_star_morph | 0.7736 | 0.8723 | 0.8480 | 0.1520 | -0.51 | 20.2 |
| 3_tta4 | 0.7786 | 0.8755 | 0.8515 | 0.0534 | +0.14 | 46.7 |
| 4_tta4_morph | 0.7786 | 0.8755 | 0.8515 | 0.1485 | +0.14 | 47.5 |
| 5_confidence_adaptive_thr | 0.7735 | 0.8723 | 0.8480 | 0.1520 | -0.51 | 19.4 |
| 6_temperature_scaled | 0.7596 | 0.8634 | 0.8415 | 0.0090 | -2.30 | 19.4 |
| 7_bilateral_refine | 0.7731 | 0.8720 | 0.8478 | 0.0581 | -0.56 | 20.4 |
| 8_concept_guided | 0.7741 | 0.8727 | 0.8481 | 0.0725 | -0.43 | 20.7 |
| 9_concept_guided_morph | 0.7741 | 0.8727 | 0.8481 | 0.1519 | -0.43 | 21.6 |
| 10_lbr_net | 0.7397 | 0.8504 | 0.8314 | 0.0196 | -4.85 | 44.9 |
| 11_tta4_lbr_net | 0.7437 | 0.8530 | 0.8343 | 0.0123 | -4.35 | 72.2 |
| 12_concept_lbr | 0.7459 | 0.8545 | 0.8347 | 0.0470 | -4.06 | 46.2 |

## Parts

- **A** Threshold + morphology
- **B** TTA-4
- **C** Entropy / ECE / temperature / adaptive thr
- **D** LBR-Net → `checkpoints/vit_lbr_net.pt`
- **E** Bilateral refine
- **F** Concept-guided refine (ViT-specific)

## vs Lasana post-process

Same Team-4 methods; this folder targets the **ViT+concept** checkpoint and adds concept-guided refinement using the paper concept maps.

## Findings (this run)

1. **Best method: TTA-4** → IoU **0.7786** (+0.14% vs thr=0.5), Acc **85.2%**.
2. ViT raw thr=0.5 is already strong (IoU 0.778); classical thr*/morph did **not** beat it on this test split (val t*=0.70 slightly overfit).
3. **Temperature scaling** improved ECE (0.059 → **0.009**) but hurt IoU — good for calibration story, not mask overlap.
4. Concept-guided refine alone did not beat TTA; LBR-Net v1 still needs better training (hurt IoU, same pattern as Lasana LBR).
5. Recommended report pipeline for now: **ViT baseline or TTA-4**; keep ECE/temperature as reliability analysis.
