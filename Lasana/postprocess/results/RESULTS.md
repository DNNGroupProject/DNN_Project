# Post-Processing Research Results (executed)

**Base:** `Lasana/checkpoints/lasana_unet_best.keras`  
**Data:** 1200 pairs → train 959 / val 121 / test 120 (seed 42)  
**Hardware:** CPU (no GPU)  
**Study guide:** `Lasana/POST_PROCESSING_RESEARCH.md`

Best validation threshold **t\* = 0.65**

## Comparison table

| Method | IoU | Dice | Acc | ECE | ΔIoU% | ms/img | Verdict |
|--------|-----|------|-----|-----|-------|--------|---------|
| 0_baseline_thr0.5 | **0.7379** | 0.8492 | 0.8111 | **0.027** | 0.00 | 116.6 | baseline |
| 1_thr_star | 0.7383 | 0.8494 | **0.8180** | 0.027 | **+0.05** | 116.6 | small win |
| 2_thr_star_morph | **0.7383** | **0.8495** | 0.8182 | 0.182* | **+0.07** | ~117 | best classical |
| 3_tta4 | 0.7382 | 0.8494 | 0.8185 | 0.028 | +0.05 | ~466 | not worth 4× cost |
| 4_tta4_morph | 0.7382 | 0.8494 | 0.8186 | 0.182* | +0.05 | ~466 | same |
| 5_confidence_adaptive_thr | 0.7329 | 0.8459 | 0.8183 | 0.182* | **−0.67** | 116.6 | hurts — redesign |
| 6_temperature_scaled | 0.7383 | 0.8494 | 0.8180 | 0.027 | +0.05 | 116.6 | calibration OK |
| 7_bilateral_refine | 0.7383 | 0.8494 | 0.8183 | 0.028 | +0.06 | ~117 | tiny gain |
| 8_lbr_net (8 epochs) | 0.7270 | 0.8419 | 0.8172 | 0.041 | **−1.48** | ~117 | needs better train |
| 9_tta4_lbr_net | 0.7279 | 0.8425 | 0.8180 | 0.038 | −1.36 | ~466 | same |
| 10_lbr_morph | 0.7268 | 0.8418 | 0.8172 | 0.183* | −1.50 | ~117 | same |

\*ECE on **hard** 0/1 masks looks worse by definition — trust ECE on soft probabilities.

## Parts completed

| Part | Method | Status |
|------|--------|--------|
| **A** | Threshold sweep + morphology | Done |
| **B** | TTA-4 | Done |
| **C** | Entropy uncertainty + adaptive thr + temperature / ECE | Done |
| **D** | LBR-Net trained → `checkpoints/lbr_net.keras` | Done (needs improvement) |
| **E** | Bilateral refine (DenseCRF substitute) | Done |

`pydensecrf` failed to build on Windows/Python 3.12 — bilateral filter used as edge-aware stand-in.

## Research conclusions (this run)

1. **Best cheap win:** `thr*=0.65` + light morphology → **ΔIoU% ≈ +0.07%**, accuracy **81.1% → 81.8%**.
2. **TTA** did not beat thr\* enough to justify **~4×** latency on this subset.
3. **Naive adaptive threshold** on high entropy **hurt** IoU — confidence policies need careful design (abstention / human review, not just harsher thr).
4. **Probabilities are fairly calibrated** (ECE ≈ 0.027).
5. **LBR-Net v1 (8 epochs)** reduced IoU — negative result is still research: next iterations should use longer training, boundary-only loss, and validation early-stopping on IoU (not just loss).

## Artifacts

```text
Lasana/postprocess/
  run_research.py          # Parts A–E (+ D)
  run_lbr_only.py          # retrain LBR
  results/
    comparison_table.csv
    RESULTS.md             # this file
    threshold_sweep.json
    test_entropy_maps.npy  # if Part C saved it
    run_console.log
    lbr_console.log
Lasana/checkpoints/
  lasana_unet_best.keras
  lbr_net.keras
```

## Next experiments (for you / next run)

1. Retarget LBR: monitor **val IoU**, train 30+ epochs, stronger boundary weighting.  
2. Install DenseCRF in WSL/Linux or use a pure-Python mean-field demo.  
3. True bagging: train 3 U-Nets with seeds 0/1/2 (needs GPU).  
4. MC Dropout: add Dropout to U-Net and finetune.  
5. Confidence = **abstain** high-entropy pixels and report coverage vs IoU on the rest.
