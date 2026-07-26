# ViT_XAI_Postprocess — Team 4 refinement for the ViT + Concept model

Located at **`Lasana/ViT_XAI_Postprocess/`** (next to `Lasana/postprocess/`).  
Same post-processing research role, applied to the **trained ViT + guided-concept** segmenter.

| Item | Detail |
|------|--------|
| **Base model** | `../ViT_XAI_Segmentation/checkpoints/vit_concept_seg_best.pt` (project root) |
| **Base test (raw)** | IoU ≈ 0.761, Dice ≈ 0.864, Acc ≈ 82.5% (full-data train) |
| **Your role** | Post-processing & refinement after the probability map |
| **Sister folder** | `Lasana/postprocess/` = same methods on CNN U-Net |

---

## Why under Lasana?

Keeps all **Team-4 post-processing** work together:

| Folder | Backbone | Purpose |
|--------|----------|---------|
| `Lasana/postprocess/` | CNN U-Net | Team-4 methods on Lasana |
| **`Lasana/ViT_XAI_Postprocess/`** | **ViT + concepts** | Same methods + concept-guided refine |

ViT **training** stays in project-root `ViT_XAI_Segmentation/`; only refinement lives here.

---

## Pipeline

```text
RGB tile
  → ViT + Concept Seg (frozen checkpoint)
  → probability map P (+ concept maps)
  → ★ THIS FOLDER ★
       A threshold/morph
       B TTA
       C uncertainty / ECE
       D LBR-Net
       E bilateral
       F concept-guided refine   ← ViT-specific
  → final mask + metrics (ΔIoU%, ECE, ms)
```

---

## Parts (what / why)

| Part | Method | Why |
|------|--------|-----|
| **A** | Threshold sweep + morphology | Cheap IoU/Acc win; t=0.5 is arbitrary |
| **B** | TTA-4 (flip average) | Free ensemble without new weights |
| **C** | Entropy, ECE, temperature, adaptive thr | Reliability / confidence decisions |
| **D** | LBR-Net (tiny residual refiner) | Novel learnable boundary cleanup |
| **E** | Bilateral refine | Edge-aware CRF-style substitute |
| **F** | Concept-guided refine | Use paper concept maps (dense↑ clearing↓) |

Loss for LBR: boundary-weighted BCE + Dice (same idea as Lasana LBR).

---

## How to run

```bash
cd Lasana/ViT_XAI_Postprocess

# Default: 1500 samples (CPU-friendly). Full data:
#   set VIT_PP_MAX_SAMPLES=0
python run_research.py
```

Optional env vars:

| Env | Default | Meaning |
|-----|---------|---------|
| `VIT_PP_MAX_SAMPLES` | `1500` | `0` = all pairs |
| `VIT_PP_BATCH` | `4` | batch size |
| `VIT_PP_LBR_EPOCHS` | `10` | LBR training epochs |
| `VIT_PP_LBR_TRAIN_CAP` | `800` | max train tiles for LBR |

### Outputs

```text
Lasana/ViT_XAI_Postprocess/
  run_research.py
  README.md
  checkpoints/vit_lbr_net.pt
  results/
    comparison_table.csv
    RESULTS.md
    threshold_sweep.json
    test_entropy_maps.npy
```

---

## Metrics tracked

- IoU, Dice, Pixel accuracy  
- **ΔIoU %** vs thr=0.5 baseline  
- **ECE** (calibration)  
- **ms/image** overhead  

Compare side-by-side with `Lasana/postprocess/results/RESULTS.md`.

---

## Relation to learning order

Same topics as `Lasana/LEARNING_ORDER.md` (stages 4–11).  
Extra here: **concept-guided refinement** using the XAI concept layer from arXiv:2101.03919.
