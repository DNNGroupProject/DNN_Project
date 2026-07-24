# Baseline comparison (Person 4)

| Model | Dice | IoU | F1 | AAMO | Params | GFLOPs | FPS |
|-------|------|-----|----|------|--------|--------|-----|
| U-Net (CNN baseline) | 0.8492 | 0.7379 | 0.8492 | n/a | 1951105 | n/a | 3.48 |
| SegFormer-B0 (no attention loss) | - | - | - | pending | - | - | - |
| SegFormer-B0 + Attention Consistency Loss | - | - | - | pending | - | - | - |
| SegFormer-B0 + Attention Consistency + Boundary Loss | - | - | - | pending | - | - | - |

Notes:
- U-Net row: Lasana checkpoint `lasana_unet_best.keras`, 1200-sample split (seed 42), thr=0.5, test split.
- AAMO is n/a for U-Net (no attention). SegFormer AAMO needs Person 2 Grad-Rollout maps.
- GFLOPs = n/a until `keras-flops` / `thop` is installed; params and FPS are measured.
- Fill SegFormer rows with: `python evaluate.py --model segformer` after Person 2 delivers the checkpoint.
