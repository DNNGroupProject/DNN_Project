# Person 4 — Evaluation Lead

Shared evaluation package for the research proposal:

> **Explainability-Guided SegFormer for Forest Cover Segmentation**  
> Using Attention Consistency Supervision

You own metrics, efficiency numbers, AAMO, baseline/ablation tables — not model training.

---

## Role (from the proposal)

| Phase | Weeks | Deliverable |
|-------|-------|-------------|
| Phase 1 | 1 | Shared eval: Dice, IoU, F1, params, FLOPs |
| Phase 1 | 2–3 | Evaluate U-Net + vanilla SegFormer-B0; comparison table |
| Phase 1 | 4 | Finalize short-paper tables/figures; sanity-check |
| Phase 2 | 7–9 | **AAMO** metric; full ablation; multi-seed mean±std |
| Phase 2 | 10 | Support boundary-refinement evaluation (stretch) |

---

## Layout

```text
Person4_Evaluation/
  README.md
  config.py
  metrics.py
  efficiency.py
  aamo.py
  evaluate.py
  ablation_runner.py
  adapters/
    __init__.py
    base.py
    unet_keras.py
    segformer_stub.py
  docs/
    PERSON4_GUIDE.md
  results/
    baseline_comparison.csv
    baseline_comparison.md
```

---

## Quick start

```bash
cd Person4_Evaluation

# Week 1–3: evaluate existing Lasana U-Net
python evaluate.py --model unet

# Optional: limit samples for a fast CPU smoke run
python evaluate.py --model unet --max-samples 200

# AAMO unit check (no SegFormer needed)
python aamo.py

# Ablation scaffold (fills available rows; stubs wait on Person 2/3)
python ablation_runner.py
```

SegFormer evaluation needs Person 2’s checkpoint + attention hooks. Until then:

```bash
python evaluate.py --model segformer
# → clear "waiting on Person 2" error
```

AAMO can be tested on precomputed attention maps:

```bash
python evaluate.py --model unet --attention-npy path/to/attention.npy
```

---

## Dependencies on teammates

| Person | You need from them |
|--------|--------------------|
| 1 | Clean split / U-Net baseline (Lasana checkpoint already usable) |
| 2 | SegFormer-B0 checkpoint + attention-map API |
| 3 | Explainability-guided SegFormer checkpoint |
| 5 | Your tables go into the IEEE paper |

---

## Outputs

- `results/baseline_comparison.csv` / `.md` — proposal Table 2 style  
- `results/prediction_grid.png` — image | GT | prediction  
- `results/ablation_mean_std.csv` — multi-seed scaffold  

See [docs/PERSON4_GUIDE.md](docs/PERSON4_GUIDE.md) for beginner explanations and AAMO math.
