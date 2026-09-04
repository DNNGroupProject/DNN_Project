# Person 4 — Evaluation Lead (Phase 2)

Role: **Evaluation Lead**. Proposal §6.2.2, Weeks 7–10: implement AAMO
(done in Phase 1), run the complete ablation across all model configurations,
and report results with multiple seeds (mean ± std).

| Weeks | Deliverable | Status |
|---|---|---|
| 7–9 | Fold full-scale U-Net / SegFormer-B0 / SegFormer-B0+L_att into one table | Done — `results/baseline_comparison_full_scale.md` |
| 7–9 | Attention-consistency row uses Dinura's λ2 sweep winner (`l2_1_mse`) | Done |
| 7–9 | Multi-seed mean±std reporting | Done for DeepLabV3+ (seeds 42/43/44); U-Net/SegFormer remain Seeds=1 pending GPU reruns |
| 10 | Boundary Loss ablation row | Pending — blocked on Person 5 integration |

## Layout

```
Phase2/Lasana-Person4/
  README.md
  fold_full_scale_results.py   consolidates Table-1 rows (no model loading)
  train_deeplab_multiseed.py   DeepLabV3+ seeds 42/43/44 + ablation tables
  checkpoints/                 Deeplab seed 43/44 weights (gitignored *.pt)
  results/
    baseline_comparison_full_scale.csv / .md
    ablation_per_seed.csv
    ablation_mean_std.csv / .md
    deeplab_multiseed.json
  tests/
    test_fold_results.py
```

Phase 1 code under `Phase1/Lasana-Person4_Evaluation/` is a **frozen short-paper
snapshot** (CONTRIBUTING.md). This Phase 2 folder imports its metrics /
adapters via `sys.path` and does not edit those files. Multi-seed aggregation
uses a local `aggregate_mean_std` (sample std, ddof=1) rather than the Phase 1
helper.

## Quick start / Reproducing

```bash
cd Phase2/Lasana-Person4

# Unit tests (no GPU, no weights)
python tests/test_fold_results.py

# Fold Chanupa + Kalana + Dinura winner + DeepLab into one table
python fold_full_scale_results.py

# Train DeepLab seeds 43/44 (seed 42 reuses Phase 1 ckpt), evaluate all three,
# write ablation_per_seed + ablation_mean_std
python train_deeplab_multiseed.py

# Or aggregate only, if seed 43/44 checkpoints already exist
python train_deeplab_multiseed.py --skip-train
```

Smoke defaults (override with env vars): `DEEPLAB_MAX_SAMPLES=400`,
`DEEPLAB_EPOCHS=5`, `DEEPLAB_BATCH=2`.

## Dependencies on teammates

| Direction | What |
|---|---|
| I need | Kalana's full-scale SegFormer vanilla metrics (`Phase2/Kalana-Person2/results/`) |
| I need | Dinura's λ2 sweep winner `l2_1_mse` (`Phase2/Dinura-Person3/results/`) |
| I need | Chanupa's U-Net full-scale row (already in Phase 1 Person 4 results) |
| I hand off | `results/baseline_comparison_full_scale.md` + `results/ablation_mean_std.md` for Person 5's Week 11–12 paper assembly |

## Results

U-Net / SegFormer / L_att rows share the 3576/766/766 seed-42 test set; DeepLabV3+
is a 400-sample CPU-smoke subset (seed-42 Dice 0.7821 from `deeplab_multiseed.json`).

| Model | Dice | IoU | AAMO | Seeds |
|---|---|---|---|---|
| U-Net (CNN baseline) | 0.8615 | 0.7568 | n/a | 1 |
| SegFormer-B0 (no attention loss) | 0.8743 | 0.7766 | 0.0334 | 1 |
| SegFormer-B0 + Attention Consistency (λ2=1.0 MSE) | 0.8577 | 0.7508 | 0.7476 | 1 |
| SegFormer-B0 + Attention + Boundary Loss | — | — | pending | 0 |
| DeepLabV3+ (MobileNetV3) — extra baseline | 0.7862 ± 0.0193 | 0.6481 ± 0.0264 | n/a | 3 |

Selection rule for the attention row (Dinura): max test AAMO, then max Dice.
Winner run tag `l2_1_mse` supersedes Kalana's default-λ2=0.3 attention numbers.

DeepLabV3+ is CPU smoke-scale (400 samples / 5 epochs) and exists to exercise
the multi-seed aggregation pipeline. U-Net / SegFormer full-scale weights live
on Drive (GitHub 100 MiB limit); extra seeds for those rows need Colab GPU
reruns from Person 1/2/3.

## Cross-folder edits

**None.** No file outside this folder was modified. Phase 1 Person 4 results
are left untouched on purpose.
