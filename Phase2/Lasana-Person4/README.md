# Person 4 — Evaluation Lead (Phase 2)

Role: **Evaluation Lead**. Proposal §6.2.2, Weeks 7–10: implement AAMO
(done in Phase 1), run the complete ablation across all model configurations,
and report results with multiple seeds (mean ± std).

| Weeks | Deliverable | Status |
|---|---|---|
| 7–9 | Fold full-scale U-Net / SegFormer-B0 / SegFormer-B0+L_att into one table | Done — `results/baseline_comparison_full_scale.md` |
| 7–9 | Attention-consistency row uses Dinura's λ2 sweep winner (`l2_1_mse`) | Done |
| 7–9 | Multi-seed mean±std reporting | Done for DeepLabV3+ (seeds 42/43/44); SegFormer rows still Seeds=1 pending GPU jobs |
| 10 | Boundary Loss ablation row | Done — folded from Person 5 λ3=0.2 sweep winner |
| Phase 2/3 | **Job 5: vanilla SegFormer seed 44** (shared multi-seed Colab) | **Next — run in Colab, send Drive folder to Dhinanjaya** |

## Layout

```
Phase2/Lasana-Person4/
  README.md
  fold_full_scale_results.py   consolidates Table-1 rows (no model loading)
  train_deeplab_multiseed.py   DeepLabV3+ seeds 42/43/44 + ablation tables
  vanilla_seed44_colab.ipynb   Job 5 Colab notebook (pre-filled config)
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

## Job 5 — vanilla SegFormer-B0, seed 44 (Colab)

Assigned in `Phase2/Dhinanjaya-Person5/multiseed_kickoff.md`. Supplies the
extra seed for the vanilla SegFormer-B0 Table 1 row. Config is already filled
in `vanilla_seed44_colab.ipynb`:

| Field | Value |
|---|---|
| VARIANT | `vanilla` |
| SEED | `44` |
| LAMBDA2 / ATT_MODE / LAMBDA3 | `1.0` / `mse` / `0.0` (ignored for vanilla) |
| OUTPUT_TAG | `vanilla_seed44` |
| Drive output | `MyDrive/multiseed_outputs_vanilla_seed44/` |

### Colab steps

1. Build the shared zip locally (once; ~172 MB, gitignored — do **not** commit):

   ```bash
   python Phase2/Dhinanjaya-Person5/make_multiseed_colab_zip.py
   ```

   Output: `Phase2/Dhinanjaya-Person5/multiseed_train_colab.zip`

2. Upload that zip to Google Drive as `MyDrive/multiseed_train_colab.zip`
   (same zip everyone uses — do not rebuild a private copy).

3. Upload `Phase2/Lasana-Person4/vanilla_seed44_colab.ipynb` to Colab
   (or open it from Drive).

4. Runtime → Change runtime type → **GPU (T4 or better)** → Save.

5. Runtime → **Run all**. Training is ~2h on a T4.

6. When finished, share the whole
   `MyDrive/multiseed_outputs_vanilla_seed44/` folder back to Dhinanjaya
   (Drive link or zip). Do **not** merge results into the repo yourself.

### Send-back files (required)

| File | Needed? |
|------|---------|
| `checkpoints/segformer_b0_vanilla_best.pt` | **Yes** |
| `results/train_summary_vanilla.json` | **Yes** |
| `results/training_log_vanilla.csv` | **Yes** |
| `results/eval_vanilla.json` | **Yes** |
| `results/baseline_comparison.csv` | **Yes** |
| `checkpoints/segformer_b0_vanilla_last.pt` | Nice to have |
| `results/baseline_comparison.md` | Nice to have |
| `results/prediction_grid_vanilla.png` | Nice to have |

Sending the whole folder as one zip is simplest.

## Quick start / Reproducing (local eval tables)

```bash
cd Phase2/Lasana-Person4

# Unit tests (no GPU, no weights)
python tests/test_fold_results.py

# Fold Chanupa + Kalana + Dinura winner + Boundary + DeepLab into one table
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
| I need | Dhinanjaya's Boundary Loss handoff row (`Phase2/Dhinanjaya-Person5/results/`) |
| I hand off | Job 5 Drive folder `multiseed_outputs_vanilla_seed44/` to Dhinanjaya |
| I hand off | `results/baseline_comparison_full_scale.md` + `results/ablation_mean_std.md` for paper assembly |

## Results

U-Net / SegFormer / L_att / Boundary rows share the 3576/766/766 seed-42 test set;
DeepLabV3+ is a 400-sample CPU-smoke subset (seed-42 Dice 0.7821 from
`deeplab_multiseed.json`).

| Model | Dice | IoU | AAMO | Seeds |
|---|---|---|---|---|
| U-Net (CNN baseline) | 0.8615 | 0.7568 | n/a | 1 |
| SegFormer-B0 (no attention loss) | 0.8743 | 0.7766 | 0.0334 | 1 |
| SegFormer-B0 + Attention Consistency (λ2=1.0 MSE) | 0.8577 | 0.7508 | 0.7476 | 1 |
| SegFormer-B0 + Attention + Boundary Loss | 0.8669 | 0.7650 | 0.6218 | 1 |
| DeepLabV3+ (MobileNetV3) — extra baseline | 0.7862 ± 0.0193 | 0.6481 ± 0.0264 | n/a | 3 |

Selection rule for the attention row (Dinura): max test AAMO, then max Dice.
Winner run tag `l2_1_mse` supersedes Kalana's default-λ2=0.3 attention numbers.

DeepLabV3+ is CPU smoke-scale (400 samples / 5 epochs) and exists to exercise
the multi-seed aggregation pipeline. Full-scale SegFormer extra seeds are the
shared Colab jobs in `multiseed_kickoff.md` (Lasana = Job 5).

## Cross-folder edits

**None.** No file outside this folder was modified. Phase 1 Person 4 results
are left untouched on purpose.
