# Person 3 — Loss & Training (Phase 2)

Role: **Loss & Training**. Proposal §6.2.2 Weeks 7–9:

> Implement and tune the Attention Consistency Loss (λ sweep); train the
> full explainability-guided SegFormer-B0 model to convergence.

Kickoff: [`Phase2/Dhinanjaya-Person5/phase2_dinura_kickoff.md`](../Dhinanjaya-Person5/phase2_dinura_kickoff.md).

| Weeks | Deliverable | Status |
|---|---|---|
| 7–9 | λ2 sweep + optional KL at best λ2 | Scaffold ready; λ2=0.3 MSE seeded from Kalana; remaining cells need Colab GPU |
| 10 | Support Lasana's ablation / interpret attention results | Unblocked once winning config is final |

## Why this exists

Kalana's `Phase2/Kalana-Person2/` run is **one fixed** config (λ2=0.3, σ=8,
MSE) and is already the paper's Table 1 attention row. This folder is the
**tuning** work the proposal assigns Person 3: multiple full-scale
attention-variant trainings across λ2 ∈ {0.1, 0.3, 0.5, 1.0}, then
(optional) KL at the best λ2.

λ2=0.3 / MSE is **not re-trained** here — it is seeded from Kalana's
finished metrics so the sweep table starts with a real baseline cell.
Train 0.1 / 0.5 / 1.0 (and optional KL) on Colab.

This is the critical-path blocker for Lasana's Week 10 multi-seed ablation
and for wiring Dhinanjaya's Boundary Refinement Module into training.

## Layout

```
paths.py                    Sweep-aware paths (run tags l2_<λ>_<mode>)
train_full_scale.py         GPU trainer (same path as Kalana; writes via paths.*)
eval_full_scale.py          Person 4 metrics; CUDA-safe after GFLOPs
run_lambda_sweep.py         Train attention variant across λ2 values
eval_lambda_sweep.py        Eval every completed cell + rebuild table
aggregate_sweep.py          sweep_comparison.{csv,md} + winning_config.json
seed_from_kalana.py         Copy Kalana λ2=0.3 metrics into runs/l2_0.3_mse/
lambda_sweep_colab.ipynb    Colab GPU entrypoint
make_colab_zip.py           Builds lambda_sweep.zip (code + dataset)
tests/test_sweep_helpers.py Tag / winner-selection unit tests
checkpoints/runs/l2_*/      Per-cell weights (Drive; *.pt gitignored)
results/runs/l2_*/          Per-cell train/eval logs
results/sweep_comparison.*  Aggregated table
results/winning_config.json Current pick
```

## Quick start

```bash
# from Phase2/Dinura-Person3/
python seed_from_kalana.py
python aggregate_sweep.py
python tests/test_sweep_helpers.py
```

Full-scale remaining cells (GPU — Colab):

```bash
python make_colab_zip.py
# Upload lambda_sweep.zip → MyDrive/lambda_sweep.zip
# Open lambda_sweep_colab.ipynb, Runtime → GPU, Run all
```

Or locally once a GPU exists:

```bash
python run_lambda_sweep.py --lambda2 0.1 0.5 1.0 --att-mode mse
python eval_lambda_sweep.py --lambda2 0.1 0.3 0.5 1.0 --att-mode mse
# optional KL at the winning λ2:
python run_lambda_sweep.py --lambda2 0.5 --att-mode kl   # example
python eval_lambda_sweep.py --lambda2 0.5 --att-mode kl
```

Each att-variant epoch is ~6 min on T4 (batch 1 + Grad-Rollout). Budget
~2 hours per λ2 cell × 3 remaining MSE cells ≈ 6 hours, plus optional KL.

## Selection rule

Highest **test AAMO**, then highest **test Dice**; MSE preferred over KL on
ties. Documented again in `results/sweep_comparison.md`.

## Dependencies on teammates

| Direction | What |
|---|---|
| I need | Own Phase 1 package `Phase1/Dinura-Person3/attention_consistency/` |
| I need | Person 4 `metrics` / `aamo` / `efficiency` (import only) |
| I need | Dataset at `Phase1/Kalana-Person2/{images,masks}` |
| I need | Kalana's λ2=0.3 metrics (and optionally his Drive `.pt`) as the seeded cell |
| I hand off | `results/winning_config.json` + checkpoint path to Dhinanjaya and Lasana |

## Cross-folder edits

**None.** Kalana's Phase 2 folder is read-only (seeded from). Phase 1 package
is import-only.

## After remaining Colab cells finish

1. Download `results/runs/` (and optionally `checkpoints/runs/`) from Drive
   into this folder.
2. `python aggregate_sweep.py`
3. Ping Dhinanjaya with `winning_config.json` + checkpoint location.
