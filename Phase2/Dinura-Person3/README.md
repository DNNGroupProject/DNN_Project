# Person 3 — Loss & Training (Phase 2)

Role: **Loss & Training**. Proposal §6.2.2 Weeks 7–9:

> Implement and tune the Attention Consistency Loss (λ sweep); train the
> full explainability-guided SegFormer-B0 model to convergence.

Kickoff: `Phase2/Dhinanjaya-Person5/phase2_dinura_kickoff.md`.

| Weeks | Deliverable | Status |
|---|---|---|
| 7–9 | λ2 sweep + optional KL comparison at best λ2 | Scaffold + λ2=0.3 MSE seeded from Kalana; remaining cells need Colab GPU |
| 10 | Support Lasana's ablation / interpret attention results | Ready once winning config is final |

## How this differs from Kalana's run

Kalana's `Phase2/Kalana-Person2/` run is **one fixed** config (λ2=0.3, σ=8,
MSE, seed 42) and is already the paper's Table 1 attention row. This folder
is the **tuning** work: multiple full-scale attention-variant trainings
across λ2 ∈ {0.1, 0.3, 0.5, 1.0}, then (optional) KL at the best λ2.

λ2=0.3 / MSE is **not re-trained** — it is seeded from Kalana's finished
metrics (`python seed_from_kalana.py`) so the sweep table starts with a
real baseline cell. Train 0.1 / 0.5 / 1.0 (and optional KL) on Colab.

## Layout

```
paths.py                    Sweep-aware path helpers (run tags l2_<λ>_<mode>)
train_full_scale.py         Same GPU trainer as Kalana (writes via paths.*)
eval_full_scale.py          Person 4 metrics; CUDA-safe after GFLOPs
run_lambda_sweep.py         Train attention variant across λ2 values
eval_lambda_sweep.py        Eval every completed cell + rebuild table
aggregate_sweep.py          sweep_comparison.{csv,md} + winning_config.json
seed_from_kalana.py         Copy Kalana λ2=0.3 metrics into runs/l2_0.3_mse/
lambda_sweep_colab.ipynb    Colab GPU entrypoint (unzip Drive zip)
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
# Upload lambda_sweep.zip to Drive as MyDrive/lambda_sweep.zip
# Open lambda_sweep_colab.ipynb, Runtime → GPU, Run all
```

Or locally once a GPU exists:

```bash
python run_lambda_sweep.py --lambda2 0.1 0.5 1.0 --att-mode mse
python eval_lambda_sweep.py --lambda2 0.1 0.3 0.5 1.0 --att-mode mse
# optional KL at the winning λ2, e.g. if winner is 0.5:
python run_lambda_sweep.py --lambda2 0.5 --att-mode kl
python eval_lambda_sweep.py --lambda2 0.5 --att-mode kl
```

Each att-variant epoch is ~6 minutes on T4 (batch 1 + Grad-Rollout). Budget
~2 hours per λ2 cell × 3 remaining MSE cells ≈ 6 hours, plus optional KL.

## Selection rule

Highest **test AAMO**, then highest **test Dice**; MSE preferred over KL on
ties. Documented again in `results/sweep_comparison.md`.

## Dependencies on teammates

| Direction | What |
|---|---|
| I need | `Phase1/Dinura-Person3/attention_consistency/` (own Phase 1 package) |
| I need | Person 4 `metrics` / `aamo` / `efficiency` (import only) |
| I need | Dataset at `Phase1/Kalana-Person2/{images,masks}` |
| I need | Kalana's λ2=0.3 metrics (and optionally his Drive `.pt`) as the seeded cell |
| I hand off | `results/winning_config.json` + checkpoint path to Dhinanjaya (Boundary Refinement) and Lasana (ablation) |

## Cross-folder edits

**None.** Kalana's Phase 2 folder is read-only (seeded from). Phase 1 package
is import-only.

## After the remaining Colab cells finish

1. Download `results/runs/` (and optionally `checkpoints/runs/`) from Drive
   into this folder.
2. `python aggregate_sweep.py`
3. Ping Dhinanjaya with `winning_config.json` + checkpoint location.
