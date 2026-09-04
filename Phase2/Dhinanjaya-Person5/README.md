# Person 5 — Boundary Refinement Module (Phase 2 stretch goal)

Proposal Section 3.4 / Section 6.2: if the core contribution (Sections
3.1–3.3, done in Phase 1) is validated with time remaining, add a Boundary
Refinement Module — boundaries extracted from both prediction and ground
truth via morphological gradient, supervised with a boundary Dice loss —
since most forest-segmentation errors occur near canopy edges.

## Why this is here, in Phase 2, before the rest of Phase 2 has started

Of Person 5's Phase 2 tasks (proposal §6.2.2), everything else — overseeing
Person 3's λ-sweep, managing the Person 3→Person 4 checkpoint handoff —
depends on training/ablation runs that haven't happened yet. This module
doesn't: like Person 3's Attention Consistency Loss in Phase 1, it's pure
tensor ops that can be built and unit-tested against dummy data before any
trained model exists. Built now, ahead of its proposal-scheduled Week 10
slot, so it's ready the moment Person 3's model is ready to integrate it.

## Layout

```
boundary_refinement/
  boundary_ops.py   morphological_gradient_boundary() — differentiable
                     dilation/erosion via max/min pooling (§3.4)
  loss.py             BoundaryDiceLoss, total_objective_with_boundary()
run_boundary_sweep.py       train the λ3 sweep cells (fixed λ2=1.0/MSE)
eval_boundary_sweep.py       evaluate completed cells
aggregate_boundary_sweep.py   pick the winner, write results/boundary_sweep_comparison.{csv,md}
make_boundary_colab_zip.py    bundle everything for a Colab GPU run
boundary_sweep_colab.ipynb     the Colab notebook itself
results/
  boundary_sweep_comparison.{csv,md}   sweep table (once run)
  boundary_winning_config.json           winner + selection rule (once run)
tests/
  test_boundary_refinement.py     10 tests, dummy tensors only, no model needed
  test_boundary_sweep_helpers.py   7 tests, boundary_tag/pick_winner logic only
```

## Design notes

- **Differentiable, not binary, morphology.** Classic dilation/erosion
  operate on binary images and aren't differentiable. `L_boundary` needs to
  backpropagate into the segmentation decoder through the *predicted* soft
  mask `P`, not just the hard ground-truth mask `Y`, so dilation/erosion
  are implemented as max/min pooling (`F.max_pool2d`, `-F.max_pool2d(-x)`)
  instead — differentiable via the pooling subgradient, and reduces to the
  standard binary morphological gradient when the input actually is binary
  (i.e. `Y`).
- **No dependency beyond `torch`.** Unlike `Phase1/Dinura-Person3/attention_consistency`
  (needs `transformers` for the SegFormer model itself), this module only
  needs plain tensor ops, so it's usable in any environment that has
  `torch` — verified in the `anaconda3/envs/ml` env, which has `torch`
  but not `transformers`.
- **Kept out of `Phase1/Dinura-Person3/`.** Per `CONTRIBUTING.md`'s
  cross-folder-edit rule, this doesn't touch Person 3's existing
  `loss.py`/`total_objective` — `total_objective_with_boundary` here is a
  drop-in replacement to swap in once the module is wired into the real
  training loop (Week 10), not an edit to Person 3's file.

## Reproducing

```bash
# from Phase2/Dhinanjaya-Person5/
python tests/test_boundary_refinement.py       # 10/10, no model needed
python tests/test_boundary_sweep_helpers.py     # 7/7, no model needed

# Actual GPU training (needs transformers + CUDA — this repo runs it on
# Colab, same as Dinura's λ2 sweep and Kalana's full-scale run):
python make_boundary_colab_zip.py               # builds boundary_sweep.zip
# upload boundary_sweep.zip to MyDrive/, open boundary_sweep_colab.ipynb, Run all

# Or locally on a machine with a GPU + transformers installed:
python run_boundary_sweep.py --lambda3 0.1 0.2 0.5 --lambda2 1.0 --att-mode mse
python eval_boundary_sweep.py --lambda3 0.1 0.2 0.5 --lambda2 1.0 --att-mode mse
python aggregate_boundary_sweep.py
```

## Cross-folder edits (per CONTRIBUTING.md)

Wired Boundary Loss into **`Phase2/Dinura-Person3/train_full_scale.py`** as
an opt-in `--lambda3`/`--boundary-kernel` CLI flag (default `lambda3=0.0`,
which keeps that file's behavior and output CSV schema bit-identical to
before — verified against `run_lambda_sweep.py`'s existing `SweepArgs`
shim, which has no `lambda3` attribute at all, via `getattr(..., 0.0)`
fallbacks throughout). The `boundary_refinement` import itself is lazy
(only triggered when `lambda3>0`, inside `train_variant`), and its path
resolution tries both the real repo layout (`Phase2/Dhinanjaya-Person5/`,
a sibling of `Dinura-Person3/`) and a flat Colab-bundle layout
(`boundary_refinement/` dropped alongside `train_full_scale.py` — see
`make_boundary_colab_zip.py`) — so Dinura's existing `lambda_sweep.zip` /
`lambda_sweep_colab.ipynb` pipeline (which has no `boundary_refinement/`
bundled) is completely unaffected. Verified via
`Phase2/Dinura-Person3/tests/test_boundary_integration.py`.

The sweep orchestration itself (`run_boundary_sweep.py` etc., this folder)
does **not** edit Dinura's folder further — it imports his
`train_full_scale.py`/`eval_full_scale.py`/`paths.py` and reuses his
`paths.py`'s output roots, so a λ3 sweep's raw per-cell checkpoints/results
land under `Phase2/Dinura-Person3/{checkpoints,results}/runs/l2_1_mse_bnd<λ3>/`
(exactly where a direct `train_full_scale.py --lambda3 ...` CLI run would
put them) — only the summary table (`results/boundary_sweep_comparison.*`)
is written into this folder.

## What's NOT done yet

- **The actual GPU training run.** This machine has neither `transformers`
  nor a GPU (see `dev_environment_notes`), so the sweep code above is
  verified via stub-based unit tests only (`test_boundary_integration.py`,
  `test_boundary_sweep_helpers.py`) — not against the real SegFormer model
  or real data. Needs Colab (or similar), ~2h/cell × 3 cells on a T4.
- Once that run + `aggregate_boundary_sweep.py` produce a winner, hand its
  `baseline_comparison.csv`-style row to
  `Phase2/Lasana-Person4/fold_full_scale_results.py`'s TODO (path already
  named there: `Phase2/Dhinanjaya-Person5/results/baseline_comparison.csv`)
  to close out the last pending Phase 2 ablation row.
- `kernel_size` (default 3, i.e. a ~1-pixel boundary band) not swept —
  worth trying 5/7 in a follow-up if λ3 alone doesn't move Dice/IoU much.
