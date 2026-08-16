# Phase 2 kickoff — for Dinura (Person 3), 2026-08-16

## Why this doc exists now, not two weeks ago

Chanupa and Kalana both got a Phase 2 kickoff doc already
(`phase2_chanupa_kickoff.md`, `phase2_kalana_kickoff.md`). You hadn't —
this closes that gap. Unlike their roles, yours is the actual **critical
path**: both Lasana's Week 10 ablation study and my own Boundary
Refinement Module integration are blocked on the checkpoint your task
below produces.

## What the proposal actually assigns you

Section 6.2.2, Person 3 — Loss & Training, Weeks 7–9:

> Implement and tune the Attention Consistency Loss (λ sweep); train the
> full explainability-guided SegFormer-B0 model to convergence.

Week 10:

> Support the ablation study; help interpret attention-related results.

## How this is different from what Kalana is running on Colab right now

Kalana is currently running `Phase2/Kalana-Person2/segformer_full_scale_colab.ipynb`
— full-scale (5,108 images, 20 epochs), but with **one fixed
hyperparameter set**: λ2=0.3, σ=8, MSE, one seed. That's the proposal's
*initial* value for λ2, run at full scale instead of the 400-image smoke
run — useful, and it'll unblock the paper's headline Table 1 / Figure 2
numbers, but it is not the λ-sweep. Your Weeks 7–9 task is the tuning
work itself: multiple training runs across a few λ2 values, reviewing
validation Dice/IoU and the L_att training curve per run, to find the
best-performing config. That's what's supposed to end up as the "real"
Attention Consistency checkpoint — Kalana's default-λ2 run is a baseline
data point, not necessarily the final one.

Two things worth knowing before you start:

- **The MSE/KL choice is already built.** The proposal's Person 5 task
  ("implement the KL-divergence variant... so Person 3 can compare both
  formulations") is done — `attention_consistency/loss.py` already has
  both formulations, unit-tested. You can sweep λ2 on MSE first (matches
  the paper's current numbers) and, time permitting, repeat at the best
  λ2 with KL to compare.
- **The split is already aligned** (fixed 2026-08-15, commit `43d8480`):
  `attention_consistency/data.py`'s `make_splits` now matches
  `Chanupa-Person1/dataset.py` element-for-element at 3576/766/766, seed
  42 — the same held-out test set as the U-Net baseline and Kalana's run.
  No action needed here, just don't second-guess it if the numbers look
  slightly different from an earlier local run.

## What's already usable — you don't need to write a training script

`Phase2/Kalana-Person2/train_full_scale.py` already exposes `--lambda2`
and `--att-mode` (`mse`/`kl`) as CLI flags, plus `--seed`, `--epochs`,
`--sigma`. It imports `load_pairs` / `make_splits` /
`AttentionConsistencyLoss` / `grad_rollout_attention_map` directly from
your own `Phase1/Dinura-Person3/attention_consistency/`, unmodified — so
running it with a different `--lambda2` per pass is your λ-sweep, no new
code required. One thing to change: per `CONTRIBUTING.md`'s phase
boundary, write your sweep's checkpoints/logs to a new
`Phase2/Dinura-Person3/` folder rather than into `Kalana-Person2/results/`
— copy `paths.py` and repoint `CKPT_DIR`/`RESULTS_DIR`, same pattern
Kalana used to avoid writing into your folder.

## Checklist

- [ ] Pick λ2 sweep values (proposal's initial value is 0.3 — try e.g.
      0.1, 0.3, 0.5, 1.0, or narrower once you see the trend)
- [ ] Run full-scale attention-variant training at each λ2 (GPU — Colab,
      same setup as Kalana's notebook)
- [ ] Record val Dice/IoU and the L_att curve per run
- [ ] Optionally repeat the best λ2 with `--att-mode kl` to compare
      against MSE
- [ ] Pick the best config, write down why (a short README in your
      Phase 2 folder is enough)
- [ ] Support Lasana's Week 10 ablation study — help interpret
      attention-related results once they start
- [ ] Ping Dhinanjaya with the winning config + checkpoint location —
      needed for the Boundary Refinement Module integration and the
      paper's methodology/hyperparameter section
