# Multi-seed sweeps + MSE-vs-KL — shared GPU job (Phase 2/3, all 5 of us)

**Why:** Table 1's vanilla/attention/boundary rows are all still single-seed
(only DeepLabV3+ has real multi-seed mean±std). We also never ran the
KL-divergence variant of `L_att` — only MSE has ever been evaluated, despite
both being implemented and unit-tested since Phase 1. Neither is required
for the Sep 11 short paper, but both matter for the full paper.

**What's needed:** 7 independent training runs, each ~2h on a T4. One
shared Colab bundle covers all of them — you just edit one config cell to
pick your job.

## How to run your job

1. Upload `Phase2/Dhinanjaya-Person5/multiseed_train_colab.zip` (171.6 MB)
   to `MyDrive/multiseed_train_colab.zip`.
2. Upload `Phase2/Dhinanjaya-Person5/multiseed_train_colab.ipynb` to Colab.
3. Runtime → Change runtime type → GPU (T4+) → Save.
4. Edit the **Config** cell near the top to your assigned job's values
   (table below), then Runtime → **Run all**.
5. When it finishes (~2h), share the whole
   `MyDrive/multiseed_outputs_<OUTPUT_TAG>/` folder back to Dhinanjaya
   (Drive share link, or zip + send).

Everyone uploads the **same** zip — don't rebuild it, just reuse the one
committed to the repo.

## Assignment

| # | Person | Job | VARIANT | SEED | LAMBDA2 | ATT_MODE | LAMBDA3 | OUTPUT_TAG |
|---|---------|-----|---------|------|---------|----------|---------|------------|
| 1 | Kalana | vanilla, seed 43 | `vanilla` | 43 | 1.0 | mse | 0.0 | `vanilla_seed43` |
| 2 | Kalana | attention (MSE), seed 43 | `att` | 43 | 1.0 | mse | 0.0 | `att_mse_seed43` |
| 3 | Dinura | attention (MSE), seed 44 | `att` | 44 | 1.0 | mse | 0.0 | `att_mse_seed44` |
| 4 | Dinura | attention (**KL**), seed 42 | `att` | 42 | 1.0 | kl | 0.0 | `att_kl_seed42` |
| 5 | Lasana | vanilla, seed 44 | `vanilla` | 44 | 1.0 | mse | 0.0 | `vanilla_seed44` |
| 6 | Chanupa | boundary, seed 44 | `att` | 44 | 1.0 | mse | 0.2 | `boundary_seed44` |
| 7 | Dhinanjaya | boundary, seed 43 | `att` | 43 | 1.0 | mse | 0.2 | `boundary_seed43` |

(Kalana and Dinura each get 2 jobs since they own the vanilla/attention and
λ2/loss-formulation infrastructure respectively; everyone else gets 1.)

**Job 4 is the one that actually matters most** — it's the only KL run,
answering "does MSE vs KL change anything" for the first time. The other
6 are the 2nd/3rd seed for each of the three rows already in Table 1
(vanilla, attention-MSE, boundary), giving real mean±std instead of a bare
single-run number.

## After your job finishes

Send the output folder back — don't try to merge results into the repo
yourself. Once all 7 are back, Dhinanjaya will fold them into
`ablation_mean_std.md` (extending Lasana's existing multi-seed aggregation
pattern) and a new KL-vs-MSE comparison table, then update the paper.

## Notes

- `--variant vanilla` ignores `LAMBDA2`/`ATT_MODE`/`LAMBDA3` — only used for
  `--variant att` jobs. Left as `1.0`/`mse`/`0.0` in the vanilla rows above
  for clarity, doesn't affect the run.
- Each job trains fresh from the ImageNet-pretrained backbone — no
  checkpoint seeding needed (unlike the original λ2/λ3 sweeps, these are
  all independent single-cell runs).
- The boundary jobs (6, 7) already include the attention loss at λ2=1.0 —
  `train_full_scale.py` adds `L_boundary` on top of `L_att`, not instead of it.
- If your session disconnects partway through, re-upload the zip and
  re-run — training isn't resumable mid-epoch, so you'll lose progress
  since the last completed epoch, but the training cell skips your
  variant entirely if its checkpoint already exists in your `OUTPUT_TAG`
  folder, so a *finished* job won't get silently retrained if you re-run
  the notebook.
