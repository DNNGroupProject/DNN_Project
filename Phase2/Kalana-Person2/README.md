# Person 2 — Transformer Lead (Phase 2)

Role: **Transformer Lead**. Proposal §6.2.2 only assigns Person 2 a Weeks 5–6
support task (*"Support integration of the attention-extraction pipeline with
Person 3's loss implementation"*) and nothing after Week 6. The full-scale
SegFormer GPU run below was handed over on 2026-08-15
(`Phase2/Dhinanjaya-Person5/phase2_kalana_kickoff.md`) because it is the
paper's blocking numbers, and this role already owns the closest equivalent
Colab (`Phase1/Kalana-Person2/segformer_baseline_scratch_colab.ipynb`).

| Weeks | Deliverable | Status |
|---|---|---|
| 5–6 | Support attention-extraction ↔ Person 3 loss integration | Pipeline already lives in `Phase1/Dinura-Person3/attention_consistency/` — this folder consumes it, does not reimplement it |
| 5–9 (handoff) | Full-scale SegFormer-B0 train + eval (vanilla and +Attention Consistency Loss) | Notebook + wrappers ready; GPU numbers not in yet — run the Colab |

## What this folder is for

`Phase1/Dinura-Person3/segformer_full_scale_colab.ipynb` is complete but has
never been executed (`execution_count: null` on every cell). Every SegFormer
AAMO/Dice/IoU number currently in the paper is the 400-image / 8-epoch CPU
smoke run. This folder is the Phase 2 place that actually runs that job, with
writes landing **here** rather than back into Person 3's or Person 4's Phase 1
trees.

Same code path as the smoke trainer (`train_segformer_smoke.py` /
`eval_segformer.py` / `generate_attention_figures.py`): SegFormer-B0, stage-4
Grad-Rollout, Attention Consistency Loss, Person 4's Dice/IoU/F1/AAMO/
efficiency modules. Nothing of that is rewritten. What *is* different:

- Output dirs are `Phase2/Kalana-Person2/{checkpoints,results}/`.
- Person 4's comparison table and checkpoints are **not** overwritten
  (Person 3's `eval_segformer.py` copies into `Lasana-Person4_Evaluation/`;
  `eval_full_scale.py` here does not).
- Split is the real audited 5,108-pair 70/15/15: **3576 / 766 / 766**, seed
  42 — the same held-out 766 images Chanupa's U-Net baseline already used
  (Person 3's `data.py` was aligned to that algorithm on 2026-08-15).

`Phase1/Dhinanjaya-Person5/split_fix_notice_kalana.md` is FYI only. The Phase 1
scratch notebook's `make_splits()` was already the Chanupa algorithm; it is
not changed here.

## Layout

```
paths.py                         Repo or unzipped-bundle path helpers
train_full_scale.py              Person 3's smoke trainer, GPU + this folder's dirs
eval_full_scale.py               Person 4 metrics; writes only this folder's results/
generate_full_scale_figures.py   Attention-drift figures → results/attention_drift_figures/
segformer_full_scale_colab.ipynb Colab GPU run (unzip Drive zip, train, eval, figures)
make_colab_zip.py                Builds segformer_full_scale.zip (code + dataset only)
tests/test_split_identity.py     Pre-flight: 3576/766/766 matches Chanupa's U-Net split
checkpoints/                     Full-scale weights (Drive; do not commit .pt)
results/                         Train logs, eval JSON, comparison table, figures
```

## Quick start / Reproducing

From `Phase2/Kalana-Person2/`:

```bash
python tests/test_split_identity.py
```

No GPU, no `transformers`. Uses the committed mask filenames when the dataset
is present; otherwise a synthetic filename list (same algorithm).

Full-scale GPU (the paper run) — Colab, Runtime → GPU. **Do not upload the
whole repo.**

```bash
python make_colab_zip.py    # writes segformer_full_scale.zip (~170 MB)
```

1. Upload `segformer_full_scale.zip` to Drive as `MyDrive/segformer_full_scale.zip`.
2. Open `segformer_full_scale_colab.ipynb` in Colab (GPU runtime) and Run all.
   It unzips to `/content/segformer_full_scale` and writes checkpoints/results
   to `MyDrive/segformer_full_scale_outputs`.
3. Copy the JSON / markdown / figures back into git; leave `.pt` files on
   Drive (they will blow GitHub's 100 MiB limit — same issue Chanupa hit on
   the U-Net checkpoint).

The zip is gitignored (too big for GitHub). Rebuild it any time with
`python make_colab_zip.py`.

Equivalent local commands once a GPU and `transformers` exist, still from
this folder:

```bash
python train_full_scale.py --variant both
python eval_full_scale.py --variant both
python generate_full_scale_figures.py --n 3
```

Hyperparameters (proposal §4.1 / Person 3 notebook, full audited split):

| | |
|---|---|
| Split | 3576 / 766 / 766, seed 42 |
| Epochs | 20 |
| Batch | 16 (vanilla); attention variant stays batch 1 (Grad-Rollout constraint) |
| lr | 6e-5 |
| λ2 / σ / att mode | 0.3 / 8 / mse |

## Dependencies on teammates

| Direction | What |
|---|---|
| I need | `Phase1/Dinura-Person3/attention_consistency/` (model, hooks, rollout, loss, data). Read-only. |
| I need | `Phase1/Lasana-Person4_Evaluation/{metrics,aamo,efficiency,evaluate}.py` for the eval formulas. Import-only; no writes. |
| I need | `Phase1/Kalana-Person2/{images,masks}` — the 5,108-pair dataset. |
| I need | `Phase1/Chanupa-Person1/dataset.py` — only the split-identity test compares against it. |
| I hand off | `results/eval_{vanilla,att}.json`, `results/baseline_comparison.{csv,md}`, attention-drift figures, and (on Drive) `checkpoints/segformer_b0_{vanilla,att}_{best,last}.pt` to Dhinanjaya for the paper table / Figure 2. |

## Results

Not trained yet. Smoke-scale numbers (400 / 60 / 60, 8 epochs, CPU) still live
only in `Phase1/Dinura-Person3/results/` and must **not** be copied here as if
they were this run. After the Colab pass, this section gets the full-scale
rows.

## Cross-folder edits

**None.** No file outside `Phase2/Kalana-Person2/` was modified. Person 3's
unexecuted notebook is left as the Phase 1 snapshot; this folder is the
Phase 2 run location, matching `CONTRIBUTING.md`'s phase boundary.

## After this run lands

Ping Dhinanjaya. Paper abstract, Table 1, Figure 2, the smoke-scale caveat
paragraphs, and Person 4's comparison table are his follow-up — listed in
`Phase1/Dhinanjaya-Person5/full_scale_segformer_todo.md` ("After this is done").
Not edited from here.
