# Full-scale SegFormer results — what needs to be done, checked 2026-08-15

**Update 2026-08-15:** the split-size mismatch flagged below is now fixed
in code, not just documented. `attention_consistency/data.py`'s
`list_pairs`/`make_splits` previously used NumPy's `RandomState.shuffle` +
sklearn's `train_test_split` — a different RNG and a different
partitioning algorithm from `Chanupa-Person1/dataset.py`'s stdlib
`random.shuffle` + front-slice. Same seed and same counts did **not** mean
the same held-out images. Rewrote `data.py` to use the identical algorithm
(sort mask filenames, `random.seed(42)`, `random.shuffle`, front-slice
val/test/train) and verified directly against the real `Kalana-Person2/masks`
directory (5,108 files): at n_train=3576/n_val=766/n_test=766, the
resulting train/val/test file lists are **exactly identical**, element for
element, to Chanupa's `dataset.py` split. Also updated
`segformer_full_scale_colab.ipynb`'s `Args`/`EvalArgs` from the old
proposal-rounded 3500/750/750 to the real 3576/766/766. Not yet run — the
notebook still needs a GPU pass — but whoever runs it next will
automatically train/eval on the same held-out test set as the U-Net
baseline and augmentation ablation, no extra step required on their end.

## Status: not started

Checked against `origin/main` @ `0d2e3c0` (2026-08-15). `git log` on
`Phase1/Kalana-Person2/` and `Phase1/Dinura-Person3/` shows nothing since
`f66c283` (2026-08-09, Person 5's mask-binarization fix — not a training
run). Opened `Phase1/Dinura-Person3/segformer_full_scale_colab.ipynb`
directly: every code cell has `execution_count: null` and no outputs — the
notebook has never actually been run. It's a ready-to-go script, not a
result.

Every number currently in the paper for both SegFormer variants — the
headline AAMO jump (0.0144→0.432) and the Dice/IoU deltas — comes from
`Phase1/Dinura-Person3/results/train_summary_vanilla.json` /
`train_summary_att.json`: **400 training images, 8 epochs**, confirmed
directly from those files (`n_train: 400, n_val: 60, epochs: 8`). The paper
already says this out loud (§Results, "Limitations of these preliminary
numbers") but it's still the only data we have.

## What the notebook already does (nothing to design, just to run)

`segformer_full_scale_colab.ipynb` is a complete, unexecuted Colab script,
5 steps, reusing `attention_consistency/` and `train_segformer_smoke.py` /
`eval_segformer.py` / `generate_attention_figures.py` directly (no
reimplementation — same code path as the CPU smoke run, just bigger
numbers and moved onto GPU):

1. Mount Drive, `pip install transformers thop`, point `data_mod.IMG_DIR`/
   `MASK_DIR` at the uploaded `Kalana-Person2/{images,masks}`.
2. Train both variants (`vanilla`, `att`) at **3500/750/750 split, 20
   epochs, batch 16, lr 6e-5, λ2=0.3, σ=8, seed=42** — same
   hyperparameters already documented in the paper's Implementation
   Details paragraph, just at full scale.
3. Evaluate both checkpoints with Person 4's `metrics.py`/`aamo.py`/
   `efficiency.py` — same tooling as the U-Net/DeepLab rows, so numbers
   are directly comparable.
4. Regenerate the qualitative attention-drift figures
   (`generate_attention_figures.py --n 3`).
5. Everything writes straight back to
   `Phase1/Dinura-Person3/{results,checkpoints}/` on Drive (the notebook
   `sys.path.insert`s into the Drive copy) — nothing to copy manually
   afterward.

The only manual step is Step 0: upload `Phase1/Dinura-Person3/` and
`Phase1/Kalana-Person2/{images,masks}` to Google Drive and open the
notebook in Colab with a GPU runtime. From there it runs end-to-end.

## One thing to flag before running it: split-size mismatch

The notebook hardcodes **3500/750/750** (`Args.n_train/n_val/n_test` in
Step 2, `EvalArgs` in Step 3) — the proposal's rough "~5,000 images, 70/15/15"
figure. But the dataset's actual audited size is **5,108** images
(`Phase2/Chanupa-Person1/results/mask_audit.md`), and both Chanupa's U-Net
baseline and the augmentation ablation already use the real stratified
split off that count: **3576/766/766**, same seed (42). If the SegFormer
full-scale run uses 3500/750/750 instead, its test set is not quite the
same 766 held-out images the U-Net numbers were measured on — a ~2%
mismatch, probably not paper-breaking, but worth either (a) fixing the
notebook's `Args`/`EvalArgs` to 3576/766/766 before running, so every
model in the comparison table shares one true test set, or (b) explicitly
noting the discrepancy in the paper if left as-is. Recommend (a) — it's a
two-number edit in cells 7 and 9, before any GPU time is spent.

## Owner: Kalana (assigned 2026-08-15)

Not explicitly assigned in the proposal — Phase 2 (§6.2.2) only gives
Kalana a Weeks 5–6 support task (integrating the attention-extraction
pipeline with Person 3's loss implementation) and nothing after, leaving
him with real open bandwidth from Week 7 onward. Assigning this to him
rather than leaving it as "whoever has GPU time": he already owns the
SegFormer pipeline as Transformer Lead and has already built and run the
closest equivalent full-scale job himself
(`Kalana-Person2/segformer_baseline_scratch_colab.ipynb`) — same GPU/Colab
setup, same `SEED=42`/`VAL_SPLIT`/`TEST_SPLIT`/shuffle-then-front-slice
split algorithm this notebook now also uses (see the split fix above).
Least new context to load of anyone on the team.

Needs an actual GPU runtime (~20 epochs × 2 variants at batch 16 — budget
real wall-clock time, the 8-epoch/400-image CPU smoke run alone took
669.5s per variant per `train_summary_vanilla.json`). Dinura no longer
needs to run this themselves — see `split_fix_notice_dinura.md`, updated
to reflect the handoff — but the notebook and checkpoints still live under
`Dinura-Person3/`, so Kalana will need read/write access there (or Drive
copies) to run it and land results back in the right place.

## After this is done

Ping Dhinanjaya. This updates, in `paper_acm/main.tex` and
`short_paper_draft.md`:
- The abstract's headline numbers (currently the smoke-scale AAMO/Dice/IoU).
- Table 1 / the results table — replace with full-scale rows, update the
  caption (currently says "CPU/local smoke runs").
- Figure 2 (`attention_drift_02_post_train.png`) — regenerate from the
  full-scale checkpoints; the current caption already flags the vanilla
  model's "corner artifact" as plausibly an undertraining effect worth
  revisiting once this run converges.
- The "Limitations of these preliminary numbers" paragraph and the
  Threats-to-Validity §(1) Scale point — both currently exist specifically
  to caveat the smoke-scale numbers and should be revised or removed once
  real ones land.
- `results/baseline_comparison.md` / `ablation_mean_std.md` (Person 4,
  Lasana's files) — need the new SegFormer rows merged in too.
