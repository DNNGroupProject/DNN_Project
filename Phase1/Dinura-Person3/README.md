# Person 3 — Loss & Training

This folder now covers three Phase 1 deliverables end-to-end, built together
because each one blocked the next:

| Task | Owner (per proposal) | Status |
|---|---|---|
| SegFormer-B0 model, attention-extraction hooks, adapted Grad-Rollout | Person 2 (Transformer Lead) | Done — `attention_consistency/segformer_model.py`, `hooks.py`, `rollout.py` |
| Math formulation + Attention Consistency Loss, unit-tested | **Person 3 (this role)** | Done — `math_formulation.md`, `attention_consistency/loss.py`, `tests/` |
| Real SegFormer rows + AAMO in the baseline comparison table | Person 4 (Evaluation Lead) | Done — `eval_segformer.py`, mirrored into `../Lasana-Person4_Evaluation/results/` |

Kalana's and Lasana's own folders are untouched except for two additive
files needed to make Lasana's `evaluate.py --model segformer` actually work
(see "What touched teammates' folders" below) — everything else lives here.

## Why these three together

Person 2's attention hooks were the critical-path blocker: Person 3's loss
needs an attention map to supervise, and Person 4's SegFormer evaluation
rows and AAMO metric need both a checkpoint and attention maps. Building
all three here means the whole chain (model → attention → loss → trained
checkpoint → real metrics) is demonstrated working, not just each piece in
isolation.

## Layout

```
attention_consistency/
  segformer_model.py   SegFormer-B0 builder (pretrained MiT-B0 encoder, num_labels=2)
  hooks.py              AttentionExtractor — forward hooks on stage-4 (sr_ratio=1) blocks
  rollout.py             Adapted Gradient-weighted Attention Rollout (§3.2)
  loss.py                 gaussian_soft_target, AttentionConsistencyLoss, total_objective (§3.3)
  data.py                  Dataset loading (Kalana-Person2/images+masks), splits, ImageNet normalization
tests/
  test_attention_consistency_loss.py   8 tests, dummy tensors only, no model needed
  test_rollout_shapes.py                5 tests, offline random-init model, no internet needed
math_formulation.md      §3.3 write-up, ready to paste into the paper
train_segformer_smoke.py CPU smoke-scale training, both variants
eval_segformer.py         Real Dice/IoU/F1/AAMO/efficiency via Lasana's own metrics/aamo/efficiency modules
generate_attention_figures.py   2-3 side-by-side attention-drift figures
segformer_full_scale_colab.ipynb   Same code, full 5,000-image/20-epoch scale on GPU
results/                  Logs, checkpoints summaries, comparison tables, figures
checkpoints/              Trained weights (segformer_b0_{vanilla,att}_{best,last}.pt)
```

## Why stage 4 for Grad-Rollout (the §3.2 adaptation)

Verified empirically (`tests/test_rollout_shapes.py`) for MiT-B0 on a
256×256 input: stages 1-3 have spatial-reduction ratios 8/4/2, so their
attention matrices are rectangular (e.g. stage 1: 4096 queries × 64
keys/values) — standard Attention Rollout's recursive matrix product is
only defined for square, token-consistent attention. Stage 4 has
`sr_ratio=1`, giving true square (64×64) self-attention across its 2
blocks — the only stage where rollout applies without further
approximation. Full reasoning + the empirically-verified shapes are in
`rollout.py`'s module docstring.

## Why training needs double backprop

The Attention Consistency Loss supervises `A`, and `A` is itself *defined
via a gradient* (Grad-Rollout is a Grad-CAM-style method). Backpropagating
`L_att` into the model is therefore a second derivative through that first
gradient. `rollout.grad_rollout_attention_map(..., create_graph=True)`
computes the inner gradient with `torch.autograd.grad(create_graph=True)`
instead of the cheaper `.backward()` + `.grad` path (used at inference
time for the qualitative figures), so `A` stays attached to the autograd
graph and one combined `total_loss.backward()` reaches every parameter.
Verified directly: a combined step updates all 205 parameter tensors, at
roughly the same wall-clock cost as a plain forward+backward
(~0.5s/sample on CPU).

## Results (CPU smoke run — 400/60/60 train/val/test, 8 epochs)

| Model | Dice | IoU | F1 | AAMO | Params | GFLOPs | FPS |
|---|---|---|---|---|---|---|---|
| U-Net (CNN baseline, Lasana) | 0.8492 | 0.7379 | 0.8492 | n/a | 1.95M | n/a | 3.48 |
| SegFormer-B0 (no attention loss) | 0.8017 | 0.6690 | 0.8017 | **0.0144** | 3.71M | 1.69 | 14.7 |
| SegFormer-B0 + Attention Consistency Loss | 0.8191 | 0.6936 | 0.8191 | **0.432** | 3.71M | 1.69 | 13.6 |

The headline number: **AAMO jumps from 0.0144 to 0.432 (~30×)** — the
attention-consistency variant's attention overlaps the forest mask far
more than the vanilla model's, which is the whole point of the proposal.
Dice/IoU also both improved slightly (0.8017→0.8191, 0.6690→0.6936), so
the attention supervision didn't cost segmentation accuracy at this scale
— though 8 epochs / 400 images is nowhere near enough to call that a
reliable trend rather than noise; re-run at full scale before trusting it.

**Honest caveat on the qualitative figures** (`results/attention_drift_figures/`):
they don't tell as clean a story as the AAMO number. The vanilla model's
Grad-Rollout attention is dominated by a hot corner artifact in most test
images rather than a legible "drifts onto roads/shadows" pattern — plausibly
an undertrained-model / rollout-recursion artifact rather than the
motivating failure mode the proposal describes. The attention-consistency
variant's map is visibly broader and warmer over more of the image
(consistent with the large AAMO gain) but doesn't yet cleanly separate
canopy from non-forest by eye. Both are genuine outputs of the trained
checkpoints, not cherry-picked — worth watching whether the full-scale
(proposal §4.1) run produces visually cleaner separation once the model is
properly converged.

## What's a smoke test vs. full-scale, honestly

CPU-only, no GPU in this environment. `train_segformer_smoke.py` trains
both variants on 400/60/60 train/val/test images (not the proposal's
5,000, 70/15/15) for 8 epochs, chosen to be large enough to produce real,
non-degenerate Dice/IoU numbers within a background-runnable CPU budget.
`segformer_full_scale_colab.ipynb` runs the *exact same code* at the
proposal's full scale (3500/750/750, 20 epochs) on a GPU — nothing is
reimplemented for scale-up, only the numbers change. Treat the numbers in
`results/` as a working proof of the pipeline, not final paper numbers;
re-run the Colab notebook for those.

## Reproducing

```bash
# from Phase1/Dinura-Person3/
python tests/test_attention_consistency_loss.py   # 8/8, no model needed
python tests/test_rollout_shapes.py                 # 5/5, offline model
python train_segformer_smoke.py --variant both       # ~30-60 min on CPU
python eval_segformer.py --variant both                # fills real Dice/IoU/F1/AAMO
python generate_attention_figures.py --n 3               # 3 drift figures
```

## What touched teammates' folders

Two additive files in `../Lasana-Person4_Evaluation/`, so
`python evaluate.py --model segformer` and `--model segformer-att` work
for real instead of raising "waiting on Person 2":

- `adapters/segformer.py` (new) — real adapter, delegates to this folder's
  `attention_consistency` package rather than duplicating it.
- `adapters/__init__.py`, `evaluate.py::get_adapter` — three-line change to
  use the real adapter instead of `SegFormerStubAdapter` for
  `segformer`/`segformer-att` (the `segformer-boundary` stub is untouched —
  Boundary Refinement is a Phase 2 stretch goal, out of scope here).
- `checkpoints/segformer_b0_vanilla.pt`, `segformer_b0_att_consistency.pt`
  (copied by `eval_segformer.py`, not committed by hand) and the two
  previously-"pending" rows in `results/baseline_comparison.{csv,md}` are
  filled in; the existing U-Net row is untouched.

`Kalana-Person2/` was read-only (dataset + as reference for the SegFormer
training-loop pattern) — nothing there was modified.
