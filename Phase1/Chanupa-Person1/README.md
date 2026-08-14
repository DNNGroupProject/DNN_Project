# Person 1 — Data & Baselines (Phase 1)

Role: prepare the dataset and provide the CNN baseline the SegFormer work is
measured against. Proposal §6.1.2:

| Weeks | Deliverable | Status |
|---|---|---|
| 1–2 | Clean and split the dataset (train/val/test) | Done — split lives in the notebook, seed 42 |
| 1–2 | Shared augmentation pipeline | Done — [`shared/augmentation.py`](../../shared/augmentation.py) |
| 1–2 | Train the U-Net baseline to a working Dice/IoU | Done — results below |
| 3–4 | Qualitative dataset/result figures for the paper | Done — `prediction_grid.png`, `training_curves.png` |
| 3–4 | Augmentation ablation (with vs. without) | Harness done + smoke run; full-scale run still owed — see below |
| 3–4 | Support debugging as needed | Ongoing — continues in [`Phase2/Chanupa-Person1/`](../../Phase2/Chanupa-Person1/) |

## Layout

```
unet_baseline_colab.ipynb        U-Net from scratch, full dataset, 20 epochs (GPU/Colab)
unet_model.py                    UNet + dice_iou_score + load_unet, lifted out of the notebook
dataset.py                       ForestSegDataset + seeded split, ditto, plus the augment hook
augmentation_ablation.py         Trains with vs. without augmentation, writes the table
make_fixture_checkpoint.py       Writes the untrained loader-test checkpoint below
tests/test_augmentation.py       15 tests for shared/augmentation.py, synthetic arrays only
tests/test_unet_model.py         9 tests for the loader helpers and the fixture
checkpoints/                     unet_fixture_random.pt (untrained); the real .pt goes here too
test_metrics.txt                 Final test numbers + params/GFLOPs
training_log.csv                 Per-epoch train/val loss, Dice, IoU (20 rows)
training_curves.png              Loss and Dice/IoU curves
prediction_grid.png              Image / ground truth / prediction triptychs
results/                         The ablation table (.csv + .md) and its per-epoch log
```

`unet_model.py` and `dataset.py` are verbatim lifts of the notebook's Steps
1–4 — same model, same pairing, same seeded split (it reproduces 3576/766/766
exactly). They exist because a notebook cell isn't importable, and both the
ablation here and Person 4's future `adapters/unet_torch.py` need to build
this model outside Colab. Don't change them without changing the notebook.

## Reproducing

The notebook is Colab-only (it mounts Drive and expects a GPU runtime):

1. Upload a folder containing `images/` and `masks/` to Google Drive. The same
   5,108 pairs are committed at `Phase1/Kalana-Person2/{images,masks}`.
2. Open `unet_baseline_colab.ipynb`, set `DATA_ROOT` in Step 1 to that folder.
3. Runtime → Change runtime type → GPU (T4).
4. Run all. Step 1 asserts every mask has a matching image before training
   starts, so a bad upload fails fast rather than halfway through epoch 1.

Outputs land in `RESULTS_DIR` on Drive; the four result files above were copied
back into this folder from there.

The ablation and the tests run locally, no Drive and no GPU — they read the
5,108 pairs committed at `Phase1/Kalana-Person2/{images,masks}`:

```bash
python tests/test_augmentation.py && python tests/test_unet_model.py
```

```bash
python augmentation_ablation.py
```

Defaults are smoke scale (600 pairs, 8 epochs, a narrowed U-Net) so it
finishes on a laptop CPU in about half an hour. The full-scale run needs a
GPU — same script, from a Colab cell:

```bash
python augmentation_ablation.py --subset 0 --epochs 20 --features 64,128,256,512 --batch-size 8 --device cuda
```

The notebook is a deliberate structural clone of Kalana's SegFormer notebook —
same dataset class, same split (seed 42, 70/15/15), same loss, same metric,
same training loop. **Only the model differs.** That is what makes the Dice/IoU
directly comparable to the SegFormer runs, so don't "improve" one of those
shared pieces here without doing it in both.

## Results

Full dataset, 5,108 pairs, train/val/test = 3576/766/766, 20 epochs, batch 8,
lr 1e-3, 256×256, seed 42:

| Metric | Value |
|---|---|
| Best val Dice | 0.8661 |
| Test Dice | 0.8563 |
| Test IoU | 0.7534 |
| Test loss | 0.4302 |
| Params | 31.04 M |
| GFLOPs @ 1×3×256×256 | 109.48 |

Caveat: this is a full-scale GPU run on the complete dataset, not a smoke test
— but it is a **single seed**, so treat the third decimal as noise. Val Dice
bounces between 0.81 and 0.86 over the last few epochs (`training_log.csv`),
which is the level of run-to-run variation to expect.

### Note on which U-Net is which

`Phase1/Lasana-Person4_Evaluation/results/baseline_comparison.md` lists a row
"U-Net (CNN baseline)" at **1,951,105 params, Dice 0.8492**. That is *not* this
model — it's Lasana's Keras U-Net (`Lasana/checkpoints/lasana_unet_best.keras`,
loaded via `config.py:12-13`). This one is a PyTorch U-Net at **31.04 M params,
Dice 0.8563**.

Two different baselines are being reported under one label, and the 1.95M
number has already propagated into the short-paper draft. Whichever is meant to
be *the* U-Net baseline, the paper should say which, since a 16× parameter
difference matters to the efficiency argument. Flagging rather than editing —
that table is Person 4's.

## Augmentation ablation (Weeks 3–4)

`augmentation_ablation.py` trains the U-Net twice and reports the difference.
Same split, same seed, same starting weights, same loop — the only thing that
changes is whether `shared/augmentation.py` is applied to the training set.
Val and test are never augmented.

**Smoke scale only so far** — 600 pairs, 8 epochs, a narrowed U-Net
(16/32/64/128), on CPU. Full table in
[`results/augmentation_ablation.md`](results/augmentation_ablation.md):

| Arm | Best val Dice | Test Dice | Test IoU | Test loss |
|---|---|---|---|---|
| No augmentation | 0.8474 | 0.8091 | 0.6972 | 0.4854 |
| With augmentation | 0.8246 | 0.8008 | 0.6855 | 0.5146 |
| Delta (aug − none) | −0.0228 | −0.0083 | −0.0117 | +0.0292 |

**Augmentation came out worse here, and that is not yet an argument against
it.** The augmented arm is also behind on *training* Dice (0.8067 vs 0.8156)
and training loss (0.5172 vs 0.4977) — it is fitting the training set less
well, not generalising worse from an equal fit. That is the signature of a run
that hasn't converged, which is what you'd expect: augmentation is a
regulariser, it pays off by holding back overfitting, and at 8 epochs on 420
images there is no overfitting to hold back. The no-augmentation arm's val Dice
was still climbing at epoch 7. All the extra input variance can do at this
length is slow it down.

What the run does establish: the harness is correct and reproducible. Both arms
start from identical weights, the deltas are non-zero and stable across re-runs
at the same seed, and the tests confirm the transforms keep the mask aligned
and binary.

**The number for the paper is the full-scale run** — 5,108 pairs, 20 epochs,
the real 64/128/256/512 U-Net, on a GPU (the command is under "Reproducing").
Until that exists, don't quote the table above as the augmentation result, and
treat it as single-seed either way.

## Getting the weights out of Colab

`unet_baseline_reconciliation.md` asks me to commit this baseline's trained
weights, because right now they only exist in a Colab session and nothing
downstream can load them. `checkpoints/` is the landing spot. The steps:

1. Open `unet_baseline_colab.ipynb` in Colab. If the original session is still
   alive, jump to step 3 — Step 5 already saved the best checkpoint to
   `CHECKPOINT_OUT`, i.e. `/content/drive/MyDrive/unet_baseline.pt`.
2. If the session is gone: Runtime → Change runtime type → GPU (T4), set
   `DATA_ROOT`, Run all. Seed 42 and the split are fixed, so it reproduces the
   run above rather than making a new one.
3. **It won't fit in git as-is.** The file on Drive is 119 MiB (≈125 MB —
   31,037,698 params × 4 bytes) and GitHub hard-rejects anything over 100 MiB,
   so a straight commit bounces on push. Nothing else in the repo is close:
   every `.pt` under `Phase1/` is 14.9 MB, and the largest checkpoint anywhere
   is 23.6 MB. There is no `.gitattributes`, so no Git LFS either.
4. **Don't cast the whole `state_dict` to fp16.** The obvious fix costs 10
   Dice points. Casting only the bulk weight tensors costs nothing. Both
   measured on the full 766-image test set:

   | | Size | Test Dice | Test IoU |
   |---|---|---|---|
   | fp32 original | 124 MB | 0.8563 | 0.7534 |
   | blanket `{k: v.half()}` | 62 MB | **0.7510** | **0.6095** |
   | weights-only cast (below) | 60 MB | **0.8563** | **0.7534** |

   The damage is in the BatchNorm buffers, not the conv weights. fp16 tops
   out at 65504 and loses normals below ~6e-5, and BN divides by
   `sqrt(running_var + eps)` — so one buffer hitting `inf` or `0` wrecks that
   layer while every weight around it is fine. The U-Net has 18 BN layers.
   Skipping the small tensors leaves them at full precision, and since the
   conv weights are essentially all 31 M parameters the file barely grows:

   ```python
   sd = torch.load("/content/drive/MyDrive/unet_baseline.pt", map_location="cpu")
   out = {k: (v.half() if v.is_floating_point() and v.numel() >= 10_000 else v)
          for k, v in sd.items()}
   torch.save(out, "/content/drive/MyDrive/unet_baseline_best.pt")
   ```

   This is what `checkpoints/unet_baseline_best.pt` is. `load_unet()` casts it
   back to fp32 on the way in, so nothing downstream has to know.
5. Verify anything you regenerate, in Colab — on a laptop CPU the 766-image
   test pass takes over half an hour, on a T4 it takes 25 seconds. Load with
   `load_unet()`, run the test set, confirm **0.8563 / 0.7534**. The
   blanket-fp16 file looked perfectly healthy until it was measured.
6. The file lands at `checkpoints/unet_baseline_best.pt` — the
   `<model>_<variant>_{best,last}.pt` naming from
   [CONTRIBUTING.md](../../CONTRIBUTING.md). Keep the untouched fp32 copy on
   Drive (`unet_baseline_20ep.pt`); it is the only full-precision
   copy that exists. 60 MB is over GitHub's 50 MB *warning* threshold but
   well under the 100 MiB hard limit, so expect a warning on push, not a
   rejection — no Git LFS needed.

### Don't wait for it — there's a fixture

`checkpoints/unet_fixture_random.pt` (0.5 MB, committed) is an **untrained,
random-init** U-Net saved in exactly the format the real checkpoint uses: a
bare `state_dict`, same key names, same structure. It exists so
`adapters/unet_torch.py` can be written and tested today instead of waiting on
the decision above — every line of the loading path is exercisable against it,
and swapping in the trained file later changes nothing but the path.

**It produces noise, not predictions.** The guard against anyone mistaking it
for the baseline is structural rather than just a naming convention: the
fixture is (4,8,16,32) wide while the real model is (64,128,256,512), so code
that assumes the trained architecture hits a shape mismatch and fails loudly.
`tests/test_unet_model.py` pins that behaviour.

Load either file with the same call — it infers the width from the checkpoint
and casts any half-precision weights back to fp32, so whichever route the real
file ends up shipping by, this side doesn't change:

```python
from unet_model import load_unet

model = load_unet("checkpoints/unet_fixture_random.pt")   # or the real one
preds = model(batch).argmax(dim=1)                         # eval mode already set
```

Regenerate it with `python make_fixture_checkpoint.py` (seeded, so it rewrites
byte-identically and won't churn the diff).

### The contract

What Person 4 needs to write `adapters/unet_torch.py` against the real file:

| | |
|---|---|
| Architecture | `unet_model.UNet(in_channels=3, out_channels=2, features=(64,128,256,512))` |
| File contents | a plain `state_dict`, weight tensors fp16 / BN buffers fp32 — `load_unet()` handles the cast |
| Input | `1×3×256×256`, ImageNet-normalized, mask thresholded at >127 |
| Output | 2 logit channels; `argmax(dim=1)` for the forest mask |

## Dependencies on teammates

| Direction | What |
|---|---|
| I need | Nothing — the notebook is self-contained given the dataset. |
| I hand off | The U-Net baseline row (Dice/IoU/params/GFLOPs) to Person 4's comparison table, plus `unet_model.load_unet` and the fixture checkpoint so their adapter can be built and tested before the real weights land; `shared/augmentation.py` to Persons 2 and 3 if they want augmented runs; the dataset split convention to Persons 2 and 3. |

## What's still owed

**The full-scale augmentation ablation**, and nothing else. The harness and the
smoke run are done; the 20-epoch, full-dataset, both-arms run needs a GPU, and
that is the number the paper should quote rather than the smoke one above.

It doesn't touch the committed baseline numbers: `augment` defaults to off
everywhere, and the notebook was not modified.

While confirming the checkpoint, the full test pass was re-run through
`dataset.py` + `unet_model.py` rather than the notebook and reproduced
`loss=0.4302 dice=0.8563 iou=0.7534` exactly — so both the committed baseline
and the extracted modules are verified against each other.

## Cross-folder edits

Two additive changes outside this folder, both in `shared/` (the one place in
the repo meant to be imported by more than one person):

- **`shared/augmentation.py`** — new file, the Weeks 1–2 deliverable. Put here
  rather than in this folder because the proposal calls it the *shared*
  pipeline and `shared/` is the only importable package in the repo
  (`__init__.py`, no hyphen in the path); from `Phase1/Chanupa-Person1/` every
  teammate would need a `sys.path` insert. Nothing existing was modified.
- **`shared/README.md`** — added a section describing it, and retitled the file
  from "Shared experiment tracking" to "Shared utilities" since it now covers
  two things. The W&B content is unchanged.
