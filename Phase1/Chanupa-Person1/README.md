# Person 1 — Data & Baselines (Phase 1)

Role: prepare the dataset and provide the CNN baseline the SegFormer work is
measured against. Proposal §6.1.2:

| Weeks | Deliverable | Status |
|---|---|---|
| 1–2 | Clean and split the dataset (train/val/test) | Done — split lives in the notebook, seed 42 |
| 1–2 | Shared augmentation pipeline | **Not done** — see "What's missing" |
| 1–2 | Train the U-Net baseline to a working Dice/IoU | Done — results below |
| 3–4 | Qualitative dataset/result figures for the paper | Done — `prediction_grid.png`, `training_curves.png` |
| 3–4 | Augmentation ablation (with vs. without) | **Not done** — blocked on the pipeline above |
| 3–4 | Support debugging as needed | Ongoing — continues in [`Phase2/Chanupa-Person1/`](../../Phase2/Chanupa-Person1/) |

## Layout

```
unet_baseline_colab.ipynb   U-Net from scratch, full dataset, 20 epochs (GPU/Colab)
test_metrics.txt             Final test numbers + params/GFLOPs
training_log.csv             Per-epoch train/val loss, Dice, IoU (20 rows)
training_curves.png          Loss and Dice/IoU curves
prediction_grid.png          Image / ground truth / prediction triptychs
```

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

## Dependencies on teammates

| Direction | What |
|---|---|
| I need | Nothing — the notebook is self-contained given the dataset. |
| I hand off | The U-Net baseline row (Dice/IoU/params/GFLOPs) to Person 4's comparison table, and the dataset split convention to Persons 2 and 3. |

## What's missing

**No augmentation anywhere in this folder.** The notebook says so explicitly in
its header ("no augmentation"), and no Phase 1 PyTorch pipeline in the repo has
any — the only augmentation code that exists is the pre-Phase 1
`Lasana/train_lasana.py::augment_pair` (tf.image). Both the shared augmentation
pipeline (Weeks 1–2) and the with/without ablation (Weeks 3–4) are therefore
still outstanding. The ablation can't start until the pipeline exists, and
adding either now would change the baseline numbers above, so it needs to be a
deliberate re-run rather than a quiet edit.

## Cross-folder edits

None.
