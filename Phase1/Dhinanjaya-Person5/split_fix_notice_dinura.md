# For Dinura — split-algorithm fix in `attention_consistency/data.py`, 2026-08-15

## What was wrong

While writing up the outstanding full-scale-SegFormer-run task
(`full_scale_segformer_todo.md`), found that
`attention_consistency/data.py`'s `make_splits`/`list_pairs` used **NumPy's
`RandomState.shuffle` + sklearn's `train_test_split`** to build the
train/val/test split. `Chanupa-Person1/dataset.py` — which trained the
U-Net baseline and the augmentation ablation — uses a different algorithm
entirely: sort the mask filenames, `random.seed(42)` + stdlib
`random.shuffle`, then front-slice val/test/train off the shuffled list.

Same seed (42), same split sizes, but **a different RNG and a different
partitioning algorithm do not select the same images.** That meant the
full-scale SegFormer run, once you ran it, would have been evaluated on a
different 766-image test set than the U-Net/DeepLab/augmentation-ablation
rows already in the paper — quietly breaking the "every baseline is scored
on the same held-out data" comparison the paper leans on.

(Worth knowing: Kalana's own `segformer_baseline_scratch_colab.ipynb`
already uses the correct algorithm — same SEED/VAL_SPLIT/TEST_SPLIT/
shuffle-then-front-slice as Chanupa's `dataset.py`. Your `data.py` is the
one file that had drifted from that convention, not the team's split logic
in general.)

## What was fixed (commit `43d8480`, pushed to `main`)

- `attention_consistency/data.py`: `list_pairs` now sorts *mask* filenames
  (matching `dataset.py`'s source-of-truth list, not the image-file list),
  shuffles with stdlib `random` under the given seed, and takes the first
  `max_samples`. `make_splits` now front-slices val → test → train off that
  shuffled pool instead of calling `train_test_split` twice. Mask→image
  filename mapping (`_image_path_for`) now matches `dataset.py`'s
  `mask_name_to_image_name` exactly (`"_mask"→"_sat"` substring replace).
  Function signatures are unchanged — `list_pairs(max_samples, seed)`,
  `make_splits(n_train, n_val, n_test, seed)` returning the same
  `{"train": ..., "val": ..., "test": ...}` shape — so
  `train_segformer_smoke.py`, `eval_segformer.py`, and
  `generate_attention_figures.py` all keep working unmodified.
- `segformer_full_scale_colab.ipynb`: `Args`/`EvalArgs` changed from the
  proposal-rounded `3500, 750, 750` to the real audited split
  `3576, 766, 766` (5,108 total pairs × 70/15/15).

## Verification done

Wrote a standalone script (no repo dependencies) that runs both the old
Chanupa-style algorithm and the fixed `data.py` algorithm directly against
the real `Kalana-Person2/masks` directory (5,108 files) at
n_train=3576/n_val=766/n_test=766. Train, val, and test lists came back
**element-for-element identical** between the two. That's the actual claim
being made here — not just "same sizes," the literal same 766 images end
up in each split.

## What I could *not* verify

This machine's conda `ml` env is missing both `cv2` and `transformers`
(matches an earlier note about that env), so I couldn't import
`attention_consistency/data.py` end-to-end through `load_pairs`/
`to_model_input` — those two functions are untouched by this fix, so risk
should be low, but whoever runs the notebook should confirm the first real
run prints `len(train)/len(val)/len(test)` as exactly 3576/766/766 and that
the three sets don't overlap, before spending 20 epochs × 2 variants of GPU
time on it.

## Context — Kalana is picking up the actual run

You don't need to run `segformer_full_scale_colab.ipynb` yourself. Kalana
has open Phase 2 bandwidth (his proposal-assigned role only covers Weeks
5–6) and already has working full-scale Colab infrastructure for this
exact pipeline (`Kalana-Person2/segformer_baseline_scratch_colab.ipynb`),
so he's taking the full-scale run — see `full_scale_segformer_todo.md`'s
Owner section. The notebook and checkpoints still live under
`Dinura-Person3/`, so he'll need access there (or Drive copies) to run it
and land results back in the right place — flag if that's awkward and we
can adjust. This note stays here mainly so you know why `data.py` changed
under you and that the fix is verified, in case anything else in your
folder still imports it.
