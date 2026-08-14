# For Kalana — FYI on a split-consistency fix, no action needed, 2026-08-15

## Short version

While auditing the outstanding full-scale SegFormer run
(`full_scale_segformer_todo.md`), found that Dinura's
`attention_consistency/data.py` (used by `train_segformer_smoke.py`/
`eval_segformer.py`/the full-scale Colab notebook) built its train/val/test
split with a different shuffle algorithm than Chanupa's `dataset.py` (the
U-Net baseline and augmentation ablation) — same seed, same split sizes,
but a different RNG, so it would have quietly scored SegFormer on a
different held-out test set than the other baselines in the paper. Fixed
in commit `43d8480` — full writeup in `split_fix_notice_dinura.md` if you
want the details.

## Why you're getting this note

Checked your two notebooks against the same question and **neither needed
any change**:

- `segformer_baseline_scratch_colab.ipynb`'s `make_splits()` (cell 10:
  `SEED=42`, `VAL_SPLIT=TEST_SPLIT=0.15`, sort mask files, stdlib
  `random.shuffle`, front-slice val→test→train) is already the *exact*
  algorithm Chanupa's `dataset.py` uses. If anything, this looks like the
  original reference implementation the rest of the team converged on —
  Dinura's copy of it in `attention_consistency/data.py` is what had
  drifted (NumPy `RandomState` + sklearn `train_test_split` instead), not
  yours.
- `segformer_attention.ipynb` defines its own local `list_mask_files()` /
  `mask_name_to_image_name()` helpers rather than importing Dinura's
  `data.py`, so it was never exposed to the mismatched algorithm either.

So: nothing broken on your end, nothing to fix. Flagging only because your
scratch SegFormer run (`segformer_b0_scratch.pt`, Table 2 results) is a
third real SegFormer training pass we have sitting around — if it's ever
worth pulling into the paper's comparison table alongside the
attention-consistency variants, this note is the reason its test split
should already line up with the U-Net baseline's, since it's built the
same way.
