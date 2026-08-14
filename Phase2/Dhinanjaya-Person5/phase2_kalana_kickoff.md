# Phase 2 kickoff — for Kalana (Person 2), 2026-08-15

## What the proposal actually assigns you

Section 6.2.2's Phase 2 work-distribution table gives Person 2 —
Transformer Lead a role that only covers **Weeks 5–6**:

> Support integration of the attention-extraction pipeline with Person 3's
> loss implementation.

Nothing is assigned to you for Weeks 7–12 — the Gantt chart (§6.2.3) shows
you blank from W7 onward. That leaves you with real open bandwidth, which
is why the task below is being handed to you rather than left as
"whoever has GPU time."

## Your actual Phase 2 task: the full-scale SegFormer run

`Phase1/Dinura-Person3/segformer_full_scale_colab.ipynb` has never been
executed — every code cell shows `execution_count: null`. Every number
currently in the paper for both SegFormer variants (the headline AAMO
jump 0.0144→0.432, the Dice/IoU deltas) is still an 8-epoch/400-image CPU
smoke run. This full-scale run (5,108 images, 20 epochs, GPU) is what
turns those into real numbers, and it's the single biggest thing blocking
the paper's headline results right now.

Full technical details, what the notebook already does, and one important
fix already applied are in `../../Phase1/Dhinanjaya-Person5/full_scale_segformer_todo.md`
— read that before starting. Short version:

- The notebook is complete and ready to run (mount Drive, train both
  variants, evaluate, generate figures, save back to Drive) — nothing to
  design, just execute on a GPU runtime.
- **Split algorithm fixed 2026-08-15** (commit `43d8480`): the notebook's
  `Args`/`EvalArgs` now use the real audited split (3576/766/766, not the
  proposal-rounded 3500/750/750), and `attention_consistency/data.py`'s
  shuffle algorithm was rewritten to match `Chanupa-Person1/dataset.py`'s
  exactly (verified: element-for-element identical test set at those
  counts). This means your run will land on the *same* held-out 766
  images the U-Net baseline and augmentation ablation already used — the
  apples-to-apples comparison the paper needs.
- Why you specifically: you already built and ran the closest equivalent
  — `Kalana-Person2/segformer_baseline_scratch_colab.ipynb` — same GPU/
  Colab setup, and (it turns out) the exact same split algorithm this
  notebook now also uses. Least new context to load of anyone on the team.

## One practical wrinkle

The notebook and its output paths (`results/`, `checkpoints/`) live under
`Dinura-Person3/`, not your own folder. You'll need access there (or a
Drive copy of that folder alongside your own `Kalana-Person2/images,masks`)
to run it and land results in the right place — see Step 0 of the
notebook. Dinura's been told you're picking this up
(`split_fix_notice_dinura.md`); flag it to Dhinanjaya if folder access is
awkward and we'll sort it out.

## After it's done

Ping Dhinanjaya — the paper's abstract, results table, Figure 2, and two
caveat paragraphs all currently exist specifically to flag these numbers
as smoke-scale-only, and need updating once real ones land. Full list in
`full_scale_segformer_todo.md`'s "After this is done" section.
