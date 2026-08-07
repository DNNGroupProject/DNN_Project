# Week 4 Slide Deck Outline

Draft outline for the Phase 1 short-paper presentation. Content pulled from
`short_paper_draft.md` — build slides from that once it's reviewed. Aim for
~10-12 slides, one idea per slide.

1. **Title** — "Explainability-Guided SegFormer for Forest Cover
   Segmentation Using Attention Consistency Supervision." Team names/roles.

2. **Motivation** — Forest segmentation matters (deforestation tracking,
   ecological monitoring). Transformer attention is usually only visualized
   after training, never supervised. One sentence: "what if attention was a
   training target, not just a picture?"

3. **Problem statement** — Attention drifts onto roads/shadows/buildings
   instead of canopy → hurts both boundary accuracy and trust in
   explanations. No prior forest-segmentation work supervises or measures
   this quantitatively.

4. **Related work, two buckets** (one slide, table or two-column layout) —
   Attention-as-architecture (accuracy only) vs. attention-as-explainability
   (supervised, but wrong domain) → the gap this project fills. Pull
   directly from `related_work.md` §2.1/2.2 tables.

5. **Contribution summary** — 3 bullets: Attention Consistency Loss,
   SegFormer-adapted Grad-Rollout, AAMO metric.

6. **Pipeline diagram** — the mermaid flowchart from `intro_and_pipeline.md`
   (image → encoder → decoder → P, and encoder stage 4 → rollout → A,
   mask → gaussian → A*, both feeding the combined loss). This is the
   single most important slide — spend the most prep time getting the
   diagram legible.

7. **Why only stage 4?** — One slide, the sr_ratio square-attention argument
   (table of stage shapes: 4096×64, 1024×64, 256×64, 64×64 — only the last
   is square). Keep it visual/short; full reasoning is in the paper, not
   the talk.

8. **Math formulation** — `L = L_dice + λ1·L_bce + λ2·L_att`, `A* =
   Gaussian(Y)`, `L_att = MSE(A, A*)` or `KL(A‖A*)`. Don't over-explain;
   this audience has seen the proposal.

9. **Preliminary results table** — the baseline comparison table from
   §5 of the short paper (U-Net / vanilla SegFormer / +Attention
   Consistency). Bold the AAMO jump (0.0144 → 0.432) as the headline
   number.

10. **Qualitative attention-drift figures** — 2-3 side-by-side images from
    `Phase1/Dinura-Person3/results/attention_drift_figures/`. State the
    honest caveat live rather than let someone ask: vanilla model's map
    shows a hot-corner artifact, not yet the clean roads/shadows story —
    likely undertraining at this scale, revisit after the full-scale run.

11. **Honest scale caveat** — one slide, explicit: numbers so far are an
    8-epoch/400-image CPU smoke test proving the pipeline works end-to-end,
    not final results. Full 5,000-image/20-epoch GPU run is the real
    number to trust — cite `segformer_full_scale_colab.ipynb`.

12. **Next steps (Phase 2)** — λ-sweep, MSE vs. KL comparison (already
    implemented, not yet run), full multi-seed ablation study, optional
    Boundary Refinement Module, full IEEE paper.

## Presentation notes

- Whoever presents should be ready to explain the double-backprop point
  (§3.3 note) only if asked — it's an implementation detail, not something
  to preempt in the talk.
- Keep slide 6 (pipeline diagram) and slide 9 (results table) on screen
  longest; those are what reviewers will remember.
- Don't present the ablation_mean_std.md numbers as-is — they're still a
  stale scaffold (0 seeds recorded). Use `baseline_comparison.md`'s numbers
  instead, and say "single run" out loud rather than implying multi-seed
  results exist yet.
