# Person 5 — Phase 2 next steps (own tracking), 2026-08-16

Companion to the four kickoff docs sent to the rest of the team
(`phase2_chanupa_kickoff.md`, `phase2_kalana_kickoff.md`,
`phase2_dinura_kickoff.md`, `phase2_lasana_kickoff.md`). This is my own
Section 6.2.2 scope, with what's already done vs. still blocked.

## Already done, ahead of schedule

- [x] **KL-divergence variant of the attention loss.** Proposal assigns
      this to me ("implement the KL-divergence variant... so Person 3 can
      compare both formulations"). Dinura already built it themselves in
      Phase 1 (`attention_consistency/loss.py`, both MSE and KL, unit
      tested) — nothing left for me to implement here.
- [x] **Boundary Refinement Module built and unit-tested**
      (`Phase2/Dhinanjaya-Person5/boundary_refinement/`, 10/10 tests,
      dummy tensors only — no trained model needed for this part).
      Scheduled for Week 10 in the proposal; done early because it's pure
      tensor ops with no dependency on anyone else's unfinished work.
- [x] **U-Net baseline reconciliation** — Chanupa's checkpoint export +
      Lasana's PyTorch adapter, merged, paper updated (2026-08-16).

## Blocked — waiting on Dinura and/or Lasana

- [ ] **Technically oversee Dinura's λ-sweep** — review val Dice/IoU and
      L_att curves, sanity-check training behavior. Can't start until
      Dinura's runs exist (`phase2_dinura_kickoff.md` just sent).
- [ ] **Wire the Boundary Refinement Module into the real training loop.**
      Needs Dinura's tuned attention-consistency checkpoint as the base
      model to integrate `total_objective_with_boundary` into — the
      module itself doesn't need a model to unit-test, but integration
      does.
- [ ] **Tune `lambda3`** (boundary-loss weight, currently a 0.2
      placeholder) and sweep `kernel_size` (currently 3) once integrated.
- [ ] **Manage the Person 3 → Person 4 handoff** — confirm checkpoints,
      configs, and metric definitions (especially AAMO) are used
      consistently across every ablation row; resolve any
      training/evaluation code discrepancies. Needs both Dinura's
      checkpoint and Lasana actively running the ablation study.
- [ ] **Weeks 11–12: finish integrating the Boundary Refinement Module,
      hand the resulting checkpoint to Lasana** for the final ablation
      row.
- [ ] **Assemble the full paper** (results, discussion, conclusion)
      directly from Lasana's verified multi-seed ablation table;
      coordinate final figures; circulate for full co-author review
      before submission (course §6.1.2). This is the full IEEE-target
      paper, separate from the ACM short-paper deadline below.

## Not blocked on anyone — course requirements, short paper (due Aug 23)

Separate from the Phase 2 proposal scope above, but time-sensitive and
100% actionable right now:

- [ ] Colour-highlighting + margin-comment submission version (course
      §11, one colour per author) — zero progress so far.
- [ ] Substantial human rewrite of the AI-drafted paper prose, so it
      counts as my identifiable individual contribution per §11 — the
      prose volume keeps growing (augmentation-ablation paragraph,
      U-Net-row update, etc.) without this happening yet.
- [ ] Slide deck (outline exists in `slide_deck_outline.md`, no deck
      built).
- [ ] Verify the CCS concept IDs in `paper_acm/main.tex` against ACM's
      real generator (dl.acm.org/ccs) — currently best-effort guesses.
- [ ] Circulate the current draft to Persons 1–4 for co-author review of
      their own sections (course §6.1.2) — not yet done even once.
