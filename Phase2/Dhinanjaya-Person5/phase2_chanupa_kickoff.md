# Phase 2 kickoff — for Chanupa (Person 1), 2026-08-15

## Update 2026-08-16 — FYI, full-scale SegFormer results landed in the paper

Kalana's full-scale run (`Phase2/Kalana-Person2/`) is merged into
`paper_acm/main.tex` (commit `0f08f39`, pushed to `main`): abstract, Table 1,
Figure 2, and the caveat paragraphs all now use the real 5,108-image/
20-epoch numbers instead of the smoke-scale ones. Checkpoints stayed on
Drive, not git, so no checkpoint-size issue came up — the flag below about
your fp16 fix wasn't needed this time, but still stands if a future
checkpoint does hit the 100 MiB wall. Nothing needed from you here.

## What the proposal actually assigns you

Section 6.2.2's Phase 2 work-distribution table gives Person 1 — Data &
Baselines a **light support role, Weeks 5–9**:

> Assist with data-related debugging and any additional
> augmentation/preprocessing needs as they arise.

That's it — no new core deliverable assigned to you in Phase 2. Your
Phase 1 scope (U-Net baseline, dataset audit, augmentation ablation — the
last two both extended-scope work you took on ahead of schedule) already
covered the substantial Person-1 contribution; Phase 2's core work
(Attention Consistency Loss tuning, full ablation study, AAMO) belongs to
Dinura and Lasana.

## What "as they arise" means in practice right now

Nothing is currently blocked on you. Flagging two things you're well
positioned for if they do come up, precisely because you already solved
adjacent problems:

- **Checkpoint size limits.** You already hit and solved GitHub's 100 MiB
  limit for `unet_baseline_best.pt` (selective fp16 casting that skips
  BatchNorm buffers — see `unet_baseline_reconciliation.md`). If the
  full-scale SegFormer checkpoints (Kalana's about to train these, see
  `full_scale_segformer_todo.md`) hit the same wall, you're the fastest
  person on the team to ask.
- **Augmentation.** `shared/augmentation.py` is your pipeline. If anyone
  wants the same with/without-augmentation ablation run on a SegFormer
  variant instead of just the U-Net, that's an extension of work you've
  already built and validated, not a new task from scratch.

Otherwise: this doc exists to close the loop on your Phase 2 role
formally (so the paper's contribution record stays accurate per the
course's §11 individual-contribution requirement), not to hand you new
work. If you want more to do, ping Dhinanjaya — there's real spare
capacity in your proposal-assigned Weeks 5–9 slot.
