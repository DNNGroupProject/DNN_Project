# Phase 2 kickoff — for Lasana (Person 4), 2026-08-16

## Update 2026-08-16 — Kalana's results landed, paper's Table 1 updated

Kalana's full-scale SegFormer run (`Phase2/Kalana-Person2/results/`) is in
now — `baseline_comparison.{csv,md}` there has both variants (SegFormer-B0:
Dice 0.8743/IoU 0.7766/AAMO 0.0334; +Attention: Dice 0.8690/IoU 0.7684/AAMO
0.5752), and I've already folded those rows into `paper_acm/main.tex`
Table 1 directly (commit `0f08f39`) since it was blocking the paper text.
Your checklist item "fold the SegFormer-B0/+Attention rows into
`baseline_comparison.md` alongside your U-Net row" is still worth doing for
`Phase1/Lasana-Person4_Evaluation/results/baseline_comparison.md` /
`ablation_mean_std.md` specifically (Kalana's file is in his own Phase 2
folder, not yours) — that's the one your multi-seed ablation table builds
on. Everything's still single-seed, so the Week 10 multi-seed task is
unblocked on the SegFormer side now (still waiting on Dinura's λ-sweep for
the tuned attention config).

## Where you actually stand

Your Phase 1 work is done and merged: `adapters/unet_torch.py`, the
re-pointed `config.py`/`evaluate.py`, and the regenerated
`baseline_comparison.md` (U-Net now Dice 0.8615/IoU 0.7568, Chanupa's
PyTorch checkpoint, dataset-wide reduction) landed on `main` 2026-08-16.
The paper's U-Net row and efficiency prose are already updated to match —
nothing further needed there.

## What the proposal assigns you for Phase 2

Section 6.2.2, Person 4 — Evaluation Lead, Week 10:

> Implement the AAMO metric; run the complete ablation study across all
> model configurations; report results with multiple seeds (mean ± std).

The AAMO half is already done — you built `aamo.py` in Phase 1 with 10
passing unit tests. What's left is the ablation study itself, and it's
currently blocked, not idle-blocked-on-nothing:

- **Kalana** is running the full-scale SegFormer training on Colab right
  now (vanilla + attention, fixed λ2=0.3) — not done yet.
- **Dinura** hasn't started their λ2-sweep yet (`phase2_dinura_kickoff.md`,
  sent today) — the "real" tuned attention-consistency checkpoint your
  ablation study should use comes from that, not necessarily Kalana's
  default-hyperparameter run.

Nothing to do on the ablation study until at least Kalana's run lands.

## What "all model configurations" means here

Matching the table already in the paper (`baseline_comparison.md` /
`paper_acm/main.tex` Table 1):

| Config | Status |
|---|---|
| U-Net (CNN) | Done (yours, merged) |
| DeepLabV3+ (extra baseline) | Done |
| SegFormer-B0 (vanilla) | Training now (Kalana, Colab) |
| SegFormer-B0 + Attention Consistency | Training now (Kalana, default λ2); may be superseded by Dinura's tuned config |
| SegFormer-B0 + Attention Consistency + Boundary Loss | Blocked — needs my Boundary Refinement Module wired into training first |

## The multi-seed part is the actual new work

Everything currently in the repo — U-Net, DeepLabV3+, both SegFormer
variants — is single-seed. That's an explicit Threats-to-Validity item in
the paper right now ("no run has been repeated with a different seed").
Your Week 10 job is what closes it: once checkpoints exist for a config,
re-run training/eval across 2–3 seeds and report mean ± std, updating
`results/ablation_mean_std.md` (currently reconciled with
`baseline_comparison.md` but only at "Seeds: 1" for everything except the
still-pending Boundary Loss row).

## Checklist

- [ ] Nothing to run yet — waiting on Kalana's Colab run (in progress)
      and Dinura's λ-sweep (not started)
- [ ] Once Kalana's results land: fold the SegFormer-B0 / +Attention
      full-scale rows into `baseline_comparison.md` alongside your
      already-merged U-Net row
- [ ] Once Dinura's tuned config lands: decide (with Dhinanjaya) whether
      it replaces or supplements Kalana's default-λ2 attention row
- [ ] Re-run eval across ≥2–3 seeds per config that's ready, report
      mean ± std, update `results/ablation_mean_std.md`
- [ ] Ping Dhinanjaya once the multi-seed table is ready — his Week 11–12
      task ("assemble the full paper directly from Person 4's verified
      ablation table") depends on this being final
