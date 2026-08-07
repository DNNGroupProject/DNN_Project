# Person 5 — Boundary Refinement Module (Phase 2 stretch goal)

Proposal Section 3.4 / Section 6.2: if the core contribution (Sections
3.1–3.3, done in Phase 1) is validated with time remaining, add a Boundary
Refinement Module — boundaries extracted from both prediction and ground
truth via morphological gradient, supervised with a boundary Dice loss —
since most forest-segmentation errors occur near canopy edges.

## Why this is here, in Phase 2, before the rest of Phase 2 has started

Of Person 5's Phase 2 tasks (proposal §6.2.2), everything else — overseeing
Person 3's λ-sweep, managing the Person 3→Person 4 checkpoint handoff —
depends on training/ablation runs that haven't happened yet. This module
doesn't: like Person 3's Attention Consistency Loss in Phase 1, it's pure
tensor ops that can be built and unit-tested against dummy data before any
trained model exists. Built now, ahead of its proposal-scheduled Week 10
slot, so it's ready the moment Person 3's model is ready to integrate it.

## Layout

```
boundary_refinement/
  boundary_ops.py   morphological_gradient_boundary() — differentiable
                     dilation/erosion via max/min pooling (§3.4)
  loss.py             BoundaryDiceLoss, total_objective_with_boundary()
tests/
  test_boundary_refinement.py   10 tests, dummy tensors only, no model needed
```

## Design notes

- **Differentiable, not binary, morphology.** Classic dilation/erosion
  operate on binary images and aren't differentiable. `L_boundary` needs to
  backpropagate into the segmentation decoder through the *predicted* soft
  mask `P`, not just the hard ground-truth mask `Y`, so dilation/erosion
  are implemented as max/min pooling (`F.max_pool2d`, `-F.max_pool2d(-x)`)
  instead — differentiable via the pooling subgradient, and reduces to the
  standard binary morphological gradient when the input actually is binary
  (i.e. `Y`).
- **No dependency beyond `torch`.** Unlike `Phase1/Dinura-Person3/attention_consistency`
  (needs `transformers` for the SegFormer model itself), this module only
  needs plain tensor ops, so it's usable in any environment that has
  `torch` — verified in the `anaconda3/envs/ml` env, which has `torch`
  but not `transformers`.
- **Kept out of `Phase1/Dinura-Person3/`.** Per `CONTRIBUTING.md`'s
  cross-folder-edit rule, this doesn't touch Person 3's existing
  `loss.py`/`total_objective` — `total_objective_with_boundary` here is a
  drop-in replacement to swap in once the module is wired into the real
  training loop (Week 10), not an edit to Person 3's file.

## Reproducing

```bash
# from Phase2/Dhinanjaya-Person5/
python tests/test_boundary_refinement.py   # 10/10, no model needed
```

## What's NOT done yet (deliberately, per proposal §6.2's Week 10 slot)

- Not wired into Person 3's actual training loop — that requires Person
  3's model/checkpoint and is Person 3's/Person 5's joint integration work
  once Person 3's core loss tuning (Weeks 5–6) is done.
- `lambda3` (boundary-loss weight) is an untuned placeholder (0.2) — tune
  alongside `lambda1`/`lambda2` on the validation set once integrated.
- `kernel_size` (default 3, i.e. a ~1-pixel boundary band) not yet swept —
  worth trying 5/7 to see if a wider boundary band helps or just adds noise.
