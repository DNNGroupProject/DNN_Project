# Integration review — Person 2 ↔ Person 3 attention/loss pipeline

Technical oversight pass (Person 5 Wk2–3 task) checking that Person 2's
attention-extraction pipeline and Person 3's loss implementation are
consistent with each other and with proposal §3.2/§3.3.

## What was checked

- `Phase1/Kalana-Person2/segformer_attention.ipynb` (Person 2, original
  `AttentionExtractor` + `grad_rollout_attention_map` implementation, Task 1)
  vs. `Phase1/Dinura-Person3/attention_consistency/hooks.py` and `rollout.py`
  (the copy integrated into Person 3's training pipeline).
- `Phase1/Dinura-Person3/attention_consistency/loss.py` vs.
  `Phase1/Dinura-Person3/math_formulation.md` vs. proposal §3.3.
- `Phase1/Lasana-Person4_Evaluation/aamo.py` vs. the `A` definition Rollout
  produces, to confirm AAMO consumes attention maps in the range/format
  Rollout actually emits.

## Findings

**Consistent — no action needed.**

1. **Attention extraction/rollout**: Person 3's `hooks.py`/`rollout.py` is a
   direct carry-over of Person 2's notebook implementation (same
   `stage_index=4` restriction, same `MIT_B0_STAGE_CONFIG` reasoning, same
   Grad-Rollout recursion and row-mean relevance readout). No divergent
   second implementation exists — good, this was the main integration risk
   (two people independently reimplementing §3.2 slightly differently).
2. **Attention map contract**: `grad_rollout_attention_map` returns
   `A ∈ [0,1]^(256×256)` (min-max normalized by max after clamping to
   ≥0, upsampled bilinearly from the 8×8 stage-4 grid). `loss.py`'s
   `AttentionConsistencyLoss.forward` and `aamo.compute_aamo` both consume
   that shape/range correctly — `aamo.normalize_attention` re-normalizes
   defensively even though Rollout's output should already be in range, so
   there's no silent mismatch if that changes.
3. **Loss formulation**: `loss.py` implements exactly `L = L_dice + λ1·L_bce
   + λ2·L_att` with `λ1=1, λ2=0.3` defaults and `L_att` as MSE or KL,
   matching `math_formulation.md` and proposal §3.3 term-for-term.
4. **Double backprop**: `rollout.py`'s `create_graph=True` path is used
   correctly by the training loop to keep `A` differentiable w.r.t. model
   parameters for the combined loss — documented and verified (per Person
   3's README) to update all parameter tensors in one `backward()`.

## Note for Phase 2 planning

Person 3 already implemented the **KL-divergence variant** of `L_att`
(`AttentionConsistencyLoss(mode="kl")`) in Phase 1, ahead of the proposal's
Phase 2 Week 5–6 assignment (originally scoped as a Person 5 task — "implement
the KL-divergence variant... so Person 3 can compare both formulations").
Only MSE has been used for the reported checkpoints/results so far
(`results/train_summary_att.json`); the KL path is implemented and unit-
tested but not yet run. Phase 2's λ-sweep should include an MSE-vs-KL
comparison using the existing `mode="kl"` path rather than reimplementing it.
