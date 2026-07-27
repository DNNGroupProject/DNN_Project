# 3.3 Mathematical Formulation

*(Person 3 deliverable — ready to paste into the short paper / full paper §3.3. Implementation: `attention_consistency/loss.py`, `attention_consistency/rollout.py`.)*

Let the input image be `X ∈ R^(256×256×3)` with ground-truth binary mask `Y ∈ {0,1}^(256×256)`.

```
F = E(X),  F = {F1, F2, F3, F4}     (SegFormer encoder features, one per stage)
P = D(F),  P ∈ [0,1]^(256×256)      (segmentation decoder output)
A = Rollout(F), A ∈ [0,1]^(256×256) (adapted attention map)
A* = Gaussian(Y)                    (soft attention target)
```

**A — adapted attention map.** Computed by `rollout.grad_rollout_attention_map`:
restricted to encoder stage 4 (the only stage with spatial-reduction ratio
`sr=1`, so its attention matrix is square and token-consistent across
blocks — see `rollout.py`'s module docstring for the full architectural
justification), gradient-weighted per Chefer et al. (arXiv:2101.03919),
rolled out across the stage's 2 blocks, reduced to a per-token relevance
vector by row-mean (no CLS token in SegFormer to read off directly), then
upsampled from the 8×8 stage-4 grid to 256×256 and min-max normalized.

**A\* — soft attention target.** Computed by `loss.gaussian_soft_target`:
a 2D Gaussian blur (σ, tuned on the validation set — see below) applied to
the binary mask `Y`, then re-normalized to `[0,1]`. This softens the hard
mask boundary so the loss does not penalize attention for spilling a few
pixels into canopy edges — only for attending to clearly off-target
regions (roads, shadows, built structures).

### Loss terms

```
L_dice = 1 − (2·|P ∩ Y|) / (|P| + |Y|)
L_bce  = −Σ [Y·log P + (1−Y)·log(1−P)]
L_att  = MSE(A, A*)   or   KL(A ‖ A*)
```

Both `L_att` variants are implemented in `loss.AttentionConsistencyLoss`
(`mode="mse"` or `mode="kl"`), selectable per run. MSE penalizes pointwise
squared deviation between the raw attention and soft-target maps; the KL
variant first renormalizes both `A` and `A*` per-sample into probability
distributions over the image (`Σ = 1`) and measures `KL(A ‖ A*)`, which
penalizes attention mass placed where the soft target has near-zero
density more sharply than MSE does — useful if roads/shadows should be
suppressed more aggressively than MSE alone achieves.

### Total training objective

```
L = L_dice + λ1·L_bce + λ2·L_att
```

with initial values **λ1 = 1, λ2 = 0.3**, implemented in
`loss.total_objective`, to be tuned experimentally via validation-set
performance (per proposal §3.3 — this is the Phase 2 λ-sweep, Person 3's
weeks 5–6 task).

### Unit testing without a trained model

Per the Phase 1 plan, Person 3 has "no trained model to attach a loss to
yet" in weeks 1–2. Accordingly `tests/test_attention_consistency_loss.py`
exercises `gaussian_soft_target` and `AttentionConsistencyLoss` entirely
against synthetic dummy attention maps and masks — no SegFormer forward
pass required — checking:

1. `gaussian_soft_target` spreads mass around a mask's foreground and
   stays within `[0,1]`.
2. `L_att` is (near) zero when `A` already equals `A*`.
3. `L_att` is strictly larger when attention is placed *entirely off* the
   forest mask than when it is placed *on* the forest mask, for both MSE
   and KL modes — this is the property the whole loss exists to enforce
   (penalize attention drifting onto roads/shadows/buildings).
4. Gradients flow through `AttentionConsistencyLoss` back to `A` (needed
   since `A` depends on model parameters via Grad-Rollout).

`attention_consistency/rollout.py` and `hooks.py` (Person 2's task) are
additionally smoke-tested with a from-scratch (randomly initialized, no
internet required) SegFormer-B0 in `tests/test_rollout_shapes.py`, to lock
down tensor shapes independent of pretrained-weight downloads.
