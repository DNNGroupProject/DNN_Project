> **Status: Week 4 work-in-progress draft**, assembled by Person 5 from
> `related_work.md`, `intro_and_pipeline.md`, the proposal PDF, and the
> real Phase 1 results in `Phase1/Dinura-Person3/README.md` and
> `Phase1/Lasana-Person4_Evaluation/results/`. **Superseded as the
> submission target by `paper_acm/main.tex`** — the course requires the ACM
> template, not IEEE, and that version is already built and verified to
> compile (3 pages incl. references, within the 4-page limit). This
> markdown file is kept as the staging/content source; not yet reviewed by
> the rest of the team (§6.1.2 review step still pending). Citation numbers
> are provisional pending final bibliography.
>
> **Known unresolved issue:** the U-Net baseline row below conflates two
> different models (Lasana's Keras U-Net vs. Chanupa's PyTorch U-Net) — see
> `unet_baseline_reconciliation.md`. Not fixed in this draft yet; waiting on
> Chanupa (checkpoint) and Lasana (adapter + re-eval).

# Explainability-Guided SegFormer for Forest Cover Segmentation Using Attention Consistency Supervision

*[Author names / affiliations — placeholder, fill in before submission]*

## Abstract

Vision Transformer (ViT)-based segmentation models achieve strong accuracy
but their internal attention is typically used only for post-hoc
visualization, not as a supervised training signal. Prior work applying
attention supervision for interpretability has focused on medical imaging
and autonomous-driving domains, while forest/non-forest segmentation from
aerial and satellite imagery still relies on attention purely as an
architectural feature-refinement mechanism, with interpretability evaluated
qualitatively if at all. This paper presents an explainability-guided
SegFormer-B0 architecture that treats attention as a first-class training
target: an Attention Consistency Loss aligns the model's internal attention
with the ground-truth forest mask during training, encouraging the network
to focus on canopy regions rather than roads, shadows, or built structures.
Because SegFormer's encoder uses spatial-reduction attention rather than
standard multi-head self-attention, this work adapts Gradient-weighted
Attention Rollout to the SegFormer architecture, restricting it to the
final encoder stage where attention is square and token-consistent. A new
quantitative metric, Average Attention-Mask Overlap (AAMO), is introduced to
evaluate interpretability numerically rather than only visually. In
preliminary small-scale experiments (8-epoch, 400-image CPU runs), the
proposed method raises AAMO from 0.0144 to 0.432 relative to a vanilla
SegFormer-B0 baseline while also slightly improving Dice (0.8017→0.8191)
and IoU (0.6690→0.6936), against a U-Net baseline (Dice 0.8492). Full-scale
validation (5,108 images, complete ablation study) is ongoing (Phase 2).

## 1. Introduction

*(See `intro_and_pipeline.md` §1.1–1.3 for full text — summarized here.)*

Semantic segmentation of forest cover from aerial and satellite imagery
underpins ecological monitoring, deforestation tracking, and land-use
management. Transformer-based models such as SegFormer [1] have been
applied to this task for their ability to capture long-range spatial
dependencies, but — as Section 2 details — the great majority of
transformer segmentation work, including recent forest-specific studies
[6]–[8], uses attention purely as an architectural mechanism to improve
accuracy, evaluating explainability only through post-hoc qualitative
heatmap inspection, if at all.

**Problem statement.** Transformer attention in segmentation models is not
explicitly guided during training and frequently attends to task-irrelevant
regions — roads, shadows, buildings — rather than the class of interest,
reducing both accuracy at ambiguous boundaries and the trustworthiness of
the model's explanations. No existing work addresses this specifically for
forest/non-forest segmentation, nor evaluates attention faithfulness
quantitatively in this domain.

**Objectives.**
1. Design an Attention Consistency Loss supervising SegFormer-B0's internal
   attention against the ground-truth forest mask during training.
2. Adapt Gradient-weighted Attention Rollout to SegFormer's spatial-reduction
   attention.
3. Define and evaluate AAMO, a quantitative interpretability metric,
   alongside standard segmentation metrics.
4. Demonstrate that explainability-guided training improves both
   segmentation accuracy and attention faithfulness relative to a vanilla
   SegFormer-B0 baseline and a U-Net baseline.

## 2. Related Work

*(Full text in `related_work.md` — included verbatim below for draft
assembly; merge citation numbering with the final bibliography before
submission.)*

Forest and aerial segmentation research has focused on accuracy — better
encoders, attention modules, and datasets [1], [6]–[10] — with attention's
role as an explanation left largely unexamined. The relevant literature
splits into two families.

**2.1 Attention-as-Architecture.** ForResANeXt [6], Attention-Refined
PP-LiteSeg [7], and an attention-gated U-Net for deforestation mapping [8]
all embed attention purely as a feature-refinement mechanism, improving
accuracy without supervising or evaluating attention as an explanation.

**2.2 Attention-as-Explainability.** eX-ViT [3], TransAttUnet [4], and work
on guiding attention in end-to-end driving models [5] all supervise
attention as a training signal, but outside the forest/remote-sensing
domain and without a quantitative attention-mask overlap metric. This work
also builds directly on Chefer et al.'s Gradient-weighted Attention Rollout
[2] for extracting the attention map to supervise.

**2.3 Gap.** No prior work combines (a) attention supervision as a training
signal, (b) the forest/non-forest remote-sensing domain, and (c) a
quantitative interpretability metric evaluated alongside standard
segmentation metrics — the combined gap this paper addresses.

## 3. Proposed Methodology

*(Full pipeline diagram and implementation notes in `intro_and_pipeline.md`
§Pipeline.)*

### 3.1 Architecture Overview

The pipeline extends SegFormer-B0 with an explainability-supervision path
applied during training only; at inference the model runs as standard
SegFormer-B0 with negligible overhead:

- Input RGB image (256×256×3) → Patch Embedding → SegFormer Encoder (4 stages)
- Encoder features → MLP Segmentation Decoder → Predicted mask `P`
- Encoder attention (stage 4 only) → Adapted Grad-Rollout → Attention map `A`
- Ground-truth mask `Y` → Gaussian smoothing → Soft attention target `A*`
- `A` vs. `A*` → Attention Consistency Loss → combined with segmentation loss

### 3.2 Handling SegFormer's Spatial-Reduction Attention

Standard Attention Rollout assumes full multi-head self-attention across all
tokens. SegFormer's Efficient Self-Attention reduces keys/values by a
spatial-reduction ratio at earlier stages (sr = 8, 4, 2 for stages 1–3),
making those stages' attention matrices rectangular. Only stage 4
(`sr_ratio = 1`) has true square, token-consistent 64×64 attention — the
only stage where Rollout's recursive matrix product is valid without
further approximation. This restriction was verified empirically on
MiT-B0 at 256×256 input resolution (`rollout.py`, `test_rollout_shapes.py`)
and is the concrete implementation of the adaptation.

### 3.3 Mathematical Formulation

Let input `X ∈ R^(256×256×3)`, ground-truth binary mask `Y ∈ {0,1}^(256×256)`.

```
F = E(X),  F = {F1, F2, F3, F4}     (SegFormer encoder features)
P = D(F),  P ∈ [0,1]^(256×256)      (segmentation decoder output)
A = Rollout(F), A ∈ [0,1]^(256×256) (adapted attention map, stage 4 only)
A* = Gaussian(Y)                    (soft attention target, σ tuned on val)
```

Loss terms:

```
L_dice = 1 − (2|P ∩ Y|) / (|P| + |Y|)
L_bce  = −Σ [Y log P + (1−Y) log(1−P)]
L_att  = MSE(A, A*)  or  KL(A ‖ A*)
L = L_dice + λ1·L_bce + λ2·L_att        (λ1 = 1, λ2 = 0.3 initial)
```

Both `L_att` variants (MSE and KL) are implemented and unit-tested
(`attention_consistency/loss.py`); only MSE has been used for the reported
checkpoints so far. Training requires double backpropagation, since `A` is
itself defined via a gradient (Grad-Rollout is Grad-CAM-style) — verified to
update all model parameters in one combined `backward()` at roughly the
cost of a plain forward+backward pass.

### 3.4 Stretch Goal: Boundary Refinement

If the core contribution is validated with time remaining, a Boundary
Refinement Module will add a boundary Dice loss `L_boundary` (morphological-
gradient-extracted boundaries from prediction and ground truth), since most
forest segmentation errors occur near canopy edges. Scheduled as a Phase 2
Week 10 stretch goal, not a Phase 1 deliverable.

## 4. Experimental Design

**Dataset.** 5,108 RGB aerial images, 256×256, binary forest/non-forest
masks (proposal §4.1 estimated ~5,000; the audited actual count is 5,108 —
see `Phase2/Chanupa-Person1/results/mask_audit.md`); 70/15/15 stratified
train/val/test split. Dataset integrity was verified directly rather than
assumed: 0 orphaned image/mask pairs, 0 unreadable files, and mask values
are already effectively binary (no pixel more than 9 grey levels from pure
black/white; the 32–223 mid-gray band is empty). *Current Phase 1 results
below use a 400/60/60 CPU-smoke-scale subset, not the full 5,108 — see
caveat under Results.*

**Baselines.** U-Net (CNN) and vanilla SegFormer-B0 (no attention loss),
isolating the effect of the proposed attention supervision, plus DeepLabV3+
(MobileNetV3 backbone) as an optional additional CNN reference point.

**Metrics.** Dice, IoU, Precision, Recall, F1 (segmentation quality); params,
FLOPs, FPS (efficiency); AAMO (interpretability — normalized overlap between
thresholded attention and ground-truth forest mask).

## 5. Preliminary Results

CPU smoke run, 400/60/60 train/val/test images, 8 epochs
(`Phase1/Lasana-Person4_Evaluation/results/baseline_comparison.md`):

| Model | Dice | IoU | F1 | AAMO | Params | GFLOPs | FPS |
|---|---|---|---|---|---|---|---|
| U-Net (CNN baseline) | 0.8492 | 0.7379 | 0.8492 | n/a | 1.95M | n/a | 3.48 |
| DeepLabV3+ (MobileNetV3, extra baseline) | 0.7369 | 0.5834 | 0.7369 | n/a | 11.02M | n/a | 8.51 |
| SegFormer-B0 (no attention loss) | 0.8017 | 0.6690 | 0.8017 | 0.0144 | 3.71M | 1.69 | 12.45 |
| SegFormer-B0 + Attention Consistency Loss | 0.8191 | 0.6936 | 0.8191 | **0.432** | 3.71M | 1.69 | 14.55 |
| SegFormer-B0 + Attention Consistency + Boundary Loss | – | – | – | pending | – | – | – |

**Headline result:** AAMO increases ~30× (0.0144 → 0.432) with the
attention-consistency variant, while Dice/IoU also improve slightly over
vanilla SegFormer-B0 — the attention supervision does not appear to cost
segmentation accuracy at this scale. DeepLabV3+, included as an optional
extra CNN reference point beyond U-Net, trails both U-Net and SegFormer-B0
on Dice/IoU on a 200-image smoke run, with a strong precision/recall
imbalance (precision 0.592, recall 0.976) — it over-predicts forest broadly
rather than delineating boundaries precisely.

**Caveats (must be resolved before final submission):**
- 8 epochs / 400 images is a smoke-scale proof of pipeline correctness, not
  a reliable trend — re-run at the full 5,108-image/20-epoch scale
  (`segformer_full_scale_colab.ipynb`) before treating these numbers as
  final. Same caveat applies to DeepLabV3+'s 200-image smoke run.
- Qualitative attention-drift figures (`results/attention_drift_figures/`)
  don't yet tell as clean a story as the AAMO number: the vanilla model's
  attention shows a "hot corner" artifact rather than the motivating
  roads/shadows failure mode, plausibly from undertraining. Worth revisiting
  once the full-scale run converges properly.
- **Resolved 2026-08-09:** the multi-seed ablation table
  (`Phase1/Lasana-Person4_Evaluation/results/ablation_mean_std.md`) now
  matches the single-run numbers in `baseline_comparison.md` (Seeds: 1 for
  each trained model; only the boundary-loss row remains genuinely
  pending). Still only single-seed, though — true mean±std still needs
  multiple seeds per config, which is Phase 2 scope.
- **Unresolved:** the U-Net row above conflates two different models — see
  the status note at the top of this file and
  `unet_baseline_reconciliation.md`.

## 6. Expected Contributions

- An Attention Consistency Loss supervising transformer attention using
  segmentation ground truth, applied — to the best of the current
  literature review — for the first time in the forest/non-forest
  remote-sensing domain.
- An adaptation of Gradient-weighted Attention Rollout to SegFormer's
  spatial-reduction attention mechanism.
- AAMO, a quantitative interpretability metric usable alongside standard
  segmentation metrics in future forest-segmentation studies.
- An empirical comparison of accuracy and attention faithfulness between
  explainability-guided and accuracy-only training.

## 7. Conclusion and Next Steps

Preliminary small-scale results support the core hypothesis: supervising
SegFormer's attention against the ground-truth mask substantially improves
attention-mask overlap without costing segmentation accuracy. Phase 2
(Weeks 5–12) will validate this at full scale — tuning λ via a sweep,
comparing the already-implemented MSE and KL-divergence formulations of
`L_att`, running the complete multi-seed ablation study, and, time
permitting, adding the Boundary Refinement Module — culminating in the full
IEEE paper.

## References

*(See `related_work.md` for the full annotated list; numbering below matches it.)*

[1] E. Xie, W. Wang, Z. Yu, A. Anandkumar, J. M. Alvarez, and P. Luo,
"SegFormer: Simple and Efficient Design for Semantic Segmentation with
Transformers," *NeurIPS*, 2021.
[2] H. Chefer, S. Gur, and L. Wolf, "Transformer Interpretability Beyond
Attention Visualization," *arXiv:2101.03919*, 2021.
[3] Y. Zhang et al., "eX-ViT: A Novel eXplainable Vision Transformer for
Weakly Supervised Semantic Segmentation," *Pattern Recognition*, 2023.
[4] B. Chen et al., "TransAttUnet: Multi-level Attention-guided U-Net with
Transformer for Medical Image Segmentation," *arXiv:2107.05274*, 2022.
[5] "Guiding Attention in End-to-End Driving Models," *arXiv:2405.00242*, 2024.
[6] "ForResANeXt: Forest/non-forest segmentation with aggregated residual
attention network in satellite imagery," *ScienceDirect*, 2026.
[7] "Enhancing Cross-Regional Generalization in UAV Forest Segmentation
Across Plantation and Natural Forests with Attention-Refined PP-LiteSeg
Networks," *Remote Sensing*, 2026.
[8] "An attention-based U-Net for detecting deforestation within satellite
sensor imagery," *ScienceDirect*, 2022.
[9] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks
for Biomedical Image Segmentation," *MICCAI*, 2015.
[10] B.-C.-Z. Blaga and S. Nedevschi, "Forest Inspection Dataset for Aerial
Semantic Segmentation and Depth Estimation," *arXiv:2403.06621*, 2024.

---

## Open items before this is submission-ready

- [x] Port into the course-required ACM template (not IEEE — see `paper_acm/main.tex`, compiles at 3 pages incl. references).
- [x] Author names, affiliations — filled in `paper_acm/main.tex` (real names, no registration numbers required for the short-paper phase per the course spec).
- [x] Reconcile `baseline_comparison.md` vs. `ablation_mean_std.md` (see §5 caveat — resolved 2026-08-09).
- [ ] Circulate to Persons 1–4 for co-author review of their sections (§6.1.2 of the proposal).
- [ ] Resolve the U-Net baseline conflation (`unet_baseline_reconciliation.md`) — blocked on Chanupa (checkpoint export) and Lasana (adapter + re-eval).
- [ ] Replace CPU-smoke-scale numbers with full-scale Colab results once available.
- [ ] Apply §11's colour-highlighting + margin-comment requirement (one colour per author) before Moodle submission — not started on any draft yet.
- [ ] Rewrite AI-assisted prose (this draft, `related_work.md`, `intro_and_pipeline.md`) substantially in your own words so it counts as your identifiable individual contribution per §11.
- [ ] Slide deck summarizing this draft — see `slide_deck_outline.md`.
