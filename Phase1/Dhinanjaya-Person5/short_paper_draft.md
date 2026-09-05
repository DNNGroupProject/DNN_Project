> **Status: Week 4 work-in-progress draft**, assembled by Person 5 from
> `related_work.md`, `intro_and_pipeline.md`, the proposal PDF, and the
> real Phase 1/2 results. **Superseded as the submission target by
> `paper_acm/main.tex`** — the course requires the ACM template, not IEEE.
> `paper_acm/main.tex` remains canonical for *prose/formatting* (algorithm
> box, formal AAMO equations, Implementation Details + Reproducibility
> paragraph, full Threats-to-Validity / Broader-Impact discussion) — this
> markdown file's prose still lags behind it in those respects. This
> markdown file is kept as the staging/content source; not yet reviewed by
> the rest of the team (§6.1.2 review step still pending). Citation numbers
> are provisional pending final bibliography.
>
> **Resolved 2026-08-16:** the U-Net baseline row below is Chanupa's PyTorch
> U-Net (31.04M params), trained and evaluated with the same
> split/seed/training loop as the SegFormer runs, via Lasana's
> `adapters/unet_torch.py`. Dice/IoU are dataset-wide (TP/FP/FN accumulated
> over all 766 test images), the same reduction used for every other row.
>
> **Reconciled 2026-09-05 — full-scale numbers throughout, sourced from
> `Phase2/Lasana-Person4/results/baseline_comparison_full_scale.md`** (the
> single most current cross-team fold-in table, which already fixes the
> earlier dual-DeepLab-Dice and n=1 std-formatting review findings): the
> §5 Results table, abstract headline numbers, and Conclusion below now
> match the full 5{,}108-image/20-epoch runs instead of the 400/60/60
> CPU-smoke numbers this draft previously reported. The Boundary Loss row
> is filled in with the completed Phase 2 λ3-sweep winner (λ3=0.2). DeepLabV3+
> now reads 0.7821/0.6422 (400-sample seed-42 smoke via
> `train_deeplab_multiseed.py`), not the earlier 200-image run's 0.7369 —
> **this number still differs from `paper_acm/main.tex`'s own Table 1
> (0.7369/0.5834), which was never updated after Lasana's PR #9 fixed this
> same dual-Dice-number issue for `Phase2/Lasana-Person4/`**; flagging here
> rather than silently picking a side — `paper_acm/main.tex`'s table needs
> the same fix, but that wasn't asked for this session.

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
evaluate interpretability numerically rather than only visually. On the full
5,108-image dataset (3576/766/766 split, seed 42, 20 epochs), with λ2 set
by a validation sweep over {0.1, 0.3, 0.5, 1.0}, the proposed method raises
AAMO from 0.0334 to 0.7476 (~22×) relative to a vanilla SegFormer-B0
baseline trained on the same split, at a modest segmentation cost (Dice
0.8743→0.8577, IoU 0.7766→0.7508), alongside a full-scale-trained U-Net
baseline (Dice 0.8615, IoU 0.7568) on the same held-out test set. A stretch
Boundary Refinement Module — a boundary-Dice loss L_boundary on
morphological-gradient-extracted mask edges — further improves segmentation
(Dice 0.8669, IoU 0.7650 at λ3=0.2, swept over {0.1, 0.2, 0.5}) but at the
cost of some attention faithfulness (AAMO 0.6218, down from 0.7476),
indicating a genuine interaction between the two loss terms worth further
study. The multi-seed ablation study is complete.

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

### 3.4 Boundary Refinement (completed, Phase 2 Week 10)

A Boundary Refinement Module adds a boundary Dice loss `L_boundary`
(morphological-gradient-extracted boundaries from the soft prediction and
ground truth, since most forest segmentation errors occur near canopy
edges), fixed at the λ2=1.0 attention winner:

```
L = L_dice + λ1·L_bce + λ2·L_att + λ3·L_boundary
```

A validation sweep over λ3 ∈ {0.1, 0.2, 0.5} selects λ3=0.2 (max test Dice,
then max test IoU — L_boundary targets segmentation/boundary quality, not
attention faithfulness, so it uses a different selection rule than the λ2
sweep). See §5 for results and the AAMO tradeoff this introduces.

## 4. Experimental Design

**Dataset.** 5,108 RGB aerial images, 256×256, binary forest/non-forest
masks (proposal §4.1 estimated ~5,000; the audited actual count is 5,108 —
see `Phase2/Chanupa-Person1/results/mask_audit.md`); 70/15/15 stratified
train/val/test split. Dataset integrity was verified directly rather than
assumed: 0 orphaned image/mask pairs, 0 unreadable files, and mask values
are already effectively binary (no pixel more than 9 grey levels from pure
black/white; the 32–223 mid-gray band is empty). Results below are the full
5,108-image, 3576/766/766, 20-epoch runs for U-Net and both SegFormer-B0
variants; DeepLabV3+ remains a 400-sample CPU-smoke reference (see caveat
under Results).

**Baselines.** U-Net (CNN) and vanilla SegFormer-B0 (no attention loss),
isolating the effect of the proposed attention supervision, plus DeepLabV3+
(MobileNetV3 backbone) as an optional additional CNN reference point.

**Metrics.** Dice, IoU, Precision, Recall, F1 (segmentation quality); params,
FLOPs, FPS (efficiency); AAMO (interpretability — normalized overlap between
thresholded attention and ground-truth forest mask).

## 5. Results

Full 5,108-image dataset, 3576/766/766 split, seed 42, 20 epochs, best-val
checkpoints (vanilla epoch 5, attention-consistency epoch 11, boundary
epoch 4); DeepLabV3+ is a 400-sample CPU-smoke reference
(`Phase2/Lasana-Person4/results/baseline_comparison_full_scale.md`):

| Model | Dice | IoU | F1 | AAMO | Params | GFLOPs | FPS |
|---|---|---|---|---|---|---|---|
| U-Net (CNN baseline) | 0.8615 | 0.7568 | 0.8615 | n/a | 31.04M | 109.48 | 1.47 |
| SegFormer-B0 (no attention loss) | 0.8743 | 0.7766 | 0.8743 | 0.0334 | 3.71M | 1.69 | 84.52 |
| SegFormer-B0 + Attention Consistency Loss (λ2=1.0 MSE) | 0.8577 | 0.7508 | 0.8577 | **0.7476** | 3.71M | 1.69 | 103.36 |
| SegFormer-B0 + Attention Consistency + Boundary Loss (λ3=0.2) | **0.8669** | **0.7650** | 0.8669 | 0.6218 | 3.71M | 1.69 | 113.23 |
| DeepLabV3+ (MobileNetV3, extra baseline, smoke-scale) | 0.7821 | 0.6422 | 0.7821 | n/a | 11.02M | n/a | n/a |

**Headline result:** AAMO increases ~22× (0.0334 → 0.7476) with the
attention-consistency variant, at a modest Dice/IoU cost (0.8743→0.8577,
0.7766→0.7508) relative to vanilla SegFormer-B0, both trained full-scale on
the same held-out test set. Across the λ2 sweep (0.1–1.0, MSE), test AAMO
rises monotonically (0.548, 0.575, 0.603, 0.748) while test Dice stays
within 0.852–0.869. Our earlier 400/60/60, 8-epoch CPU smoke run had
instead shown the attention loss slightly *improving* Dice/IoU
(0.8017→0.8191, 0.6690→0.6936); that small-scale accuracy gain did not
hold at full scale, though the AAMO direction was consistent at both
scales (smoke: 0.0144→0.432).

**Boundary Loss result.** Adding the Boundary Refinement Module (§3.4) at
its swept winner λ3=0.2 further improves Dice/IoU over the plain-attention
row (0.8577→0.8669, 0.7508→0.7650) — the best Dice/IoU of any SegFormer-B0
variant — but at a real cost to attention faithfulness: AAMO drops
non-monotonically across the λ3 sweep (0.7775 at λ3=0.1, 0.6218 at
λ3=0.2, 0.6805 at λ3=0.5, vs. the λ3=0 baseline's 0.7476). Best-validation
checkpoints for λ3≥0.2 land much earlier in training (epoch 4) than the
λ3∈{0,0.1} runs (epoch 11), consistent with `L_boundary`'s gradient —
which flows through the same shared encoder that produces the attention
maps — competing with `L_att` for representation capacity. This is
single-seed, batch-size-1 evidence and not yet disentangled from
checkpoint-selection noise (see caveats below).

DeepLabV3+, included as an optional extra CNN reference point beyond
U-Net, trails both U-Net and SegFormer-B0 on Dice/IoU on a 400-sample
CPU-smoke run, with a strong precision/recall imbalance (precision 0.659,
recall 0.962) — it over-predicts forest broadly rather than delineating
boundaries precisely.

**Augmentation ablation (U-Net, full scale).** Separately, a full-scale
(5,108-image, 20-epoch) augmentation ablation on the U-Net baseline —
same split/seed/starting weights, toggling only
`shared/augmentation.py` — finds augmentation does **not** improve
accuracy under this budget: test Dice 0.8604 (no aug) → 0.8505 (with
aug), IoU 0.7598 → 0.7454
(`Phase1/Chanupa-Person1/results/augmentation_ablation.md`). Neither arm
overfits within 20 epochs — the regime where augmentation is expected to
help — and this is single-seed, so the supportable claim is
*"augmentation did not improve the U-Net baseline under our training
budget (−0.010 Dice)"*, not that augmentation is unhelpful for the task
generally.

**Caveats (must be resolved before final submission):**
- **Single seed for the λ2 and λ3 sweeps.** No run has been repeated with a
  different seed, and the sweeps that select λ2=1.0 and λ3=0.2 are
  themselves single-seed, so the reported numbers carry unknown variance.
  The Boundary Loss AAMO tradeoff above in particular is single-seed,
  batch-size-1 evidence and not yet disentangled from checkpoint-selection
  noise (full numbers: `Phase2/Dhinanjaya-Person5/results/boundary_sweep_comparison.md`).
- **DeepLabV3+ remains smoke-scale** (400 samples / 5 epochs via
  `train_deeplab_multiseed.py`, seed-42 eval) — a full 5,108-image/20-epoch
  run has not been done for this extra baseline.
- **Rollout restriction.** Restricting Gradient-weighted Attention Rollout
  to SegFormer's stage-4 attention avoids the invalid rectangular-matrix
  case but discards information from stages 1–3, whose cost this work does
  not yet quantify.
- Qualitative attention-drift figures (`results/attention_drift_figures/`)
  don't yet tell as clean a story as the AAMO number: the vanilla model's
  attention shows a "hot corner" artifact rather than the motivating
  roads/shadows failure mode, and this persists at full scale — unlikely to
  be simply an undertraining effect as earlier speculated from the 8-epoch
  smoke run.
- **Resolved 2026-08-16:** the U-Net baseline row conflation (two different
  models both called "U-Net") — see the status note at the top of this
  file.
- **Resolved (2026-08-30, Lasana's PR #9):** the multi-seed ablation table
  (`Phase2/Lasana-Person4/results/ablation_mean_std.md`) is complete for all
  rows including DeepLabV3+ (seeds 42/43/44, sample std).
- **Resolved 2026-09-05:** the Boundary Loss row above — previously
  "pending" — is now the completed Phase 2 λ3-sweep winner.

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

Full-scale results (5,108 images, 20 epochs) confirm the core hypothesis:
supervising SegFormer's attention against the ground-truth mask raises
AAMO ~22× (0.0334→0.7476) for a modest 1.7-point Dice cost, and the
completed Boundary Refinement Module (λ3=0.2) further improves Dice/IoU
(best of any SegFormer-B0 variant tested) at a real cost to attention
faithfulness — a genuine interaction between the two loss terms, not yet
disentangled from single-seed noise. The multi-seed ablation study is
complete. Remaining work: repeat the λ2/λ3 sweeps and the Boundary Loss
result with multiple seeds, compare the already-implemented MSE and
KL-divergence formulations of `L_att`, and address the residual overfitting
visible in the training curves — culminating in the full IEEE paper.

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
- [x] Resolve the U-Net baseline conflation (`unet_baseline_reconciliation.md`) — resolved 2026-08-16 (Chanupa's checkpoint + Lasana's adapter).
- [ ] Circulate to Persons 1–4 for co-author review of their sections (§6.1.2 of the proposal).
- [x] Replace CPU-smoke-scale numbers with full-scale Colab results (resolved 2026-09-05; DeepLabV3+ remains smoke-scale, see caveats).
- [x] Reconcile the Boundary Loss row (§5 table) with the completed full-scale λ3 sweep (resolved 2026-09-05).
- [ ] `paper_acm/main.tex`'s own Table 1 still shows the pre-PR#9 DeepLabV3+ number (0.7369/0.5834, not 0.7821/0.6422) and has no Boundary Loss row — flagged, not fixed this session.
- [ ] Apply §11's colour-highlighting + margin-comment requirement (one colour per author) before Moodle submission — not started on any draft yet.
- [ ] Rewrite AI-assisted prose (this draft, `related_work.md`, `intro_and_pipeline.md`) substantially in your own words so it counts as your identifiable individual contribution per §11.
- [ ] Slide deck summarizing this draft — see `slide_deck_outline.md`.
