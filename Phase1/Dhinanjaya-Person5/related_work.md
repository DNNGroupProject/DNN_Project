# 2. Related Work

Semantic segmentation of forest cover from aerial and satellite imagery has been
approached almost exclusively as an accuracy problem: better encoders, better
attention modules, better datasets. U-Net [9] remains the dominant convolutional
baseline for this task, and more recent forest-specific datasets such as the
WildUAV forest inspection benchmark [10] have focused effort on acquiring
larger and more varied labeled imagery (real and simulated, multiple altitudes
and illumination conditions) to improve segmentation accuracy on models such as
HRNet and PointFlow, without examining what the underlying network attends to
or why. Transformer-based segmentation models such as SegFormer [1] have since
been adopted in this domain for their ability to capture long-range spatial
dependencies, but the role attention plays inside these models — and whether it
is trustworthy — has received comparatively little scrutiny. The literature
relevant to this proposal splits into two distinct families, a distinction that
is central to positioning this project's contribution.

## 2.1 Attention-as-Architecture: Attention Used to Improve Accuracy

The first family treats attention purely as a feature-refinement mechanism
embedded inside the network, evaluated only by its effect on segmentation
accuracy. ForResANeXt [6] embeds lightweight attention in residual blocks with
attention-gated skip connections and a Focal Dice loss for forest/non-forest
segmentation in satellite imagery, but attention is never inspected as an
explanation — only as an accuracy lever. Attention-Refined PP-LiteSeg [7]
fuses multi-branch channel, spatial, and pixel attention for cross-regional UAV
forest segmentation across plantation and natural forests, improving
generalization but again without any interpretability supervision or
evaluation. Closer to the present domain, an attention-based U-Net for
deforestation mapping [8] adds standard attention gates to a U-Net trained on
Sentinel-2 imagery; the gates improve feature selection at the encoder-decoder
skip connections but are neither used nor measured as an explanation of the
model's decisions. Across all three works, attention maps are not supervised
during training and are not evaluated against any ground truth of what the
model *should* be attending to — they are an architectural convenience, not an
explainability artifact.

## 2.2 Attention-as-Explainability: Attention Used as a Supervised Signal

The second family treats attention as an interpretability output and
explicitly supervises it during training, but does so outside the remote-sensing
domain and without a quantitative attention-mask overlap metric. eX-ViT [3]
regularizes Vision Transformer attention maps with an attribute-guided
attention loss under weak supervision, but targets natural-image classification
rather than dense per-pixel segmentation. TransAttUnet [4] integrates
multi-level guided attention into a U-Net/Transformer hybrid for medical image
segmentation, which shares this proposal's goal of steering attention toward
task-relevant regions, but the medical-imaging domain does not surface the
canopy/road/shadow confusion characteristic of forest cover segmentation, nor
does it address the resulting class imbalance. Work on guiding attention in
end-to-end driving models [5] applies an explicit attention loss during
training to align attention with task-relevant regions of the driving scene,
demonstrating that attention supervision is viable as a training signal
outside classification, but again in a domain (autonomous driving perception)
with no segmentation-mask supervision and no interpretability metric analogous
to the one proposed here. Underpinning the attention-extraction side of all
of this family's work is Attention Rollout and its gradient-weighted variant
[2], introduced for standard multi-head self-attention in Vision Transformers;
this proposal adapts it for the first time to SegFormer's spatial-reduction
attention, since Rollout's assumption of full-resolution multi-head
self-attention at every stage does not hold for SegFormer's Efficient
Self-Attention.

## 2.3 Research Gap

No identified prior work combines (a) attention supervision as a training
signal, (b) the forest/non-forest remote-sensing domain, and (c) a
quantitative interpretability metric evaluated alongside standard segmentation
metrics. The attention-as-architecture literature ([6], [7], [8]) operates in
the correct domain but never supervises or measures attention as an
explanation. The attention-as-explainability literature ([3], [4], [5])
supervises attention as a training signal but never in the forest-segmentation
domain, and none of the three reports a quantitative attention-fidelity metric
comparable across model configurations. This proposal addresses that combined
gap: an Attention Consistency Loss that supervises SegFormer's internal
attention against the ground-truth forest mask, an adaptation of
Gradient-weighted Attention Rollout to SegFormer's spatial-reduction attention,
and a new quantitative metric — Average Attention-Mask Overlap (AAMO) — that
lets attention fidelity be reported and compared the same way Dice or IoU are.

---

### References (draft numbering — reconcile with full paper bibliography before submission)

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
