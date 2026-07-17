# Post-Processing & Refinement — Complete Research Study Guide

**Team Member 4 · Refinement & Ensembling**  
**Project:** Binary forest / non-forest semantic segmentation  
**Baseline:** Lasana U-Net (sigmoid probability map → threshold 0.5)  
**Source plan:** `architecture_improvements.docx` §4  

This document is written so you can **learn the science**, then implement. Read sections in order. Each major topic has: intuition → math → forest use → what to implement → papers.

---

# Part 0 — The big picture

## 0.1 What problem do you own?

Your teammates improve the **network** (architecture, loss, data, multi-scale).  
You improve what happens **after** the network speaks.

```text
Image x  →  U-Net  →  P(x) ∈ [0,1]^{H×W}   ← probability that pixel is forest
                         │
                         ▼
              ★ YOUR RESEARCH LAYER ★
         threshold · morph · CRF · TTA · ensemble
         uncertainty · LBR-Net · prune/quantize
                         │
                         ▼
              Ŷ mask + optional uncertainty U(x)
```

**Research question**

> Given a trained segmenter, which post-processing / ensembling / uncertainty methods improve forest masks the most, how trustworthy are the probabilities, and can a tiny learnable refiner beat classical methods with little latency?

## 0.2 Why forests need this

| Failure mode | Why it happens | What can fix it |
|--------------|----------------|-----------------|
| Jagged canopy edges | CNN receptive field + downsample | CRF, active contours, LBR-Net |
| Speckles / holes | Local noise, JPEG mask artifacts | Morphology, TTA, ensemble |
| Wrong operating point | Fixed thr=0.5 | Threshold sweep on val IoU |
| “Looks confident but wrong” | Poor calibration | Temperature scaling, ECE study |
| Slow deployment | Large float32 model | Prune / quantize |

## 0.3 Lasana starting point (known numbers)

On a CPU subset run (~1200 images): test **accuracy 81.1%**, **IoU 0.735**, **Dice 0.768**.  
Your job: raise IoU/Dice further **without** retraining a huge new backbone (unless you ensemble several).

---

# Part 1 — Prerequisites you must understand first

## 1.1 Probability map vs hard mask

U-Net ends with **sigmoid**:

\[
P_{ij} = \sigma(z_{ij}) = \frac{1}{1+e^{-z_{ij}}} \in (0,1)
\]

Hard mask (current Lasana):

\[
\hat{Y}_{ij} = \mathbf{1}[P_{ij} > 0.5]
\]

**Insight:** 0.5 is arbitrary. The best threshold for **IoU** is often ≠ 0.5 when classes are imbalanced or the model is biased.

## 1.2 IoU and Dice (refresh)

For binary masks (TP, FP, FN = pixel counts):

\[
\mathrm{IoU} = \frac{TP}{TP+FP+FN}, \qquad
\mathrm{Dice} = \frac{2\,TP}{2\,TP+FP+FN}
\]

Relation: \(\mathrm{Dice} = \frac{2\,\mathrm{IoU}}{1+\mathrm{IoU}}\).

**Pixel accuracy** can look high while IoU is mediocre. Always report **IoU / Dice** as primary.

## 1.3 Post-processing improvement percentage

\[
\Delta\mathrm{IoU}\% = \frac{\mathrm{IoU}_{\text{method}} - \mathrm{IoU}_{\text{baseline}}}{\mathrm{IoU}_{\text{baseline}}} \times 100
\]

Example: baseline IoU 0.735 → method 0.760 → ΔIoU% ≈ **+3.4%**.

Also always report **absolute** ΔIoU and **extra milliseconds**.

---

# Part 2 — Method 1: Threshold tuning + morphology

## 2.1 Threshold sweep

**Idea:** On **validation** only, try many thresholds; pick \(t^\*\) that maximizes IoU; freeze \(t^\*\); evaluate once on test.

```text
for t in [0.30, 0.35, …, 0.70]:
    mask = (P_val > t)
    score = IoU(mask, Y_val)
pick t* = argmax score
```

**Why it works:** If the model systematically under-predicts forest, \(t^\* < 0.5\) recovers recall; if it over-predicts, \(t^\* > 0.5\) cuts false positives.

**Learn:** precision–recall curve; IoU vs threshold plot (put this figure in your paper).

## 2.2 Morphological cleanup

Operate on the **binary** mask:

| Op | Effect on forest mask |
|----|------------------------|
| **Erosion** | Shrink forest; remove thin protrusions |
| **Dilation** | Expand forest; fill gaps |
| **Opening** = erode→dilate | Remove small FP blobs |
| **Closing** = dilate→erode | Fill small holes inside forest |
| **Remove small components** | Drop connected regions with area < N |

**Forest tip:** Prefer small kernels (3×3 or 5×5). Large kernels destroy thin tree corridors.

**Implement:** `cv2.morphologyEx`, `skimage.morphology.remove_small_objects`.

**Expected gain:** Often +0.3–1.5 IoU if the raw mask is noisy; sometimes ~0 if U-Net already clean.

---

# Part 3 — Method 2: Test-Time Augmentation (TTA)

## 3.1 Intuition

Training used flips/rotations. At test time, the model may still be slightly asymmetric.  
**TTA** = run the same weights on several transformed views, map predictions back, average.

## 3.2 Formal recipe

Let \(\mathcal{T} = \{T_1,\ldots,T_K\}\) be transforms with inverses \(T_k^{-1}\).

\[
P_{\mathrm{TTA}}(x) = \frac{1}{K}\sum_{k=1}^{K} T_k^{-1}\big(\mathrm{Model}(T_k(x))\big)
\]

Then threshold \(P_{\mathrm{TTA}}\).

## 3.3 Practical TTA sets for forests

| Set | Transforms | Cost |
|-----|------------|------|
| TTA-2 | identity, hflip | 2× |
| TTA-4 | + vflip, hvflip | 4× |
| TTA-8 | + rot90×{0,1,2,3} | 8× |

**Rule:** Geometric transforms must use the **same** transform on the mask/prob when inverting. Photometric TTA (brightness) has identity inverse on the mask path.

**Expected gain:** Often +0.5–2 IoU; diminishing returns after TTA-4.  
**Tradeoff:** Latency ×K — always plot IoU vs ms.

---

# Part 4 — Method 3: Dense Conditional Random Fields (CRF)

## 4.1 What is a CRF? (learn this carefully)

A **Conditional Random Field** models \(P(Y \mid X)\): labels \(Y\) given image \(X\).

Energy of a labeling:

\[
E(Y) = \underbrace{\sum_i \psi_u(y_i)}_{\text{unary}} + \underbrace{\sum_{i<j} \psi_p(y_i,y_j)}_{\text{pairwise}}
\]

Inference ≈ find \(Y\) minimizing \(E\) (MAP).

- **Unary** \(\psi_u\): from your U-Net — “this pixel looks like forest according to the CNN”  
- **Pairwise** \(\psi_p\): “nearby similar-colored pixels should share a label”

## 4.2 Dense CRF (Krähenbühl & Koltun, NIPS 2011)

Classic pairwise uses Gaussian kernels on position and color:

\[
\begin{aligned}
k(f_i,f_j) &=
w_1 \exp\!\Big(-\frac{|p_i-p_j|^2}{2\theta_\alpha^2}-\frac{|I_i-I_j|^2}{2\theta_\beta^2}\Big) \\
&\quad + w_2 \exp\!\Big(-\frac{|p_i-p_j|^2}{2\theta_\gamma^2}\Big)
\end{aligned}
\]

- First term (**bilateral**): same label if **close in space and color** (respects edges)  
- Second term (**smoothness**): mild spatial smoothing  

**Fully connected** = every pixel pairs with every other → theoretically huge, but mean-field + high-dim Gaussian filtering makes it ~linear in #pixels (~0.2s historically on VOC-sized images).

## 4.3 How it helps forests

- Smooths salt-and-pepper noise inside canopy / clearings  
- Keeps boundaries aligned with real RGB edges (roads, rivers, field edges)  
- Can hurt thin linear features if \(\theta\) too large (over-smooth)

## 4.4 Implementation notes

- Library: **`pydensecrf`** (common for research prototypes)  
- Unary from probs: convert \(P\) to energies \(-\log P\), \(-\log(1-P)\)  
- Tune \(w_1,w_2,\theta_\alpha,\theta_\beta,\theta_\gamma\) on **validation IoU**  
- Modern note: as CNNs got stronger, CRF gains shrank (NeurIPS 2021 discussion), but for a **student baseline U-Net** CRF often still helps boundaries — you should measure, not assume  

**Paper to read:** Krähenbühl & Koltun, *Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials*, NIPS 2011.  
Also: DeepLab papers (CRF as post-process section).

---

# Part 5 — Method 4: Ensemble methods

## 5.1 Why ensembles work (bias–variance)

A single U-Net has **variance**: different seeds → different errors.  
Averaging diverse predictors reduces variance if errors are not perfectly correlated.

## 5.2 Bagging (different random seeds)

Train \(M\) models \(f_1,\ldots,f_M\) (different seeds / slight data order).

\[
P_{\mathrm{bag}}(x)=\frac{1}{M}\sum_{m=1}^{M} f_m(x)
\]

**Forest practice:** M=3 is a good start; M=5 if you can afford it.  
**Report:** IoU gain vs \(M\times\) training and \(M\times\) inference cost.

## 5.3 Boosting / cascade (coarse → fine)

**Idea:** Model-1 predicts; Model-2 focuses on residuals / hard pixels (often boundaries).

```text
P1 = Model1(x)
P2 = Model2(x, P1)      # e.g. concat image + P1
P  = P1 + α·(P2 - 0.5) # or learned fusion
```

Or train Model-2 only on a **boundary band** (pixels near GT edges).

This is close in spirit to your **novel LBR-Net** (see Part 8).

## 5.4 Stacking (meta-learner)

Hold out a stacking set. Collect predictions from base models as features:

\[
z_{ij} = [P^{(1)}_{ij},\, P^{(2)}_{ij},\, P^{(3)}_{ij},\, R_{ij},\, G_{ij},\, B_{ij}]
\]

Train a **tiny** meta-model \(g(z)\) (1×1 conv or shallow MLP) to predict the label.

**Critical:** Meta-train on data the base models did **not** overfit, or use out-of-fold predictions — otherwise stacking cheats.

## 5.5 What to claim in the paper

Not “ensemble is good” — claim a **Pareto story**:

> Bagging M=3 gives +X IoU at +Y ms; TTA-4 gives +Z IoU at lower storage cost (one model).

---

# Part 6 — Method 5: Active contours (snakes)

## 6.1 Intuition

Treat the forest boundary as a curve \(C\). Evolve \(C\) to minimize energy:

\[
E(C) = E_{\mathrm{internal}}(C) + E_{\mathrm{external}}(C)
\]

- **Internal:** prefer smooth, not wildly wiggly curves  
- **External:** snap to image edges / dark–bright transitions  

Classical: Kass et al. *Snakes* (1988).  
Practical modern variant: **morphological geodesic active contours** in `skimage`.

## 6.2 Pipeline for your project

1. Get coarse mask from U-Net  
2. Extract contour  
3. Evolve with morphological snake for N iterations  
4. Rasterize back to mask  

## 6.3 Pros / cons

| Pros | Cons |
|------|------|
| Explicit boundary polish | Sensitive to initialization |
| No second neural net | Hyperparameter fiddly |
| Good qualitative demos | Often smaller IoU gain than TTA/CRF |

Treat as **optional 5th/6th method** in your systematic comparison, not the main novelty.

---

# Part 7 — Uncertainty estimation & confidence decisions

## 7.1 Two kinds of uncertainty

| Type | Meaning | Forest example |
|------|---------|----------------|
| **Aleatoric** | Noise in data / labels | Mixed pixels at canopy edge, JPEG mask noise |
| **Epistemic** | Model doesn’t know | Rare shadow patterns, unseen sensor |

You mainly estimate **predictive uncertainty** from stochastic inference or ensembles.

## 7.2 Monte Carlo Dropout (Gal & Ghahramani, ICML 2016)

**Training:** model has Dropout layers.  
**Test:** keep Dropout **ON**; run \(T\) forwards with different masks.

\[
\bar{P} = \frac{1}{T}\sum_{t=1}^{T} P^{(t)}, \qquad
U = \frac{1}{T}\sum_{t=1}^{T}\big(P^{(t)}-\bar{P}\big)^2
\]

Or use **predictive entropy**:

\[
H = -\bar{P}\log\bar{P} - (1-\bar{P})\log(1-\bar{P})
\]

- High \(U\) or \(H\) at edges is expected  
- High uncertainty **inside** clear forest → possible failure / OOD  

**Note:** Lasana’s current U-Net may lack Dropout — you may need to **add Dropout** and finetune, or use ensemble variance instead.

## 7.3 Ensemble variance

From bagged models:

\[
U(x)=\mathrm{Var}_m\big[f_m(x)\big]
\]

Often stronger epistemic signal than MC Dropout alone.

## 7.4 Calibration & ECE (Guo et al., ICML 2017)

**Calibration:** if the model says 0.8, it should be correct ~80% of the time.

**Expected Calibration Error (ECE):**

1. Bin pixels (or images) by confidence into \(B\) bins  
2. For each bin \(b\): compute accuracy \(\mathrm{acc}(b)\) and mean confidence \(\mathrm{conf}(b)\)  
3.  

\[
\mathrm{ECE} = \sum_{b=1}^{B} \frac{n_b}{N}\,\big|\mathrm{acc}(b)-\mathrm{conf}(b)\big|
\]

Lower is better. Plot a **reliability diagram** (confidence vs accuracy).

**Temperature scaling (simple fix):** learn scalar \(T>0\) on val:

\[
P_T = \sigma(z / T)
\]

Often reduces ECE without changing argmax much (for multi-class); for binary, still useful for confidence.

## 7.5 Confidence-based decision making (your expected contribution)

Examples to implement and discuss:

1. **Adaptive threshold:** use stricter threshold where uncertainty is high  
2. **Abstention:** mark pixels with \(U > \tau\) as “unknown” for human review  
3. **Quality gate:** if mean image uncertainty high → reject tile / request re-inference with TTA  

This turns uncertainty from a pretty heatmap into a **system feature**.

---

# Part 8 — ★ Novel contribution: Lightweight Boundary Refinement Network

## 8.1 Motivation from literature

**SegFix** (CVPR 2020 workshop / arXiv 2007.04269): boundary pixels are less reliable than interior pixels; refine by replacing boundary labels using interior predictions along an offset direction. Model-agnostic, fast, complementary to DenseCRF.

You do **not** copy SegFix. You propose a **tiny learnable module** specialized for forest canopy — call it e.g. **LBR-Net** (Lightweight Boundary Refiner).

## 8.2 Design goals

| Goal | Target |
|------|--------|
| Params | ≪ U-Net (aim &lt; 5% of U-Net, or &lt; 0.5M) |
| Input | RGB + coarse \(P\) (+ optional uncertainty) |
| Output | Refined \(P'\) or residual \(\Delta P\) |
| Training | Boundary-weighted BCE+Dice |
| Latency | Small overhead (ideally &lt; +20% ms) |

## 8.3 Architecture sketch (learnable, simple)

```text
Input: concat[RGB, P_coarse]     # shape H×W×4
  → DepthwiseSeparableConv 3×3, 32 ch
  → ReLU + BN
  → DepthwiseSeparableConv 3×3, 32 ch
  → ReLU + BN
  → Conv 1×1 → 1 channel
  → residual: P_refined = σ( logit(P_coarse) + α·Δ )
```

Optional: apply loss **only in a boundary band**:

```text
band = dilate(GT) XOR erode(GT)   # ring around edges
loss = BCEDice(P_refined, GT) weighted by band
```

## 8.4 Why this is a valid “novel” team contribution

- Classical CRF is fixed kernels; LBR-Net **learns** forest-specific refinement  
- Complements TTA/ensembles (orthogonal axes)  
- Easy ablations: U-Net | +CRF | +SegFix-style | +LBR-Net | +TTA+LBR-Net  
- Efficiency story pairs well with pruning/quantization  

## 8.5 Related reading (inspiration)

1. SegFix — boundary refinement by interior transfer  
2. PointRend — adaptive refinement at uncertain locations (heavier idea)  
3. CascadePSP — powerful but **not** lightweight (contrast against it)  
4. Guided filter / bilateral residual papers — classical learnable smoothing  

---

# Part 9 — Method 6: Pruning & quantization

## 9.1 Pruning

Remove weights/channels with small magnitude → fewer FLOPs / smaller models.

- **Unstructured:** sparse weights (needs sparse kernels to be fast)  
- **Structured (channel prune):** actually faster on GPUs/CPUs  

Workflow: train → prune → finetune → measure IoU drop.

## 9.2 Quantization

Store weights/activations as INT8 instead of FP32.

- **PTQ** (post-training quantization): calibrate on a few batches, no full retrain  
- **QAT** (quantization-aware training): better accuracy, more work  

Tools: TFLite, ONNX Runtime, PyTorch `quantize_dynamic` / QAT APIs.

## 9.3 What to report

| Model | IoU | Size (MB) | ms/image | ΔIoU vs FP32 |
|-------|-----|-----------|----------|--------------|
| U-Net FP32 | | | | 0 |
| U-Net INT8 | | | | |
| Ensemble FP32 | | | | |
| Best pipeline + INT8 | | | | |

This supports the “deployment efficiency” bullet in the architecture doc.

---

# Part 10 — Metrics dashboard (memorize these)

| Metric | Formula / meaning | You use it for |
|--------|-------------------|----------------|
| IoU / Dice | Overlap | Primary quality |
| ΔIoU % | Relative gain vs baseline | Post-process improvement % |
| Precision / Recall | FP vs FN | Threshold & morphology analysis |
| Boundary IoU / Hausdorff | Edge quality | CRF / LBR-Net proof |
| ECE | Calibration gap | Uncertainty study |
| ms/image, overhead × | Latency | Cost of each method |
| Params / MB | Model size | LBR-Net & quantization |

---

# Part 11 — Experimental protocol (do science, not vibes)

1. Fix **one** base checkpoint (Lasana best or Team-1 best).  
2. Tune **all** method hyperparameters on **validation only**.  
3. Touch test set **once** per final config.  
4. Same split as team (seed 42, 80/10/10).  
5. Report mean ± std over 3 seeds when randomness exists (dropout, bagging).  
6. Same hardware for all timing.  
7. Qualitative grid: Image | GT | Baseline | Method | Uncertainty.

### Minimum comparison table (≥5 methods)

| # | Method |
|---|--------|
| 0 | U-Net + thr=0.5 (baseline) |
| 1 | + thr* + morphology |
| 2 | + TTA-4 |
| 3 | + DenseCRF |
| 4 | + Bagging (M=3) |
| 5 | + LBR-Net (ours) |
| 6 | + TTA + LBR-Net (optional combo) |
| — | + Uncertainty/ECE analysis (orthogonal column) |

---

# Part 12 — Topics to learn (curriculum)

Study in this order. Check boxes as you go.

### Week foundation

- [ ] Sigmoid, logits, thresholding  
- [ ] IoU, Dice, precision, recall  
- [ ] Why accuracy is misleading  
- [ ] Train/val/test leakage  

### Classical processing

- [ ] Morphology (open/close/components)  
- [ ] Contours and edge maps (Sobel/Canny)  
- [ ] Bilateral filter intuition  

### Probabilistic models

- [ ] Unary vs pairwise energy  
- [ ] DenseCRF idea + mean-field (high level)  
- [ ] Hands-on: one image through `pydensecrf`  

### Ensembles & TTA

- [ ] Bias–variance  
- [ ] Bagging vs boosting vs stacking  
- [ ] Inverse geometric transforms  

### Uncertainty

- [ ] Epistemic vs aleatoric  
- [ ] MC Dropout paper idea  
- [ ] Entropy / variance maps  
- [ ] Temperature scaling  
- [ ] ECE + reliability diagrams  

### Learnable refinement

- [ ] Depthwise separable convs  
- [ ] Residual refinement  
- [ ] Boundary band sampling  
- [ ] SegFix abstract + figures  

### Efficiency

- [ ] Structured pruning concept  
- [ ] INT8 PTQ concept  
- [ ] Profiling latency  

### Paper skills

- [ ] Ablation tables  
- [ ] Pareto curves (IoU vs ms)  
- [ ] Failure case analysis  

---

# Part 13 — Core papers / resources

| # | Paper / resource | Why |
|---|------------------|-----|
| 1 | Krähenbühl & Koltun, NIPS 2011 — DenseCRF | CRF post-process foundation |
| 2 | Chen et al., DeepLab (v1/v2) — CRF sections | How SOTA used CRF historically |
| 3 | Gal & Ghahramani, ICML 2016 — MC Dropout | Uncertainty via dropout |
| 4 | Guo et al., ICML 2017 — Calibration / ECE | ECE definition & temperature scaling |
| 5 | SegFix, arXiv:2007.04269 | Boundary refinement inspiration |
| 6 | PointRend, Kirillov et al., CVPR 2020 | Adaptive refinement idea |
| 7 | skimage morphological snakes docs | Active contours practice |
| 8 | TFLite / ONNX quantization tutorials | Deployment |

---

# Part 14 — 8–10 week plan (aligned with architecture doc)

| Weeks | Learn + build | Output |
|-------|---------------|--------|
| 1–2 | Threshold, morph, TTA | First ΔIoU table + plots |
| 3–4 | DenseCRF + bagging | 4-method comparison |
| 5–6 | MC Dropout / ensemble U + ECE | Uncertainty maps + calibration |
| 7–8 | LBR-Net design + train | Novel module + ablation |
| 9 | Prune / INT8 best pipeline | Efficiency table |
| 10 | Writing + integrate with team | Paper section draft |

Phase with team: individual modules → combine with best arch/loss/aug/multi-scale → write paper.

---

# Part 15 — How you connect to other members

| Member | Topic | Sync |
|--------|-------|------|
| 1 | Architecture | You post-process their best U-Net/Attention-U-Net |
| 2 | Loss | Boundary losses may reduce need for CRF; still measure |
| 3 | Augmentation | Keep TTA transforms consistent with train augs |
| 5 | Multi-scale | Refine fused multi-scale probs; ensemble across scales |

**Full-system story for the paper:** best network + best loss + aug + **your refinement stack**.

---

# Part 16 — Expected contributions (from the doc — your checklist)

- [ ] Systematic comparison of **5+** post-processing methods  
- [ ] **Novel:** lightweight boundary refinement network  
- [ ] Uncertainty maps for reliability assessment  
- [ ] Confidence-based decision making  
- [ ] Track: post-processing improvement %, ECE, inference overhead  
- [ ] (Bonus) pruning & quantization for deployment  

---

# Part 17 — Suggested code layout

```text
Lasana/
  POST_PROCESSING_RESEARCH.md          ← this study guide
  postprocess/
    01_threshold_morph.py
    02_tta.py
    03_dense_crf.py
    04_ensemble_bagging.py
    05_uncertainty.py
    06_lbr_net.py
    07_quantize_benchmark.py
    eval_compare.py
    results/
```

Start from checkpoint: `Lasana/checkpoints/lasana_unet_best.keras`.

---

# Part 18 — One-page memory sheet

| Need | Use |
|------|-----|
| Quick IoU bump, zero new models | thr* + morph, then TTA |
| Sharper edges from RGB | DenseCRF |
| Best accuracy, accept cost | Bagging M=3–5 |
| Reliability / human-in-loop | MC Dropout or ensemble variance + ECE |
| Paper novelty | LBR-Net (tiny residual boundary head) |
| Deploy on edge | Prune + INT8; measure IoU drop |

**Baseline to beat:** thr=0.5 single U-Net.  
**Primary score:** IoU (and Dice).  
**Honesty metrics:** ΔIoU %, ECE, ms overhead.

---

*End of study guide. Learn Parts 1→7 first, implement Part 11’s table, then build Part 8 (LBR-Net) as your novel piece.*
