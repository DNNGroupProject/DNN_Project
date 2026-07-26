# Your Complete Learning Order — Team Member 4 (Post-Processing & Refinement)

This is the **full ordered list** of everything you need to learn to understand *your* part of the project.  
Follow stages **1 → 12** in order. Do not skip early stages.

**Your job in one line:** improve the forest mask *after* U-Net outputs probabilities (threshold, cleanup, TTA, CRF, ensembles, uncertainty, tiny refiner, efficiency).

**Local materials**
- Theory: `Lasana/POST_PROCESSING_RESEARCH.md`
- Experiments you already ran: `Lasana/postprocess/results/RESULTS.md`
- Code: `Lasana/postprocess/run_research.py`
- Baseline model: Lasana U-Net

---

## How to use this roadmap

For each topic:
1. Learn the idea (intuition)
2. Learn the formula / algorithm
3. Connect it to forest masks
4. Look at your experiment result (if you already ran it)
5. Check the box only when you can explain it out loud without notes

---

# STAGE 1 — Project context (what you own)

**Goal:** know where your work sits in the team pipeline.

| # | Learn this | Why |
|---|------------|-----|
| 1.1 | What semantic segmentation is (per-pixel classification) | Your whole project |
| 1.2 | Input image vs output mask | Pipeline basics |
| 1.3 | Forest = 1, background = 0 | Binary setup |
| 1.4 | Teammate roles: Arch / Loss / Aug / **You** / Multi-scale | Who does what |
| 1.5 | Your layer starts *after* U-Net probabilities | Scope of your research |
| 1.6 | Difference between improving the network vs refining predictions | Avoid overlapping teammates |

**Done when:** you can draw: `Image → U-Net → P → YOUR METHODS → final mask`.

---

# STAGE 2 — Deep learning essentials for segmentation

**Goal:** read U-Net outputs without confusion.

| # | Learn this | Why |
|---|------------|-----|
| 2.1 | CNN basics (conv, pool, upsample) | U-Net building blocks |
| 2.2 | Encoder–decoder idea | Why resolution is lost/restored |
| 2.3 | Skip connections (U-Net) | Why Lasana beat Kalana |
| 2.4 | Logits vs probabilities | Before/after sigmoid |
| 2.5 | Sigmoid activation | Binary forest probability |
| 2.6 | BCE loss (intuition) | Pixel-wise training |
| 2.7 | Dice loss (intuition) | Overlap training |
| 2.8 | Train / validation / test split | No cheating when tuning post-process |
| 2.9 | Overfitting | Why val-only tuning matters for thresholds |

**Done when:** you understand `P = sigmoid(z)` and why thresholding is a separate decision.

---

# STAGE 3 — Metrics (your report language)

**Goal:** never use only accuracy; speak in IoU/Dice/ECE.

| # | Learn this | Formula / idea | You use it for |
|---|------------|----------------|----------------|
| 3.1 | TP, FP, TN, FN | confusion at pixel level | All metrics |
| 3.2 | Pixel accuracy | (TP+TN)/all | Weak metric |
| 3.3 | Precision | TP/(TP+FP) | False trees |
| 3.4 | Recall | TP/(TP+FN) | Missed forest |
| 3.5 | **IoU (Jaccard)** | TP/(TP+FP+FN) | **Primary score** |
| 3.6 | **Dice** | 2TP/(2TP+FP+FN) | Paper report |
| 3.7 | Relationship Dice ↔ IoU | Dice = 2·IoU/(1+IoU) | Sanity check |
| 3.8 | **ΔIoU %** | relative improvement vs baseline | Your “improvement %” |
| 3.9 | Boundary IoU / Hausdorff (intro) | edge quality | CRF / LBR claims |
| 3.10 | Latency (ms/image) | wall-clock | Overhead reporting |

**Your numbers to remember**
- Baseline IoU ≈ **0.738**, Acc ≈ **81.1%**
- Best cheap classical: thr\*+morph ≈ **+0.07% ΔIoU**, Acc ≈ **81.8%**

**Done when:** you can explain why 90% accuracy can still mean weak forest masks.

---

# STAGE 4 — Probability maps & thresholding (Part A core)

**Goal:** master the simplest post-process (and the one that already helped you).

| # | Learn this | Details |
|---|------------|---------|
| 4.1 | Probability map \(P_{ij} \in [0,1]\) | Soft prediction |
| 4.2 | Hard mask via threshold | \(\hat Y = 1[P > t]\) |
| 4.3 | Why \(t=0.5\) is arbitrary | Bias / class balance |
| 4.4 | Threshold sweep on **validation** | Pick \(t^\*\) by max IoU |
| 4.5 | Apply \(t^\*\) once on test | Scientific protocol |
| 4.6 | Precision–Recall vs threshold curve | Visual understanding |
| 4.7 | IoU vs threshold curve | Your `threshold_sweep.json` |

**Your result:** best \(t^\* = 0.65\) (not 0.5).

**Done when:** you can reinvent the threshold sweep from scratch.

---

# STAGE 5 — Classical image processing cleanup (Part A)

**Goal:** clean speckles/holes without another neural net.

| # | Learn this | Forest effect |
|---|------------|---------------|
| 5.1 | Binary morphology idea | Shape cleanup |
| 5.2 | Erosion | Shrink forest / remove thin noise |
| 5.3 | Dilation | Expand / fill gaps |
| 5.4 | Opening (erode→dilate) | Remove small false-positive blobs |
| 5.5 | Closing (dilate→erode) | Fill small holes in canopy |
| 5.6 | Structuring element / kernel size | 3×3 safe; large kernels destroy thin corridors |
| 5.7 | Connected components | Find separate blobs |
| 5.8 | Remove small objects by area | Drop tiny FP regions |
| 5.9 | OpenCV/`skimage` morphology APIs | Implementation |

**Your result:** thr\* + morph was the **best classical** method (+0.07% ΔIoU).

**Done when:** you can choose open vs close for a noisy mask and justify it.

---

# STAGE 6 — Test-Time Augmentation (Part B)

**Goal:** understand “free ensemble” at inference using one model.

| # | Learn this | Details |
|---|------------|---------|
| 6.1 | What TTA is | Augment test image, predict, invert, average |
| 6.2 | Why train augs make TTA useful | Model saw flips/rots |
| 6.3 | Geometric transforms | flip / rot90 |
| 6.4 | Inverse transforms on probability maps | Must undo geometry |
| 6.5 | Soft probability averaging | Better than voting hard masks |
| 6.6 | TTA-2 / TTA-4 / TTA-8 cost | Latency ×K |
| 6.7 | Pareto thinking: IoU vs ms | Is gain worth cost? |

**Your result:** TTA-4 ≈ same IoU as thr\*, but ~**4× slower** → not worth it on this run.

**Done when:** you can write the TTA average formula and explain inverse flips.

---

# STAGE 7 — Edge-aware refinement & CRF (Part E + real CRF theory)

**Goal:** understand how image edges guide label cleanup.

### 7A — What you already ran (bilateral substitute)

| # | Learn this |
|---|------------|
| 7.1 | Bilateral filter (smooth while keeping edges) |
| 7.2 | Why RGB edges matter for canopy boundaries |
| 7.3 | Gradients / Sobel (edge strength) |
| 7.4 | Limitation vs true DenseCRF |

### 7B — What you must still learn (Dense CRF)

| # | Learn this | Details |
|---|------------|---------|
| 7.5 | What a Conditional Random Field is | Labels depend on neighbors + image |
| 7.6 | Unary potential | From U-Net probabilities |
| 7.7 | Pairwise potential | Similar color + nearby → same label |
| 7.8 | Dense CRF (fully connected pairwise) | Classic post-process |
| 7.9 | Mean-field inference (high-level) | How DenseCRF is computed efficiently |
| 7.10 | Hyperparameters \(\theta_\alpha,\theta_\beta,\theta_\gamma\) | Must tune on val |
| 7.11 | When CRF helps vs over-smooths | Thin tree lines risk |

**Paper:** Krähenbühl & Koltun, NIPS 2011 (DenseCRF).  
Also skim DeepLab papers’ CRF post-process section.

**Done when:** you can explain unary vs pairwise and why CRF respects road/river edges.

---

# STAGE 8 — Ensembles (bagging / boosting / stacking)

**Goal:** understand combining multiple predictions (your architecture-doc requirement).

| # | Learn this | Idea |
|---|------------|------|
| 8.1 | Bias vs variance | Why averaging helps |
| 8.2 | **Bagging** | Same model, different seeds → average \(P\) |
| 8.3 | Soft voting vs hard voting | Prefer soft probs |
| 8.4 | **Boosting / cascade** | Model-2 fixes Model-1 hard regions |
| 8.5 | **Stacking** | Meta-learner on stacked predictions |
| 8.6 | Diversity of models | Correlated errors don’t help |
| 8.7 | Cost: train ×M and infer ×M | Always report latency |
| 8.8 | Out-of-fold stacking discipline | Avoid leakage |

**Relate to your work:** TTA is a cheap single-model “pseudo-ensemble.” True bagging needs multiple U-Nets (better on GPU).

**Done when:** you can compare bagging vs TTA vs stacking in one paragraph.

---

# STAGE 9 — Uncertainty & calibration (Part C)

**Goal:** know when the model is unsure; measure trustworthiness.

| # | Learn this | Details |
|---|------------|---------|
| 9.1 | Confidence vs correctness | Not the same |
| 9.2 | Aleatoric uncertainty | Noise in data/labels (fuzzy canopy edge) |
| 9.3 | Epistemic uncertainty | Model ignorance (rare patterns) |
| 9.4 | Predictive entropy | \(H = -p\log p-(1-p)\log(1-p)\) |
| 9.5 | Variance across samples | Ensemble / MC Dropout |
| 9.6 | **Monte Carlo Dropout** | Dropout ON at test, \(T\) forwards |
| 9.7 | Uncertainty maps (visualization) | Edges often high entropy |
| 9.8 | **Calibration** | “80% confident” ≈ 80% correct |
| 9.9 | Reliability diagram | Plot confidence vs accuracy |
| 9.10 | **ECE (Expected Calibration Error)** | Main calibration metric |
| 9.11 | Temperature scaling | Simple calibration fix \(z/T\) |
| 9.12 | Confidence-based decisions | Abstain / review / adaptive policies |
| 9.13 | Why naive high-entropy stricter threshold can hurt | Your −0.67% ΔIoU lesson |

**Papers**
- Gal & Ghahramani, ICML 2016 (MC Dropout)
- Guo et al., ICML 2017 (calibration / ECE)

**Your results**
- Soft-prob ECE ≈ **0.027** (fairly calibrated)
- Adaptive threshold policy **hurt** IoU → redesign as abstention, not harder thr

**Done when:** you can define ECE and explain a reliability diagram.

---

# STAGE 10 — Active contours (optional classical method)

**Goal:** know snakes enough to compare them in your “5+ methods” table.

| # | Learn this |
|---|------------|
| 10.1 | Contour / snake as a curve around forest |
| 10.2 | Internal energy (smoothness) |
| 10.3 | External energy (snap to edges) |
| 10.4 | Morphological / geodesic active contours overview |
| 10.5 | Pros/cons vs CRF and LBR-Net |

**Done when:** you can say when snakes help boundaries and why they are fiddly.

---

# STAGE 11 — Novel piece: Lightweight Boundary Refinement Network (Part D)

**Goal:** understand your paper contribution idea and why v1 failed.

| # | Learn this | Details |
|---|------------|---------|
| 11.1 | Why boundaries are hardest | CNN blur / mixed pixels |
| 11.2 | Residual refinement idea | Predict correction \(\Delta\), not full mask from scratch |
| 11.3 | Depthwise / Separable convolutions | Keep model tiny |
| 11.4 | Boundary band / distance transform | Train focus on edges |
| 11.5 | Guided by RGB + coarse \(P\) | Inputs to LBR-Net |
| 11.6 | Parameter budget (≪ U-Net) | “Lightweight” claim |
| 11.7 | Early stopping on **val IoU** | Not only loss |
| 11.8 | Why short training can hurt IoU | Your −1.48% result |
| 11.9 | Related ideas: SegFix, PointRend (read abstracts) | Inspiration, don’t copy |
| 11.10 | Ablation design | U-Net vs +CRF vs +LBR vs +TTA+LBR |

**Your result:** LBR-Net v1 (8 epochs) **dropped** IoU — still useful science; next: longer train, boundary-weighted loss, IoU early-stop.

**Done when:** you can propose 3 concrete upgrades to LBR-Net v2.

---

# STAGE 12 — Deployment efficiency (architecture-doc bullet)

**Goal:** finish the “refinement & ensembling” track with speed/size.

| # | Learn this | Details |
|---|------------|---------|
| 12.1 | Why FLOPs / params / MB matter | Real maps / edge devices |
| 12.2 | Structured vs unstructured pruning | What actually speeds up |
| 12.3 | Quantization (FP32 → INT8) | PTQ vs QAT |
| 12.4 | Knowledge distillation (intro) | Big ensemble → small student |
| 12.5 | Measure IoU drop vs speedup | Never optimize size blindly |

**Done when:** you can design a “best accuracy” vs “best mobile” table.

---

# STAGE 13 — Scientific method for *your* experiments

**Goal:** write a paper-ready section, not vibes.

| # | Learn this |
|---|------------|
| 13.1 | Freeze one base checkpoint |
| 13.2 | Tune hyperparameters on validation only |
| 13.3 | Report test once |
| 13.4 | Always include baseline thr=0.5 |
| 13.5 | Report ΔIoU%, ECE, ms overhead together |
| 13.6 | Qualitative grids: Image \| GT \| Baseline \| Method \| Uncertainty |
| 13.7 | Failure cases (shadows, thin corridors, water edges) |
| 13.8 | Ablation tables + Pareto curves (IoU vs ms) |

**Done when:** you can fill this table from memory:

| Method | IoU | ΔIoU% | ECE | ms |

---

## Master checklist (complete order)

Copy this and tick as you learn:

### Foundations
- [ ] 1. Project role / pipeline position  
- [ ] 2. CNN + U-Net + sigmoid + train/val/test  
- [ ] 3. IoU, Dice, ΔIoU%, latency  

### Core post-process (must master)
- [ ] 4. Thresholding & \(t^\*\) sweep  
- [ ] 5. Morphology + connected components  
- [ ] 6. TTA  

### Research-level methods
- [ ] 7. Bilateral + **Dense CRF theory**  
- [ ] 8. Bagging / boosting / stacking  
- [ ] 9. Uncertainty, entropy, ECE, temperature scaling, confidence decisions  
- [ ] 10. Active contours (overview)  
- [ ] 11. LBR-Net (novel module) + why v1 failed  
- [ ] 12. Pruning / quantization  

### Communication
- [ ] 13. Experimental protocol & paper tables  

---

## Suggested weekly learning schedule (8 weeks)

| Week | Learn stages | Practice |
|------|--------------|----------|
| 1 | 1–3 | Re-read Lasana README + RESULTS table |
| 2 | 4–5 | Re-run threshold/morph code; plot IoU vs t |
| 3 | 6 | Implement TTA yourself; measure ms |
| 4 | 7 | Read DenseCRF paper + try bilateral vs CRF |
| 5 | 8 | Sketch bagging plan (3 seeds) |
| 6 | 9 | Plot entropy maps + ECE/reliability diagram |
| 7 | 10–11 | Redesign LBR-Net v2 training |
| 8 | 12–13 | Efficiency table + write your paper subsection draft |

---

## Minimum vocabulary you must know cold

| Term | One-line meaning |
|------|------------------|
| Probability map | Soft forest score per pixel |
| Threshold \(t\) | Cut to make hard mask |
| IoU | Overlap quality |
| Morphology | Shape cleanup with erode/dilate |
| TTA | Average predictions over test augs |
| CRF | Probabilistic label smoothing using image edges |
| Ensemble | Combine several models’ predictions |
| Entropy | Uncertainty from probability |
| ECE | How well confidence matches accuracy |
| LBR-Net | Your tiny learnable boundary refiner |
| ΔIoU% | Relative improvement vs baseline |

---

## What to open while studying

1. This file (order)  
2. `POST_PROCESSING_RESEARCH.md` (deep theory)  
3. `postprocess/results/RESULTS.md` (your real numbers)  
4. `postprocess/run_research.py` (how it was coded)  

---

**Bottom line:** learn **Stages 1→6** until solid, then **7→9** for research depth, then **11** as your novelty, then **12–13** for a complete contribution. That is the complete ordered set for *your* part.
