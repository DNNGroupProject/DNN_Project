# ViT + Concept-Guided XAI for Forest Segmentation

**Separate module** created from advisor feedback (ignore unrelated first-sentence comments).  
**Goal:** move beyond plain CNNs → **Vision Transformer (ViT) segmentation**, and use an **existing XAI method** to *improve and explain* forest segmentation.

| Item | Detail |
|------|--------|
| **Advisor reference paper** | Wickramanayake, Hsu, Lee — *Comprehensible Convolutional Neural Networks via Guided Concept Learning*, arXiv:[2101.03919](https://arxiv.org/abs/2101.03919) (local: `../2101.03919v2.pdf`) |
| **Task** | Binary forest / non-forest semantic segmentation |
| **Dataset** | Same Forest Segmented data as Lasana (`Lasana/dataset/...`) |
| **Base vs this folder** | Lasana = CNN U-Net; **this folder = ViT + concept XAI** |

---

## 1. What the advisor asked (and what we did)

| Advisor point | Our response in this folder |
|---------------|-----------------------------|
| CNNs may no longer be SOTA for segmentation → explore **ViT** | Built a **Vision Transformer encoder** + dense segmentation head |
| Use an **existing XAI method** to improve forest segmentation with ViT | Adapted **guided concept learning** from arXiv:2101.03919 into the ViT pipeline |
| Paper as reference for XAI improving models | Same loss structure: accuracy + concept uniqueness + mapping consistency; explanations as **concept contributions** |

---

## 2. What the paper does (and why it matters)

Paper title: **Comprehensible CNNs via Guided Concept Learning** (2021).

### Problem the paper solves
- Post-hoc XAI (Grad-CAM, etc.) is applied *after* training and may not match the true decision process.
- Interpretable CNNs often learn features that are **not human-consistent** (partial/overlapping parts).

### Paper idea (CNN classification)
1. Add a **concept layer** (1×1 conv) after the last convolutional features.
2. Guide each concept filter to match a **word phrase** from image captions.
3. Train with:

\[
L = L_A + \lambda (L_u + L_m)
\]

| Symbol | Meaning |
|--------|---------|
| \(L_A\) | Classification accuracy (cross-entropy) |
| \(L_u\) | **Concept uniqueness** — one filter ↔ one phrase |
| \(L_m\) | **Mapping consistency** — visual feature close to its phrase in joint embedding (+ counter-image) |
| GAP + FC | Decision = weighted sum of concepts → **built-in explanation** |

### Why we use it for *our* project
- It is an **existing, published XAI-for-improvement** method (not only visualization).
- Concepts are **trained into the model**, so explanations come from the classifier itself.
- We keep the same scientific core, but change backbone and task as the advisor requested.

---

## 3. What we changed vs the paper (important)

| Paper (2101.03919) | This project adaptation |
|--------------------|-------------------------|
| CNN backbone (VGG/ResNet/DenseNet) | **ViT encoder** (transformer patches + attention) |
| Image **classification** (birds/flowers) | Dense **segmentation** (forest mask) |
| Caption **word phrases** (CUB/Flowers) | **Pseudo forest concepts** (no captions in Forest Segmented) |
| Explain class score | Explain **forest decision** + pixel mask |
| Attention not central | ViT **attention maps** as extra transformer-native XAI |

### Forest concepts we use (pseudo-labels)

Because Forest Segmented has **no text descriptions**, we derive soft concept indicators from RGB + mask (see `pseudo_concepts.py`):

| Concept | Intuition |
|---------|-----------|
| `dense_canopy` | High forest coverage + green vegetation |
| `sparse_trees` | Mixed / mid forest ratio |
| `shadow_region` | Dark areas overlapping forest |
| `open_clearing` | Bright non-forest |
| `forest_boundary` | Edge band between forest and clear |

These play the role of the paper’s phrase indicators \(z_k\) for \(L_u\) / \(L_m\).

---

## 4. Pipeline (end-to-end)

```text
RGB satellite patch (224×224)
        │
        ▼
┌─────────────────────────────┐
│ Patch embedding (16×16)     │
│ + positional encoding       │
│ + ViT transformer blocks    │  ← attention = XAI signal
└──────────────┬──────────────┘
               │ token features
               ▼
┌─────────────────────────────┐
│ Concept layer (1×1 → K maps)│  ← paper idea
│ GAP → concept activations   │
└──────────────┬──────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
 Seg head (upsample)   Concept FC
 → forest mask P       → contributions %
       │                │
       ▼                ▼
 Final mask      Human-readable explanation
 "forest because dense_canopy 61%, boundary 22%, ..."
```

### Training objective (adapted)

\[
L = L_A + \lambda (L_u + L_m)
\]

- \(L_A\): **BCE + Dice** on the segmentation mask (+ light aux CE from concepts → forest/background)
- \(L_u\): uniqueness between GAP concepts and pseudo-label vector \(z\)
- \(L_m\): embedding consistency + **counter-image** batch shuffle (as in the paper)

So XAI is not only plotted — it **regularizes learning**.

---

## 5. Architecture details

### 5.1 Vision Transformer encoder
- Image size: **224**
- Patch size: **16** → \(14\times14 = 196\) tokens
- Embed dim / depth / heads: configurable in `config.py` (default lightweight for CPU: 192 / 6 / 3)
- CLS token + patch tokens; segmentation uses patch tokens reshaped to a feature grid

**Why ViT (advisor):** global self-attention models long-range context (canopy patterns across the tile) better than local CNN kernels alone; attention weights are naturally inspectable.

### 5.2 Concept layer (from paper)
- 1×1 projection: token feature channels → **K concept maps**
- Global Average Pooling → concept scores
- Joint embedding of concept maps ↔ concept name embeddings for \(L_m\)

### 5.3 Segmentation head
- Fuse concept maps + ViT features
- Small conv decoder → logits upsampled to 224×224
- Sigmoid → forest probability

### 5.4 Explanation head (paper §III-C)
- Contribution of concept \(k\) to forest class ≈ \(\hat v_k \times W_{k,\text{forest}}\)
- Softmax → percentage contributions for reports / figures

---

## 6. Why this is better aligned with the advisor than only CNN U-Net

| Approach | Backbone | XAI | Improves training? |
|----------|----------|-----|--------------------|
| Kalana / Lasana | CNN U-Net | Mostly post-hoc / none | — |
| **This folder** | **ViT** | Guided concepts + attention | **Yes** (extra losses) |
| Paper original | CNN | Guided concepts | Yes (classification) |

We are **not replacing** Lasana; this is a parallel research track for the advisor’s ViT + XAI direction.

---

## 7. Folder layout

```text
ViT_XAI_Segmentation/
  README.md                 ← this file (what / why)
  requirements.txt
  config.py                 ← paths, ViT size, λ, concepts
  pseudo_concepts.py        ← forest concept pseudo-labels
  dataset.py                ← loader + aug + concepts
  model.py                  ← ViT + concept layer + losses
  train.py                  ← training loop
  xai_explain.py            ← attention + concept contribution figures
  checkpoints/              ← best weights after train
  results/                  ← metrics JSON + explanation PNGs
```

---

## 8. How to run

### 8.1 Install

```bash
pip install -r ViT_XAI_Segmentation/requirements.txt
```

### 8.2 Dataset

Uses existing data:

```text
Lasana/dataset/Forest Segmented/Forest Segmented/{images,masks}
```

Optional CPU smoke test — in `config.py` set:

```python
MAX_SAMPLES = 800
NUM_EPOCHS = 10
```

### 8.3 Train

```bash
cd ViT_XAI_Segmentation
python train.py
```

Outputs:
- `checkpoints/vit_concept_seg_best.pt` (best val IoU)
- `results/train_results.json` (test IoU / Dice / Acc + history)

### 8.4 Explain (XAI)

```bash
python xai_explain.py --n 4
```

Outputs under `results/explanations/`:
- image | GT | prediction | ViT attention overlay
- top concept overlays + contribution bar chart
- short text explanation per sample

---

## 9. Metrics to report (for paper / viva)

| Metric | Role |
|--------|------|
| IoU / Dice / Pixel Acc | Segmentation quality vs Lasana CNN |
| Concept uniqueness (qualitative maps) | Are concepts distinct? |
| Contribution explanations | Human-readable “why forest?” |
| Attention overlays | Transformer XAI |
| Ablation: \(L_A\) only vs \(L_A+L_u+L_m\) | Prove XAI losses help (as in paper Table II) |

Suggested ablation table:

| Model | Backbone | Losses | Test IoU | Explainable? |
|-------|----------|--------|----------|--------------|
| Lasana U-Net | CNN | BCE+Dice | (your number) | Limited |
| ViT-Seg | ViT | \(L_A\) only | | Attention only |
| **ViT-Concept-Seg (ours)** | ViT | \(L_A+\lambda(L_u+L_m)\) | | Attention + concepts |

---

## 10. What vs Why (quick sheet)

| What | Why |
|------|-----|
| Separate folder from Lasana | Advisor-directed new track; keep CNN baseline intact |
| ViT encoder | Follow “explore ViT for segmentation” guidance |
| Concept layer + \(L_u,L_m\) | Use **existing XAI method from the cited paper** to *improve* learning, not only visualize |
| Pseudo forest concepts | Dataset has no CUB-style captions; keep paper mechanism |
| Attention rollout figures | Extra ViT-native explanations |
| BCE+Dice as \(L_A\) | Segmentation needs overlap loss, not only CE |
| Counter-image shuffle | Faithful to paper’s mapping consistency |

---

## 11. Relation to your Team-4 post-processing work

| Track | Folder | Focus |
|-------|--------|-------|
| CNN baseline + refinement | `Lasana/` | U-Net, TTA, morph, LBR-Net |
| **ViT + guided XAI** | **`ViT_XAI_Segmentation/`** | Advisor paper + transformer backbone |

You can later combine: **ViT-Concept-Seg → then Lasana-style post-process**.

---

## 12. References

1. S. Wickramanayake, W. Hsu, M. L. Lee, *Comprehensible Convolutional Neural Networks via Guided Concept Learning*, arXiv:2101.03919 (PDF in project root).
2. Dosovitskiy et al., *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale* (ViT).
3. Project Lasana U-Net baseline (CNN comparison).

---

## 13. Honest limitations (good for discussion with advisor)

1. Pseudo-concepts are weaker than real captions — future work: LLM/captioner or expert tags for forest tiles.  
2. Default ViT is **small** (CPU-friendly); scale embed/depth on GPU for stronger SOTA-style results.  
3. This is a **research adaptation**, not a claim of beating every modern SegFormer/Mask2Former out of the box — the contribution is **ViT + paper-style guided concepts for forest segmentation**.

---

*Created to implement advisor feedback: ViT-based segmentation + XAI method from `2101.03919v2.pdf` to improve and explain forest masks.*
