# DNN_Project — Explainability-Guided SegFormer for Forest Cover Segmentation

Semester 5 undergraduate research project. Core idea: supervise a SegFormer-B0's
internal attention against the ground-truth forest mask during training (an
**Attention Consistency Loss**), using an adaptation of Gradient-weighted
Attention Rollout for SegFormer's spatial-reduction attention, and a new
quantitative interpretability metric (**AAMO**). Benchmarked against U-Net and
vanilla SegFormer-B0 baselines. Full motivation, methodology, and math are in
[`Research_Proposal_extended.pdf`](Research_Proposal_extended.pdf); see also
[`Research_Proposal .pdf`](<Research_Proposal .pdf>) for the original version.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the repo's folder/naming/documentation
conventions before adding new work.

## Team

| # | Name | Role |
|---|---|---|
| 1 | Chanupa | Data & Baselines |
| 2 | Kalana | Transformer Lead |
| 3 | Dinura | Loss & Training |
| 4 | Lasana | Evaluation Lead |
| 5 | Dhinanjaya | Writing & Technical Integration Lead |

## Repo map

### Current work — `Phase1/`

One folder per person, `Phase1/<Name>-Person<N>[_Role]/`, each with its own
`README.md` describing what's inside. Start there.

| Folder | Owner | Contents |
|---|---|---|
| [`Phase1/Chanupa-Person1/`](Phase1/Chanupa-Person1/) | Person 1 | U-Net baseline notebook + full-dataset results |
| [`Phase1/Kalana-Person2/`](Phase1/Kalana-Person2/) | Person 2 | SegFormer-B0 training, dataset images/masks, checkpoints |
| [`Phase1/Dinura-Person3/`](Phase1/Dinura-Person3/) | Person 3 | Attention hooks, adapted Grad-Rollout, Attention Consistency Loss, math formulation, tests |
| [`Phase1/Lasana-Person4_Evaluation/`](Phase1/Lasana-Person4_Evaluation/) | Person 4 | Shared eval package: metrics, efficiency, AAMO, ablation runner, comparison tables |
| [`Phase1/Dhinanjaya-Person5/`](Phase1/Dhinanjaya-Person5/) | Person 5 (you) | Writing (related work, paper drafts), integration notes |

### Shared utilities

- [`shared/`](shared/) — shared Weights & Biases experiment-tracking config, opt-in for all four technical members. See [`shared/README.md`](shared/README.md).

### Reference material

- [`Literature_Review_Papers/`](Literature_Review_Papers/) — source PDFs cited in the proposal's related work.
- [`Research_Proposal_extended.pdf`](Research_Proposal_extended.pdf), [`Research_Proposal .pdf`](<Research_Proposal .pdf>) — the proposal documents.

### Pre-Phase1 / exploratory (superseded, kept for history)

These predate the `Phase1/<Name>-PersonN/` convention and are **not** where
current work happens — don't build on them, but don't delete them either,
they contain early results/notes that informed the Phase 1 folders above.

- `Dinura/` — early dev iterations (`dev0`, `dev1`) and `project1-Do-not-use/` (explicitly marked unused by its own README).
- `Lasana/` — pre-Phase1 U-Net training + ViT XAI postprocessing research; superseded by `Phase1/Lasana-Person4_Evaluation/`.
- `ViT_XAI_Segmentation/` — early standalone ViT explainability experiment (pseudo-concepts, not the SegFormer/AAMO pipeline used now).

### Coming — `Phase2/`

Phase 2 (Weeks 5–12: full ablation study + full paper) will follow the same
`Phase2/<Name>-Person<N>/` convention once it starts. See
[CONTRIBUTING.md](CONTRIBUTING.md).
