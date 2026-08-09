# Shared utilities

Two things live here, both meant to be imported by more than one person:

- [`augmentation.py`](augmentation.py) — the shared train-time augmentation
  pipeline (Person 1, proposal §6.1.2 Weeks 1–2).
- [`experiment_tracking.py`](experiment_tracking.py) — shared Weights & Biases
  config, opt-in.

---

## Augmentation pipeline

Flips, 90° rotations, brightness and contrast, applied to an (image, mask)
pair. numpy only — no torch, no cv2, no albumentations — so it imports from
anyone's pipeline without adding a dependency.

```python
import numpy as np
from shared.augmentation import augment_pair

rng = np.random.default_rng(42)      # once, e.g. in your Dataset.__init__
image, mask = augment_pair(image, mask, rng)
```

Call it **after** `/255.0` and **before** the ImageNet normalize: brightness
and contrast only mean something while pixels are in [0,1]. Pass one seeded
generator for the whole run so results are reproducible, and use
`num_workers=0` (or reseed per worker) — forked DataLoader workers each
inherit a copy of the generator and would replay identical augmentations.

Geometric ops move the image *and* the mask; photometric ops move the image
only. There is deliberately no arbitrary-angle rotation or scaling: those
interpolate, and interpolating a label map manufactures exactly the gray
boundary pixels that [`Phase2/Chanupa-Person1/results/mask_audit.md`](../Phase2/Chanupa-Person1/results/mask_audit.md)
measured as absent from this dataset. Flips and `rot90` leave the mask exactly
binary.

`AugmentConfig` holds the probabilities and ranges; `NO_AUGMENT` turns
everything off and makes `augment_pair` an identity function.

Used by [`Phase1/Chanupa-Person1/dataset.py`](../Phase1/Chanupa-Person1/dataset.py)
and measured by its
[augmentation ablation](../Phase1/Chanupa-Person1/README.md#augmentation-ablation-weeks-34).
Tests: `python Phase1/Chanupa-Person1/tests/test_augmentation.py`.

---

## Experiment tracking (Weights & Biases)

One shared W&B project — `dnn-forest-segformer-xai` — so all four technical
members' runs (baselines, attention-loss training, ablation study) are
comparable in one place instead of scattered across separate CSV logs.

This is **opt-in and non-invasive**: nobody's existing training scripts have
been modified. `shared/experiment_tracking.py` no-ops safely if `wandb` isn't
installed or `WANDB_DISABLED=1` is set, so adding the two lines below never
breaks a script that already works.

## One-time setup

```bash
pip install wandb
wandb login   # paste your API key from wandb.ai/authorize
```

Ask Person 5 (or whoever creates the team's W&B org) for the entity/team
name, then either set it once per shell:

```bash
export WANDB_ENTITY=<team-entity-name>
```

or hardcode it in `shared/experiment_tracking.py`'s `ENTITY` constant.

## Adding it to your script

```python
from shared.experiment_tracking import init_run, log, finish

run = init_run(
    person="person3",                        # person1/2/3/4
    task="attention_consistency_train",       # short slug for this run
    config={"lambda1": 1.0, "lambda2": 0.3, "sigma": 8.0, "epochs": 8},
)

for epoch in range(epochs):
    ...
    log({"epoch": epoch, "train_loss": loss, "dice": dice, "iou": iou})

finish()
```

Suggested `task` slugs, matching each person's existing result files so W&B
runs map cleanly onto `results/*.csv`:

| Person | Suggested `task` slugs |
|---|---|
| 1 — Data & Baselines | `unet_baseline` |
| 2 — Transformer Lead | `segformer_vanilla` |
| 3 — Loss & Training | `attention_consistency_train` (log `lambda1`, `lambda2`, `sigma`, `mode` in config so MSE vs. KL runs are filterable) |
| 4 — Evaluation Lead | `ablation` (one run per ablation row, log the full metrics dict — Dice/IoU/F1/AAMO/params/FLOPs/FPS — from `evaluate.py`) |

## Why this instead of each person's own CSV logs

CSV logs (`results/training_log*.csv`) stay as the source of truth committed
to the repo — this doesn't replace them. W&B is additive, for:

- Comparing loss curves across people's runs without manually merging CSVs.
- The Phase 2 λ-sweep and multi-seed ablation study (Person 3 §6.2, Person 4
  §6.2) — many runs that are much easier to compare in one dashboard than in
  separate files.
- Attention-drift and prediction-grid figures logged as `wandb.Image` so
  they're browsable alongside the metrics that produced them.

## Not done yet

- No team W&B entity has been created — whoever sets one up should update
  `WANDB_ENTITY` here and share the invite link in the group chat.
- Existing training scripts (`train_segformer_smoke.py`,
  `unet_baseline_colab.ipynb`, `ablation_runner.py`, etc.) haven't been wired
  up — each owner should add the three lines above to their own script
  rather than have them added for them, since only the owner can verify the
  run actually completes in their environment (CPU smoke vs. Colab GPU).
