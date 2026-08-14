# U-Net baseline reconciliation — action needed from Chanupa & Lasana

## The problem

Two different U-Net models are both currently labeled **"U-Net (CNN baseline)"** in our results and the paper draft. This was flagged by Chanupa in `Phase1/Chanupa-Person1/README.md` ("Note on which U-Net is which") but not yet resolved.

| | Lasana's U-Net (currently used) | Chanupa's U-Net (proposed replacement) |
|---|---|---|
| Framework | Keras | PyTorch |
| Params | 1,951,105 (1.95M) | 31,040,000 (31.04M) |
| Test Dice | 0.8492 | 0.8563 |
| Test IoU | 0.7379 | 0.7534 |
| Test F1 | 0.8492 | (same as Dice) |
| GFLOPs | n/a (not recorded) | 109.48 @ 1×3×256×256 |
| FPS | 3.48 | not recorded |
| Training data | Unclear — checkpoint lives in `Lasana/checkpoints/lasana_unet_best.keras`, no committed training log/config in the Phase 1 folder | Full dataset, 5,108 pairs, split 3576/766/766, seed 42, 20 epochs, batch 8, lr 1e-3, 256×256 |
| Source location | `Lasana/` — **pre-Phase1 folder, marked superseded in the repo root `README.md`** ("early... research; superseded by `Phase1/Lasana-Person4_Evaluation/`... don't build on them") | `Phase1/Chanupa-Person1/unet_baseline_colab.ipynb` |
| Checkpoint committed to git? | Yes (`.keras` file) | **No** — only the notebook, `test_metrics.txt`, `training_log.csv`, and result images are committed. No `.pt`/`.pth` weights file exists in the repo. |

## Decision

We're going with **Chanupa's PyTorch U-Net** as the official baseline going forward. Two reasons:

1. **Apples-to-apples comparison.** Chanupa's notebook is a deliberate structural clone of Kalana's SegFormer training notebook — same dataset class, same split (seed 42, 70/15/15), same loss, same metric, same training loop. Only the model differs. That's exactly what a baseline is supposed to be for, and it's not true of Lasana's checkpoint (no matching training config committed anywhere in Phase 1).
2. **Superseded folder.** The currently-used checkpoint lives inside `Lasana/`, which the repo's own root `README.md` explicitly marks as superseded pre-Phase1 work ("don't build on them"). We're currently building the official baseline table on top of a folder we agreed not to build on.

Trade-off to be aware of: Chanupa's model is ~16× larger (31.04M vs 1.95M params) with real GFLOPs recorded (109.48) vs. none for Lasana's. If the paper leans on the "SegFormer is far more parameter-efficient than the CNN baseline" story, that comparison gets less dramatic with the bigger U-Net. Numbers will need re-checking against the SegFormer efficiency claims once this is done.

## Where the current (wrong) number is already live

The 1.95M / 0.8492 numbers have already propagated into:
- `Phase1/Lasana-Person4_Evaluation/results/baseline_comparison.md` and `.csv`
- `Phase1/Lasana-Person4_Evaluation/config.py:23` — the actual live code:
  ```python
  UNET_CKPT = PROJECT / "Lasana" / "checkpoints" / "lasana_unet_best.keras"
  ```
  This is what `evaluate.py --model unet` currently loads — it's not just a stale doc reference, it's what's producing today's numbers.
- The paper draft: `Phase1/Dhinanjaya-Person5/short_paper_draft.md:36,185` and the compiled `Phase1/Dhinanjaya-Person5/paper_acm/main.tex:71,250`. **I'll hold off updating the paper's numbers until the new checkpoint + eval numbers exist**, so we don't end up updating it twice.

## What's needed — Chanupa

- [x] Export and commit the trained PyTorch U-Net weights (`.pt`/`.pth`) from the Colab run into `Phase1/Chanupa-Person1/checkpoints/` (or wherever fits the folder convention). Right now the trained model only lives in the Colab session/Drive — nothing downstream can load it without this.

### Done — Chanupa's reply

**The checkpoint is in:** `Phase1/Chanupa-Person1/checkpoints/unet_baseline_best.pt`, 59 MB. Load it with `unet_model.load_unet(path)` — it infers the architecture width from the checkpoint and handles the dtype, so the adapter is a thin wrapper over one call.

Two things had to happen first. The architecture only existed inside a notebook cell, so even with the `.pt` in hand there was nothing to load a `state_dict` into — it's now `Phase1/Chanupa-Person1/unet_model.py`, a verbatim lift that still reads 31,037,698 params. `dataset.py` is the matching lift of the loader and split.

**The size problem, and a trap in it.** The fp32 file is 124 MB, over GitHub's 100 MiB hard limit. The obvious fix — casting the whole `state_dict` to fp16 — produces a healthy-looking 62 MB file that **silently drops test Dice 0.8563 → 0.7510 and IoU 0.7534 → 0.6095**. Nothing errors. The damage is in the BatchNorm buffers, not the conv weights: fp16 caps at 65504 and loses normals below ~6e-5, and BN divides by `sqrt(running_var + eps)`. Casting only tensors with ≥10,000 elements leaves the 36 buffers at full precision, lands at 59 MB, and measures **0.8563 / 0.7534 exactly**. Worth knowing if the SegFormer checkpoints ever need shrinking. Full table in `Phase1/Chanupa-Person1/README.md`.

**Two confirmations that came out of verifying it**, both relevant beyond this item:

- **The committed baseline reproduces exactly.** Re-running the 766-image test set through `dataset.py` + `unet_model.py` (not the notebook) gives `loss=0.4302 dice=0.8563 iou=0.7534`, matching `test_metrics.txt` to four decimals, on a 3576/766/766 split. The number going into the paper is solid.
- **Measure any checkpoint you regenerate.** 25 seconds on a T4. The blanket-fp16 file passed every structural check.

There's also `checkpoints/unet_fixture_random.pt` (0.5 MB, untrained, deliberately a different width) as a fast fixture for loader tests — it fails loudly if anything mistakes it for the baseline.

## What's needed — Lasana

- [ ] Write a PyTorch U-Net adapter (e.g. `adapters/unet_torch.py`) analogous to the existing `adapters/unet_keras.py`, so `evaluate.py --model unet` can load Chanupa's checkpoint instead of (or alongside) the Keras one.
- [ ] Benchmark FPS for Chanupa's model — `efficiency.py` already has `measure_fps`, just needs to run against the new adapter. GFLOPs (109.48) is already known from Chanupa's notebook; worth double-checking it against `efficiency.py`'s own GFLOPs measurement for consistency.
- [ ] Once the adapter works, re-point `config.py:23`'s `UNET_CKPT` at the new checkpoint, re-run `evaluate.py --model unet`, and regenerate `results/baseline_comparison.md`/`.csv` with the new numbers.

## After both are done

Ping me (Dhinanjaya) once the new `baseline_comparison.md` is regenerated — I'll update the paper draft's U-Net row and the efficiency-comparison prose to match, and re-verify the ACM PDF still compiles within the page limit.

---

## Status

Chanupa's half is done; Lasana's three items are unblocked and outstanding. Nothing is waiting on Chanupa.

One thing for the paper, Dhinanjaya, separate from this reconciliation: the augmentation ablation is now run at full scale (`Phase1/Chanupa-Person1/results/augmentation_ablation.md`). Augmentation **does not help** the U-Net baseline — test Dice 0.8604 → 0.8505 with it on. Supportable claim: *"augmentation did not improve the U-Net baseline under our training budget (−0.010 Dice)."* Not supportable: that augmentation doesn't help this task — neither arm ever overfits in 20 epochs, which is the regime where it would pay, and it's a single seed. The caveats are written up in that folder's README.
