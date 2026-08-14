# Lasana's U-Net adapter — what's needed, checked 2026-08-14

## Status: not started

Checked against `origin/main` @ `f9befcb` (2026-08-14). No commits from Lasana since
`289d716` (DeepLabV3+ extra baseline, 2026-08-09). Specifically:

- `Phase1/Lasana-Person4_Evaluation/adapters/unet_torch.py` does not exist.
- `config.py:23`'s `UNET_CKPT` still points at `Lasana/checkpoints/lasana_unet_best.keras`
  (the superseded Keras model).
- `results/baseline_comparison.md` still shows the old numbers (1.95M params, Dice 0.8492,
  FPS 3.48) — see [[unet_baseline_reconciliation]] for why those are the wrong numbers to
  ship in the paper.

This doc exists because Chanupa's half is now done (checkpoint committed, verified
reproducing exactly) and nothing is blocking Lasana anymore — see below for exactly what
to load and how.

## Checkpoints available (Chanupa-Person1, committed)

| File | Size | What it is |
|---|---|---|
| `Phase1/Chanupa-Person1/checkpoints/unet_baseline_best.pt` | 59 MB | The real baseline. state_dict only, selectively fp16 (tensors ≥10k elements) to fit under GitHub's 100 MiB limit — `load_unet()` below casts back to fp32 on load, so this is transparent to callers. Reproduces test Dice **0.8563**, IoU **0.7534** exactly (verified independently, matches `test_metrics.txt` to 4 decimals). |
| `Phase1/Chanupa-Person1/checkpoints/unet_fixture_random.pt` | 0.5 MB | Untrained, deliberately different channel widths. Use this for adapter unit tests (load-path correctness), never for real numbers — it fails loudly if mistaken for the baseline. |

Model code lives in `Phase1/Chanupa-Person1/unet_model.py` (not in Lasana's folder —
import it across folders, same pattern `adapters/segformer.py` already uses for
Dinura-Person3's code, see below). Key pieces:

```python
from unet_model import load_unet, UNet

model = load_unet("path/to/unet_baseline_best.pt", device="cuda")  # eval mode, fp32, correct width — one call
```

`load_unet` already handles the two things a caller would otherwise have to guess:
architecture width (read back out of the checkpoint's `downs.0.block.0.weight` shape) and
dtype (half-precision tensors are cast back to fp32 on the way in). Don't reimplement
either — just call it.

**Output shape gotcha**: the model head is **2 channels** (background, forest), not 1 —
matches the SegFormer/DeepLab convention in this repo so `F.cross_entropy`/argmax stay
consistent. Forest probability is `softmax(logits, dim=1)[:, 1]`, i.e. identical to
`adapters/deeplab_model.py`'s `forest_prob_from_logits` — reuse that function rather than
writing a new one.

## What to do — three items

### 1. Write `Phase1/Lasana-Person4_Evaluation/adapters/unet_torch.py`

Clone `adapters/deeplabv3.py` (`DeepLabV3Adapter`) — it's the closest existing template:
same `ModelAdapter` interface, same torch device handling, same `_to_tensor` /
`predict_dataset` / `count_params` / `estimate_gflops` / `measure_speed` shape. Swap in:

- `load()`: `self.model = load_unet(checkpoint or self.checkpoint, device=str(self.device))`
  instead of `build_deeplabv3()` + manual `load_state_dict`.
- Cross-folder import, same pattern as `adapters/segformer.py` uses for Dinura-Person3
  (`sys.path.insert` at module top, not a package install):
  ```python
  import sys
  from pathlib import Path
  _CHANUPA_DIR = Path(__file__).resolve().parents[2] / "Chanupa-Person1"
  if str(_CHANUPA_DIR) not in sys.path:
      sys.path.insert(0, str(_CHANUPA_DIR))
  from unet_model import load_unet
  from adapters.deeplab_model import forest_prob_from_logits  # reuse, don't duplicate
  ```
- `predict_dataset`: same normalization question as DeepLab — check whether
  `unet_baseline_colab.ipynb` trained with plain `/255` scaling or ImageNet
  mean/std normalization before picking `_to_tensor`'s transform. (DeepLab uses ImageNet
  norm because torchvision's pretrained backbone expects it; this U-Net was trained from
  scratch, so it almost certainly does NOT want ImageNet normalization — check the
  notebook/`dataset.py`'s transform before assuming either way.)
- `count_params`/`estimate_gflops`/`measure_speed`: identical to `DeepLabV3Adapter`,
  just reuse `count_torch_params`/`gflops_torch`/`measure_fps` from `efficiency.py`.

Add it to `evaluate.py`'s model registry (wherever `--model deeplab` is wired in) as
`--model unet_torch` or replace the existing `--model unet` Keras path — your call on
naming, but the reconciliation doc's decision was to make this the one true baseline going
forward, so it likely should just become what `--model unet` means.

Write a couple of adapter tests against `unet_fixture_random.pt` (loads without error,
correct output shape) rather than the 59 MB real checkpoint, matching how
`Phase1/Chanupa-Person1/tests/test_unet_model.py` already tests `unet_model.py` itself.

### 2. Benchmark FPS

`efficiency.py` already has `measure_fps` — `DeepLabV3Adapter.measure_speed()` (lines
105–115 of `adapters/deeplabv3.py`) is the exact pattern to copy: build a dummy
`(1, 3, 256, 256)` tensor, warm up once, call `measure_fps(_run, warmup=config.FPS_WARMUP,
runs=config.FPS_RUNS)`. GFLOPs (109.48) is already known from Chanupa's notebook via
`thop` — worth cross-checking against `gflops_torch()`'s own measurement for consistency,
but not blocking.

### 3. Re-point config and regenerate results

- `config.py:23`: change `UNET_CKPT` to
  `PHASE1 / "Chanupa-Person1" / "checkpoints" / "unet_baseline_best.pt"`.
- Run `python evaluate.py --model unet_torch` (or whatever flag you land on from item 1).
- Regenerate `results/baseline_comparison.md` and `.csv` with the new row: expect Dice
  0.8563, IoU 0.7534, Params 31,037,698, GFLOPs 109.48, plus your new FPS measurement.

## After this is done

Ping Dhinanjaya once `baseline_comparison.md` is regenerated — the paper draft's U-Net row
(`short_paper_draft.md`, `paper_acm/main.tex`) and the SegFormer-parameter-efficiency prose
still cite the old 1.95M/0.8492 numbers on purpose, held back until these real ones land.
Also flagging separately: Chanupa's full-scale augmentation ablation
(`Phase1/Chanupa-Person1/results/augmentation_ablation.md`) isn't in the paper yet either —
different task, not blocking this one, but worth batching into the same paper-update pass.
