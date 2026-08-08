# Person 1 — Data support (Phase 2)

Role: **Data & Baselines**. Proposal §6.2.2, Weeks 5–9: *"Light support role:
assist with data-related debugging and any additional augmentation/preprocessing
needs as they arise."*

| Weeks | Deliverable | Status |
|---|---|---|
| 5–9 | Data-related debugging support | Ongoing — first item is the mask audit below |
| 5–9 | Additional augmentation / preprocessing as needed | Not requested yet |

## Why the mask audit exists

The team believed the masks needed careful preprocessing to be usable. The
rationale is written down in `Lasana/README.md` (§4, "Correct mask
preprocessing"): resizing masks with the default bilinear filter supposedly
"creates gray in-between pixels on boundaries", and `/255.0` without a
threshold is unsafe because "JPEG masks are not perfectly 0/255". That
reasoning is why every loader in the repo does `INTER_NEAREST` + `> 127`, and
why `Phase1/Kalana-Person2/main.py:123-130` — which does neither — looked like
a bug worth fixing.

Nobody had measured it. There is no data-validation utility anywhere in the
repo, and the only mask sanity check that exists is a single `.unique()` print
inside one notebook. So before changing any loader, I measured the data.

## What the audit found

Full run over all 5,108 pairs (334,757,888 mask pixels) — see
[`results/mask_audit.md`](results/mask_audit.md):

- **Every image and every mask is exactly 256×256.** `IMG_SIZE = 256`
  everywhere, so every `resize` call in the repo is a no-op. The
  bilinear-on-a-label-map concern is real in principle but never executes on
  this dataset.
- **Only 20 of 256 grey levels occur at all**: `0–9` and `246–255`. No pixel
  sits more than 9 levels from pure black or pure white.
- **Mid-gray pixels (32–223): exactly 0.** Not "rounds to 0.0000%" — the count
  is zero. The "gray in-between pixels" the docs describe do not exist here.
- `mask / 255.0` and `mask > 127` differ by a **mean of 0.000098** per pixel,
  and **0.0000%** of pixels differ by more than 0.1.
- Pairing is clean: 0 orphan images, 0 orphan masks, 0 unreadable files.
- Forest pixel ratio 0.6117 ± 0.3306 — the dataset leans forest-majority, and
  the per-mask spread is wide enough that anyone reporting pixel accuracy
  should expect it to flatter the model.

**Conclusion: the mask normalization difference is a real code-quality problem
but not a correctness problem on this dataset.** No committed metric is wrong
because of it. Anyone re-running a Phase 1 result should expect the same
numbers whether the threshold is applied or not.

That conclusion is dataset-specific, not general. It stops holding the moment
`IMG_SIZE` changes, a differently-sized dataset is swapped in, or masks are
re-exported at a lower JPEG quality — which is exactly why the audit is a
committed script rather than a one-off note.

## Layout

```
mask_audit.py                    CLI audit: pairing, geometry, value spread,
                                  binarisation cost, class balance
results/mask_audit.csv           The table, machine-readable
results/mask_audit.md            The same table, renders in GitHub/PRs
tests/test_mask_audit.py         15 tests, synthetic arrays only
```

## Reproducing

```bash
python tests/test_mask_audit.py
```

```bash
python mask_audit.py
```

Both from `Phase2/Chanupa-Person1/`. Needs `numpy` and `Pillow` only —
deliberately not `cv2`, which most loaders here import but which isn't
installed in every environment the team works in.

Defaults to `Phase1/Kalana-Person2/{images,masks}` (5,108 pairs, genuinely
tracked in git, so this works on a fresh clone with no dataset download).
Override with `--images` / `--masks`, speed it up with `--sample-every 10`.

The script exits **0** when integrity checks pass, **1** when they fail (orphan
pairs, unreadable files, or a size that disagrees with `--expect-size`), and
**2** when a directory doesn't exist — so it can be dropped in front of a
training run as a pre-flight gate.

## Dependencies on teammates

| Direction | What |
|---|---|
| I need | Nothing. The audit runs off the committed dataset. |
| I hand off | The measured mask/dataset properties below, for anyone writing the paper's dataset section or deciding on a loader. |

## Cross-folder edits

**None.** No file outside this folder was modified. The three findings below
are reported to their owners rather than changed by me.

### Reported, not changed

**1. `Phase1/Kalana-Person2/main.py:123-130` (Person 2)** — the mask is
resized with the default bilinear filter and then divided by 255 under a
comment that says "Convert mask values to 0 or 1", which it does not do. The
matching change, if Person 2 wants it:

```python
mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
mask = (mask > 127).astype(np.float32)
```

Worth doing so the code matches its comment and survives a change of
`IMG_SIZE`, but per the audit it **changes no current result** — do not
re-run anything or restate any number on account of it.

**2. `Lasana/README.md:127-134`** — the justification for thresholding
("JPEG masks are not perfectly 0/255", "soft interpolation creates gray
in-between pixels on boundaries") does not match the data: zero mid-gray
pixels exist, and no pixel is more than 9 levels off an extreme. The advice it
gives is still the right advice; only the stated reason is wrong. Worth
correcting before the paper's dataset section inherits it.

**3. `Phase1/Lasana-Person4_Evaluation/config.py:8` (Person 4)** — `DATA_DIR`
points at `Lasana/dataset/Forest Segmented/Forest Segmented`, which is
gitignored (`.gitignore:8`) and absent from the checkout, so `evaluate.py`
cannot run as committed. Person 3 already documents working around this by
reading `Phase1/Kalana-Person2/{images,masks}` instead
(`attention_consistency/data.py:1-7`). The two evaluation paths therefore point
at different data roots. Not fixed here because changing it changes which data
Person 4's eval reads, which is their call.

## Not done

No augmentation pipeline and no augmentation/preprocessing changes — none have
been requested yet, and this branch was scoped to the mask question. The
proposal's augmentation ablation (§6.1.2, Weeks 3–4) also remains outstanding;
see [`Phase1/Chanupa-Person1/README.md`](../../Phase1/Chanupa-Person1/README.md).
