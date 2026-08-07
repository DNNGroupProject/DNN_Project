# Repo Structure & Conventions

This documents the pattern already in use across `Phase1/Dinura-Person3/` and
`Phase1/Lasana-Person4_Evaluation/` — the two most fully-built-out folders —
formalized here so Phase 2 and any remaining Phase 1 work stays consistent
and integrates cleanly across people.

## Folder layout

One folder per person per phase: **`Phase<N>/<FirstName>-Person<M>[_Role]/`**
(e.g. `Phase1/Dinura-Person3/`, `Phase2/Lasana-Person4/`). Work in your own
folder. If you need to touch a teammate's folder (e.g. wiring your output
into their pipeline), keep the edit minimal and additive, and document it —
see "Cross-folder edits" below.

Within a person's folder, use these subfolder names when applicable so
tooling and reviewers know where to look:

```
<Name>-Person<N>/
  README.md            Required. See "README contents" below.
  <module code>.py      Your code, flat or in a package (e.g. attention_consistency/)
  tests/                 Unit tests, runnable without GPU/internet where possible
  checkpoints/            Trained weights: <model>_<variant>_{best,last}.{pt,keras}
  results/                 Logs, metrics tables (.csv + .md), figures (.png)
  *_colab.ipynb            Full-scale GPU version of the same code, if local runs are CPU-smoke-scale only
```

## README contents

Every person folder's `README.md` should cover, in this order (see
`Phase1/Dinura-Person3/README.md` and `Phase1/Lasana-Person4_Evaluation/README.md`
for worked examples):

1. **Role summary** — one line, plus a weeks/deliverables table copied from the proposal's §6 work distribution.
2. **Layout** — a tree of what's in the folder and what each file does.
3. **Quick start / Reproducing** — exact commands to run tests, training, eval, in that order.
4. **Dependencies on teammates** — a table of whose output you need and what you hand off.
5. **Results**, if any exist yet — table of metrics, with an honest caveat noting scale (CPU smoke test vs. full-scale GPU run) so numbers aren't mistaken for final paper results.
6. **Cross-folder edits**, if you touched anyone else's folder — see below.

## Cross-folder edits

If your work requires changing a teammate's folder (e.g. Person 3 adding a
real SegFormer adapter into Person 4's `adapters/` so evaluation stops
stubbing it out), keep it to the minimum additive change, and list exactly
what changed in your own README under "What touched teammates' folders" —
don't silently modify someone else's files. Prefer adding a new file over
editing an existing one; if you must edit an existing file, keep the diff
small enough that the owner can review it at a glance.

## Naming for checkpoints & results

- Checkpoints: `<model>_<variant>_{best,last}.pt` (PyTorch) or `.keras`
  (Keras), e.g. `segformer_b0_att_best.pt`, `segformer_b0_vanilla_last.pt`.
- Comparison/ablation tables: commit both a `.csv` (machine-readable) and a
  `.md` (renders directly in GitHub/PRs) version of the same table.
- Figures: descriptive names, e.g. `prediction_grid_segformer-att.png`,
  `attention_drift_01.png` — not `figure1.png`.

## Tests

Unit tests should run without a GPU and, where possible, without internet
access (offline/randomly-initialized models rather than downloading
pretrained weights) — see `Phase1/Dinura-Person3/tests/` for the pattern.
This keeps tests runnable in CI or on anyone's laptop, not just the Colab
GPU environment used for full-scale training.

## Commit messages

Prefer `<Verb phrase> (Person <N> task <M>)`, matching the task numbering in
the proposal's §6 work-distribution tables, e.g.:

```
Add SegFormer attention hooks and Grad-Rollout (Person 2 task 1).
Integrate Attention Consistency Loss and train preliminary checkpoints (Person 2 task 3).
```

This makes it possible to trace any commit back to the exact proposal
deliverable it implements. Not all existing history follows this — that's
fine, don't rewrite it — but use it going forward.

## Phase boundaries

Phase 1 (`Phase1/`) is the Week 1–4 short-paper scope: baselines, the core
Attention Consistency Loss + adapted Rollout, a first-pass AAMO, and a
work-in-progress short paper. Phase 2 (`Phase2/`) is Weeks 5–12: tuned loss
(λ-sweep, KL-divergence variant), full multi-seed ablation study, optional
Boundary Refinement Module, and the full IEEE paper. Don't mix Phase 2 work
into `Phase1/` folders once Phase 2 starts — create the parallel
`Phase2/<Name>-Person<N>/` folder instead, so Phase 1's short-paper numbers
stay reproducible as a fixed snapshot.
