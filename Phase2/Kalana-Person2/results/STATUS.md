# Full-scale results — not here yet

This directory is empty of metrics on purpose. The paper's current SegFormer
numbers (AAMO 0.0144 → 0.432, etc.) are the Person 3 CPU smoke run
(400 images / 8 epochs) and live under `Phase1/Dinura-Person3/results/`.
Copying them here would make them look like this run.

After `segformer_full_scale_colab.ipynb` finishes, this folder should contain:

- `train_summary_{vanilla,att}.json`
- `training_log_{vanilla,att}.csv`
- `eval_{vanilla,att}.json`
- `baseline_comparison.{csv,md}`
- `prediction_grid_{vanilla,att}.png`
- `attention_drift_figures/attention_drift_0N_full_scale.png`

Then ping Dhinanjaya — those files are what replace the smoke-scale rows.
