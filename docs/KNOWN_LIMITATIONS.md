# Known limitations

## Blocking

* **No real data available.** The processed CSV is missing (see
  `DATA_PROVENANCE.md`). All verification so far uses synthetic fixtures; the
  pipeline has not been trained on the TCGA cohort.
* **Label provenance unknown.** `metastasis_status` is not constructed in any
  supplied notebook. Until its definition is recovered, results cannot be
  interpreted clinically even if the CSV is found.
* **R pipeline unexecuted.** `scripts/build_data.R` is a careful reconstruction
  but was not run (no R toolchain in the verification environment).

## Statistical

* **Small, imbalanced cohort.** With 220 patients and 33 positives, the 20%
  hold-out contains ~7 metastatic patients. A single misclassification moves
  recall by ~14 points; test-set metrics will have wide uncertainty. Report
  nested-CV mean ± SD alongside the hold-out numbers, and prefer ROC-AUC,
  PR-AUC and balanced accuracy over accuracy (a constant "non-metastatic"
  predictor scores ~85% accuracy).
* **No external validation.** All patients come from TCGA LUAD/LUSC.
* **Whole-cohort, label-free preprocessing** in the R build (top-SD gene
  filter, mutation-rate filter, missForest imputation) is not fold-isolated.
* **Threshold** is fixed at 0.5 on the meta-classifier probability; it is not
  tuned or calibrated. Probabilities from RF on SMOTE-augmented data are not
  calibrated risks.
* **Nested-CV cost.** Each outer fold refits the entire stack, including its
  inner OOF loop (5 × 6 autoencoder fits plus the final model). About 2
  minutes on a laptop CPU at the reported data size; acceptable, but heavier
  configurations will scale linearly.

## Fidelity to the original

* Results from the corrected pipeline are **not expected to match** the
  91-94% accuracies in the write-up: those were measured on SMOTE-augmented
  test splits containing synthetic neighbours of training patients, with
  stacking meta-features produced in-sample, and the ensemble figure value was
  typed into the plotting cell.
* The autoencoder noise level was changed from σ = 50 (applied twice) to
  σ = 0.5 (applied once). `configs/legacy_hyperparameters.yaml` keeps 50 for a
  sensitivity check.
* The RNA branch follows the Final notebook (RF on the 32-d latent code); the
  LASSO notebook's variant was a bug (latent indices used as gene indices) and
  has no meaningful reproduction.
* Stacking uses probabilities rather than hard labels by default.
* `legacy` mode re-runs only the per-modality leaky protocol. The original
  fusion is not re-implemented because it has no coherent leakage-free
  interpretation.

## Engineering

* Keras determinism: `keras.utils.set_random_seed` is set per fit, but
  oneDNN/threading can still introduce tiny run-to-run differences on CPU. Set
  `TF_ENABLE_ONEDNN_OPTS=0` for closer reproducibility.
* Tested on Windows 11, Python 3.12.10, CPU only. Python 3.11 (as requested in
  the brief) was not installed on the build machine; `pyproject.toml` allows
  3.10-3.12.
* The artifact format stores DAE weights inside the `base_rs.joblib` pickle.
  Pickles are not portable across major library versions and must only be
  loaded from trusted sources.

## Use

Research and educational use only. Not a medical device. Not validated for
any clinical decision, diagnosis, or prognosis for any individual patient.
