# NSCLC metastasis prediction from TCGA multi-omics

A clean, tested reconstruction of my 2023–2024 research project that predicts
metastatic status in non-small-cell lung cancer (TCGA LUAD + LUSC) by fusing
four molecular data types. I originally built it as a set of Google Colab
notebooks. This repository is a transformation of those notebooks into an installable Python
package, keeps the original architecture, and **corrects the data-leakage and
evaluation errors in the original analysis**.

> **Status.** The code, tests and documentation are complete. **No model has
> been trained on the real cohort yet**: the processed TCGA table the notebooks
> used isn't available, and the notebooks never define the metastasis label (see
> [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md)). The accuracies reported
> in the original work have **not been independently reproduced**.
> See [`RESULTS.md`](RESULTS.md).
>
> Research code only. Not for clinical use.

---

## Architecture

```
                 ┌──────────── per patient ─────────────┐
 rs_ (RNA-seq) ─▶ impute ▸ SMOTE ▸ scale ▸ Denoising AE (64→32→64) ▸ RF ─┐
 cn_ (CNV)     ─▶ impute ▸ SMOTE ▸ LASSO-CV  | PCA(100)          ▸ RF ─┤  out-of-fold
 mu_ (mutation)─▶ impute ▸ SMOTE ▸ LASSO-CV  | PCA(100)          ▸ RF ─┤  P(metastasis)
 pp_ (RPPA)    ─▶ impute ▸ SMOTE ▸ LASSO-CV  | PCA(100)          ▸ RF ─┘        │
                                                                                ▼
                                                     Random-forest meta-classifier
```

* **Data:** TCGA LUAD/LUSC via `curatedTCGAData` 2.1.1. The assays are
  `RNASeq2GeneNorm`, `GISTIC_ThresholdedByGene`, `Mutation` and `RPPAArray`,
  aligned by patient. The write-up reports 220 patients (33 metastatic) and
  1,558 features.
* **RNA-seq branch:** a denoising autoencoder (Gaussian noise, 64→32 encoder,
  symmetric decoder, 50 epochs) compresses the expression profile. A random
  forest then classifies the 32-dimensional latent code.
* **CNV, mutation and RPPA branches:** a random forest (100 trees, seed 42)
  runs after either LASSO-CV feature selection (the later notebook, used in
  `configs/baseline.yaml`) or 100-component PCA (the earlier notebook, used in
  `configs/pca_variant.yaml`).
* **Fusion:** a random-forest meta-classifier trained on each modality's
  out-of-fold metastasis probability.

## What was corrected

| Original notebook | This repository |
|---|---|
| SMOTE applied to the **whole dataset before** the train/test split, so test sets contained synthetic copies of training patients | Patients are split first. SMOTE, imputation, scaling, LASSO, PCA and the autoencoder are all fitted **inside training folds** (`imblearn` pipelines) |
| Stacking meta-features were predictions on patients the base models had **already been trained on** | **Out-of-fold cross-fitting**: each patient's meta-features come from models that never saw that patient's label |
| `StackingClassifier` silently **refit** the "loaded" omics models on a 4-column table of predictions | The four modality pipelines and the meta-model are separate, explicitly fitted objects aligned by patient ID |
| The RNA branch used latent-unit indices (0–31) to select **gene columns**, so the autoencoder had no effect | The RF consumes the encoder's latent representation |
| Autoencoder noise σ = 50 applied twice, with the test set used as validation data | Noise applied once with σ = 0.5 (configurable). No test data during training |
| Accuracy on SMOTE-balanced test splits. The ensemble's 94.32% was typed into the plotting cell | Accuracy, balanced accuracy, ROC-AUC, PR-AUC, precision, recall, F1 and κ on real held-out patients, plus nested 5-fold CV and split manifests |
| `.h5` / `.joblib` / `.pkl` mismatch, no saved scaler, variables overwritten across cells | One versioned model-artifact directory holds every fitted step |

The cell-by-cell audit is in [`docs/REPRODUCTION_AUDIT.md`](docs/REPRODUCTION_AUDIT.md).

## Repository layout

```
reference_notebooks/     original Colab notebooks (source preserved, outputs stripped)
src/nsclc_metastasis/
  config.py              typed YAML configuration
  data.py                loading, schema validation, patient alignment
  preprocessing.py       FoldSafeSMOTE, LassoSelector, SafePCA
  autoencoder.py         denoising autoencoder (Keras) as a sklearn transformer
  modality_models.py     per-modality imblearn pipelines
  stacking.py            out-of-fold stacking ensemble
  evaluation.py          hold-out + nested CV protocol, metrics, plots
  inference.py           artifact save/load, prediction
  legacy.py              ORIGINAL leaky protocol (comparison only)
  synthetic.py           SYNTHETIC data generator for tests
  cli.py                 command-line interface
scripts/build_data.R     TCGA download + preprocessing (from NSCLCData.ipynb)
configs/                 baseline.yaml · pca_variant.yaml · legacy_hyperparameters.yaml
tests/                   pytest suite (synthetic fixtures only)
docs/                    REPRODUCTION_AUDIT · DATA_PROVENANCE · KNOWN_LIMITATIONS
RESULTS.md               reported vs. verified vs. blocked
```

## Setup (Windows PowerShell)

```powershell
git clone https://github.com/smokyfishy/NSCLC-Metastasis.git
cd NSCLC-Metastasis
py -3.12 -m venv .venv            # Python 3.10–3.12 supported; verified on 3.12.10
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
# exact versions used for verification:
# python -m pip install -r requirements.txt
pytest -q
```

On macOS or Linux, use `python3 -m venv .venv` and `source .venv/bin/activate`.
Everything runs on CPU. If you need closer run-to-run determinism from
TensorFlow, set `$env:TF_ENABLE_ONEDNN_OPTS = "0"`.

## Usage

Put the processed table at `data/raw/NSCLCMultiOmicsDataProcessed_reordered.csv`
(git-ignored). The required schema is in
[`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md).

```powershell
# 1. Check the schema and compare counts with the write-up
python -m nsclc_metastasis.cli validate-data --data data/raw/NSCLCMultiOmicsDataProcessed_reordered.csv

# 2. Train: hold-out split, nested 5-fold CV, final fit, reports and artifact (~2 min on CPU)
python -m nsclc_metastasis.cli train --data data/raw/NSCLCMultiOmicsDataProcessed_reordered.csv `
    --config configs/baseline.yaml --model-dir artifacts/model

# 3. Re-score the saved model on its recorded held-out patients
python -m nsclc_metastasis.cli evaluate --model artifacts/model --data data/raw/NSCLCMultiOmicsDataProcessed_reordered.csv

# 4. Predict for new patients (the label column is optional)
python -m nsclc_metastasis.cli predict --model artifacts/model --data path\to\new_patients.csv --out reports/predictions.csv

# Optional: the PCA variant, and the ORIGINAL leaky protocol for comparison
python -m nsclc_metastasis.cli train --data <csv> --config configs/pca_variant.yaml --model-dir artifacts/model_pca
python -m nsclc_metastasis.cli legacy --data <csv>
```

`nsclc-metastasis <command>` is installed as an equivalent console script.

To try the full pipeline without real data, generate a **synthetic** table with
the reported dimensions. The resulting metrics mean nothing.

```powershell
python -m nsclc_metastasis.cli make-synthetic --reported-shape --out data/processed/SYNTHETIC_reported_shape.csv
python -m nsclc_metastasis.cli train --data data/processed/SYNTHETIC_reported_shape.csv --model-dir artifacts/SYNTHETIC_model
```

### Programmatic use

```python
from nsclc_metastasis.config import load_config
from nsclc_metastasis.data import load_dataset
from nsclc_metastasis.evaluation import run_experiment, write_reports
from nsclc_metastasis.inference import save_model, load_model

cfg = load_config("configs/baseline.yaml")
ds = load_dataset("data/raw/NSCLCMultiOmicsDataProcessed_reordered.csv", cfg.data)
result = run_experiment(ds, cfg)            # split -> nested CV -> final fit -> test
write_reports(result, cfg, out_dir=__import__("pathlib").Path("reports/baseline"))
save_model(result.model, "artifacts/model", ds, result.split_manifest)

model = load_model("artifacts/model")
model.predict_detail(ds.X.head())          # per-modality probabilities + ensemble
```

## Evaluation protocol

1. **Hold-out split.** One stratified, patient-level 80/20 split (seed 42).
   The test patients are left untouched until step 4.
2. **Nested CV.** 5-fold stratified CV on the training partition. In each
   fold the entire stack is refit, including its inner 5-fold out-of-fold
   loop, SMOTE, LASSO/PCA and the autoencoder. Per-fold scores go to
   `cv_fold_metrics.csv`.
3. **Final fit.** The final model is fitted on the whole training partition.
4. **Test.** The final model is scored once on the held-out patients.

`train` writes the following to `reports/<config name>/`: `metrics.json`,
`cv_fold_metrics.csv`, `split_manifest.csv` (each patient's partition and
fold), `train_oof_meta_features.csv`, `test_predictions.csv`,
`feature_importance_<modality>.csv`, and ROC, PR, confusion-matrix and
modality-importance plots.

With about 7 metastatic patients in a 44-patient test set, hold-out
estimates are noisy. Read them alongside the nested-CV mean ± SD.

## Tests

The `pytest` suite uses **synthetic data only**. It covers:

* Schema handling: prefixes, IDs, labels, missing values, column alignment.
* Leakage guards: held-out rows never reach SMOTE or LASSO, SMOTE runs only
  at fit time, and the stacking features are truly out-of-fold (a
  memorisation test).
* Each modality pipeline: LASSO selecting no features, PCA clamping, and
  the autoencoder's latent shape and pickling.
* The stack: patient-ID alignment and invariance to row and column order.
* Artifacts and CLI: model save/load round-trip, and train → evaluate →
  predict end to end.

## Data and ethics

TCGA open-access data are governed by NIH GDC policies. This repository
doesn't contain patient-level data or trained weights, and `.gitignore`
blocks them. The model is a research artifact trained on a small
retrospective cohort. It must not be used to inform diagnosis or treatment.

## Original materials

* Notebooks: [`reference_notebooks/`](reference_notebooks/) (Colab links in `SOURCES.md`)
* Reconstruction brief: [`docs/RECONSTRUCTION_BRIEF.md`](docs/RECONSTRUCTION_BRIEF.md)
