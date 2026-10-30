# Reproduction audit

This document maps every substantive cell of the four original Colab notebooks to
the module that now implements it, records the original hyperparameters, and
lists each defect found and how it was handled. Cell numbers are 0-based
indices into `reference_notebooks/*.ipynb` (outputs were stripped before
these notebooks were added; source is unchanged).

| Notebook | Role | Cells |
|---|---|---|
| `NSCLCData.ipynb` | R: TCGA download, cleaning, merge, CSV export | 77 |
| `Final of NSCLCMultiomics.ipynb` | Python: EDA, 4 modality models (PCA variant), stacking | 53 |
| `LASSO of NSCLCMultiomics.ipynb` | Python: later LASSO variant, CV stacking experiments, figures | 63 |
| `Copy of NSCLCMultiomics.ipynb` | Historical variant of the LASSO notebook + a "deep multi-view" MLP experiment | 53 |

---

## 1. Data preparation — `NSCLCData.ipynb` → `scripts/build_data.R`

| Cells | Original behaviour | New implementation | Notes / defects |
|---|---|---|---|
| 1-12 | Install/load `curatedTCGAData`, `MultiAssayExperiment`, `RaggedExperiment`, `missForest`, `dplyr`, `purrr`, `doParallel`, `UpSetR`, `TCGAretriever`, `correlation`, `tictoc` | Package list at top of `build_data.R` | `TCGAretriever`, `correlation`, `tictoc`, `UpSetR` were never needed for the export; omitted. |
| 13-14 | `curatedTCGAData(c("LUAD","LUSC"), c("GISTIC_ThresholdedByGene","Mutation","RNASeq2GeneNorm","RPPAArray"), version="2.1.1")` | Same call | Unchanged. |
| 18 | Flatten `colData`; drop clinical columns with >30% NA; keep histology ∈ {LUAD NOS, LUSC NOS} | Same thresholds | Original filter lists the LUSC string twice (harmless). |
| 19, 30 | Copy-number gene symbols taken from `LUAD_GISTIC_ThresholdedByGene-20160128` row metadata and assigned to `listAssays[[1]]` | Symbols taken from each GISTIC experiment's own `rowData` | **Positional indexing**: `listAssays[[1]]`, `[[2]]`, `[[3]]`, `[[7]]` were used for CN, mutation, RPPA, RNA-seq. With 8 experiments (4 assays × 2 cohorts) the mapping depends on ExperimentList order and probably mixed LUAD-only and LUSC-only assays. The rebuild selects experiments **by name** and column-binds LUAD + LUSC on shared genes. This may change cohort composition; see DATA_PROVENANCE. |
| 25 | RNA-seq: truncate barcodes, `log2(pmax(x,0)+0.1)`, transpose, keep top-2000 SD genes | Same transform and SD filter | SD ranking uses the whole cohort (label-free; not fold-isolated). Final CSV has 401 `rs_` columns, not 2000 — the reduction came from the `batchone.csv` intersection (cells 67-74). |
| 37-45 | Mutation `RaggedExperiment` reshaped into a binary gene × patient matrix via `coalesce` and `foreach(i=1:232, j=1:231)`; keep genes with mutation rate > 0.02 | Long table (patient, gene) → binary matrix; same 0.02 threshold | Hard-coded loop bounds; comment says "less than 5%" while code uses 2%. 2% kept. |
| 47-56 | RPPA: drop proteins with >50% NA; `missForest(maxiter=10, ntree=100)`, `set.seed(4578)` | Same | Imputation is fitted on the whole cohort before any split (label-free). Python pipeline adds in-fold median imputation for any remaining NaN. |
| 57-60 | Prefix columns `pp_`, `mu_`, `rs_`, `cn_` | Same | Cell 57 comment says "rs_" but code writes "pp_" (code is correct). |
| 61 | Intersect patients across the four blocks, `cbind` | Same | |
| 63 | `write.csv(dfcolData, "metastasisstatusb1.csv")` | — | Clinical table exported, but **no cell constructs `metastasis_status`**. The label provenance is undocumented. |
| 66 | Random column shuffle selecting 860/249/604/218 columns (`set.seed(123)`) | Not reproduced | Result (`finalDataset`) is never used afterwards. |
| 67-74 | Read `batchone.csv` (origin unknown), intersect columns with `finalData`, export `batchone_filtered.csv` / `batchtwo_filtered.csv` | Optional `--feature-list` | `batchone.csv` is not in the bundle. Its header determined the final feature set. |
| 75-76 | Read `NSCLCMultiOmicsDataProcessed.csv`, reorder columns `rs_, cn_, mu_, pp_`, write `..._reordered.csv` with `row.names = FALSE` | Writes `patient_id` as first column | `NSCLCMultiOmicsDataProcessed.csv` is read but never written by this notebook. **Patient IDs were dropped** by `row.names = FALSE`, so the modelling CSV cannot be traced back to TCGA barcodes. |

## 2. Exploratory analysis (Final 0-14, LASSO 0-14) — not part of the package

t-tests and linear regression on MYH11, CNV–RNA correlation clustermap, PCA/KMeans,
t-SNE with Kendall τ between the two t-SNE axes (not with the label, despite the
comment). These are descriptive, fitted on all patients, and do not feed the
model. They are left in the reference notebooks. Leakage note: any feature
choice inspired by these full-cohort plots would leak; none was carried over.

## 3. RNA-seq branch (`rs_`) → `autoencoder.py`, `modality_models.py`

| Source | Original | Corrected |
|---|---|---|
| Final 15 / LASSO 16 | `SMOTE(...).fit_resample(X_rs, y)` on all patients, then `train_test_split(test_size=0.2, random_state=42)` | Split first; SMOTE runs inside the pipeline on training folds only (`FoldSafeSMOTE`) |
| | `sampling_strategy="auto"` (Final), `0.6` (LASSO, Copy) | `0.6` in `baseline.yaml`, `auto` in `pca_variant.yaml` |
| | `StandardScaler` fitted on (augmented) training split | `StandardScaler` inside the pipeline (post-SMOTE, like the original) |
| | `noise_factor = 50`; inputs pre-noised with `K.random_normal` **and** `GaussianNoise(50)` layer | Single `GaussianNoise` layer, `noise_stddev = 0.5` (configurable; 50 kept in `legacy_hyperparameters.yaml`) |
| | Dense 64 → 32 (relu) → 64 → n (linear), MSE, Adam, 50 epochs, batch 32 | Unchanged |
| | `validation_data=(X_test_noisy, X_test_std)` | No test data used during training |
| | Seeds 39 (`random`, `numpy`, `tf`) in LASSO nb; none in Final | `keras.utils.set_random_seed(39)` per fit |
| Final 15 | RF(`random_state=42`, 100 trees) on the **32-d encoder output** | Same — this is what the package implements |
| LASSO 16 / Copy 16 | `top_features_idx = X_train_encoded.var(axis=0).argsort()[:100]` then `RF.fit(X_train_std[:, top_features_idx])` | **Bug, not reproduced.** `argsort` over 32 latent units returns indices 0-31; slicing the 401 standardised *gene* columns with them selects the first 32 genes in an arbitrary order. The autoencoder had no influence on the classifier. Variable name says "top 500", comment says "top 100", actual count was 32. |
| Final 16-19 | Uses `forest_model_top_500_features`, `top_features_idx` | **Undefined in the Final notebook** (cell-order dependency on the LASSO/Copy notebook state). Cell 19 also predicts on **unscaled** `X_new` with a model trained on standardised inputs. |
| Final 51-52 | `predict_with_models` loads `random_forest_model.pkl` (never saved; models were `.joblib`), uses `scaler`/`loaded_autoencoder` before cell 52 defines it, passes noisy input to the full autoencoder (reconstruction, not latent) | Replaced by `inference.load_model` / `predict_dataset` |
| Copy 17 | "Deep multi-view" MLP on PCA(100) of RNA-seq with noise 0.5 | Historical experiment; not part of the published architecture; not reimplemented |

## 4. CNV / mutation / RPPA branches → `preprocessing.py`, `modality_models.py`

| | CNV (`cn_`) | Mutation (`mu_`) | RPPA (`pp_`) |
|---|---|---|---|
| Final (PCA variant) cells | 20-24 | 25-29 | 30-36 |
| LASSO variant cells | 24-28 | 29-33 | 34-40 |
| SMOTE strategy (Final / LASSO) | auto / 0.6 | auto / auto | auto / 0.3 |
| Reducer (Final) | `PCA(n_components=100, random_state=42)` | same | same |
| Reducer (LASSO) | `LassoCV(cv=5, max_iter=10000, random_state=42)`, keep `coef_ != 0` | same | same |
| Classifier | `RandomForestClassifier(n_estimators=100, random_state=42)` | same | same |

Defects and fixes:

* **SMOTE before split** in every branch → in-pipeline `FoldSafeSMOTE`.
* `pca`, `lasso`, `X_train`, `X_test`, `y_train`, `y_test`, `predictions` are
  re-bound by successive cells, so later cells depend on execution order
  (e.g. LASSO 27 saves `pca` — at that point the 2-component EDA PCA from
  cell 10 — as the CNV `pca_model.joblib`; LASSO 42 iterates over whichever
  `X_train` was assigned last). → Each modality is an independent pipeline
  object keyed by modality name.
* Zero LASSO-selected features would make `X_train[selected_features]` empty and
  crash the RF → `LassoSelector(on_empty="all"|"error")`, recorded in
  `fallback_used_`.
* `PCA(100)` fails when a fold has <100 samples or a modality has <100 columns
  → `SafePCA` clamps to `min(n_samples, n_features)` with a warning.
* Final 31 / LASSO 35: RPPA "top 30 proteins" index `X_train_pp.columns` with RF
  importances computed in PCA/LASSO space — the names do not correspond to the
  importances. → `modality_models.feature_importance` maps importances to the
  LASSO-selected column names, or labels them `PC1..PCk` for PCA.
* "New dataset" cells (Final 24/29/36, LASSO 28/33/40) evaluate on the full
  cohort that includes the training patients and report it as external
  accuracy. → Removed; `evaluate --subset test` scores only recorded held-out
  patients.
* Several labels mislabel the method ("with Autoencoder and SMOTE for df_mu" for
  a PCA/LASSO model; "SMOTE and PCA" printed for the LASSO model).

## 5. Fusion → `stacking.py`

| Source | Original | Problem | Corrected |
|---|---|---|---|
| Final 37-39, LASSO 41, 44 | Base models predict **hard labels** for all 220 patients (including their own training patients); 4-column table split 80/20; `StackingClassifier(estimators=[loaded RF models], final_estimator=RF(100), passthrough=True, cv=5)`; reported `cross_val_predict` accuracy on the meta **training** rows | (a) meta-features for training patients come from models fitted on those patients → leakage; (b) `StackingClassifier` clones and **refits** its estimators on the 4-column table, so the omics models are never used as omics models — the "ensemble" is RF-on-RF over 4 binary columns; (c) the number reported is not a test-set score | OOF cross-fitting on the training partition; RF meta-classifier on 4 OOF probability columns aligned by patient ID; base pipelines refit on the full training partition; scored once on held-out patients |
| LASSO 42 | Same stacking inside a 5-fold loop over `X_train` — but `X_train` at that point is whatever modality table was last assigned | Shape/semantic mismatch depending on execution order | — |
| LASSO 45 | Builds a nested `StackingClassifier` of RF/GB/LR/SVM/KNN, then immediately overwrites it; 60/40 split; reports test accuracy/AUC | Same leakage; dead code | Not reproduced; RF meta-learner kept |
| LASSO 50 | CV loop + refit + test evaluation on the 4-column prediction table | Same leakage | — |
| Final 42 / LASSO 53 | `roc_curve(y_train, y_pred_stacking)` with **hard labels** | ROC of a binary score is a single point; AUC is not meaningful | ROC/PR use ensemble probabilities on held-out patients |
| LASSO 61 | Bar chart with `'Multi-Model Ensemble': 94.32`, `'CNV Model': 81.67`, `'RS Model': 83.33` typed in as constants | Figure values are not computed by the notebook | All numbers in `reports/` are computed by code |
| Final 40 / LASSO 51 | χ² test on the training confusion matrix | Training-set association only | Not reproduced |

Meta-feature choice: the original used hard 0/1 predictions; the default now
uses positive-class probabilities (`stacking.meta_features: proba`).
`legacy_hyperparameters.yaml` restores `label`.

## 6. Evaluation → `evaluation.py`

| Original | Corrected |
|---|---|
| Accuracy, classification report, confusion matrix and Cohen's κ on a SMOTE-augmented test split (synthetic patients included, class balance artificial) | Accuracy, balanced accuracy, ROC-AUC, PR-AUC, precision, recall, F1, κ and confusion matrix on **real held-out patients only** at natural prevalence |
| Single 80/20 split per model; different splits per modality (different SMOTE outputs) | One patient-level stratified split shared by all modalities; nested 5-fold CV on the training partition for variance estimates |
| No split manifests | `split_manifest.csv` with partition and outer-fold membership per patient; `train_oof_meta_features.csv` with stacking fold membership |

## 7. Artifacts → `inference.py`

Original: `denoising_autoencoder_model.h5`, `random_forest_model_top_500_features.joblib`,
`random_forest_model.joblib` + `pca_model.joblib` (CNV), `*_mu.joblib`, `*_pp.joblib`,
`lasso_model*.joblib`; loader expects `random_forest_model.pkl`. No scaler was
saved, so the RNA model could not be applied to new data.

New: a single artifact **directory** (`manifest.json`, `config.yaml`,
`base_<modality>.joblib` full pipelines including imputer/scaler/selector/DAE
weights, `meta.joblib`, `rs_autoencoder.keras` export, `split_manifest.csv`).

## 8. Dependencies observed in the notebooks

Python: pandas, numpy, scipy (`ttest_ind`, `kendalltau`, `chi2_contingency`),
scikit-learn (RandomForest, PCA, LassoCV, StandardScaler, StackingClassifier,
StratifiedKFold, KMeans, TSNE, metrics), imbalanced-learn (SMOTE),
TensorFlow/Keras (Dense, GaussianNoise, Model), seaborn, matplotlib, joblib.
Versions were not recorded (Colab defaults, 2023-2024).

R: curatedTCGAData 2.1.1 data release, MultiAssayExperiment, SummarizedExperiment,
RaggedExperiment, missForest, dplyr, purrr, plyr, doParallel, UpSetR.

## 9. Departures from the original methodology (summary)

1. Patient split precedes all fitting; SMOTE, imputation, scaling, LASSO, PCA and the autoencoder are fitted on training folds only.
2. Stacking meta-features are out-of-fold positive-class probabilities (original: in-sample hard labels); base models are separate fitted pipelines, not refit by `StackingClassifier`.
3. The RNA branch classifies the 32-d latent code (Final notebook behaviour), not gene columns indexed by latent-unit indices (LASSO notebook bug).
4. Autoencoder noise applied once with σ = 0.5 (original σ = 50, applied twice); no test data in `validation_data`.
5. In-fold median imputation added; `FoldSafeSMOTE`, `LassoSelector` and `SafePCA` add explicit handling for small folds, empty selection and insufficient dimensions.
6. Evaluation on real held-out patients with imbalance-aware metrics, nested CV, and split manifests.
7. Data build selects assays by name, keeps primary-tumour samples, retains patient IDs, and refuses to fabricate labels.
