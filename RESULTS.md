# Results

This file keeps three kinds of statement strictly apart. Nothing in section 1
has been reproduced; nothing in section 2 is a measurement on patient data.

---

## 1. Historical claims — *reported, not independently reproduced*

From the original notebooks and the 2023–2024 write-up.

| Claim | Source | Status |
|---|---|---|
| Cohort: 220 TCGA LUAD/LUSC patients, 33 metastatic / 187 non-metastatic; 1,558 features (813 `cn_`, 401 `rs_`, 126 `mu_`, 218 `pp_`) | Research write-up | Not verified: the processed CSV is unavailable |
| Multi-model ensemble accuracy **94.32%** | `LASSO of NSCLCMultiomics.ipynb`, cell 61 | **Hard-coded constant** in a plotting cell, not computed in any surviving cell |
| CNV model accuracy **81.67%**, RNA-seq model accuracy **83.33%** | Same cell | Hard-coded constants |
| Ensemble/modality accuracies of roughly 91–94% and Cohen's κ compared against 0.81 ("almost perfect agreement") | Write-up; Final cells 48-49; LASSO cells 59-60 | Measured on SMOTE-augmented test splits (SMOTE before split) with in-sample stacking meta-features. Expected to be optimistic. See `docs/REPRODUCTION_AUDIT.md` §5-6 |
| Benchmark "average accuracy of previous approaches: 85%" | Final cell 48 (plot annotation) | Source not cited in the notebooks |

Why these numbers can't be taken at face value: on the reported class
balance, a model that always predicts "non-metastatic" scores 85% accuracy
(187/220). The original test splits were also drawn after SMOTE, so they
contained synthetic interpolations of training patients.

## 2. Verified local measurements

All of these come from the build machine: Windows 11, CPU, Python 3.12.10,
TensorFlow 2.20.0 / Keras 3.15.1, scikit-learn 1.9.1, imbalanced-learn 0.14.2.
Exact versions are in `requirements.txt`.

### 2a. Software verification

| Check | Result |
|---|---|
| `pytest -q` (46 tests, synthetic fixtures) | **46 passed** |
| Leakage guards: held-out rows never reach SMOTE or LASSO fits, SMOTE never runs at predict time, stacking meta-features are out-of-fold | Pass |
| Model artifact save → load reproduces predictions; CLI train → evaluate → predict | Pass |
| Full-scale smoke run of `configs/baseline.yaml` (real hyperparameters, 50 epochs, 5×5 nested CV) on **synthetic** data shaped like the cohort (220 × 1,558, 33 positive) | Completed in about 2 minutes on CPU, and wrote every report and artifact file |
| `configs/pca_variant.yaml` (train, `--skip-cv`) and `configs/legacy_hyperparameters.yaml` (`legacy` command) on the same synthetic table | Completed |
| Original σ = 50 autoencoder noise | Keras 3's built-in `GaussianNoise` rejects stddev > 1, so a custom layer with the same semantics was added and tested |

### 2b. Synthetic-data illustration (not evidence about NSCLC)

On the synthetic 220-patient table, the legacy protocol (SMOTE before split)
scored the mutation model at ROC-AUC ≈ 1.00 and accuracy 0.99 on its
augmented test split. The corrected pipeline scored the same modality at
ROC-AUC 0.95 on the 44 held-out rows and 0.80 ± 0.07 in nested 5-fold CV. The only
point is that the two protocols give different answers on the same data. The
values themselves mean nothing, because the data are random numbers with an
injected signal.

**No metrics on real patient data have been produced.**

## 3. Not yet reproduced / blocked

| Item | Blocker | What unblocks it |
|---|---|---|
| Training and evaluating the corrected pipeline on the real cohort | `NSCLCMultiOmicsDataProcessed_reordered.csv` is unavailable | Place the file in `data/raw/`, then run `validate-data` and `train` (README, Usage) |
| Confirming the 220 / 33 / 1,558 counts | Same | `validate-data` prints the comparison automatically |
| Meaning of `metastasis_status` | No notebook defines it | Recover the original label derivation (plausibly TCGA `pathologic_M`, unverified) |
| Rebuilding the dataset from TCGA | `scripts/build_data.R` not executed (no R toolchain); `batchone.csv` feature list and labels missing | Install R + Bioconductor (`docs/DATA_PROVENANCE.md`) and supply `--labels` and optionally `--feature-list` |
| Quantifying the original protocol's optimism on real data | Needs the real CSV | `python -m nsclc_metastasis.cli legacy --data <csv>` next to `train` |
| Python 3.11 verification | 3.11 not installed on the build machine | `py -3.11 -m venv .venv` and re-run `pytest` |

When real results exist, add them to section 2 with the data file's SHA-256
(written to `reports/<config>/metrics.json`), the config name, the commit
hash, and both nested-CV mean ± SD and hold-out values.
