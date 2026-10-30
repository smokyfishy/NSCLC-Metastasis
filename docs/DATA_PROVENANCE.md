# Data provenance

## What the model expects

A single CSV, one row per patient:

| Column(s) | Type | Notes |
|---|---|---|
| `patient_id` (recommended) | string | TCGA patient barcode. Also accepted: `sample_id`, `barcode`, `bcr_patient_barcode`. If absent, row positions are used and a warning is printed. |
| `rs_<GENE>` | float | RNASeq2GeneNorm, `log2(max(x,0)+0.1)` |
| `cn_<GENE>` | float/int | GISTIC thresholded copy number (−2…2) |
| `mu_<GENE>` | 0/1 | Gene carries ≥1 somatic mutation |
| `pp_<PROTEIN>` | float | RPPA level (missForest-imputed in the original) |
| `metastasis_status` | 0/1 | 1 = metastatic. Required for `train`/`evaluate`, not for `predict`. |

Check any candidate file with:

```powershell
python -m nsclc_metastasis.cli validate-data --data data/raw/NSCLCMultiOmicsDataProcessed_reordered.csv
```

The report compares observed counts with those in the research write-up
(220 patients; 33 metastatic / 187 non-metastatic; 813 `cn_`, 401 `rs_`,
126 `mu_`, 218 `pp_`). Mismatches are reported, never "fixed" by dropping rows.

## Status of the original file

`NSCLCMultiOmicsDataProcessed_reordered.csv` — the file every modelling
notebook reads from `/content/` — **is not available**:

* not included in the supplied notebook bundle;
* not found by name in the connected Google Drive (per the bundle's `SOURCES.md`);
* not found under the local user profile (`Downloads`, `Documents`, `OneDrive`, `Desktop`) when this repository was built.

The Google Sheet titled "NSCLC Metastasis Prediction Data Set" has 21 columns
and is **not** the 1,558-feature table; it must not be substituted.

**Consequently, no model in this repository has been trained on real data,
and no performance number for the real cohort has been reproduced.**

To proceed, place the original file at
`data/raw/NSCLCMultiOmicsDataProcessed_reordered.csv` (git-ignored) and run
`validate-data`, then `train`.

## Rebuilding from TCGA (`scripts/build_data.R`)

`build_data.R` reconstructs `NSCLCData.ipynb` against curatedTCGAData 2.1.1.
It has **not been executed** in the verification environment (no R available).

### Setup (Windows)

1. Install R ≥ 4.3 from CRAN and Rtools matching the R version.
2. In R:
   ```r
   install.packages(c("BiocManager", "missForest", "dplyr", "purrr"))
   BiocManager::install(c("curatedTCGAData", "MultiAssayExperiment",
                          "SummarizedExperiment", "RaggedExperiment"))
   ```
3. From the repository root (PowerShell):
   ```powershell
   Rscript scripts/build_data.R --out data/processed/NSCLCMultiOmics_rebuilt.csv --labels data/raw/metastasis_labels.csv
   ```
   The first run downloads several GB into the ExperimentHub cache.

### What cannot be reconstructed from the notebooks

1. **The label.** No cell in `NSCLCData.ipynb` creates `metastasis_status`.
   Cell 63 writes the filtered clinical table (`metastasisstatusb1.csv`), so the
   label was presumably derived from it outside the notebook. A plausible
   source is the TCGA pathologic M stage (`pathologic_M`: M1/M1a/M1b = 1,
   M0 = 0, MX/NA = excluded), but this is **a hypothesis, not a verified
   definition**. M stage at diagnosis is distant metastasis at presentation,
   not metastasis during follow-up. The R script therefore requires an
   explicit `--labels` file and stops otherwise.
2. **The final feature set.** The 2,000 top-SD genes and ~all-gene CN/mutation
   matrices were reduced to 401/813/126 columns by intersecting with the
   columns of `batchone.csv` (cells 67-74), whose origin is undocumented. If
   that file's header is recovered, pass it via `--feature-list`.
3. **The intermediate `NSCLCMultiOmicsDataProcessed.csv`** read in cell 75 is
   never written by the notebook.
4. **Which cohort each assay came from.** The notebook selected assays by
   position (`listAssays[[1]]`, `[[2]]`, `[[3]]`, `[[7]]`) and took CN gene
   symbols from the LUAD GISTIC experiment. With LUAD+LUSC loaded there are 8
   experiments, so some modalities probably came from LUAD only. The script
   selects assays by name and combines LUAD+LUSC, which may yield a different
   (likely larger) cohort.

### Intentional differences from the notebook

| Notebook | `build_data.R` | Reason |
|---|---|---|
| Positional assay selection | By name, LUAD + LUSC combined | Positional indices are ambiguous |
| Barcode truncation keeps all sample types | Primary tumour (`01`) only; `--keep-all-sample-types` restores original | Normal-tissue aliquots otherwise collide with tumour aliquots after truncation |
| `row.names = FALSE` | `patient_id` column written | Enables patient-level split manifests and traceability |
| Custom `coalesce`/`foreach(1:232)` mutation reshape | Long table → binary matrix | Same intent; hard-coded bounds removed |
| Label absent | `--labels` required | Do not fabricate outcomes |

### Label-free, whole-cohort steps that remain

RNA-seq top-SD gene selection, mutation-rate filtering, RPPA NA filtering and
missForest imputation are computed on all patients before any train/test
split. They do not use labels, so they cannot leak outcome information
directly, but they do let test patients influence the feature space. This is
noted in `KNOWN_LIMITATIONS.md`.

## Data handling policy

* `data/raw/*`, `data/processed/*`, `artifacts/*` and `reports/*` are
  git-ignored. Do not commit patient-level tables or trained weights.
* TCGA open-access data are subject to the NIH GDC Data Use policies.
* Files whose names start with `SYNTHETIC` are generated by
  `nsclc_metastasis.synthetic` and contain no patient information.
