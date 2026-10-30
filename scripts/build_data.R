#!/usr/bin/env Rscript
# =============================================================================
# build_data.R — rebuild the processed TCGA LUAD/LUSC multi-omics table
#
# Reconstructed from reference_notebooks/NSCLCData.ipynb (cells 11-76).
# STATUS: NOT EXECUTED in this repository's verification run (no R toolchain
# was available). Treat as a documented, reviewable reconstruction; see
# docs/DATA_PROVENANCE.md for every known divergence from the notebook.
#
# Usage (PowerShell or bash):
#   Rscript scripts/build_data.R --out data/processed/NSCLCMultiOmics_rebuilt.csv `
#       [--labels data/raw/metastasis_labels.csv] [--feature-list data/raw/batchone_columns.txt] `
#       [--keep-all-sample-types]
#
#   --labels        CSV with columns patient_id,metastasis_status (0/1). REQUIRED to
#                   produce a training table. The notebook never derives this label,
#                   so this script will not invent one.
#   --feature-list  Optional text file, one column name per line (e.g. the header of
#                   the original batchone.csv). Restricts features exactly as notebook
#                   cells 67-74 did by intersecting with batchone.csv columns.
#   --keep-all-sample-types  Reproduce the notebook literally: keep every TCGA sample
#                   type after barcode truncation (normal-tissue aliquots can then
#                   collide with tumour aliquots). Default keeps primary tumour (01) only.
# =============================================================================

suppressPackageStartupMessages({
  library(curatedTCGAData)      # Bioconductor
  library(MultiAssayExperiment)
  library(SummarizedExperiment)
  library(RaggedExperiment)
  library(missForest)
  library(dplyr)
  library(purrr)
})

# ----------------------------------------------------------------- arguments
args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  i <- match(flag, args)
  if (is.na(i)) return(default)
  if (i == length(args)) stop("Missing value for ", flag)
  args[[i + 1]]
}
out_path      <- get_arg("--out", "data/processed/NSCLCMultiOmics_rebuilt.csv")
labels_path   <- get_arg("--labels")
feature_list  <- get_arg("--feature-list")
primary_only  <- !("--keep-all-sample-types" %in% args)

# Parameters copied from the notebook
CURATED_VERSION <- "2.1.1"                                  # cell 13-14
ASSAYS <- c("GISTIC_ThresholdedByGene", "Mutation", "RNASeq2GeneNorm", "RPPAArray")
COLDATA_MAX_NA_FRACTION <- 0.3                              # cell 18
RNA_TOP_SD <- 2000                                          # cell 25
RNA_PSEUDOCOUNT <- 0.1                                      # cell 25: log2(pmax(x, 0) + 0.1)
MUT_MIN_RATE <- 0.02                                        # cell 45 (comment says 5%, code uses 2%)
RPPA_MAX_NA_FRACTION <- 0.5                                 # cell 54
MISSFOREST_SEED <- 4578; MISSFOREST_MAXITER <- 10; MISSFOREST_NTREE <- 100
HISTOLOGY_KEEP <- c(
  "lung adenocarcinoma- not otherwise specified (nos)",
  "lung squamous cell carcinoma- not otherwise specified (nos)"
)

# Barcode helpers. The notebook truncated aliquot barcodes with
#   gsub("(\\-[[:alnum:]]+\\-[[:alnum:]]+\\-[[:alnum:]]+\\-[0-9]+)$", "", x)
# (TCGA-XX-XXXX-01A-11R-A262-07 -> TCGA-XX-XXXX) and then replaced "-"/"_" by ".".
to_patient <- function(x) gsub("\\-|\\_", ".", substr(x, 1, 12))
sample_type <- function(x) substr(x, 14, 15)

select_samples <- function(m) {
  bc <- colnames(m)
  keep <- if (primary_only) sample_type(bc) == "01" else rep(TRUE, length(bc))
  m <- m[, keep, drop = FALSE]
  pid <- to_patient(colnames(m))
  dup <- duplicated(pid)
  if (any(dup)) message(sprintf("  dropping %d duplicate aliquots (first kept)", sum(dup)))
  m <- m[, !dup, drop = FALSE]
  colnames(m) <- pid[!dup]
  m
}

# Experiments are selected by NAME. The notebook used positional indices
# (listAssays[[1]], [[2]], [[3]], [[7]]) whose meaning depends on the
# ExperimentList order; see docs/DATA_PROVENANCE.md.
experiments_for <- function(mae, assay) {
  nm <- grep(paste0("_", assay, "-"), names(experiments(mae)), value = TRUE)
  if (length(nm) == 0) stop("No experiment found for assay ", assay)
  message("  ", assay, ": ", paste(nm, collapse = ", "))
  nm
}

cbind_common_rows <- function(mats) {
  common <- Reduce(intersect, lapply(mats, rownames))
  do.call(cbind, lapply(mats, function(m) m[common, , drop = FALSE]))
}

# ---------------------------------------------------------------- download
message("Downloading curatedTCGAData ", CURATED_VERSION, " (LUAD, LUSC) ...")
mae <- curatedTCGAData(c("LUAD", "LUSC"), ASSAYS, dry.run = FALSE, version = CURATED_VERSION)

# ---------------------------------------------------------------- clinical
cd <- as.data.frame(colData(mae))
rownames(cd) <- gsub("\\-|\\_", ".", rownames(cd))
colnames(cd) <- gsub("\\-|\\_", ".", colnames(cd))
na_frac <- vapply(cd, function(v) mean(is.na(v)), numeric(1))
cd <- cd[, na_frac <= COLDATA_MAX_NA_FRACTION, drop = FALSE]
cd <- cd[cd$histological.type %in% HISTOLOGY_KEEP, , drop = FALSE]
message("Patients after histology filter: ", nrow(cd))

# ---------------------------------------------------------------- RNA-seq
message("RNA-seq ...")
rna <- cbind_common_rows(lapply(experiments_for(mae, "RNASeq2GeneNorm"),
                                function(n) select_samples(assay(mae[[n]]))))
rna <- log2(pmax(rna, 0) + RNA_PSEUDOCOUNT)
rna <- t(rna)
sds <- apply(rna, 2, sd)
rna <- rna[, order(sds, decreasing = TRUE)[seq_len(min(RNA_TOP_SD, ncol(rna)))]]
# NOTE: SD ranking uses the whole cohort (label-free, but not fold-isolated).

# ---------------------------------------------------------------- copy number
message("Copy number ...")
cn_list <- lapply(experiments_for(mae, "GISTIC_ThresholdedByGene"), function(n) {
  se <- mae[[n]]
  m <- assay(se)
  sym <- rowData(se)$Gene.Symbol
  if (!is.null(sym)) rownames(m) <- sym
  m <- m[!duplicated(rownames(m)), , drop = FALSE]
  select_samples(m)
})
cn <- t(cbind_common_rows(cn_list))
cn <- cn[intersect(rownames(cn), rownames(cd)), , drop = FALSE]

# ---------------------------------------------------------------- mutation
# Binary patient x gene matrix: 1 if the gene carries any recorded mutation.
# Equivalent in intent to notebook cells 37-45 (which reshaped the
# RaggedExperiment with hard-coded loop bounds 1:232 / 1:231).
message("Mutation ...")
mut_long <- bind_rows(lapply(experiments_for(mae, "Mutation"), function(n) {
  re <- mae[[n]]
  gr <- unlist(as(re, "GRangesList"))
  sym_col <- intersect(c("Hugo_Symbol", "gene", "Gene"), colnames(mcols(gr)))[1]
  if (is.na(sym_col)) stop("No gene symbol column in mutation assay ", n)
  data.frame(barcode = names(gr), gene = as.character(mcols(gr)[[sym_col]]))
}))
mut_long <- mut_long[!is.na(mut_long$gene) & mut_long$gene != "", ]
if (primary_only) mut_long <- mut_long[sample_type(mut_long$barcode) == "01", ]
mut_long$patient <- to_patient(mut_long$barcode)
mut_long <- distinct(mut_long, patient, gene)
mu <- as.matrix(table(mut_long$patient, mut_long$gene))
mu[mu > 0] <- 1
storage.mode(mu) <- "numeric"
mu <- mu[intersect(rownames(mu), rownames(cd)), , drop = FALSE]
mu <- mu[, colMeans(mu) > MUT_MIN_RATE, drop = FALSE]

# ---------------------------------------------------------------- RPPA
message("RPPA ...")
pp <- t(cbind_common_rows(lapply(experiments_for(mae, "RPPAArray"),
                                 function(n) select_samples(assay(mae[[n]])))))
pp <- pp[intersect(rownames(pp), rownames(cd)), , drop = FALSE]
colnames(pp) <- gsub("\\-|\\_", ".", colnames(pp))
pp <- pp[, colMeans(is.na(pp)) <= RPPA_MAX_NA_FRACTION, drop = FALSE]
if (anyNA(pp)) {
  # Label-free, but fitted on the whole cohort before any train/test split.
  set.seed(MISSFOREST_SEED)
  imp <- missForest(as.data.frame(pp), maxiter = MISSFOREST_MAXITER,
                    ntree = MISSFOREST_NTREE, parallelize = "no")
  message("  missForest OOB error: ", paste(format(imp$OOBerror), collapse = " "))
  pp <- as.matrix(imp$ximp)
}

# ---------------------------------------------------------------- merge
colnames(rna) <- paste0("rs_", colnames(rna))
colnames(cn)  <- paste0("cn_", colnames(cn))
colnames(mu)  <- paste0("mu_", colnames(mu))
colnames(pp)  <- paste0("pp_", colnames(pp))

blocks <- list(RS = rna, CN = cn, MU = mu, PP = pp)
ids <- Reduce(intersect, lapply(blocks, rownames))
message("Patients with all four modalities: ", length(ids))
final <- do.call(cbind, lapply(blocks, function(b) b[ids, , drop = FALSE]))  # rs, cn, mu, pp order

if (!is.null(feature_list)) {
  keep <- readLines(feature_list)
  keep <- intersect(colnames(final), keep)
  message("Restricting to ", length(keep), " columns from ", feature_list)
  final <- final[, keep, drop = FALSE]
}

out <- data.frame(patient_id = ids, final, check.names = FALSE)
counts <- sapply(c("rs_", "cn_", "mu_", "pp_"), function(p) sum(startsWith(colnames(out), p)))
message("Feature counts: ", paste(names(counts), counts, sep = "=", collapse = ", "))

dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
if (is.null(labels_path)) {
  feat_path <- sub("\\.csv$", "_features_only.csv", out_path)
  write.csv(out, feat_path, row.names = FALSE)
  message("\nWrote FEATURES ONLY to ", feat_path)
  stop("No --labels file supplied. metastasis_status is not derivable from NSCLCData.ipynb; ",
       "supply a verified patient_id,metastasis_status CSV (see docs/DATA_PROVENANCE.md).",
       call. = FALSE)
}

labels <- read.csv(labels_path, stringsAsFactors = FALSE)
stopifnot(all(c("patient_id", "metastasis_status") %in% colnames(labels)))
labels$patient_id <- gsub("\\-|\\_", ".", labels$patient_id)
stopifnot(all(labels$metastasis_status %in% c(0, 1)))
out <- inner_join(out, labels[, c("patient_id", "metastasis_status")], by = "patient_id")
message("Patients with labels: ", nrow(out), " (positive = ", sum(out$metastasis_status), ")")
write.csv(out, out_path, row.names = FALSE)
message("Wrote ", out_path)
writeLines(capture.output(sessionInfo()), sub("\\.csv$", "_sessionInfo.txt", out_path))
