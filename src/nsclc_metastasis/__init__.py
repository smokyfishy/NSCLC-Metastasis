"""NSCLC metastasis prediction from TCGA LUAD/LUSC multi-omics.

A leakage-corrected reconstruction of a 2023-2024 research pipeline:
four modality-specific models (RNA-seq denoising autoencoder + random forest;
copy-number, mutation and RPPA random forests with LASSO or PCA feature
reduction) fused by a random-forest meta-classifier trained on out-of-fold
probabilities.

Research code only. Not a medical device; not for clinical decisions.
"""

__version__ = "1.0.0"

MODALITIES = ("rs", "cn", "mu", "pp")
MODALITY_NAMES = {
    "rs": "RNA-seq (RNASeq2GeneNorm)",
    "cn": "Copy number (GISTIC thresholded by gene)",
    "mu": "Somatic mutation (binary)",
    "pp": "Protein (RPPA array)",
}
