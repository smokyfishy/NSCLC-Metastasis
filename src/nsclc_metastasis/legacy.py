"""HISTORICAL, LEAKY protocol — for comparison only, never the default.

Re-implements the per-modality evaluation protocol of the original notebooks
so its optimism can be measured against the corrected pipeline on the same
data:

    SMOTE(full modality table) -> train_test_split(test_size=0.2, random_state=42,
    not stratified) -> [StandardScaler + DAE | LassoCV | PCA] fitted on the
    augmented training split -> RandomForest -> metrics on the augmented test split

Test rows include SMOTE-synthesised patients interpolated from training
patients (and vice versa), so these numbers are expected to be inflated.
The original fusion step is not re-implemented: it trained the meta-model on
predictions for patients the base models had already been fitted on, which
cannot be made meaningful without changing it (see stacking.py).
"""

from __future__ import annotations

import warnings

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from .config import Config
from .data import MultiOmicsDataset
from .evaluation import compute_metrics
from .modality_models import build_modality_pipeline, positive_proba

LEGACY_WARNING = (
    "LEGACY PROTOCOL: SMOTE is applied before the train/test split. Metrics from this "
    "mode are optimistically biased and must not be reported as model performance."
)


def run_legacy(dataset: MultiOmicsDataset, cfg: Config) -> pd.DataFrame:
    warnings.warn(LEGACY_WARNING, stacklevel=2)
    rows = []
    for m in cfg.enabled_modalities():
        mc = cfg.modalities[m]
        Xm = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(dataset.modality(m))
        y = dataset.y.to_numpy()
        smote = SMOTE(sampling_strategy=mc.smote.sampling_strategy, random_state=mc.smote.random_state)
        X_aug, y_aug = smote.fit_resample(Xm, y)
        X_tr, X_te, y_tr, y_te = train_test_split(X_aug, y_aug, test_size=cfg.split.test_size, random_state=42)
        # Same reducer + RF as the corrected pipeline, minus the in-pipeline
        # imputer and SMOTE (already applied above, leakily).
        mc_nosmote = clone_modality_cfg(mc)
        pipe = build_modality_pipeline(m, mc_nosmote)
        pipe = clone(pipe).fit(X_tr, y_tr)
        s = compute_metrics(y_te, positive_proba(pipe, X_te))
        rows.append({"modality": m, "reducer": mc.reducer, "n_after_smote": int(len(y_aug)),
                     **{k: v for k, v in s.items() if k != "confusion_matrix"},
                     **{f"cm_{k}": v for k, v in s["confusion_matrix"].items()}})
    out = pd.DataFrame(rows)
    out.attrs["warning"] = LEGACY_WARNING
    return out


def clone_modality_cfg(mc):
    import copy

    mc2 = copy.deepcopy(mc)
    mc2.smote.enabled = False
    mc2.impute = "none"
    return mc2
