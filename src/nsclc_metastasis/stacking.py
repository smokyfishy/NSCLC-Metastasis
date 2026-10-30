"""Leakage-free late fusion of the four modality models.

Original approach (Final cells 37-39; LASSO cells 41-50) and its problems:

* Base models (trained on SMOTE-before-split data) predicted hard labels for
  the *entire* cohort, including patients they had been trained on.
* Those 4 label columns were split again with ``train_test_split`` and fed to
  ``StackingClassifier(estimators=[loaded RF models], passthrough=True)``.
  ``StackingClassifier`` clones and refits its estimators on whatever ``X``
  it receives, so the "loaded" omics models were silently retrained as random
  forests on the 4-column prediction table, not used as omics models at all.
* The reported ensemble score was ``cross_val_predict`` accuracy on the
  meta-training rows, and one plotting cell hard-codes 94.32%.

Corrected procedure implemented by :class:`MultiOmicsStackingModel`:

1. Within the training partition only, split patients with
   ``StratifiedKFold``. For each fold, fit every modality pipeline on the
   other folds and predict the held-out fold. This yields out-of-fold (OOF)
   positive-class probabilities: no patient's meta-feature comes from a model
   that saw that patient's label.
2. Fit the random-forest meta-classifier on the OOF table (one column per
   modality, rows aligned by patient ID).
3. Refit each modality pipeline on the whole training partition; at
   prediction time these produce the meta-features for new patients.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from .config import Config
from .data import align_columns, split_modalities
from .modality_models import build_modality_pipeline, positive_proba

log = logging.getLogger(__name__)


def safe_n_splits(y, requested: int) -> int:
    counts = np.bincount(np.asarray(y, dtype=int), minlength=2)
    n_min = int(counts.min())
    if n_min < 2:
        raise ValueError(f"Need at least 2 samples per class for stratified CV; class counts = {counts.tolist()}")
    if n_min < requested:
        log.warning("Reducing CV folds from %d to %d (minority class has %d samples)", requested, n_min, n_min)
    return min(requested, n_min)


class MultiOmicsStackingModel:
    def __init__(self, config: Config):
        self.config = config

    # ------------------------------------------------------------------ fit
    def _base_output(self, pipe, X):
        if self.config.stacking.meta_features == "proba":
            return positive_proba(pipe, X)
        return pipe.predict(X).astype(float)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MultiOmicsStackingModel":
        if not X.index.equals(y.index):
            raise ValueError("X and y must share the same patient index (order included).")
        if X.index.duplicated().any():
            raise ValueError("Duplicate patient IDs in training data.")
        cfg = self.config
        self.modalities_ = cfg.enabled_modalities()
        all_cols = split_modalities(X.columns, cfg.data.prefixes)
        self.feature_columns_ = {m: list(all_cols[m]) for m in self.modalities_}
        empty = [m for m in self.modalities_ if not self.feature_columns_[m]]
        if empty:
            raise ValueError(f"Enabled modalities have no feature columns: {empty}")

        n_splits = safe_n_splits(y, cfg.split.stacking_cv_folds)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.split.seed)
        oof = pd.DataFrame(np.nan, index=X.index, columns=self.modalities_)
        fold_of = pd.Series(-1, index=X.index, name="stacking_fold")
        templates = {m: build_modality_pipeline(m, cfg.modalities[m]) for m in self.modalities_}

        for k, (tr, va) in enumerate(skf.split(X, y)):
            fold_of.iloc[va] = k
            for m in self.modalities_:
                cols = self.feature_columns_[m]
                pipe = clone(templates[m]).fit(X.iloc[tr][cols], y.iloc[tr])
                oof.iloc[va, oof.columns.get_loc(m)] = self._base_output(pipe, X.iloc[va][cols])
            log.info("stacking OOF fold %d/%d done", k + 1, n_splits)

        if oof.isna().any().any():  # pragma: no cover - guarded by construction
            raise RuntimeError("OOF meta-feature table has gaps.")
        self.oof_meta_features_ = oof
        self.stacking_folds_ = fold_of

        f = cfg.stacking.forest
        self.meta_model_ = RandomForestClassifier(
            n_estimators=f.n_estimators, random_state=f.random_state, n_jobs=f.n_jobs, class_weight=f.class_weight
        ).fit(oof.to_numpy(), y.to_numpy())

        self.base_models_ = {}
        for m in self.modalities_:
            cols = self.feature_columns_[m]
            self.base_models_[m] = clone(templates[m]).fit(X[cols], y)
        self.training_ids_ = list(X.index)
        return self

    # -------------------------------------------------------------- predict
    def meta_features(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index, columns=self.modalities_, dtype=float)
        for m in self.modalities_:
            Xm = align_columns(X, self.feature_columns_[m])
            out[m] = self._base_output(self.base_models_[m], Xm)
        return out

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        meta = self.meta_features(X)
        proba = self.meta_model_.predict_proba(meta.to_numpy())
        classes = list(self.meta_model_.classes_)
        return proba[:, classes.index(1)] if 1 in classes else np.zeros(len(meta))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X) >= self.config.stacking.threshold).astype(int)

    def predict_detail(self, X: pd.DataFrame) -> pd.DataFrame:
        meta = self.meta_features(X)
        classes = list(self.meta_model_.classes_)
        p = self.meta_model_.predict_proba(meta.to_numpy())
        ens = p[:, classes.index(1)] if 1 in classes else np.zeros(len(meta))
        out = meta.add_prefix("base_" + self.config.stacking.meta_features + "_")
        out["ensemble_probability"] = ens
        out["predicted_label"] = (ens >= self.config.stacking.threshold).astype(int)
        return out

    @property
    def meta_feature_importance_(self) -> pd.Series:
        return pd.Series(self.meta_model_.feature_importances_, index=self.modalities_).sort_values(ascending=False)
