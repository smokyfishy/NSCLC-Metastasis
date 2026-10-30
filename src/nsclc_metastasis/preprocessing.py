"""Fold-safe preprocessing components.

All three components are scikit-learn/imbalanced-learn compatible estimators
so they can live inside an ``imblearn.pipeline.Pipeline``. That is what makes
them fold-safe: the pipeline is cloned and fitted on each training fold, and
SMOTE (a sampler) is only ever applied during ``fit`` — never to validation
or test rows.

Original notebook behaviour that these replace:

* ``SMOTE(...).fit_resample(X_all, y_all)`` followed by ``train_test_split``
  on the augmented table. Synthetic minority rows were interpolated from
  patients that later ended up in the test set, and near-duplicates of test
  patients appeared in training.
* ``LassoCV``/``PCA`` fitted once on that (already leaky) training split, then
  reused on the *whole* cohort for the "new dataset" and stacking cells.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.utils.validation import check_is_fitted

log = logging.getLogger(__name__)


class FoldSafeSMOTE(BaseEstimator):
    """SMOTE that degrades gracefully on small training folds.

    * ``k_neighbors`` is reduced to ``n_minority - 1`` when the fold has too
      few minority samples (imblearn would otherwise raise).
    * With a float ``sampling_strategy`` (minority/majority ratio, as used in
      the LASSO notebook: 0.6 for rs/cn, 0.3 for pp), folds that are already
      at or above the ratio are passed through unchanged instead of raising.
    * Folds with fewer than 2 minority samples are passed through unchanged.

    Every adjustment is recorded in ``last_resample_info_``.
    """

    def __init__(self, sampling_strategy="auto", k_neighbors=5, random_state=42):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X, y, **params):
        y_arr = np.asarray(y)
        classes, counts = np.unique(y_arr, return_counts=True)
        info = {"n_in": int(len(y_arr)), "class_counts_in": dict(zip(classes.tolist(), counts.tolist()))}
        if len(classes) < 2:
            info["action"] = "skipped: single class"
            self.last_resample_info_ = info
            return X, y
        n_min, n_maj = counts.min(), counts.max()
        if n_min < 2:
            info["action"] = "skipped: fewer than 2 minority samples"
            warnings.warn("SMOTE skipped: fewer than 2 minority samples in training fold.", stacklevel=2)
            self.last_resample_info_ = info
            return X, y
        if isinstance(self.sampling_strategy, (int, float)) and not isinstance(self.sampling_strategy, bool):
            if n_min / n_maj >= float(self.sampling_strategy):
                info["action"] = "skipped: ratio already satisfied"
                self.last_resample_info_ = info
                return X, y
        k = min(int(self.k_neighbors), int(n_min) - 1)
        if k < self.k_neighbors:
            info["k_neighbors_reduced_to"] = k
        smote = SMOTE(sampling_strategy=self.sampling_strategy, k_neighbors=k, random_state=self.random_state)
        X_res, y_res = smote.fit_resample(X, y)
        info["action"] = "resampled"
        info["n_out"] = int(len(y_res))
        self.last_resample_info_ = info
        return X_res, y_res


class LassoSelector(BaseEstimator, TransformerMixin):
    """Keep features with a non-zero ``LassoCV`` coefficient.

    Mirrors ``LassoCV(cv=5, max_iter=10000, random_state=42)`` +
    ``X.columns[lasso.coef_ != 0]`` from the LASSO notebook, but is fitted
    inside each training fold. LassoCV regresses the 0/1 label (as in the
    original); it is used only as a feature filter.

    If every coefficient is zero, ``on_empty="all"`` keeps all features and
    sets ``fallback_used_ = True``; ``on_empty="error"`` raises.
    """

    def __init__(self, cv=5, max_iter=10000, random_state=42, on_empty="all"):
        self.cv = cv
        self.max_iter = max_iter
        self.random_state = random_state
        self.on_empty = on_empty

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        cv = max(2, min(int(self.cv), len(y)))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*did not converge.*")
            self.lasso_ = LassoCV(cv=cv, max_iter=self.max_iter, random_state=self.random_state)
            self.lasso_.fit(X, y)
        support = self.lasso_.coef_ != 0
        self.fallback_used_ = False
        if not support.any():
            if self.on_empty == "error":
                raise ValueError("LassoCV selected zero features on this training fold.")
            warnings.warn(
                "LassoCV selected zero features; keeping all features for this fold "
                "(lasso.on_empty='all').",
                stacklevel=2,
            )
            support = np.ones_like(support, dtype=bool)
            self.fallback_used_ = True
        self.support_ = support
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "support_")
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        return X[:, self.support_]

    def get_support(self):
        check_is_fitted(self, "support_")
        return self.support_.copy()


class SafePCA(BaseEstimator, TransformerMixin):
    """PCA whose ``n_components`` is clamped to ``min(n_samples, n_features)``.

    The original used ``PCA(n_components=100, random_state=42)``. That is
    valid for the full cohort but fails on small folds or modalities with
    fewer than 100 columns; the effective value is stored in
    ``n_components_effective_`` and a warning is emitted when clamped.
    """

    def __init__(self, n_components=100, random_state=42):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        k = min(int(self.n_components), X.shape[0], X.shape[1])
        if k < 1:
            raise ValueError("PCA needs at least one sample and one feature.")
        if k < self.n_components:
            warnings.warn(
                f"PCA n_components clamped from {self.n_components} to {k} "
                f"(n_samples={X.shape[0]}, n_features={X.shape[1]}).",
                stacklevel=2,
            )
        self.n_components_effective_ = k
        self.pca_ = PCA(n_components=k, random_state=self.random_state).fit(X)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "pca_")
        return self.pca_.transform(np.asarray(X, dtype=float))
