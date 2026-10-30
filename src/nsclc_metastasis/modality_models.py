"""Modality-specific base models.

Each modality becomes one ``imblearn`` pipeline::

    [impute] -> [SMOTE] -> [scale] -> reducer -> RandomForestClassifier

* ``rs`` (RNA-seq): reducer = denoising autoencoder (scaling is always on,
  as in the original ``StandardScaler`` before the autoencoder).
* ``cn``/``mu``/``pp``: reducer = ``LassoSelector`` (LASSO notebook) or
  ``SafePCA`` (Final notebook), chosen in the YAML config.

Because SMOTE is a pipeline *sampler*, it only runs inside ``fit`` on the
rows the pipeline is being trained on. Imputation, scaling, LASSO, PCA and
the autoencoder are likewise fitted on training rows only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .autoencoder import DenoisingAutoencoder
from .config import ModalityConfig
from .preprocessing import FoldSafeSMOTE, LassoSelector, SafePCA


def build_modality_pipeline(modality: str, cfg: ModalityConfig) -> Pipeline:
    steps = []
    if cfg.impute != "none":
        steps.append(("impute", SimpleImputer(strategy=cfg.impute, keep_empty_features=True)))
    if cfg.smote.enabled:
        steps.append(
            (
                "smote",
                FoldSafeSMOTE(
                    sampling_strategy=cfg.smote.sampling_strategy,
                    k_neighbors=cfg.smote.k_neighbors,
                    random_state=cfg.smote.random_state,
                ),
            )
        )
    if cfg.scale or cfg.reducer == "autoencoder":
        steps.append(("scale", StandardScaler()))

    if cfg.reducer == "autoencoder":
        if modality != "rs":
            raise ValueError("autoencoder reducer is only defined for the rs modality")
        a = cfg.autoencoder
        steps.append(
            (
                "reduce",
                DenoisingAutoencoder(
                    hidden_units=a.hidden_units,
                    latent_units=a.latent_units,
                    epochs=a.epochs,
                    batch_size=a.batch_size,
                    noise_stddev=a.noise_stddev,
                    learning_rate=a.learning_rate,
                    seed=a.seed,
                    verbose=a.verbose,
                ),
            )
        )
    elif cfg.reducer == "lasso":
        lc = cfg.lasso
        steps.append(
            (
                "reduce",
                LassoSelector(cv=lc.cv, max_iter=lc.max_iter, random_state=lc.random_state, on_empty=lc.on_empty),
            )
        )
    elif cfg.reducer == "pca":
        steps.append(("reduce", SafePCA(n_components=cfg.pca.n_components, random_state=cfg.pca.random_state)))
    elif cfg.reducer != "none":
        raise ValueError(f"Unknown reducer {cfg.reducer!r}")

    f = cfg.forest
    steps.append(
        (
            "rf",
            RandomForestClassifier(
                n_estimators=f.n_estimators,
                random_state=f.random_state,
                n_jobs=f.n_jobs,
                class_weight=f.class_weight,
            ),
        )
    )
    return Pipeline(steps)


def positive_proba(pipeline, X) -> np.ndarray:
    """Probability of class 1, robust to a training fold that saw one class."""
    proba = pipeline.predict_proba(X)
    classes = list(pipeline.classes_)
    if 1 in classes:
        return proba[:, classes.index(1)]
    return np.zeros(len(proba))


def reduced_feature_names(pipeline, input_columns: list[str]) -> list[str]:
    """Names of the columns the random forest sees, in order."""
    reducer = pipeline.named_steps.get("reduce")
    if reducer is None:
        return list(input_columns)
    if isinstance(reducer, LassoSelector):
        return list(np.asarray(input_columns)[reducer.get_support()])
    if isinstance(reducer, SafePCA):
        return [f"PC{i + 1}" for i in range(reducer.n_components_effective_)]
    if isinstance(reducer, DenoisingAutoencoder):
        return [f"latent_{i + 1}" for i in range(reducer.latent_units)]
    raise TypeError(type(reducer))


def feature_importance(pipeline, input_columns: list[str]) -> pd.DataFrame:
    rf = pipeline.named_steps["rf"]
    names = reduced_feature_names(pipeline, input_columns)
    return (
        pd.DataFrame({"feature": names, "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
