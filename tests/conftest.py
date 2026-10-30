"""Shared fixtures. ALL data in this test suite is SYNTHETIC (see
nsclc_metastasis.synthetic); no test result says anything about real patients."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from nsclc_metastasis.config import load_config  # noqa: E402
from nsclc_metastasis.data import dataset_from_frame  # noqa: E402
from nsclc_metastasis.synthetic import make_synthetic_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def make_fast(cfg, ae_epochs: int = 2):
    """Shrink a real config so tests run in seconds (structure unchanged)."""
    cfg = copy.deepcopy(cfg)
    cfg.split.outer_cv_folds = 3
    cfg.split.stacking_cv_folds = 3
    for mc in cfg.modalities.values():
        mc.forest.n_estimators = 15
        mc.lasso.cv = 3
        mc.pca.n_components = 10
        mc.autoencoder.epochs = ae_epochs
        mc.autoencoder.hidden_units = 16
        mc.autoencoder.latent_units = 8
    cfg.stacking.forest.n_estimators = 25
    return cfg


@pytest.fixture(scope="session")
def baseline_cfg():
    return load_config(ROOT / "configs" / "baseline.yaml")


@pytest.fixture(scope="session")
def fast_cfg(baseline_cfg):
    return make_fast(baseline_cfg)


@pytest.fixture()
def synth_df():
    return make_synthetic_dataset(n_samples=120, positive_fraction=0.2, seed=1, signal=1.5)


@pytest.fixture()
def synth_ds(synth_df, baseline_cfg):
    return dataset_from_frame(synth_df, baseline_cfg.data)
