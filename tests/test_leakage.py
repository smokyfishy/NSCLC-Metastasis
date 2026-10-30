"""Leakage guards (SYNTHETIC data only).

These tests encode the core corrections over the original notebooks:
held-out patients never reach any fitted component, SMOTE runs only in fit,
and stacking meta-features are genuinely out-of-fold.
"""

import copy

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from nsclc_metastasis.data import dataset_from_frame
from nsclc_metastasis.evaluation import holdout_split, run_experiment
from nsclc_metastasis.modality_models import build_modality_pipeline, positive_proba
from nsclc_metastasis.preprocessing import FoldSafeSMOTE, LassoSelector
from nsclc_metastasis.stacking import MultiOmicsStackingModel
from nsclc_metastasis.synthetic import make_synthetic_dataset


def _row_keys(a):
    a = np.round(np.asarray(a, dtype=float), 8)
    return {r.tobytes() for r in a}


def test_holdout_split_is_disjoint_stratified_and_deterministic(synth_ds):
    tr1, te1 = holdout_split(synth_ds.X.index, synth_ds.y, 0.2, 42)
    tr2, te2 = holdout_split(synth_ds.X.index, synth_ds.y, 0.2, 42)
    assert not set(tr1) & set(te1)
    assert set(tr1) | set(te1) == set(synth_ds.X.index)
    assert list(te1) == list(te2)
    prev = synth_ds.y.mean()
    assert abs(synth_ds.y.loc[te1].mean() - prev) < 0.06


def test_heldout_rows_never_reach_smote_or_lasso(monkeypatch, fast_cfg, synth_ds):
    seen = []
    orig_resample = FoldSafeSMOTE.fit_resample
    orig_lasso_fit = LassoSelector.fit

    def spy_resample(self, X, y, **kw):
        seen.append(np.asarray(X).copy())
        return orig_resample(self, X, y, **kw)

    def spy_lasso(self, X, y):
        seen.append(np.asarray(X).copy())
        return orig_lasso_fit(self, X, y)

    monkeypatch.setattr(FoldSafeSMOTE, "fit_resample", spy_resample)
    monkeypatch.setattr(LassoSelector, "fit", spy_lasso)
    cfg = copy.deepcopy(fast_cfg)
    cfg.modalities["rs"].autoencoder.epochs = 1
    result = run_experiment(synth_ds, cfg)

    test_ids = result.split_manifest.query("partition == 'test'")["patient_id"]
    assert len(test_ids) > 0 and len(seen) > 0
    # Continuous modalities only: exact row collisions between patients are
    # practically impossible there, so any match means a held-out row leaked.
    for m in ("rs", "cn", "pp"):
        cols = synth_ds.modality_columns[m]
        test_rows = _row_keys(synth_ds.X.loc[test_ids, cols])
        for arr in seen:
            if arr.shape[1] == len(cols):
                assert not (_row_keys(arr) & test_rows), f"held-out {m} rows reached a fitted step"


def test_smote_is_only_applied_during_fit(monkeypatch, fast_cfg, synth_ds):
    calls = {"n": 0}
    orig = FoldSafeSMOTE.fit_resample

    def counting(self, X, y, **kw):
        calls["n"] += 1
        return orig(self, X, y, **kw)

    monkeypatch.setattr(FoldSafeSMOTE, "fit_resample", counting)
    X = synth_ds.modality("cn")
    pipe = build_modality_pipeline("cn", fast_cfg.modalities["cn"]).fit(X.iloc[:90], synth_ds.y.iloc[:90])
    n_after_fit = calls["n"]
    proba = positive_proba(pipe, X.iloc[90:])
    assert n_after_fit == 1
    assert calls["n"] == n_after_fit  # predict did not resample
    assert proba.shape == (30,)  # no synthetic rows added at inference


def test_stacking_meta_features_are_out_of_fold(fast_cfg):
    """With labels independent of features, in-sample RF probabilities look
    near-perfect (memorisation) while honest OOF probabilities hover at chance.
    The original notebook fed the meta-model the former."""
    df = make_synthetic_dataset(n_samples=200, positive_fraction=0.3, signal=0.0, seed=7)
    ds = dataset_from_frame(df)
    cfg = copy.deepcopy(fast_cfg)
    cfg.modalities["rs"].enabled = False
    model = MultiOmicsStackingModel(cfg).fit(ds.X, ds.y)

    oof = model.oof_meta_features_
    assert oof.index.equals(ds.X.index)
    assert not oof.isna().any().any()
    assert set(model.stacking_folds_.unique()) == {0, 1, 2}
    for m in ("cn", "pp"):
        in_sample = roc_auc_score(ds.y, model.meta_features(ds.X)[m])
        out_of_fold = roc_auc_score(ds.y, oof[m])
        assert in_sample > 0.95
        assert out_of_fold < 0.75


def test_meta_model_trained_on_oof_table(fast_cfg, synth_ds):
    cfg = copy.deepcopy(fast_cfg)
    cfg.modalities["rs"].enabled = False
    model = MultiOmicsStackingModel(cfg).fit(synth_ds.X, synth_ds.y)
    assert model.meta_model_.n_features_in_ == 3
    refit = copy.deepcopy(model.meta_model_).fit(model.oof_meta_features_.to_numpy(), synth_ds.y.to_numpy())
    oof = model.oof_meta_features_.to_numpy()
    np.testing.assert_allclose(refit.predict_proba(oof), model.meta_model_.predict_proba(oof))


def test_legacy_mode_warns(fast_cfg, synth_ds):
    from nsclc_metastasis.legacy import run_legacy

    cfg = copy.deepcopy(fast_cfg)
    cfg.modalities["rs"].enabled = False
    with pytest.warns(UserWarning, match="LEGACY PROTOCOL"):
        out = run_legacy(synth_ds, cfg)
    assert set(out["modality"]) == {"cn", "mu", "pp"}
