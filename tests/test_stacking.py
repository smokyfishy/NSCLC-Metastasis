import copy

import numpy as np
import pytest

from nsclc_metastasis.stacking import MultiOmicsStackingModel, safe_n_splits


@pytest.fixture()
def no_rs_cfg(fast_cfg):
    cfg = copy.deepcopy(fast_cfg)
    cfg.modalities["rs"].enabled = False
    return cfg


def test_fit_predict_shapes_and_alignment(no_rs_cfg, synth_ds):
    X, y = synth_ds.X, synth_ds.y
    model = MultiOmicsStackingModel(no_rs_cfg).fit(X.iloc[:100], y.iloc[:100])
    assert model.modalities_ == ["cn", "mu", "pp"]
    assert list(model.oof_meta_features_.columns) == ["cn", "mu", "pp"]
    assert model.oof_meta_features_.index.equals(X.index[:100])
    detail = model.predict_detail(X.iloc[100:])
    assert detail.index.equals(X.index[100:])
    assert {"ensemble_probability", "predicted_label", "base_proba_cn"} <= set(detail.columns)
    p = model.predict_proba(X.iloc[100:])
    assert p.shape == (20,) and np.all((p >= 0) & (p <= 1))
    np.testing.assert_array_equal(model.predict(X.iloc[100:]), (p >= 0.5).astype(int))


def test_meta_model_never_sees_raw_omics(no_rs_cfg, synth_ds):
    model = MultiOmicsStackingModel(no_rs_cfg).fit(synth_ds.X, synth_ds.y)
    # one meta-feature per modality; the omics models are separate fitted pipelines
    assert model.meta_model_.n_features_in_ == len(model.modalities_)
    for m, pipe in model.base_models_.items():
        assert pipe.named_steps["impute"].n_features_in_ == len(model.feature_columns_[m])


def test_prediction_is_row_order_invariant(no_rs_cfg, synth_ds):
    model = MultiOmicsStackingModel(no_rs_cfg).fit(synth_ds.X.iloc[:100], synth_ds.y.iloc[:100])
    Xt = synth_ds.X.iloc[100:]
    a = model.predict_detail(Xt)
    b = model.predict_detail(Xt.iloc[::-1]).loc[a.index]
    np.testing.assert_allclose(a["ensemble_probability"], b["ensemble_probability"])


def test_prediction_ignores_column_order_and_extra_columns(no_rs_cfg, synth_ds):
    model = MultiOmicsStackingModel(no_rs_cfg).fit(synth_ds.X.iloc[:100], synth_ds.y.iloc[:100])
    Xt = synth_ds.X.iloc[100:]
    shuffled = Xt[Xt.columns[::-1]].assign(unrelated_column=1.0)
    np.testing.assert_allclose(model.predict_proba(Xt), model.predict_proba(shuffled))


def test_label_meta_features_mode(no_rs_cfg, synth_ds):
    cfg = copy.deepcopy(no_rs_cfg)
    cfg.stacking.meta_features = "label"
    model = MultiOmicsStackingModel(cfg).fit(synth_ds.X, synth_ds.y)
    assert set(np.unique(model.oof_meta_features_.to_numpy())) <= {0.0, 1.0}


def test_misaligned_labels_rejected(no_rs_cfg, synth_ds):
    with pytest.raises(ValueError, match="same patient index"):
        MultiOmicsStackingModel(no_rs_cfg).fit(synth_ds.X, synth_ds.y.iloc[::-1])


def test_safe_n_splits():
    assert safe_n_splits([0] * 10 + [1] * 3, 5) == 3
    with pytest.raises(ValueError):
        safe_n_splits([0] * 10 + [1], 5)


def test_full_four_modality_stack_with_autoencoder(fast_cfg, synth_ds):
    pytest.importorskip("tensorflow")
    cfg = copy.deepcopy(fast_cfg)
    cfg.modalities["rs"].autoencoder.epochs = 1
    model = MultiOmicsStackingModel(cfg).fit(synth_ds.X.iloc[:100], synth_ds.y.iloc[:100])
    assert model.modalities_ == ["rs", "cn", "mu", "pp"]
    assert model.predict_detail(synth_ds.X.iloc[100:]).shape[0] == 20
