import copy
import pickle

import numpy as np
import pytest

from nsclc_metastasis.data import dataset_from_frame
from nsclc_metastasis.modality_models import (
    build_modality_pipeline,
    feature_importance,
    positive_proba,
    reduced_feature_names,
)
from nsclc_metastasis.preprocessing import FoldSafeSMOTE, LassoSelector, SafePCA
from nsclc_metastasis.synthetic import make_synthetic_dataset


def test_fold_safe_smote_reduces_k_on_tiny_minority():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(23, 4))
    y = np.array([1] * 3 + [0] * 20)
    s = FoldSafeSMOTE(k_neighbors=5, random_state=0)
    Xr, yr = s.fit_resample(X, y)
    assert s.last_resample_info_["k_neighbors_reduced_to"] == 2
    assert (yr == 1).sum() == (yr == 0).sum() == 20
    np.testing.assert_array_equal(Xr[:23], X)  # originals kept, synthetic rows appended


def test_fold_safe_smote_skips_when_ratio_already_met():
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = np.array([1] * 4 + [0] * 6)
    s = FoldSafeSMOTE(sampling_strategy=0.3)
    Xr, _ = s.fit_resample(X, y)
    assert Xr is X and s.last_resample_info_["action"].startswith("skipped")


def test_lasso_selector_selects_informative_features():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(150, 30))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    sel = LassoSelector(cv=3).fit(X, y)
    assert sel.get_support()[0]
    assert sel.transform(X).shape == (150, sel.get_support().sum())


def test_lasso_zero_selection_fallback_and_error():
    X = np.ones((40, 5))  # constant features -> every coefficient is zero
    y = np.array([0, 1] * 20)
    with pytest.warns(UserWarning, match="zero features"):
        sel = LassoSelector(cv=3, on_empty="all").fit(X, y)
    assert sel.fallback_used_ and sel.get_support().all()
    with pytest.raises(ValueError, match="zero features"):
        LassoSelector(cv=3, on_empty="error").fit(X, y)


def test_safe_pca_clamps_components():
    X = np.random.default_rng(0).normal(size=(12, 30))
    with pytest.warns(UserWarning, match="clamped"):
        p = SafePCA(n_components=100).fit(X)
    assert p.n_components_effective_ == 12
    assert p.transform(X).shape == (12, 12)


@pytest.mark.parametrize(
    "modality,reducer",
    [("cn", "lasso"), ("mu", "lasso"), ("pp", "lasso"), ("cn", "pca"), ("pp", "pca")],
)
def test_omics_pipelines_fit_predict(fast_cfg, synth_ds, modality, reducer):
    mc = copy.deepcopy(fast_cfg.modalities[modality])
    mc.reducer = reducer
    X = synth_ds.modality(modality)
    pipe = build_modality_pipeline(modality, mc).fit(X.iloc[:90], synth_ds.y.iloc[:90])
    p = positive_proba(pipe, X.iloc[90:])
    assert p.shape == (30,) and np.all((p >= 0) & (p <= 1))
    imp = feature_importance(pipe, list(X.columns))
    assert len(imp) == pipe.named_steps["rf"].n_features_in_
    if reducer == "lasso":
        assert set(imp["feature"]) <= set(X.columns)  # names map back to real columns
    else:
        assert imp["feature"].str.startswith("PC").all()


def test_pipeline_handles_missing_values(fast_cfg):
    ds = dataset_from_frame(make_synthetic_dataset(n_samples=80, missing_fraction=0.1, seed=4))
    X = ds.modality("pp")
    assert X.isna().any().any()
    pipe = build_modality_pipeline("pp", fast_cfg.modalities["pp"]).fit(X, ds.y)
    assert np.isfinite(positive_proba(pipe, X)).all()


def test_autoencoder_accepts_original_noise_level_and_noise_is_training_only(tmp_path):
    """Original sigma = 50 is rejected by Keras 3's GaussianNoise; ours must build,
    perturb inputs in training mode only, and survive a .keras round trip."""
    pytest.importorskip("tensorflow")
    import keras

    from nsclc_metastasis.autoencoder import DenoisingAutoencoder, build_autoencoder

    ae, _ = build_autoencoder(6, 4, 2, noise_stddev=50.0, seed=0)
    noise = ae.get_layer("gaussian_noise")
    x = np.ones((3, 6), dtype="float32")
    np.testing.assert_array_equal(keras.ops.convert_to_numpy(noise(x, training=False)), x)
    assert np.abs(keras.ops.convert_to_numpy(noise(x, training=True)) - x).mean() > 5

    X = np.random.default_rng(0).normal(size=(20, 6)).astype("float32")
    dae = DenoisingAutoencoder(hidden_units=4, latent_units=2, epochs=1, noise_stddev=50.0).fit(X)
    np.testing.assert_allclose(dae.transform(X), dae.transform(X))  # deterministic at inference
    dae.save_keras(tmp_path / "ae.keras")
    reloaded = keras.models.load_model(tmp_path / "ae.keras")
    np.testing.assert_allclose(keras.ops.convert_to_numpy(reloaded(X, training=False)), dae.reconstruct(X), rtol=1e-5)


def test_rs_autoencoder_branch_uses_latent_codes_and_pickles(fast_cfg, synth_ds):
    pytest.importorskip("tensorflow")
    X = synth_ds.modality("rs")
    ae_cfg = fast_cfg.modalities["rs"].autoencoder
    pipe = build_modality_pipeline("rs", fast_cfg.modalities["rs"]).fit(X.iloc[:90], synth_ds.y.iloc[:90])
    dae = pipe.named_steps["reduce"]
    # The RF sees the encoder output, not original gene columns picked by latent indices.
    assert pipe.named_steps["rf"].n_features_in_ == ae_cfg.latent_units
    assert reduced_feature_names(pipe, list(X.columns)) == [f"latent_{i + 1}" for i in range(ae_cfg.latent_units)]
    assert len(dae.loss_history_) == ae_cfg.epochs
    p1 = positive_proba(pipe, X.iloc[90:])
    z = dae.transform(pipe.named_steps["scale"].transform(X.iloc[90:].to_numpy()))
    assert z.shape == (30, ae_cfg.latent_units)
    restored = pickle.loads(pickle.dumps(pipe))
    np.testing.assert_allclose(positive_proba(restored, X.iloc[90:]), p1, rtol=1e-5)
