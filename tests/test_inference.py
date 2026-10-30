import copy
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from nsclc_metastasis.cli import main
from nsclc_metastasis.data import DataValidationError, dataset_from_frame
from nsclc_metastasis.evaluation import compute_metrics
from nsclc_metastasis.inference import load_model, predict_dataset, save_model
from nsclc_metastasis.stacking import MultiOmicsStackingModel


def test_save_load_roundtrip_reproduces_predictions(tmp_path, fast_cfg, synth_ds):
    pytest.importorskip("tensorflow")
    cfg = copy.deepcopy(fast_cfg)
    cfg.modalities["rs"].autoencoder.epochs = 1
    model = MultiOmicsStackingModel(cfg).fit(synth_ds.X.iloc[:100], synth_ds.y.iloc[:100])
    out = save_model(model, tmp_path / "model", synth_ds)
    files = {p.name for p in out.iterdir()}
    assert {"manifest.json", "config.yaml", "meta.joblib", "base_rs.joblib", "rs_autoencoder.keras"} <= files

    loaded = load_model(out)
    Xt = synth_ds.X.iloc[100:]
    np.testing.assert_allclose(loaded.predict_proba(Xt), model.predict_proba(Xt), rtol=1e-5)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["feature_columns"]["cn"] == model.feature_columns_["cn"]


def test_load_model_rejects_single_file(tmp_path):
    f = tmp_path / "model.joblib"
    f.write_bytes(b"x")
    with pytest.raises(ValueError, match="directory"):
        load_model(f)


def test_predict_requires_training_columns(tmp_path, fast_cfg, synth_df):
    cfg = copy.deepcopy(fast_cfg)
    cfg.modalities["rs"].enabled = False
    ds = dataset_from_frame(synth_df)
    model = MultiOmicsStackingModel(cfg).fit(ds.X, ds.y)
    bad = dataset_from_frame(synth_df.drop(columns=["cn_SYNGENE000", "metastasis_status"]), require_label=False)
    with pytest.raises(DataValidationError, match="missing"):
        predict_dataset(model, bad)


def test_compute_metrics_values():
    m = compute_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9])
    assert m["accuracy"] == 0.5 and m["roc_auc"] == 0.75
    assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    single = compute_metrics([0, 0], [0.2, 0.7])
    assert np.isnan(single["roc_auc"])


def test_cli_end_to_end_on_synthetic_data(tmp_path, fast_cfg, synth_df, monkeypatch):
    """train -> evaluate -> predict through the real CLI (SYNTHETIC data)."""
    pytest.importorskip("tensorflow")
    monkeypatch.chdir(tmp_path)
    cfg = copy.deepcopy(fast_cfg)
    cfg.modalities["rs"].autoencoder.epochs = 1
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg.to_dict()))
    synth_df.to_csv(tmp_path / "SYNTHETIC.csv", index=False)

    assert main(["validate-data", "--data", "SYNTHETIC.csv", "--config", "cfg.yaml"]) == 0
    assert main(["train", "--data", "SYNTHETIC.csv", "--config", "cfg.yaml",
                 "--model-dir", "model", "--reports-dir", "rep"]) == 0
    for name in ["metrics.json", "split_manifest.csv", "cv_fold_metrics.csv", "test_predictions.csv",
                 "roc_test.png", "pr_test.png", "confusion_matrix_test.png"]:
        assert (tmp_path / "rep" / name).exists(), name
    metrics = json.loads((tmp_path / "rep" / "metrics.json").read_text())
    assert metrics["synthetic_data"] is True
    assert metrics["split"]["n_test"] == 24
    cv = pd.read_csv(tmp_path / "rep" / "cv_fold_metrics.csv")
    assert sorted(cv["fold"].unique()) == [0, 1, 2]

    assert main(["evaluate", "--model", "model", "--data", "SYNTHETIC.csv", "--reports-dir", "ev"]) == 0
    ev = json.loads((tmp_path / "ev" / "evaluate_test.json").read_text())
    assert ev["ensemble"]["n"] == 24
    np.testing.assert_allclose(ev["ensemble"]["roc_auc"], metrics["test_metrics"]["ensemble"]["roc_auc"])

    unlabeled = synth_df.drop(columns=["metastasis_status"]).head(5)
    unlabeled.to_csv(tmp_path / "new.csv", index=False)
    assert main(["predict", "--model", "model", "--data", "new.csv", "--out", "pred.csv"]) == 0
    pred = pd.read_csv(tmp_path / "pred.csv")
    assert len(pred) == 5 and "ensemble_probability" in pred.columns
