import numpy as np
import pandas as pd
import pytest

from nsclc_metastasis.config import DataConfig, config_from_dict, load_config
from nsclc_metastasis.data import (
    DataValidationError,
    align_columns,
    dataset_from_frame,
    load_dataset,
    split_modalities,
)
from nsclc_metastasis.synthetic import make_synthetic_dataset


def test_prefix_split_uses_full_prefix_and_keeps_order():
    cols = ["rs_A", "cn_B", "cnv_score", "mu_C", "pp_D", "rs_E", "metastasis_status"]
    out = split_modalities(cols, DataConfig().prefixes)
    assert out == {"rs": ["rs_A", "rs_E"], "cn": ["cn_B"], "mu": ["mu_C"], "pp": ["pp_D"]}


def test_dataset_counts_and_ids(synth_df):
    ds = dataset_from_frame(synth_df)
    assert ds.n_samples == 120
    assert {m: len(c) for m, c in ds.modality_columns.items()} == {"rs": 40, "cn": 30, "mu": 20, "pp": 25}
    assert ds.id_source == "patient_id"
    assert ds.X.index.equals(ds.y.index)
    assert "metastasis_status" not in ds.X.columns and "patient_id" not in ds.X.columns


def test_missing_id_column_falls_back_to_row_ids_with_warning():
    df = make_synthetic_dataset(n_samples=30, with_ids=False)
    with pytest.warns(UserWarning, match="No patient ID column"):
        ds = dataset_from_frame(df)
    assert ds.id_source == "row_position"
    assert ds.X.index[0] == "row_0000"


def test_reported_counts_are_compared_not_enforced(synth_df):
    ds = dataset_from_frame(synth_df)
    cmp = ds.summary(DataConfig().expected)["comparison_with_reported_counts"]
    assert cmp["n_samples"] == {"reported": 220, "observed": 120, "match": False}
    assert ds.n_samples == 120  # nothing was dropped or padded


@pytest.mark.parametrize("bad", [[0, 1, 2], ["yes", "no", "no"]])
def test_non_binary_label_rejected(bad):
    df = pd.DataFrame({"rs_a": [1.0, 2.0, 3.0], "metastasis_status": bad})
    with pytest.raises(DataValidationError):
        dataset_from_frame(df)


def test_missing_label_rejected_unless_unlabeled_allowed():
    df = pd.DataFrame({"patient_id": ["a", "b"], "rs_a": [1.0, 2.0]})
    with pytest.raises(DataValidationError):
        dataset_from_frame(df)
    ds = dataset_from_frame(df, require_label=False)
    assert ds.y is None


def test_duplicate_ids_rejected():
    df = pd.DataFrame({"patient_id": ["a", "a"], "rs_a": [1.0, 2.0], "metastasis_status": [0, 1]})
    with pytest.raises(DataValidationError, match="Duplicate patient IDs"):
        dataset_from_frame(df)


def test_missing_values_are_kept_and_counted():
    df = make_synthetic_dataset(n_samples=50, missing_fraction=0.05, seed=3)
    ds = dataset_from_frame(df)
    assert sum(ds.summary()["n_missing_values"].values()) > 0
    assert ds.X.isna().any().any()  # imputation happens in-fold, not at load time


def test_align_columns_errors_on_missing_and_ignores_extra():
    X = pd.DataFrame(np.zeros((2, 3)), columns=["rs_a", "rs_b", "extra"])
    assert list(align_columns(X, ["rs_b", "rs_a"]).columns) == ["rs_b", "rs_a"]
    with pytest.raises(DataValidationError):
        align_columns(X, ["rs_a", "rs_zzz"])


def test_load_dataset_csv_roundtrip_and_hash(tmp_path, synth_df):
    p = tmp_path / "synthetic.csv"
    synth_df.to_csv(p, index=False)
    ds = load_dataset(p)
    assert ds.sha256 and len(ds.sha256) == 64
    assert ds.n_samples == len(synth_df)


def test_missing_file_message_points_to_provenance(tmp_path):
    with pytest.raises(FileNotFoundError, match="DATA_PROVENANCE"):
        load_dataset(tmp_path / "NSCLCMultiOmicsDataProcessed_reordered.csv")


def test_shipped_configs_load():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "configs"
    names = sorted(p.name for p in root.glob("*.yaml"))
    assert {"baseline.yaml", "pca_variant.yaml", "legacy_hyperparameters.yaml"} <= set(names)
    for p in root.glob("*.yaml"):
        cfg = load_config(p)
        assert cfg.enabled_modalities() == ["rs", "cn", "mu", "pp"]


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError):
        config_from_dict({"modalities": {"rs": {"reducer": "autoencoder", "typo": 1}}})
    with pytest.raises(ValueError):
        config_from_dict({"modalities": {"cn": {"reducer": "autoencoder"}}})
