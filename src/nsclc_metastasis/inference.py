"""Saving, loading and applying a trained model artifact.

A trained model is a *directory*, not a single file, because it combines
scikit-learn/imblearn pipelines (joblib) with a Keras autoencoder::

    artifacts/model/
      manifest.json          # versions, feature columns, data hash, file list
      config.yaml            # exact configuration used for training
      base_rs.joblib         # imblearn pipeline (DAE weights stored as arrays)
      base_cn.joblib, base_mu.joblib, base_pp.joblib
      meta.joblib            # random-forest meta-classifier
      rs_autoencoder.keras   # native Keras export of the DAE (inspection only)
      split_manifest.csv     # train/test membership used to build this model

The ``.keras`` file is an extra export; loading uses the joblib files, whose
``DenoisingAutoencoder`` step carries its own weights. This avoids the
original ``.h5``/``.joblib``/``.pkl`` mismatch where the prediction helper
loaded ``random_forest_model.pkl`` that was never written.
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from . import __version__
from .config import config_from_dict
from .data import MultiOmicsDataset
from .stacking import MultiOmicsStackingModel

ARTIFACT_FORMAT = "nsclc-metastasis-model/v1"
DISCLAIMER = (
    "Research use only. Predictions are outputs of an experimental model trained on a small "
    "retrospective TCGA cohort and must not be used for diagnosis or treatment decisions."
)


def _versions() -> dict:
    import imblearn
    import sklearn

    out = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "imbalanced-learn": imblearn.__version__,
        "nsclc_metastasis": __version__,
    }
    try:
        import tensorflow as tf

        out["tensorflow"] = tf.__version__
    except ImportError:
        out["tensorflow"] = None
    return out


def save_model(model: MultiOmicsStackingModel, out_dir: str | Path, dataset: MultiOmicsDataset | None = None,
               split_manifest: pd.DataFrame | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for m, pipe in model.base_models_.items():
        name = f"base_{m}.joblib"
        joblib.dump(pipe, out_dir / name)
        files[f"base_{m}"] = name
        reducer = pipe.named_steps.get("reduce")
        if m == "rs" and hasattr(reducer, "save_keras"):
            reducer.save_keras(out_dir / "rs_autoencoder.keras")
            files["rs_autoencoder_export"] = "rs_autoencoder.keras"
    joblib.dump(model.meta_model_, out_dir / "meta.joblib")
    files["meta"] = "meta.joblib"
    (out_dir / "config.yaml").write_text(yaml.safe_dump(model.config.to_dict(), sort_keys=False), encoding="utf-8")
    if split_manifest is not None:
        split_manifest.to_csv(out_dir / "split_manifest.csv", index=False)
        files["split_manifest"] = "split_manifest.csv"
    manifest = {
        "format": ARTIFACT_FORMAT,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": _versions(),
        "modalities": model.modalities_,
        "feature_columns": model.feature_columns_,
        "meta_features": model.config.stacking.meta_features,
        "threshold": model.config.stacking.threshold,
        "n_training_patients": len(model.training_ids_),
        "training_data": {
            "path": dataset.source_path if dataset else None,
            "sha256": dataset.sha256 if dataset else None,
            "id_source": dataset.id_source if dataset else None,
        },
        "files": files,
        "disclaimer": DISCLAIMER,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_dir


def load_model(model_dir: str | Path) -> MultiOmicsStackingModel:
    model_dir = Path(model_dir)
    if model_dir.is_file():
        raise ValueError(
            f"{model_dir} is a file; a trained model is a directory containing manifest.json."
        )
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json in {model_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != ARTIFACT_FORMAT:
        raise ValueError(f"Unsupported artifact format {manifest.get('format')!r}")
    cfg = config_from_dict(yaml.safe_load((model_dir / "config.yaml").read_text(encoding="utf-8")))
    model = MultiOmicsStackingModel(cfg)
    model.modalities_ = manifest["modalities"]
    model.feature_columns_ = manifest["feature_columns"]
    model.base_models_ = {m: joblib.load(model_dir / manifest["files"][f"base_{m}"]) for m in model.modalities_}
    model.meta_model_ = joblib.load(model_dir / manifest["files"]["meta"])
    model.training_ids_ = []
    model.manifest_ = manifest
    return model


def load_split_manifest(model_dir: str | Path) -> pd.DataFrame | None:
    p = Path(model_dir) / "split_manifest.csv"
    return pd.read_csv(p, dtype={"patient_id": str}) if p.exists() else None


def predict_dataset(model: MultiOmicsStackingModel, dataset: MultiOmicsDataset) -> pd.DataFrame:
    """Per-patient base-model outputs, ensemble probability and label."""
    out = model.predict_detail(dataset.X)
    out.index.name = "patient_id"
    return out
