"""Loading, schema validation and patient alignment for the processed CSV.

The modelling notebooks read ``NSCLCMultiOmicsDataProcessed_reordered.csv``:
one row per patient, feature columns prefixed ``rs_``/``cn_``/``mu_``/``pp_``
and a binary ``metastasis_status`` label. This module loads that file (or any
CSV with the same schema), assigns patient identifiers, splits features by
modality and reports how the observed shape compares with the counts in the
research write-up. It never drops rows or columns to force those counts.
"""

from __future__ import annotations

import hashlib
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import MODALITIES
from .config import DataConfig

log = logging.getLogger(__name__)

ROW_ID_PREFIX = "row_"


class DataValidationError(ValueError):
    """Raised when the input table cannot be used safely."""


@dataclass
class MultiOmicsDataset:
    X: pd.DataFrame  # index = patient id, columns = all prefixed features
    y: pd.Series | None  # same index, int 0/1; None for unlabeled inference data
    modality_columns: dict[str, list[str]]
    id_source: str  # name of the id column, or "row_position"
    source_path: str | None = None
    sha256: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return len(self.X)

    def modality(self, name: str) -> pd.DataFrame:
        return self.X[self.modality_columns[name]]

    def summary(self, expected: dict[str, int] | None = None) -> dict:
        out = {
            "source_path": self.source_path,
            "sha256": self.sha256,
            "id_source": self.id_source,
            "n_samples": self.n_samples,
            "n_features": {m: len(c) for m, c in self.modality_columns.items()},
            "n_missing_values": {
                m: int(self.X[c].isna().sum().sum()) for m, c in self.modality_columns.items()
            },
            "warnings": list(self.warnings),
        }
        if self.y is not None:
            counts = self.y.value_counts().to_dict()
            out["n_positive"] = int(counts.get(1, 0))
            out["n_negative"] = int(counts.get(0, 0))
        if expected:
            out["comparison_with_reported_counts"] = compare_with_expected(out, expected)
        return out


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_modalities(columns, prefixes: dict[str, str]) -> dict[str, list[str]]:
    """Group columns by modality prefix, preserving column order.

    The notebooks matched ``startswith('cn')`` (no underscore) for some
    modalities, which would also capture unrelated columns such as a
    hypothetical ``cnv_score``. The full prefix including the underscore is
    used here.
    """
    out = {m: [] for m in MODALITIES if m in prefixes}
    for col in columns:
        for m, p in prefixes.items():
            if str(col).startswith(p):
                out[m].append(col)
                break
    return out


def _resolve_ids(df: pd.DataFrame, cfg: DataConfig) -> tuple[pd.Index, str, list[str]]:
    msgs = []
    for col in cfg.id_columns:
        if col in df.columns:
            ids = df[col].astype(str)
            if ids.duplicated().any():
                dups = ids[ids.duplicated()].unique()[:5].tolist()
                raise DataValidationError(f"Duplicate patient IDs in column {col!r}: {dups}")
            return pd.Index(ids, name="patient_id"), col, msgs
    msg = (
        "No patient ID column found (looked for "
        f"{cfg.id_columns}); using row positions as IDs. The original R export "
        "used row.names = FALSE, so TCGA barcodes cannot be recovered from this file."
    )
    warnings.warn(msg, stacklevel=3)
    msgs.append(msg)
    ids = pd.Index([f"{ROW_ID_PREFIX}{i:04d}" for i in range(len(df))], name="patient_id")
    return ids, "row_position", msgs


def dataset_from_frame(
    df: pd.DataFrame,
    cfg: DataConfig | None = None,
    require_label: bool = True,
    source_path: str | None = None,
    sha256: str | None = None,
) -> MultiOmicsDataset:
    cfg = cfg or DataConfig()
    df = df.copy()
    # pandas writes an unnamed index column when a frame is saved with index=True
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed: ")]]

    if df.columns.duplicated().any():
        dups = df.columns[df.columns.duplicated()].unique()[:5].tolist()
        raise DataValidationError(f"Duplicate column names: {dups}")

    ids, id_source, msgs = _resolve_ids(df, cfg)
    modality_columns = split_modalities(df.columns, cfg.prefixes)
    empty = [m for m, c in modality_columns.items() if not c]
    if len(empty) == len(modality_columns):
        raise DataValidationError(
            f"No feature columns with prefixes {list(cfg.prefixes.values())} were found."
        )
    for m in empty:
        msgs.append(f"No columns found for modality {m!r} (prefix {cfg.prefixes[m]!r}).")

    feature_cols = [c for m in modality_columns for c in modality_columns[m]]
    X = df[feature_cols].copy()
    X.index = ids
    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        X[non_numeric] = X[non_numeric].apply(pd.to_numeric, errors="coerce")
        msgs.append(f"Coerced {len(non_numeric)} non-numeric feature columns to numeric.")
    X = X.astype(float)
    if np.isinf(X.to_numpy()).any():
        raise DataValidationError("Feature table contains infinite values.")

    y = None
    if cfg.label_column in df.columns:
        raw = df[cfg.label_column]
        if raw.isna().any():
            raise DataValidationError(f"{raw.isna().sum()} rows have a missing label.")
        values = set(pd.unique(raw))
        if not values <= {0, 1, 0.0, 1.0, True, False}:
            raise DataValidationError(
                f"Label column {cfg.label_column!r} must be binary 0/1; found {sorted(values)[:10]}"
            )
        y = pd.Series(raw.astype(int).to_numpy(), index=ids, name=cfg.label_column)
    elif require_label:
        raise DataValidationError(f"Label column {cfg.label_column!r} not found.")

    for m in msgs:
        log.warning(m)
    return MultiOmicsDataset(
        X=X,
        y=y,
        modality_columns=modality_columns,
        id_source=id_source,
        source_path=source_path,
        sha256=sha256,
        warnings=msgs,
    )


def load_dataset(
    path: str | Path, cfg: DataConfig | None = None, require_label: bool = True
) -> MultiOmicsDataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. The processed multi-omics CSV is not distributed "
            "with this repository; see docs/DATA_PROVENANCE.md for what is required."
        )
    df = pd.read_csv(path)
    return dataset_from_frame(
        df, cfg, require_label=require_label, source_path=str(path), sha256=file_sha256(path)
    )


def compare_with_expected(summary: dict, expected: dict[str, int]) -> dict:
    observed = {
        "n_samples": summary.get("n_samples"),
        "n_positive": summary.get("n_positive"),
        "n_negative": summary.get("n_negative"),
        **summary.get("n_features", {}),
    }
    return {
        k: {"reported": v, "observed": observed.get(k), "match": observed.get(k) == v}
        for k, v in expected.items()
    }


def align_columns(X: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    """Return ``X`` restricted to ``required`` columns in training order.

    Missing columns are an error (silently filling them would change the
    model's input distribution); extra columns are ignored.
    """
    missing = [c for c in required if c not in X.columns]
    if missing:
        raise DataValidationError(
            f"{len(missing)} required feature columns are missing, e.g. {missing[:5]}"
        )
    return X.loc[:, required]
