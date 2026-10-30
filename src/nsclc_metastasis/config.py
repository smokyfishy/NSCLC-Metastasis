"""Typed configuration loaded from YAML.

Every hyperparameter that appeared in the original notebooks is represented
here so that the corrected pipeline and the documented original values can be
compared side by side (see ``configs/`` and ``docs/REPRODUCTION_AUDIT.md``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import MODALITIES


@dataclass
class DataConfig:
    label_column: str = "metastasis_status"
    # The original CSV was written by R with row.names = FALSE, so it may have
    # no identifier column. If none of these columns exist, row-position IDs
    # are generated and a warning is emitted.
    id_columns: list[str] = field(
        default_factory=lambda: ["patient_id", "sample_id", "barcode", "bcr_patient_barcode"]
    )
    prefixes: dict[str, str] = field(
        default_factory=lambda: {"rs": "rs_", "cn": "cn_", "mu": "mu_", "pp": "pp_"}
    )
    # Counts reported in the research write-up. Used only for reporting
    # agreement/disagreement; rows and columns are never filtered to match them.
    expected: dict[str, int] = field(
        default_factory=lambda: {
            "n_samples": 220,
            "n_positive": 33,
            "n_negative": 187,
            "cn": 813,
            "rs": 401,
            "mu": 126,
            "pp": 218,
        }
    )


@dataclass
class SplitConfig:
    test_size: float = 0.2
    seed: int = 42
    outer_cv_folds: int = 5
    stacking_cv_folds: int = 5


@dataclass
class SmoteConfig:
    enabled: bool = True
    # float = minority/majority ratio after resampling; "auto" = balance fully.
    sampling_strategy: float | str = "auto"
    k_neighbors: int = 5
    random_state: int = 42


@dataclass
class LassoConfig:
    cv: int = 5
    max_iter: int = 10000
    random_state: int = 42
    # What to do if LassoCV zeroes every coefficient on a training fold:
    # "all" keeps every feature (with a warning), "error" raises.
    on_empty: str = "all"


@dataclass
class PCAConfig:
    n_components: int = 100
    random_state: int = 42


@dataclass
class AutoencoderConfig:
    hidden_units: int = 64
    latent_units: int = 32
    epochs: int = 50
    batch_size: int = 32
    noise_stddev: float = 0.5
    learning_rate: float = 0.001
    seed: int = 39
    verbose: int = 0


@dataclass
class ForestConfig:
    n_estimators: int = 100
    random_state: int = 42
    n_jobs: int = 1
    class_weight: str | None = None


@dataclass
class ModalityConfig:
    enabled: bool = True
    # "autoencoder" (rs only), "lasso", "pca" or "none"
    reducer: str = "lasso"
    scale: bool = False
    impute: str = "median"
    smote: SmoteConfig = field(default_factory=SmoteConfig)
    lasso: LassoConfig = field(default_factory=LassoConfig)
    pca: PCAConfig = field(default_factory=PCAConfig)
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    forest: ForestConfig = field(default_factory=ForestConfig)


@dataclass
class StackingConfig:
    # "proba" (positive-class probability, corrected default) or "label"
    # (hard 0/1 predictions, as in the original notebook).
    meta_features: str = "proba"
    forest: ForestConfig = field(default_factory=ForestConfig)
    threshold: float = 0.5


@dataclass
class Config:
    name: str = "baseline"
    description: str = ""
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    modalities: dict[str, ModalityConfig] = field(default_factory=dict)
    stacking: StackingConfig = field(default_factory=StackingConfig)

    def enabled_modalities(self) -> list[str]:
        return [m for m in MODALITIES if m in self.modalities and self.modalities[m].enabled]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build(cls, raw: dict[str, Any] | None):
    """Recursively build a dataclass from a dict, rejecting unknown keys."""
    raw = dict(raw or {})
    kwargs = {}
    for name, f in cls.__dataclass_fields__.items():
        if name not in raw:
            continue
        value = raw.pop(name)
        sub = _NESTED.get((cls, name))
        if sub is not None:
            value = _build(sub, value)
        kwargs[name] = value
    if raw:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(raw)}")
    return cls(**kwargs)


_NESTED = {
    (Config, "data"): DataConfig,
    (Config, "split"): SplitConfig,
    (Config, "stacking"): StackingConfig,
    (StackingConfig, "forest"): ForestConfig,
    (ModalityConfig, "smote"): SmoteConfig,
    (ModalityConfig, "lasso"): LassoConfig,
    (ModalityConfig, "pca"): PCAConfig,
    (ModalityConfig, "autoencoder"): AutoencoderConfig,
    (ModalityConfig, "forest"): ForestConfig,
}


def config_from_dict(raw: dict[str, Any]) -> Config:
    raw = dict(raw)
    modalities_raw = raw.pop("modalities", {}) or {}
    cfg = _build(Config, raw)
    unknown = set(modalities_raw) - set(MODALITIES)
    if unknown:
        raise ValueError(f"Unknown modalities in config: {sorted(unknown)}")
    cfg.modalities = {m: _build(ModalityConfig, v) for m, v in modalities_raw.items()}
    validate_config(cfg)
    return cfg


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as fh:
        return config_from_dict(yaml.safe_load(fh))


def validate_config(cfg: Config) -> None:
    if not cfg.enabled_modalities():
        raise ValueError("At least one modality must be enabled.")
    for m, mc in cfg.modalities.items():
        if mc.reducer not in {"autoencoder", "lasso", "pca", "none"}:
            raise ValueError(f"{m}: unknown reducer {mc.reducer!r}")
        if mc.reducer == "autoencoder" and m != "rs":
            raise ValueError("The denoising autoencoder branch is defined for rs only.")
        if mc.impute not in {"median", "mean", "none"}:
            raise ValueError(f"{m}: unknown impute strategy {mc.impute!r}")
        if mc.lasso.on_empty not in {"all", "error"}:
            raise ValueError(f"{m}: lasso.on_empty must be 'all' or 'error'")
    if cfg.stacking.meta_features not in {"proba", "label"}:
        raise ValueError("stacking.meta_features must be 'proba' or 'label'")
    if not 0 < cfg.split.test_size < 1:
        raise ValueError("split.test_size must be in (0, 1)")
    if cfg.split.outer_cv_folds < 2 or cfg.split.stacking_cv_folds < 2:
        raise ValueError("CV fold counts must be >= 2")
