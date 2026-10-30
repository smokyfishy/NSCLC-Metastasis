"""SYNTHETIC data generator for tests and pipeline smoke runs ONLY.

Nothing produced here resembles real patients. The generator exists so that
the code paths (schema checks, fold isolation, stacking, save/load) can be
exercised without access to TCGA-derived data. Any metric computed on this
data is meaningless as evidence about NSCLC metastasis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SYNTHETIC_ID_PREFIX = "SYNTH-"


def make_synthetic_dataset(
    n_samples: int = 120,
    positive_fraction: float = 0.15,
    n_features: dict[str, int] | None = None,
    signal: float = 1.0,
    missing_fraction: float = 0.0,
    seed: int = 0,
    with_ids: bool = True,
) -> pd.DataFrame:
    """Return a table with the processed-CSV schema filled with random values.

    A weak label-dependent shift (``signal``) is added to a few columns per
    modality so classifiers have something to learn in smoke tests.
    """
    rng = np.random.default_rng(seed)
    n_features = n_features or {"rs": 40, "cn": 30, "mu": 20, "pp": 25}
    n_pos = max(2, int(round(n_samples * positive_fraction)))
    y = np.zeros(n_samples, dtype=int)
    y[rng.choice(n_samples, n_pos, replace=False)] = 1

    blocks = {}
    for m, k in n_features.items():
        if m == "mu":
            base = rng.random((n_samples, k)) < 0.1
            vals = base.astype(float)
            informative = min(3, k)
            vals[y == 1, :informative] = (rng.random(((y == 1).sum(), informative)) < 0.5)
        elif m == "cn":
            vals = rng.integers(-2, 3, size=(n_samples, k)).astype(float)
            vals[y == 1, : min(3, k)] += signal
        else:
            vals = rng.normal(size=(n_samples, k))
            vals[y == 1, : min(4, k)] += signal
        blocks[m] = pd.DataFrame(vals, columns=[f"{m}_SYNGENE{j:03d}" for j in range(k)])

    # The original CSV ordered blocks rs, cn, mu, pp (NSCLCData.ipynb, last cells).
    order = [m for m in ("rs", "cn", "mu", "pp") if m in blocks]
    df = pd.concat([blocks[m] for m in order], axis=1)
    if missing_fraction > 0:
        mask = rng.random(df.shape) < missing_fraction
        df = df.mask(mask)
    df["metastasis_status"] = y
    if with_ids:
        df.insert(0, "patient_id", [f"{SYNTHETIC_ID_PREFIX}{i:04d}" for i in range(n_samples)])
    return df
