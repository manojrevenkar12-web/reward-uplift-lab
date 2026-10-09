"""Loader for the public Criteo Uplift Prediction dataset (v2.1).

The dataset comes from incrementality tests in online advertising: about 14 million
users, 12 anonymised features, a randomised treatment (being targeted by advertising),
and two binary outcomes, ``visit`` and ``conversion``. Roughly 85% of users are treated.

Reference: Diemert et al., "A Large Scale Benchmark for Uplift Modeling" (AdKDD 2018).
The file is distributed by Criteo AI Lab as ``criteo-research-uplift-v2.1.csv.gz``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CRITEO_FEATURES = tuple(f"f{i}" for i in range(12))
CRITEO_BINARY_COLUMNS = ("treatment", "conversion", "visit", "exposure")
CRITEO_TREATMENT_SHARE = 0.85


def load_criteo(
    path: str | Path,
    *,
    sample_frac: float | None = None,
    nrows: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Load and validate the Criteo Uplift CSV.

    Args:
        path: Path to the (optionally gzipped) CSV file.
        sample_frac: If given, keep a random fraction of rows. Sampling is uniform over
            users, so it preserves the randomisation and the treatment share.
        nrows: If given, read only the first ``nrows`` rows (useful for quick checks).
        seed: Random seed for sampling.

    Returns:
        A frame with the 12 features as float32 and the binary columns as int8.

    Raises:
        ValueError: If required columns are missing or binary columns hold other values.
    """
    if sample_frac is not None and not 0.0 < sample_frac <= 1.0:
        raise ValueError("sample_frac must be in (0, 1]")

    dtypes: dict[str, str] = {f: "float32" for f in CRITEO_FEATURES}
    dtypes.update({c: "int8" for c in CRITEO_BINARY_COLUMNS})
    header = pd.read_csv(path, nrows=0).columns
    missing = [c for c in dtypes if c not in header]
    if missing:
        raise ValueError(f"not a Criteo Uplift file; missing columns: {missing}")

    df = pd.read_csv(path, usecols=list(dtypes), dtype=dtypes, nrows=nrows)
    if sample_frac is not None and sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=seed)

    for col in CRITEO_BINARY_COLUMNS:
        bad = ~df[col].isin([0, 1])
        if bad.any():
            raise ValueError(f"column {col!r} must be binary; found {int(bad.sum())} other values")
    if df[list(CRITEO_FEATURES)].isna().any().any():
        raise ValueError("feature columns contain missing values")
    return df.reset_index(drop=True)
