"""Turn the simulated user table into a numeric feature matrix for modelling."""

from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_FEATURES = ("tenure_days", "sessions_7d", "offers_completed_90d", "pre_revenue_14d")
CATEGORICAL_FEATURES = {
    "platform": ("android", "ios", "web"),
    "country_tier": (1, 2, 3),
    "channel": ("organic", "paid_social", "referral"),
}


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build a model-ready feature matrix from pre-treatment user attributes only.

    Only attributes fixed before randomisation are used; outcomes and treatment are
    never features. Categorical levels are fixed in advance so that train and test
    matrices always have identical columns, even if a level is absent from one split.

    Args:
        df: Experiment data containing the columns in ``NUMERIC_FEATURES`` and
            ``CATEGORICAL_FEATURES``.

    Returns:
        A float matrix with one row per user, in the same order as ``df``.
    """
    missing = [c for c in (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES) if c not in df.columns]
    if missing:
        raise KeyError(f"missing feature columns: {missing}")

    out = pd.DataFrame(index=df.index)
    for col in NUMERIC_FEATURES:
        values = df[col].to_numpy(dtype=float)
        out[col] = np.log1p(values) if col in ("tenure_days", "pre_revenue_14d") else values
    for col, levels in CATEGORICAL_FEATURES.items():
        for level in levels[1:]:  # first level is the reference category
            out[f"{col}={level}"] = (df[col] == level).to_numpy(dtype=float)
    return out
