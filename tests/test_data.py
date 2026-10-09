from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uplift_lab.data import (
    CRITEO_FEATURES,
    SimulationConfig,
    feature_matrix,
    load_criteo,
    simulate_experiment,
)


def test_simulation_is_deterministic_for_a_seed() -> None:
    a = simulate_experiment(SimulationConfig(n_users=2_000, seed=3))
    b = simulate_experiment(SimulationConfig(n_users=2_000, seed=3))
    c = simulate_experiment(SimulationConfig(n_users=2_000, seed=4))
    pd.testing.assert_frame_equal(a.data, b.data)
    assert not a.data["converted"].equals(c.data["converted"])


def test_truth_is_internally_consistent(experiment) -> None:
    truth, cfg = experiment.truth, experiment.config
    np.testing.assert_allclose(truth["tau_conversion"], truth["mu1"] - truth["mu0"])
    np.testing.assert_allclose(truth["tau_revenue"], truth["payout"] * truth["tau_conversion"])
    np.testing.assert_allclose(
        truth["tau_profit"], truth["tau_revenue"] - cfg.reward_cost * truth["mu1"]
    )
    assert ((truth[["mu0", "mu1"]] >= 0) & (truth[["mu0", "mu1"]] <= 1)).all(axis=None)


def test_effects_are_heterogeneous_with_negative_responders(experiment) -> None:
    tau = experiment.truth["tau_conversion"]
    assert tau.min() < 0 < tau.max()
    assert (experiment.truth["tau_profit"] < 0).mean() > 0.2


def test_observed_outcomes_match_their_definitions(experiment) -> None:
    df = experiment.data
    np.testing.assert_allclose(df["reward_paid"], 0.5 * df["converted"] * df["treatment"])
    np.testing.assert_allclose(df["net_revenue_14d"], df["revenue_14d"] - df["reward_paid"])
    assert set(df["treatment"].unique()) == {0, 1}


def test_conversion_rates_match_potential_outcomes(experiment) -> None:
    df, truth = experiment.data, experiment.truth
    for arm, mu in ((1, "mu1"), (0, "mu0")):
        rows = df["treatment"] == arm
        observed = df.loc[rows, "converted"].mean()
        expected = truth.loc[rows, mu].mean()
        assert abs(observed - expected) < 4 * np.sqrt(expected * (1 - expected) / rows.sum())


def test_logging_loss_drops_only_treated_low_activity_users() -> None:
    healthy = simulate_experiment(SimulationConfig(n_users=20_000, seed=5))
    broken = simulate_experiment(SimulationConfig(n_users=20_000, seed=5, treated_logging_loss=0.5))
    assert broken.data["treatment"].eq(0).sum() == healthy.data["treatment"].eq(0).sum()
    lost = set(healthy.data["user_id"]) - set(broken.data["user_id"])
    lost_rows = healthy.data.set_index("user_id").loc[sorted(lost)]
    assert len(lost) > 0
    assert lost_rows["treatment"].eq(1).all()
    assert lost_rows["sessions_7d"].le(1).all()
    assert len(broken.data) == len(broken.truth)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_users": 10},
        {"treatment_share": 1.0},
        {"reward_cost": -1.0},
        {"treated_logging_loss": 1.0},
        {"payout_by_tier": {1: 1.0}},
    ],
)
def test_invalid_config_is_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        SimulationConfig(**kwargs)


def test_feature_matrix_has_stable_columns_and_no_leakage(experiment) -> None:
    full = feature_matrix(experiment.data)
    one_level = feature_matrix(experiment.data[experiment.data["platform"] == "web"])
    assert list(full.columns) == list(one_level.columns)
    forbidden = {"treatment", "converted", "revenue_14d", "net_revenue_14d", "reward_paid"}
    assert not forbidden & {c.split("=")[0] for c in full.columns}
    assert np.isfinite(full.to_numpy()).all()


def test_feature_matrix_reports_missing_columns() -> None:
    with pytest.raises(KeyError, match="missing feature columns"):
        feature_matrix(pd.DataFrame({"tenure_days": [1]}))


def _write_criteo(path: Path, n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f: rng.normal(size=n) for f in CRITEO_FEATURES})
    df["treatment"] = (rng.random(n) < 0.85).astype(int)
    df["conversion"] = (rng.random(n) < 0.01).astype(int)
    df["visit"] = (rng.random(n) < 0.05).astype(int)
    df["exposure"] = df["treatment"] * (rng.random(n) < 0.3)
    with gzip.open(path, "wt") as fh:
        df.to_csv(fh, index=False)
    return df


def test_load_criteo_reads_gzip_and_samples(tmp_path: Path) -> None:
    path = tmp_path / "criteo.csv.gz"
    _write_criteo(path)
    full = load_criteo(path)
    assert len(full) == 500
    assert full["treatment"].dtype == np.int8
    assert len(load_criteo(path, sample_frac=0.2, seed=1)) == 100


def test_load_criteo_rejects_bad_files(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"f0": [1.0], "treatment": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_criteo(path)
    good = tmp_path / "criteo.csv.gz"
    _write_criteo(good)
    with pytest.raises(ValueError, match="sample_frac"):
        load_criteo(good, sample_frac=0.0)


def test_load_criteo_rejects_non_binary_columns(tmp_path: Path) -> None:
    path = tmp_path / "criteo.csv.gz"
    df = _write_criteo(path)
    df.loc[0, "visit"] = 2
    with gzip.open(path, "wt") as fh:
        df.to_csv(fh, index=False)
    with pytest.raises(ValueError, match="must be binary"):
        load_criteo(path)
