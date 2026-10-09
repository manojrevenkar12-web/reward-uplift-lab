from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uplift_lab.cli import main
from uplift_lab.data import CRITEO_FEATURES
from uplift_lab.sql import QUERIES, load_query, run_query

# ---------------------------------------------------------------- SQL


def test_arm_summary_matches_pandas(experiment) -> None:
    df = experiment.data
    out = run_query("arm_summary", {"experiment": df}).set_index("treatment")
    expected = df.groupby("treatment").agg(
        n_users=("user_id", "size"),
        conversion_rate=("converted", "mean"),
        mean_revenue=("revenue_14d", "mean"),
        total_reward_paid=("reward_paid", "sum"),
    )
    for col in expected.columns:
        np.testing.assert_allclose(out[col], expected[col])


def test_segment_effects_match_pandas(experiment) -> None:
    df = experiment.data
    out = run_query("segment_effects", {"experiment": df})
    row = out[(out["dimension"] == "channel") & (out["segment"] == "paid_social")].iloc[0]
    sub = df[df["channel"] == "paid_social"]
    lift = (
        sub.loc[sub["treatment"] == 1, "converted"].mean()
        - sub.loc[sub["treatment"] == 0, "converted"].mean()
    )
    assert row["lift"] == pytest.approx(lift)
    assert set(out["dimension"]) == {"tenure", "activity", "channel", "country"}


def test_score_deciles_query() -> None:
    rng = np.random.default_rng(0)
    n = 10_000
    scored = pd.DataFrame(
        {
            "treatment": rng.integers(0, 2, n),
            "converted": rng.integers(0, 2, n),
            "score": rng.random(n),
        }
    )
    out = run_query("score_deciles", {"scored": scored})
    assert out["decile"].tolist() == list(range(1, 11))
    assert out["mean_predicted_uplift"].is_monotonic_decreasing


def test_every_bundled_query_loads_and_unknown_raises() -> None:
    for name in QUERIES:
        assert "SELECT" in load_query(name)
    with pytest.raises(KeyError, match="unknown query"):
        load_query("drop_tables")


# ---------------------------------------------------------------- CLI end to end


@pytest.mark.slow
def test_simulate_command_writes_consistent_outputs(tmp_path: Path) -> None:
    out = tmp_path / "sim"
    code = main(
        [
            "simulate",
            "--n-users",
            "20000",
            "--aa-splits",
            "40",
            "--peeking-sims",
            "200",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    results = json.loads((out / "results.json").read_text())  # strict JSON, no NaN
    report = (out / "REPORT.md").read_text()
    for name in (
        "variance_reduction",
        "aa_pvalues",
        "peeking",
        "targeting_curves",
        "uplift_deciles",
        "policy_values",
    ):
        assert (out / "figures" / f"{name}.png").stat().st_size > 10_000
        assert f"figures/{name}.png" in report
    assert results["uplift"]["selected_model"] in results["uplift"]["models"]
    assert "## Limitations" in report
    assert not results["srm"]["healthy"]["mismatch"]
    broken, healthy = results["srm"]["broken"], results["srm"]["healthy"]
    assert broken["n_treatment"] < healthy["n_treatment"]
    assert broken["n_control"] == healthy["n_control"]


@pytest.mark.slow
def test_criteo_command_on_a_synthetic_file(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 6_000
    df = pd.DataFrame({f: rng.normal(size=n) for f in CRITEO_FEATURES})
    df["treatment"] = (rng.random(n) < 0.85).astype(int)
    df["visit"] = (rng.random(n) < 0.04 + 0.03 * df["treatment"] * (df["f0"] > 0)).astype(int)
    df["conversion"] = (df["visit"] * (rng.random(n) < 0.1)).astype(int)
    df["exposure"] = df["treatment"]
    path = tmp_path / "criteo.csv.gz"
    with gzip.open(path, "wt") as fh:
        df.to_csv(fh, index=False)

    out = tmp_path / "criteo"
    assert main(["criteo", str(path), "--out", str(out)]) == 0
    results = json.loads((out / "results.json").read_text())
    assert results["n_users"] == n
    assert not results["srm"]["mismatch"]
    assert (out / "figures" / "criteo_qini.png").exists()
