from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from uplift_lab.data import feature_matrix
from uplift_lab.experiment import wilson_interval
from uplift_lab.uplift import (
    DRLearner,
    SLearner,
    TLearner,
    XLearner,
    budgeted_profit_policy,
    expected_incremental_profit,
    ipw_policy_gain,
    qini_curve,
    top_fraction_policy,
    true_gain_curve,
    truth_metrics,
    uplift_by_bin,
    uplift_curve,
)

# ---------------------------------------------------------------- meta-learners


@pytest.fixture(scope="module")
def split(experiment):
    df, truth = experiment.data, experiment.truth
    X = feature_matrix(df).to_numpy()
    t = df["treatment"].to_numpy()
    y = df["converted"].to_numpy(dtype=float)
    half = len(df) // 2
    return X[:half], t[:half], y[:half], X[half:], truth["tau_conversion"].to_numpy()[half:]


@pytest.mark.parametrize("learner", [SLearner, TLearner, XLearner, DRLearner])
def test_learners_recover_the_ranking_of_true_effects(learner, split) -> None:
    X_tr, t_tr, y_tr, X_te, tau_te = split
    tau_hat = learner().fit(X_tr, t_tr, y_tr).predict_uplift(X_te)
    metrics = truth_metrics(tau_te, tau_hat)
    # Thresholds are set for the weakest learner (the T-learner) on 20,000 users;
    # the study itself compares the learners properly on 100,000.
    assert metrics.spearman > 0.3
    assert metrics.normalised_gain > 0.4
    assert abs(metrics.mean_bias) < 0.01


def test_learners_accept_custom_model_factories(split) -> None:
    X_tr, t_tr, y_tr, X_te, _ = split
    learner = DRLearner(
        outcome_model=lambda: LogisticRegression(max_iter=500),
        effect_model=LinearRegression,
        n_folds=2,
    )
    assert learner.fit(X_tr, t_tr, y_tr).predict_uplift(X_te).shape == (len(X_te),)


def test_dr_pseudo_outcomes_average_to_the_ate(experiment) -> None:
    df, truth = experiment.data, experiment.truth
    X = feature_matrix(df).to_numpy()
    t = df["treatment"].to_numpy().astype(bool)
    y = df["converted"].to_numpy(dtype=float)
    pseudo = DRLearner(n_folds=3).pseudo_outcomes(X, t, y)
    se = pseudo.std(ddof=1) / np.sqrt(len(pseudo))
    assert abs(pseudo.mean() - truth["tau_conversion"].mean()) < 4 * se


def test_predicting_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="fitted"):
        TLearner().predict_uplift(np.zeros((3, 2)))


def test_learner_input_validation() -> None:
    X = np.zeros((50, 2))
    with pytest.raises(ValueError, match="binary"):
        TLearner().fit(X, np.full(50, 2), np.zeros(50))
    with pytest.raises(ValueError, match="at least 10"):
        TLearner().fit(X, np.r_[np.ones(45), np.zeros(5)], np.zeros(50))
    with pytest.raises(ValueError, match="n_folds"):
        DRLearner(n_folds=1)


# ---------------------------------------------------------------- evaluation


def _toy():
    # Treated users 0-3 convert only when score is high; control never converts.
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=float)
    t = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    score = np.array([0.9, 0.8, 0.1, 0.0, 0.95, 0.85, 0.05, 0.01])
    return y, t, score


def test_uplift_curve_ends_at_the_ate() -> None:
    y, t, score = _toy()
    curve = uplift_curve(y, t, score, n_points=5)
    assert curve.values[0] == 0.0
    assert curve.values[-1] == pytest.approx(y[t == 1].mean() - y[t == 0].mean())


def test_qini_curve_hand_example() -> None:
    y, t, score = _toy()
    curve = qini_curve(y, t, score, n_points=3)  # fractions 0, 0.5, 1
    # Top half: users with scores .95 .9 .85 .8 -> treated converts 2, control 0.
    assert curve.values[1] == pytest.approx(2 / 8)
    assert curve.values[2] == pytest.approx(2 / 8)


def test_good_scores_beat_reversed_scores() -> None:
    rng = np.random.default_rng(0)
    n = 20_000
    x = rng.random(n)
    t = (rng.random(n) < 0.5).astype(int)
    y = (rng.random(n) < 0.1 + 0.2 * x * t).astype(float)
    assert qini_curve(y, t, x).area_over_random() > 0
    assert qini_curve(y, t, -x).area_over_random() < 0


def test_truth_metrics_for_the_oracle_and_for_noise() -> None:
    rng = np.random.default_rng(0)
    tau = rng.normal(size=5_000)
    perfect = truth_metrics(tau, tau)
    assert perfect.pehe == 0.0
    assert perfect.normalised_gain == pytest.approx(1.0)
    noise = truth_metrics(tau, rng.normal(size=5_000))
    assert abs(noise.normalised_gain) < 0.1


def test_true_gain_curve_reaches_the_total_effect() -> None:
    tau = np.array([3.0, -1.0, 2.0, 0.0])
    curve = true_gain_curve(tau, tau, n_points=5)
    assert curve.values[-1] == pytest.approx(tau.mean())
    assert curve.values.max() == pytest.approx(5.0 / 4)


def test_uplift_by_bin_orders_bins_by_score() -> None:
    rng = np.random.default_rng(1)
    n = 30_000
    x = rng.random(n)
    t = (rng.random(n) < 0.5).astype(int)
    y = (rng.random(n) < 0.1 + 0.3 * x * t).astype(float)
    rows = uplift_by_bin(y, t, x, n_bins=5)
    observed = [r["observed_uplift"] for r in rows]
    assert observed == sorted(observed, reverse=True)
    assert [r["bin"] for r in rows] == [1, 2, 3, 4, 5]


def test_curve_input_validation() -> None:
    with pytest.raises(ValueError, match="equal length"):
        qini_curve([1.0], [1, 0], [0.5, 0.5])
    with pytest.raises(ValueError, match="NaN"):
        qini_curve([1.0, 0.0], [1, 0], [np.nan, 0.5])


# ---------------------------------------------------------------- policy


def test_expected_profit_formula() -> None:
    profit = expected_incremental_profit([0.1, 0.02], [0.5, 0.3], [6.0, 1.0], 0.5)
    np.testing.assert_allclose(profit, [0.6 - 0.25, 0.02 - 0.15])


def test_top_fraction_policy_treats_exact_share() -> None:
    policy = top_fraction_policy(np.arange(10.0), 0.3)
    assert policy.sum() == 3
    assert policy[-3:].all()


def test_budgeted_policy_respects_budget_and_prefers_roi() -> None:
    profit = np.array([1.0, 2.0, 0.5, -1.0])
    cost = np.array([1.0, 4.0, 0.1, 0.1])
    assert budgeted_profit_policy(profit, cost).tolist() == [True, True, True, False]
    capped = budgeted_profit_policy(profit, cost, budget=1.5)
    assert capped.tolist() == [True, False, True, False]
    assert cost[capped].sum() <= 1.5


def test_policy_against_itself_has_zero_gain() -> None:
    rng = np.random.default_rng(0)
    t = (rng.random(1_000) < 0.5).astype(int)
    pi = rng.random(1_000) < 0.4
    value = ipw_policy_gain(rng.normal(size=1_000), t, pi, pi, propensity=0.5)
    assert value.gain == 0.0
    assert value.std_error == 0.0


@pytest.mark.slow
def test_ipw_policy_gain_is_unbiased_and_covers() -> None:
    rng = np.random.default_rng(5)
    n, sims, hits, gains = 4_000, 400, 0, []
    x = rng.random(n)
    effect = 2.0 * x - 0.8  # positive for x > 0.4
    policy = x > 0.4
    truth = float(np.mean(policy * effect))
    for _ in range(sims):
        t = (rng.random(n) < 0.3).astype(int)
        y = rng.normal(size=n) + effect * t
        v = ipw_policy_gain(y, t, policy, propensity=0.3)
        gains.append(v.gain)
        hits += v.ci_low <= truth <= v.ci_high
    assert np.mean(gains) == pytest.approx(truth, abs=0.01)
    low, high = wilson_interval(hits, sims, confidence=0.999)
    assert low <= 0.95 <= high


def test_ipw_reports_truth_when_given() -> None:
    t = np.array([1, 0, 1, 0])
    value = ipw_policy_gain(
        np.ones(4), t, np.array([1, 1, 0, 0]), propensity=0.5, true_effect=np.full(4, 2.0)
    )
    assert value.true_gain == pytest.approx(1.0)
