"""Correctness tests for experiment analysis.

Deterministic tests check formulas against hand-computed values. Tests marked
``slow`` check statistical properties (unbiasedness, coverage, false-positive rates)
by simulation, with tolerances wide enough to fail only on a real defect.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from uplift_lab.experiment import (
    always_valid_p_values,
    cuped,
    difference_in_means,
    minimum_detectable_effect,
    msprt_log_lr,
    regression_adjusted,
    relative_lift,
    required_sample_size_means,
    required_sample_size_proportions,
    run_aa_tests,
    sample_ratio_mismatch,
    sequential_monitor,
    simulate_peeking,
    wilson_interval,
)

# ---------------------------------------------------------------- estimators


def test_difference_in_means_matches_hand_calculation() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 7.0])
    t = np.array([0, 0, 0, 1, 1, 1])
    est = difference_in_means(y, t)
    se = np.sqrt(np.var([4, 5, 7], ddof=1) / 3 + np.var([1, 2, 3], ddof=1) / 3)
    assert est.estimate == pytest.approx(16 / 3 - 2)
    assert est.std_error == pytest.approx(se)
    assert est.ci_low == pytest.approx(est.estimate - 1.959964 * se, rel=1e-6)
    assert est.p_value == pytest.approx(2 * stats.norm.sf(est.estimate / se))
    assert (est.n_treatment, est.n_control) == (3, 3)


def test_relative_lift_point_estimate() -> None:
    y = np.array([1.0, 1.0, 2.0, 2.0])
    t = np.array([0, 0, 1, 1])
    y = y + np.array([0.0, 0.1, 0.0, 0.1])
    est = relative_lift(y, t)
    assert est.estimate == pytest.approx(2.05 / 1.05 - 1)


@pytest.mark.parametrize(
    ("y", "t", "message"),
    [
        ([1.0, 2.0], [0, 1, 1], "equal length"),
        ([1.0, 2.0, 3.0], [0, 2, 1], "binary"),
        ([1.0, np.nan, 3.0, 4.0], [0, 0, 1, 1], "NaN"),
        ([1.0, 2.0, 3.0], [0, 1, 1], "at least two"),
    ],
)
def test_estimator_input_validation(y, t, message) -> None:
    with pytest.raises(ValueError, match=message):
        difference_in_means(y, t)


def test_relative_lift_undefined_for_zero_control_mean() -> None:
    with pytest.raises(ValueError, match="undefined"):
        relative_lift([0.0, 0.0, 1.0, 2.0], [0, 0, 1, 1])


# ---------------------------------------------------------------- SRM


def test_srm_accepts_balanced_and_flags_skewed_arms() -> None:
    balanced = np.r_[np.ones(50_100), np.zeros(49_900)].astype(int)
    skewed = np.r_[np.ones(50_800), np.zeros(49_200)].astype(int)
    assert not sample_ratio_mismatch(balanced).mismatch
    result = sample_ratio_mismatch(skewed)
    assert result.mismatch
    assert result.observed_treatment_share == pytest.approx(0.508)


def test_srm_respects_unequal_designed_allocation() -> None:
    t = np.r_[np.ones(85_000), np.zeros(15_000)].astype(int)
    assert not sample_ratio_mismatch(t, expected_treatment_share=0.85).mismatch
    assert sample_ratio_mismatch(t, expected_treatment_share=0.5).mismatch


# ---------------------------------------------------------------- A/A


def test_wilson_interval_known_value() -> None:
    low, high = wilson_interval(5, 100)
    assert low == pytest.approx(0.02154, abs=1e-4)
    assert high == pytest.approx(0.11175, abs=1e-4)


@pytest.mark.slow
def test_aa_tests_are_calibrated_on_skewed_revenue() -> None:
    rng = np.random.default_rng(0)
    y = rng.lognormal(0.0, 1.0, size=20_000)
    result = run_aa_tests(y, n_splits=400, seed=1)
    assert result.calibrated
    assert result.uniformity_p_value > 0.001


def test_aa_tests_detect_a_miscalibrated_estimator() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=5_000)

    def overconfident(yy, tt):
        est = difference_in_means(yy, tt)
        return type(est)(**{**est.to_dict(), "p_value": min(1.0, est.p_value * 0.3)})

    result = run_aa_tests(y, n_splits=300, estimator=overconfident, seed=2)
    assert not result.calibrated
    assert result.false_positive_rate > 0.1


# ---------------------------------------------------------------- power


def test_sample_size_and_mde_are_inverse() -> None:
    n = required_sample_size_means(std=2.0, mde=0.1)
    assert minimum_detectable_effect(std=2.0, n_total=n) == pytest.approx(0.1, rel=1e-3)


def test_sample_size_scales_with_inverse_square_of_effect() -> None:
    small = required_sample_size_means(std=1.0, mde=0.05)
    large = required_sample_size_means(std=1.0, mde=0.10)
    assert small / large == pytest.approx(4.0, rel=1e-3)


def test_proportion_sample_size_matches_textbook_value() -> None:
    # 10% -> 12% at alpha 0.05, power 0.8, 50/50: about 3,840 per arm.
    n = required_sample_size_proportions(0.10, 0.02)
    assert 3_800 * 2 <= n <= 3_900 * 2


def test_variance_reduction_shrinks_mde_by_sqrt_factor() -> None:
    base = minimum_detectable_effect(1.0, 10_000)
    reduced = minimum_detectable_effect(1.0, 10_000, variance_reduction=0.75)
    assert reduced == pytest.approx(base * 0.5)


def test_unbalanced_allocation_needs_more_users() -> None:
    assert required_sample_size_means(1.0, 0.1, treatment_share=0.85) > (
        required_sample_size_means(1.0, 0.1)
    )


@pytest.mark.parametrize("kwargs", [{"alpha": 0.0}, {"power": 1.0}, {"treatment_share": 0.0}])
def test_power_inputs_are_validated(kwargs) -> None:
    with pytest.raises(ValueError):
        required_sample_size_means(1.0, 0.1, **kwargs)


# ---------------------------------------------------------------- CUPED / Lin


def _cuped_data(rng: np.random.Generator, n: int, effect: float):
    x = rng.normal(size=n)
    t = (rng.random(n) < 0.5).astype(int)
    y = 2.0 * x + effect * t + rng.normal(size=n)
    return y, t, x


def test_cuped_theta_and_variance_reduction() -> None:
    rng = np.random.default_rng(0)
    y, t, x = _cuped_data(rng, 50_000, effect=0.1)
    adj = cuped(y, t, x)
    assert adj.theta == pytest.approx(2.0, abs=0.05)
    # Var(y) = 4 + 1, residual variance 1 -> about 80% reduction.
    assert adj.variance_reduction == pytest.approx(0.8, abs=0.02)


def test_regression_adjustment_matches_cuped_with_one_covariate() -> None:
    rng = np.random.default_rng(1)
    y, t, x = _cuped_data(rng, 20_000, effect=0.1)
    lin = regression_adjusted(y, t, x)
    cup = cuped(y, t, x)
    assert lin.effect.estimate == pytest.approx(cup.effect.estimate, abs=2e-3)
    assert lin.variance_reduction == pytest.approx(cup.variance_reduction, abs=0.01)


def test_cuped_rejects_constant_covariate() -> None:
    with pytest.raises(ValueError, match="constant"):
        cuped([1.0, 2.0, 3.0, 4.0], [0, 0, 1, 1], [5.0, 5.0, 5.0, 5.0])


@pytest.mark.slow
def test_adjusted_estimators_are_unbiased_and_cover_at_nominal_rate() -> None:
    rng = np.random.default_rng(2)
    effect, sims = 0.05, 400
    hits = {"dim": 0, "cuped": 0, "lin": 0}
    estimates = {"dim": [], "cuped": [], "lin": []}
    for _ in range(sims):
        y, t, x = _cuped_data(rng, 4_000, effect)
        results = {
            "dim": difference_in_means(y, t),
            "cuped": cuped(y, t, x).effect,
            "lin": regression_adjusted(y, t, x).effect,
        }
        for k, r in results.items():
            estimates[k].append(r.estimate)
            hits[k] += r.ci_low <= effect <= r.ci_high
    for k in hits:
        assert np.mean(estimates[k]) == pytest.approx(effect, abs=0.01)
        low, high = wilson_interval(hits[k], sims, confidence=0.999)
        assert low <= 0.95 <= high
    assert np.std(estimates["cuped"]) < 0.6 * np.std(estimates["dim"])


# ---------------------------------------------------------------- sequential


def test_msprt_matches_closed_form() -> None:
    est, var, tau2 = 0.3, 0.01, 0.04
    expected = np.sqrt(var / (var + tau2)) * np.exp(tau2 * est**2 / (2 * var * (var + tau2)))
    assert float(msprt_log_lr(est, var, tau2)) == pytest.approx(np.log(expected))


def test_always_valid_p_values_never_increase() -> None:
    p = always_valid_p_values([-1.0, 0.5, 0.2, 3.0, 1.0])
    assert np.all(np.diff(p) <= 0)
    assert p[0] == 1.0
    assert p[-1] == pytest.approx(np.exp(-3.0))


def test_sequential_monitor_reports_every_look() -> None:
    rng = np.random.default_rng(0)
    t = (rng.random(10_000) < 0.5).astype(int)
    y = rng.normal(size=10_000) + 0.2 * t
    out = sequential_monitor(y, t, n_looks=10, tau2=0.01)
    assert len(out) == 10
    assert out["n_users"].iloc[-1] == 10_000
    assert out["always_valid_p_value"].is_monotonic_decreasing
    assert out["naive_significant"].iloc[-1]


@pytest.mark.slow
def test_peeking_inflates_false_positives_and_msprt_controls_them() -> None:
    result = simulate_peeking(n_looks=20, n_sims=3_000, seed=3)
    assert result.fixed_horizon_rate == pytest.approx(0.05, abs=0.015)
    assert result.naive_peeking_rate > 0.15
    assert result.msprt_rate <= 0.05


def test_msprt_detects_a_real_effect_and_can_stop_early() -> None:
    result = simulate_peeking(n_looks=20, n_sims=1_000, effect=0.08, seed=4)
    assert result.msprt_rate > 0.8
    assert result.msprt_mean_stop_fraction < 0.8
