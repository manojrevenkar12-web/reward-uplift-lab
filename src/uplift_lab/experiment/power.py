"""Sample-size and minimum-detectable-effect calculations for two-arm experiments.

These answer the planning question every experiment needs before launch: how many
users must we expose to have a fair chance of detecting the effect we care about?
All formulas are two-sided normal approximations with configurable allocation.
"""

from __future__ import annotations

import math

from scipy import stats


def _check(alpha: float, power: float, treatment_share: float) -> tuple[float, float]:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    if not 0.0 < treatment_share < 1.0:
        raise ValueError("treatment_share must be in (0, 1)")
    return float(stats.norm.ppf(1 - alpha / 2)), float(stats.norm.ppf(power))


def required_sample_size_means(
    std: float,
    mde: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    treatment_share: float = 0.5,
) -> int:
    """Total users needed to detect an absolute difference in means.

    ``N = (z_{1-alpha/2} + z_{power})^2 * sigma^2 * (1/p + 1/(1-p)) / mde^2``

    Args:
        std: Standard deviation of the outcome per user (assumed equal in both arms).
        mde: Smallest absolute effect worth detecting.
        alpha: Two-sided false-positive rate.
        power: Probability of detecting an effect of size ``mde``.
        treatment_share: Share of users allocated to treatment.
    """
    z_a, z_b = _check(alpha, power, treatment_share)
    if std <= 0 or mde <= 0:
        raise ValueError("std and mde must be positive")
    p = treatment_share
    n = (z_a + z_b) ** 2 * std**2 * (1 / p + 1 / (1 - p)) / mde**2
    return math.ceil(n)


def required_sample_size_proportions(
    baseline_rate: float,
    mde: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    treatment_share: float = 0.5,
) -> int:
    """Total users needed to detect an absolute change in a conversion rate.

    Uses the pooled variance under the null for the critical value and the unpooled
    variance under the alternative for the power term.

    Args:
        baseline_rate: Conversion rate in the control arm.
        mde: Smallest absolute change in rate worth detecting (e.g. 0.01 = 1 point).
        alpha: Two-sided false-positive rate.
        power: Probability of detecting a change of size ``mde``.
        treatment_share: Share of users allocated to treatment.
    """
    z_a, z_b = _check(alpha, power, treatment_share)
    p0, p1 = baseline_rate, baseline_rate + mde
    if not (0.0 < p0 < 1.0 and 0.0 < p1 < 1.0) or mde == 0:
        raise ValueError("baseline_rate and baseline_rate + mde must lie in (0, 1); mde != 0")
    s = treatment_share
    p_bar = s * p1 + (1 - s) * p0
    null_term = z_a * math.sqrt(p_bar * (1 - p_bar) * (1 / s + 1 / (1 - s)))
    alt_term = z_b * math.sqrt(p1 * (1 - p1) / s + p0 * (1 - p0) / (1 - s))
    return math.ceil((null_term + alt_term) ** 2 / mde**2)


def minimum_detectable_effect(
    std: float,
    n_total: int,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    treatment_share: float = 0.5,
    variance_reduction: float = 0.0,
) -> float:
    """Smallest absolute effect on a mean detectable with ``n_total`` users.

    Args:
        std: Standard deviation of the outcome per user.
        n_total: Total users in the experiment.
        alpha: Two-sided false-positive rate.
        power: Target probability of detection.
        treatment_share: Share of users allocated to treatment.
        variance_reduction: Fraction of estimator variance removed by an adjustment
            such as CUPED; the MDE shrinks by ``sqrt(1 - variance_reduction)``.
    """
    z_a, z_b = _check(alpha, power, treatment_share)
    if std <= 0 or n_total <= 0:
        raise ValueError("std and n_total must be positive")
    if not 0.0 <= variance_reduction < 1.0:
        raise ValueError("variance_reduction must be in [0, 1)")
    p = treatment_share
    se = std * math.sqrt(1 / (p * n_total) + 1 / ((1 - p) * n_total))
    return (z_a + z_b) * se * math.sqrt(1 - variance_reduction)
