"""Average treatment effect estimators for randomised experiments.

All estimators use large-sample normal approximations, which are appropriate for
the user counts typical of product experiments (thousands per arm and up). Each
returns an :class:`EffectEstimate` so results can be compared like for like.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class EffectEstimate:
    """A treatment-effect estimate with its uncertainty.

    Attributes:
        method: Name of the estimator that produced the estimate.
        estimate: Point estimate of the effect (treatment minus control).
        std_error: Standard error of the estimate.
        ci_low: Lower bound of the two-sided confidence interval.
        ci_high: Upper bound of the two-sided confidence interval.
        p_value: Two-sided p-value for the null hypothesis of zero effect.
        n_treatment: Users in the treatment arm.
        n_control: Users in the control arm.
    """

    method: str
    estimate: float
    std_error: float
    ci_low: float
    ci_high: float
    p_value: float
    n_treatment: int
    n_control: int

    def to_dict(self) -> dict[str, float | int | str]:
        """Return the estimate as a plain dictionary (for JSON reports)."""
        return asdict(self)


def validate_arrays(y: ArrayLike, t: ArrayLike) -> tuple[FloatArray, NDArray[np.bool_]]:
    """Check outcome and treatment arrays and return them as float and boolean arrays.

    Raises:
        ValueError: On shape mismatch, non-binary treatment, non-finite outcomes, or an
            arm with fewer than two users.
    """
    y_arr = np.asarray(y, dtype=float)
    t_arr = np.asarray(t)
    if y_arr.ndim != 1 or t_arr.shape != y_arr.shape:
        raise ValueError("y and t must be one-dimensional arrays of equal length")
    if not np.isin(t_arr, (0, 1)).all():
        raise ValueError("treatment must be binary (0/1)")
    if not np.isfinite(y_arr).all():
        raise ValueError("outcome contains NaN or infinite values")
    treated = t_arr.astype(bool)
    if treated.sum() < 2 or (~treated).sum() < 2:
        raise ValueError("each arm needs at least two users")
    return y_arr, treated


def normal_estimate(
    method: str,
    estimate: float,
    std_error: float,
    n_treatment: int,
    n_control: int,
    alpha: float = 0.05,
) -> EffectEstimate:
    """Package a point estimate and standard error into a normal-theory estimate."""
    z = stats.norm.ppf(1 - alpha / 2)
    p_value = 2 * stats.norm.sf(abs(estimate) / std_error) if std_error > 0 else float("nan")
    return EffectEstimate(
        method=method,
        estimate=float(estimate),
        std_error=float(std_error),
        ci_low=float(estimate - z * std_error),
        ci_high=float(estimate + z * std_error),
        p_value=float(p_value),
        n_treatment=n_treatment,
        n_control=n_control,
    )


def difference_in_means(y: ArrayLike, t: ArrayLike, alpha: float = 0.05) -> EffectEstimate:
    """Estimate the average treatment effect as a difference in arm means.

    Uses the unpooled (Welch) standard error, which stays valid when the arms have
    different variances, as they do when the treatment changes the outcome spread.

    Args:
        y: Outcome per user.
        t: Treatment indicator per user (1 = treated).
        alpha: One minus the confidence level.
    """
    y_arr, treated = validate_arrays(y, t)
    y1, y0 = y_arr[treated], y_arr[~treated]
    se = np.sqrt(y1.var(ddof=1) / y1.size + y0.var(ddof=1) / y0.size)
    return normal_estimate(
        "difference_in_means", y1.mean() - y0.mean(), se, y1.size, y0.size, alpha
    )


def relative_lift(y: ArrayLike, t: ArrayLike, alpha: float = 0.05) -> EffectEstimate:
    """Estimate the relative lift ``mean_t / mean_c - 1`` with a delta-method interval.

    The arms are independent, so the variance of the ratio of means is approximated by
    ``Var(m_t)/m_c^2 + m_t^2 Var(m_c)/m_c^4``.

    Args:
        y: Outcome per user.
        t: Treatment indicator per user (1 = treated).
        alpha: One minus the confidence level.

    Raises:
        ValueError: If the control mean is zero, where the lift is undefined.
    """
    y_arr, treated = validate_arrays(y, t)
    y1, y0 = y_arr[treated], y_arr[~treated]
    m1, m0 = y1.mean(), y0.mean()
    if m0 == 0:
        raise ValueError("relative lift is undefined when the control mean is zero")
    v1, v0 = y1.var(ddof=1) / y1.size, y0.var(ddof=1) / y0.size
    se = np.sqrt(v1 / m0**2 + m1**2 * v0 / m0**4)
    return normal_estimate("relative_lift", m1 / m0 - 1, se, y1.size, y0.size, alpha)
