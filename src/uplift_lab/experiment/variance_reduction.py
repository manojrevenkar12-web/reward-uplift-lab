"""Variance reduction with pre-experiment covariates: CUPED and regression adjustment.

Randomisation makes the treatment independent of anything measured before the
experiment. Subtracting the part of the outcome that pre-period data already
predicts therefore removes noise without introducing bias, which narrows confidence
intervals. Narrower intervals mean the same decision with fewer users or less time.

References:
    Deng, Xu, Kohavi, Walker (2013). Improving the Sensitivity of Online Controlled
    Experiments by Utilizing Pre-Experiment Data. WSDM.
    Lin (2013). Agnostic notes on regression adjustments to experimental data.
    Annals of Applied Statistics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from uplift_lab.experiment.estimators import (
    EffectEstimate,
    difference_in_means,
    normal_estimate,
    validate_arrays,
)


@dataclass(frozen=True)
class AdjustedEstimate:
    """An adjusted effect estimate and how much variance the adjustment removed.

    Attributes:
        effect: The adjusted treatment-effect estimate.
        variance_reduction: ``1 - Var(adjusted) / Var(unadjusted)`` of the estimator,
            e.g. 0.6 means the adjusted estimate needs 60% fewer users for the same
            precision.
        theta: CUPED coefficient (``None`` for regression adjustment).
    """

    effect: EffectEstimate
    variance_reduction: float
    theta: float | None = None


def cuped(
    y: ArrayLike, t: ArrayLike, covariate: ArrayLike, alpha: float = 0.05
) -> AdjustedEstimate:
    """CUPED: adjust the outcome with one pre-experiment covariate.

    The adjusted outcome is ``y - theta * (x - mean(x))`` with
    ``theta = Cov(y, x) / Var(x)`` estimated on both arms pooled. Because ``x`` is
    measured before assignment, its mean is the same in both arms in expectation, so
    the adjustment leaves the effect estimate unbiased.

    Args:
        y: Outcome per user.
        t: Treatment indicator per user (1 = treated).
        covariate: Pre-experiment value of a metric correlated with the outcome,
            typically the same metric measured before the experiment started.
        alpha: One minus the confidence level.
    """
    y_arr, treated = validate_arrays(y, t)
    x = np.asarray(covariate, dtype=float)
    if x.shape != y_arr.shape:
        raise ValueError("covariate must have the same length as y")
    if not np.isfinite(x).all():
        raise ValueError("covariate contains NaN or infinite values")
    var_x = x.var(ddof=1)
    if var_x == 0:
        raise ValueError("covariate is constant; CUPED cannot reduce variance")

    theta = float(np.cov(y_arr, x, ddof=1)[0, 1] / var_x)
    y_adj = y_arr - theta * (x - x.mean())

    raw = difference_in_means(y_arr, treated.astype(int), alpha)
    adj = difference_in_means(y_adj, treated.astype(int), alpha)
    effect = normal_estimate(
        "cuped", adj.estimate, adj.std_error, adj.n_treatment, adj.n_control, alpha
    )
    return AdjustedEstimate(
        effect=effect,
        variance_reduction=1 - (adj.std_error / raw.std_error) ** 2,
        theta=theta,
    )


def regression_adjusted(
    y: ArrayLike, t: ArrayLike, covariates: ArrayLike, alpha: float = 0.05
) -> AdjustedEstimate:
    """Lin (2013) regression adjustment with several covariates.

    Fits ``y ~ 1 + t + Xc + t * Xc`` by least squares, where ``Xc`` is the centred
    covariate matrix. With centred covariates and full treatment interactions, the
    coefficient on ``t`` is a consistent estimate of the average treatment effect even
    when the linear model is wrong, and it is never less precise asymptotically than
    the plain difference in means. Standard errors are heteroskedasticity-robust (HC1).

    Args:
        y: Outcome per user.
        t: Treatment indicator per user (1 = treated).
        covariates: Pre-experiment covariates, shape ``(n_users, n_covariates)``.
        alpha: One minus the confidence level.
    """
    y_arr, treated = validate_arrays(y, t)
    X = np.asarray(covariates, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.shape[0] != y_arr.size:
        raise ValueError("covariates must have one row per user")
    if not np.isfinite(X).all():
        raise ValueError("covariates contain NaN or infinite values")

    t_col = treated.astype(float)[:, None]
    Xc = X - X.mean(axis=0)
    design = np.hstack([np.ones_like(t_col), t_col, Xc, t_col * Xc])
    n, k = design.shape

    xtx = design.T @ design
    beta = np.linalg.solve(xtx, design.T @ y_arr)
    resid = y_arr - design @ beta
    bread = np.linalg.inv(xtx)
    scaled = design * resid[:, None]
    cov = bread @ (scaled.T @ scaled) @ bread * n / (n - k)
    se = float(np.sqrt(cov[1, 1]))

    raw = difference_in_means(y_arr, treated.astype(int), alpha)
    effect = normal_estimate(
        "regression_adjusted", beta[1], se, raw.n_treatment, raw.n_control, alpha
    )
    return AdjustedEstimate(effect=effect, variance_reduction=1 - (se / raw.std_error) ** 2)
