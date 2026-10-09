"""A/A tests: check that the analysis pipeline is calibrated before trusting it.

An A/A test splits users who all received the *same* experience into two random
groups and runs the normal analysis. With no real difference, p-values should be
uniformly distributed and about 5% of tests should be "significant" at alpha = 0.05.
A higher rate means the variance estimate is wrong (e.g. users counted twice, or a
skewed metric breaking the normal approximation), and real results cannot be trusted.

This is the experimentation version of measuring an evaluation's noise floor before
setting alert thresholds.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from uplift_lab.experiment.estimators import EffectEstimate, difference_in_means

Estimator: TypeAlias = Callable[[NDArray[np.float64], NDArray[np.int_]], EffectEstimate]


@dataclass(frozen=True)
class AAResult:
    """Summary of repeated A/A splits.

    Attributes:
        p_values: P-value of each simulated split.
        alpha: Significance level used to count false positives.
        false_positive_rate: Share of splits with ``p < alpha``; should be close to alpha.
        false_positive_rate_ci: 95% Wilson interval for the false-positive rate.
        uniformity_p_value: Kolmogorov-Smirnov test of the p-values against Uniform(0, 1);
            a small value means the p-values are miscalibrated.
    """

    p_values: NDArray[np.float64]
    alpha: float
    false_positive_rate: float
    false_positive_rate_ci: tuple[float, float]
    uniformity_p_value: float

    @property
    def calibrated(self) -> bool:
        """True if the nominal alpha lies inside the false-positive-rate interval."""
        low, high = self.false_positive_rate_ci
        return low <= self.alpha <= high


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    z = float(stats.norm.ppf(1 - (1 - confidence) / 2))
    phat = successes / trials
    denom = 1 + z**2 / trials
    centre = (phat + z**2 / (2 * trials)) / denom
    half = z * np.sqrt(phat * (1 - phat) / trials + z**2 / (4 * trials**2)) / denom
    return float(centre - half), float(centre + half)


def run_aa_tests(
    y: ArrayLike,
    *,
    n_splits: int = 500,
    alpha: float = 0.05,
    treatment_share: float = 0.5,
    estimator: Estimator = difference_in_means,
    seed: int = 0,
) -> AAResult:
    """Run repeated random A/A splits on outcomes from a single, untreated group.

    Args:
        y: Outcomes of users who all had the same experience (e.g. the control arm).
        n_splits: Number of random splits to simulate.
        alpha: Significance level.
        treatment_share: Share of users placed in the fake "treatment" group.
        estimator: Analysis to validate; any callable ``(y, t) -> EffectEstimate``.
        seed: Random seed.
    """
    y_arr = np.asarray(y, dtype=float)
    if n_splits < 10:
        raise ValueError("n_splits must be at least 10")
    rng = np.random.default_rng(seed)
    p_values = np.empty(n_splits)
    for i in range(n_splits):
        t = (rng.random(y_arr.size) < treatment_share).astype(int)
        p_values[i] = estimator(y_arr, t).p_value

    false_positives = int((p_values < alpha).sum())
    return AAResult(
        p_values=p_values,
        alpha=alpha,
        false_positive_rate=false_positives / n_splits,
        false_positive_rate_ci=wilson_interval(false_positives, n_splits),
        uniformity_p_value=float(stats.kstest(p_values, "uniform").pvalue),
    )
