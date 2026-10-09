"""Peeking and always-valid inference.

Checking a fixed-horizon test every day and stopping at the first p < 0.05 inflates
the false-positive rate far above 5%: each look is another chance for noise to cross
the line. This module quantifies that inflation by simulation and implements the
mixture sequential probability ratio test (mSPRT), whose always-valid p-values can be
monitored continuously while keeping the false-positive rate at or below alpha.

The mSPRT uses a normal mixing distribution N(0, tau^2) over the true effect. With an
effect estimate ``theta`` whose variance is ``s2``, the mixture likelihood ratio is

    Lambda = sqrt(s2 / (s2 + tau2)) * exp(tau2 * theta^2 / (2 * s2 * (s2 + tau2)))

and the always-valid p-value after look k is ``min(1, 1 / max_{j<=k} Lambda_j)``.

Reference:
    Johari, Koomen, Pekelis, Walsh (2017). Peeking at A/B Tests: Why it matters, and
    what to do about it. KDD.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from uplift_lab.experiment.aa import wilson_interval
from uplift_lab.experiment.estimators import validate_arrays


def msprt_log_lr(
    estimate: NDArray[np.float64] | float,
    variance: NDArray[np.float64] | float,
    tau2: float,
) -> NDArray[np.float64]:
    """Log of the mSPRT mixture likelihood ratio against a zero effect.

    Args:
        estimate: Effect estimate(s).
        variance: Variance of the estimate(s) (squared standard error).
        tau2: Variance of the normal mixing distribution; set it near the squared
            effect size you expect, in the units of ``estimate``.
    """
    if tau2 <= 0:
        raise ValueError("tau2 must be positive")
    est = np.asarray(estimate, dtype=float)
    var = np.asarray(variance, dtype=float)
    return np.asarray(
        0.5 * np.log(var / (var + tau2)) + tau2 * est**2 / (2 * var * (var + tau2)),
        dtype=float,
    )


def always_valid_p_values(log_lr: ArrayLike) -> NDArray[np.float64]:
    """Convert a sequence of log likelihood ratios into always-valid p-values."""
    running_max = np.maximum.accumulate(np.asarray(log_lr, dtype=float))
    return np.asarray(np.minimum(1.0, np.exp(-running_max)), dtype=float)


def sequential_monitor(
    y: ArrayLike,
    t: ArrayLike,
    *,
    n_looks: int = 20,
    tau2: float,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Replay an experiment in arrival order and report both p-values at each look.

    Args:
        y: Outcome per user, in the order users entered the experiment.
        t: Treatment indicator per user, in the same order.
        n_looks: Number of equally spaced interim analyses.
        tau2: mSPRT mixing variance (see :func:`msprt_log_lr`).
        alpha: Significance level used for the ``*_significant`` flags.

    Returns:
        One row per look with sample sizes, the effect estimate, its standard error,
        the naive fixed-horizon p-value and the always-valid p-value.
    """
    y_arr, treated = validate_arrays(y, t)
    if n_looks < 1:
        raise ValueError("n_looks must be at least 1")
    ends = np.unique(np.linspace(0, y_arr.size, n_looks + 1, dtype=int)[1:])

    rows = []
    for end in ends:
        yt, yc = y_arr[:end][treated[:end]], y_arr[:end][~treated[:end]]
        if yt.size < 2 or yc.size < 2:
            continue
        estimate = yt.mean() - yc.mean()
        variance = yt.var(ddof=1) / yt.size + yc.var(ddof=1) / yc.size
        rows.append(
            {
                "n_users": int(end),
                "estimate": float(estimate),
                "std_error": float(np.sqrt(variance)),
                "naive_p_value": float(2 * stats.norm.sf(abs(estimate) / np.sqrt(variance))),
                "log_lr": float(msprt_log_lr(estimate, variance, tau2)),
            }
        )
    out = pd.DataFrame(rows)
    out["always_valid_p_value"] = always_valid_p_values(out["log_lr"].to_numpy())
    out["naive_significant"] = out["naive_p_value"] < alpha
    out["always_valid_significant"] = out["always_valid_p_value"] < alpha
    return out.drop(columns="log_lr")


@dataclass(frozen=True)
class PeekingResult:
    """Rejection rates of three decision rules on the same simulated experiments.

    Attributes:
        n_looks: Number of interim analyses per experiment.
        effect: True effect used in the simulation (0 means every rejection is false).
        fixed_horizon_rate: Rejection rate when testing once, at the end.
        naive_peeking_rate: Rejection rate when stopping at the first naive p < alpha.
        msprt_rate: Rejection rate when stopping at the first always-valid p < alpha.
        rate_cis: 95% Wilson intervals for the three rates, keyed by rule name.
        msprt_mean_stop_fraction: Among mSPRT rejections, the average share of the
            planned sample used before stopping (``nan`` if there were none).
    """

    n_looks: int
    effect: float
    fixed_horizon_rate: float
    naive_peeking_rate: float
    msprt_rate: float
    rate_cis: dict[str, tuple[float, float]]
    msprt_mean_stop_fraction: float


def simulate_peeking(
    *,
    n_per_arm: int = 10_000,
    n_looks: int = 10,
    n_sims: int = 2_000,
    effect: float = 0.0,
    sigma: float = 1.0,
    tau2: float | None = None,
    alpha: float = 0.05,
    seed: int = 0,
) -> PeekingResult:
    """Simulate experiments analysed with fixed-horizon, naive-peeking and mSPRT rules.

    Each simulated experiment collects ``n_per_arm`` users per arm in ``n_looks`` equal
    batches. Batch sums are drawn directly from their exact normal distribution, so
    thousands of experiments run in milliseconds.

    Args:
        n_per_arm: Planned users per arm.
        n_looks: Number of interim analyses (the last one is the planned end).
        n_sims: Number of simulated experiments.
        effect: True difference in means.
        sigma: Outcome standard deviation in each arm.
        tau2: mSPRT mixing variance; defaults to ``(0.05 * sigma) ** 2``.
        alpha: Significance level.
        seed: Random seed.
    """
    if n_looks < 1 or n_per_arm < n_looks:
        raise ValueError("need n_looks >= 1 and n_per_arm >= n_looks")
    rng = np.random.default_rng(seed)
    batch = n_per_arm // n_looks
    tau2 = (0.05 * sigma) ** 2 if tau2 is None else tau2

    batch_sd = sigma * np.sqrt(batch)
    sum_t = np.cumsum(rng.normal(effect * batch, batch_sd, size=(n_sims, n_looks)), axis=1)
    sum_c = np.cumsum(rng.normal(0.0, batch_sd, size=(n_sims, n_looks)), axis=1)
    n_seen = batch * np.arange(1, n_looks + 1)
    estimate = (sum_t - sum_c) / n_seen
    variance = 2 * sigma**2 / n_seen  # known-variance analysis
    z = np.abs(estimate) / np.sqrt(variance)

    z_crit = stats.norm.ppf(1 - alpha / 2)
    fixed = z[:, -1] > z_crit
    naive = (z > z_crit).any(axis=1)
    msprt_hits = msprt_log_lr(estimate, variance, tau2) >= np.log(1 / alpha)
    msprt = msprt_hits.any(axis=1)

    stop_fraction = float("nan")
    if msprt.any():
        first_hit = msprt_hits[msprt].argmax(axis=1)
        stop_fraction = float(np.mean((first_hit + 1) / n_looks))

    rates = {"fixed_horizon": fixed, "naive_peeking": naive, "msprt": msprt}
    return PeekingResult(
        n_looks=n_looks,
        effect=effect,
        fixed_horizon_rate=float(fixed.mean()),
        naive_peeking_rate=float(naive.mean()),
        msprt_rate=float(msprt.mean()),
        rate_cis={k: wilson_interval(int(v.sum()), n_sims) for k, v in rates.items()},
        msprt_mean_stop_fraction=stop_fraction,
    )
