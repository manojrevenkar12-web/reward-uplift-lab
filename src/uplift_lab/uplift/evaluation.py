"""Evaluation of uplift models: from observed data and, in simulation, from truth.

Individual effects are never observed, so uplift models cannot be scored per user.
Instead users are ranked by predicted uplift and the *observed* treatment-control
difference is measured in the top of the ranking. A good model concentrates the
incremental outcomes in the users it ranks first.

Curves are evaluated on a grid of population fractions ``phi`` (share of users
targeted, best-ranked first) and are expressed per user of the whole population:

* **uplift curve**: ``(mean_t(top) - mean_c(top)) * phi``; incremental outcome per
  user if only the top ``phi`` were treated. At ``phi = 1`` it equals the ATE.
* **Qini curve** (Radcliffe 2007): ``(Y_t(top) - Y_c(top) * N_t(top)/N_c(top)) / N``;
  incremental outcomes counted in the treated group, scaled by population size.

The area between a curve and the straight "random targeting" line summarises a model
(higher is better; zero means no better than random).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class Curve:
    """A targeting curve on a grid of population fractions."""

    fractions: FloatArray
    values: FloatArray

    def area_over_random(self) -> float:
        """Area between the curve and the straight line from (0, 0) to (1, values[-1])."""
        random_line = self.fractions * self.values[-1]
        return float(np.trapezoid(self.values - random_line, self.fractions))


def _prepare(
    y: ArrayLike, t: ArrayLike, score: ArrayLike
) -> tuple[FloatArray, FloatArray, NDArray[np.intp]]:
    y_arr = np.asarray(y, dtype=float)
    t_arr = np.asarray(t, dtype=float)
    s_arr = np.asarray(score, dtype=float)
    if not (y_arr.shape == t_arr.shape == s_arr.shape) or y_arr.ndim != 1:
        raise ValueError("y, t and score must be one-dimensional and of equal length")
    if not np.isin(t_arr, (0.0, 1.0)).all():
        raise ValueError("treatment must be binary (0/1)")
    if not np.isfinite(s_arr).all():
        raise ValueError("score contains NaN or infinite values")
    # Stable sort, descending, so ties keep a deterministic order.
    order = np.argsort(-s_arr, kind="stable")
    return y_arr[order], t_arr[order], order


def _grid_counts(n_points: int, n: int) -> tuple[FloatArray, NDArray[np.intp]]:
    fractions = np.linspace(0.0, 1.0, n_points)
    return fractions, np.ceil(fractions * n).astype(np.intp)


def _cumulative(
    y: FloatArray, t: FloatArray, k: NDArray[np.intp]
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    zero = np.zeros(1)
    yt = np.concatenate([zero, np.cumsum(y * t)])[k]
    nt = np.concatenate([zero, np.cumsum(t)])[k]
    yc = np.concatenate([zero, np.cumsum(y * (1 - t))])[k]
    nc = np.concatenate([zero, np.cumsum(1 - t)])[k]
    return yt, nt, yc, nc


def uplift_curve(y: ArrayLike, t: ArrayLike, score: ArrayLike, n_points: int = 101) -> Curve:
    """Observed uplift curve (see module docstring)."""
    y_s, t_s, _ = _prepare(y, t, score)
    fractions, k = _grid_counts(n_points, y_s.size)
    yt, nt, yc, nc = _cumulative(y_s, t_s, k)
    with np.errstate(divide="ignore", invalid="ignore"):
        diff = np.where((nt > 0) & (nc > 0), yt / nt - yc / nc, 0.0)
    return Curve(fractions, diff * fractions)


def qini_curve(y: ArrayLike, t: ArrayLike, score: ArrayLike, n_points: int = 101) -> Curve:
    """Observed Qini curve (see module docstring)."""
    y_s, t_s, _ = _prepare(y, t, score)
    fractions, k = _grid_counts(n_points, y_s.size)
    yt, nt, yc, nc = _cumulative(y_s, t_s, k)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(nc > 0, yt - yc * nt / nc, 0.0)
    return Curve(fractions, q / y_s.size)


def true_gain_curve(tau_true: ArrayLike, score: ArrayLike, n_points: int = 101) -> Curve:
    """Noise-free gain curve using known individual effects (simulation only).

    The value at ``phi`` is the total *true* effect of treating the top ``phi`` users
    by ``score``, divided by the population size.
    """
    tau = np.asarray(tau_true, dtype=float)
    s_arr = np.asarray(score, dtype=float)
    if tau.shape != s_arr.shape:
        raise ValueError("tau_true and score must have equal length")
    order = np.argsort(-s_arr, kind="stable")
    fractions, k = _grid_counts(n_points, tau.size)
    gain = np.concatenate([np.zeros(1), np.cumsum(tau[order])])[k] / tau.size
    return Curve(fractions, gain)


@dataclass(frozen=True)
class TruthMetrics:
    """How close an uplift model is to the true effects (simulation only).

    Attributes:
        pehe: Root mean squared error of the effect estimates (lower is better).
        spearman: Rank correlation between estimated and true effects.
        normalised_gain: Area of the model's true gain curve over random, divided by
            the oracle's area; 1 means perfect ranking, 0 means random.
        mean_bias: Mean estimated effect minus mean true effect.
    """

    pehe: float
    spearman: float
    normalised_gain: float
    mean_bias: float


def truth_metrics(tau_true: ArrayLike, tau_hat: ArrayLike) -> TruthMetrics:
    """Score effect estimates against the known individual effects."""
    tau = np.asarray(tau_true, dtype=float)
    est = np.asarray(tau_hat, dtype=float)
    if tau.shape != est.shape:
        raise ValueError("tau_true and tau_hat must have equal length")
    oracle_area = true_gain_curve(tau, tau).area_over_random()
    model_area = true_gain_curve(tau, est).area_over_random()
    return TruthMetrics(
        pehe=float(np.sqrt(np.mean((est - tau) ** 2))),
        spearman=float(stats.spearmanr(tau, est).statistic),
        normalised_gain=model_area / oracle_area if oracle_area > 0 else float("nan"),
        mean_bias=float(est.mean() - tau.mean()),
    )


def uplift_by_bin(
    y: ArrayLike, t: ArrayLike, score: ArrayLike, n_bins: int = 10
) -> list[dict[str, float]]:
    """Observed effect within equal-size bins of predicted uplift (bin 1 = highest).

    A well-calibrated ranking shows observed uplift falling from the first bin to the
    last. Each row reports the bin's mean predicted uplift, the observed difference in
    means and its standard error.
    """
    y_s, t_s, order = _prepare(y, t, score)
    s_sorted = np.asarray(score, dtype=float)[order]
    rows = []
    for i, idx in enumerate(np.array_split(np.arange(y_s.size), n_bins), start=1):
        yb, tb = y_s[idx], t_s[idx].astype(bool)
        n1, n0 = int(tb.sum()), int((~tb).sum())
        if n1 < 2 or n0 < 2:
            continue
        diff = yb[tb].mean() - yb[~tb].mean()
        se = np.sqrt(yb[tb].var(ddof=1) / n1 + yb[~tb].var(ddof=1) / n0)
        rows.append(
            {
                "bin": float(i),
                "n_users": float(idx.size),
                "predicted_uplift": float(s_sorted[idx].mean()),
                "observed_uplift": float(diff),
                "std_error": float(se),
            }
        )
    return rows
