"""Profit-aware targeting and unbiased off-policy evaluation.

Maximising uplift is not the business goal. A reward costs money every time a treated
user converts, *including* users who would have converted anyway. The expected
incremental profit of offering the reward to user ``x`` is

    profit(x) = payout(x) * tau(x) - reward_cost * mu1(x)

where ``tau(x)`` is the uplift in conversion probability and ``mu1(x)`` the conversion
probability with the reward. A user with positive uplift can still be unprofitable.

Any targeting policy can be evaluated on held-out *randomised* data without deploying
it: users whose random assignment happens to agree with the policy are reweighted by
the inverse assignment probability (Horvitz-Thompson). Because the assignment
probabilities are known by design, the estimate is unbiased.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]


def expected_incremental_profit(
    tau_hat: ArrayLike, mu1_hat: ArrayLike, payout: ArrayLike, reward_cost: float
) -> FloatArray:
    """Expected profit change from offering the reward, per user."""
    if reward_cost < 0:
        raise ValueError("reward_cost must be non-negative")
    tau = np.asarray(tau_hat, dtype=float)
    mu1 = np.clip(np.asarray(mu1_hat, dtype=float), 0.0, 1.0)
    return np.asarray(np.asarray(payout, dtype=float) * tau - reward_cost * mu1, dtype=float)


def top_fraction_policy(score: ArrayLike, fraction: float) -> BoolArray:
    """Treat the ``fraction`` of users with the highest score."""
    s = np.asarray(score, dtype=float)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    k = round(fraction * s.size)
    policy = np.zeros(s.size, dtype=bool)
    policy[np.argsort(-s, kind="stable")[:k]] = True
    return policy


def budgeted_profit_policy(
    profit: ArrayLike, expected_cost: ArrayLike, budget: float | None = None
) -> BoolArray:
    """Treat users with positive expected profit, best return on spend first.

    Without a budget, every user with positive expected profit is treated. With a
    budget, users are added in order of profit per euro of expected reward spend
    until the next user would exceed the budget (the greedy solution to the
    fractional knapsack, which is near-optimal when individual costs are small).

    Args:
        profit: Expected incremental profit per user.
        expected_cost: Expected reward spend per user if treated (``cost * mu1``).
        budget: Maximum total expected reward spend; ``None`` for no limit.
    """
    p = np.asarray(profit, dtype=float)
    c = np.asarray(expected_cost, dtype=float)
    if p.shape != c.shape:
        raise ValueError("profit and expected_cost must have equal length")
    if (c < 0).any():
        raise ValueError("expected_cost must be non-negative")
    candidates = p > 0
    if budget is None:
        return candidates
    if budget < 0:
        raise ValueError("budget must be non-negative")

    idx = np.flatnonzero(candidates)
    roi = p[idx] / np.maximum(c[idx], 1e-12)
    ranked = idx[np.argsort(-roi, kind="stable")]
    within_budget = np.cumsum(c[ranked]) <= budget
    policy = np.zeros(p.size, dtype=bool)
    policy[ranked[within_budget]] = True
    return policy


@dataclass(frozen=True)
class PolicyValue:
    """Estimated value of a policy relative to a baseline policy, per user.

    Attributes:
        name: Policy name.
        treated_share: Share of users the policy treats.
        gain: Estimated mean outcome under the policy minus under the baseline.
        std_error: Standard error of ``gain``.
        ci_low: Lower 95% confidence bound.
        ci_high: Upper 95% confidence bound.
        true_gain: Gain computed from known effects (simulation only, else ``None``).
    """

    name: str
    treated_share: float
    gain: float
    std_error: float
    ci_low: float
    ci_high: float
    true_gain: float | None = None

    def to_dict(self) -> dict[str, float | str | None]:
        """Return the value as a plain dictionary."""
        return asdict(self)


def ipw_policy_gain(
    outcome: ArrayLike,
    t: ArrayLike,
    policy: ArrayLike,
    baseline: ArrayLike | None = None,
    *,
    propensity: float,
    name: str = "policy",
    true_effect: ArrayLike | None = None,
    alpha: float = 0.05,
) -> PolicyValue:
    """Estimate a policy's gain over a baseline from randomised data.

    Each user contributes ``outcome * (w_policy - w_baseline)``, where
    ``w = 1{T = pi(x)} / P(T = pi(x))``. Computing the *difference* user by user,
    rather than two separate values, cancels shared noise and gives a much tighter
    interval.

    Args:
        outcome: Realised outcome per user (e.g. net revenue).
        t: Random treatment assignment per user.
        policy: Whether the policy would treat each user.
        baseline: Baseline policy; defaults to treating nobody.
        propensity: Known probability of assignment to treatment.
        name: Label for reporting.
        true_effect: Known per-user effect on ``outcome`` (simulation only); if given,
            the true gain ``mean((pi - baseline) * effect)`` is reported alongside.
        alpha: One minus the confidence level.
    """
    y = np.asarray(outcome, dtype=float)
    t_arr = np.asarray(t).astype(bool)
    pi = np.asarray(policy).astype(bool)
    base = np.zeros_like(pi) if baseline is None else np.asarray(baseline).astype(bool)
    if not (y.shape == t_arr.shape == pi.shape == base.shape):
        raise ValueError("outcome, t, policy and baseline must have equal length")
    if not 0.0 < propensity < 1.0:
        raise ValueError("propensity must be in (0, 1)")

    def weights(p: BoolArray) -> FloatArray:
        return np.where(p, t_arr / propensity, ~t_arr / (1 - propensity)).astype(float)

    contributions = y * (weights(pi) - weights(base))
    gain = float(contributions.mean())
    se = float(contributions.std(ddof=1) / np.sqrt(y.size))
    z = float(stats.norm.ppf(1 - alpha / 2))

    true_gain = None
    if true_effect is not None:
        effect = np.asarray(true_effect, dtype=float)
        true_gain = float(np.mean((pi.astype(float) - base.astype(float)) * effect))

    return PolicyValue(
        name=name,
        treated_share=float(pi.mean()),
        gain=gain,
        std_error=se,
        ci_low=gain - z * se,
        ci_high=gain + z * se,
        true_gain=true_gain,
    )
