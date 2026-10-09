"""Simulator for a randomised reward-offer experiment in a rewards app.

Real experiment data never reveals the individual treatment effect, so estimators can
only be judged indirectly. This simulator generates users together with their *true*
potential outcomes, which lets every estimator in this package be checked against
ground truth before it is trusted on real data.

Story behind the data-generating process
----------------------------------------
Users of a rewards app earn money by completing advertisers' offers. The experiment
randomly shows some users a bonus reward. A *conversion* means completing an advertiser
offer within 14 days; the advertiser pays the platform a payout that depends on the
user's country tier. If a treated user converts, the platform pays the bonus reward.

The effect of the bonus is heterogeneous by design:

* new and low-activity users respond strongly ("persuadables"),
* users acquired through paid social respond somewhat more,
* users who already complete many offers respond *negatively* (reward fatigue), and
* a positive uplift is not always profitable: in low-payout countries the reward
  can cost more than the incremental conversions earn.

Treatment does not affect engagement revenue, so the true effect on revenue comes
only through conversions. Pre-period engagement revenue is strongly correlated with
post-period revenue, which is what makes CUPED-style variance reduction work.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import expit

PLATFORMS = ("android", "ios", "web")
CHANNELS = ("organic", "paid_social", "referral")
COUNTRY_TIERS = (1, 2, 3)


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters of the simulated experiment.

    Attributes:
        n_users: Number of users randomised.
        treatment_share: Probability that a user is assigned to the reward offer.
        reward_cost: Bonus (EUR) paid when a treated user converts.
        payout_by_tier: Advertiser payout (EUR) per conversion, by country tier.
        treated_logging_loss: Share of *low-activity treated* users silently dropped from
            the logged data. Zero for a healthy experiment; a positive value simulates
            a tracking bug, which a sample-ratio-mismatch check should catch.
        seed: Random seed; the same seed always produces the same data.
    """

    n_users: int = 200_000
    treatment_share: float = 0.5
    reward_cost: float = 0.5
    payout_by_tier: dict[int, float] = field(default_factory=lambda: {1: 6.0, 2: 3.0, 3: 1.0})
    treated_logging_loss: float = 0.0
    seed: int = 7

    def __post_init__(self) -> None:
        if self.n_users < 100:
            raise ValueError("n_users must be at least 100")
        if not 0.0 < self.treatment_share < 1.0:
            raise ValueError("treatment_share must be strictly between 0 and 1")
        if self.reward_cost < 0:
            raise ValueError("reward_cost must be non-negative")
        if not 0.0 <= self.treated_logging_loss < 1.0:
            raise ValueError("treated_logging_loss must be in [0, 1)")
        if set(self.payout_by_tier) != set(COUNTRY_TIERS):
            raise ValueError(f"payout_by_tier must define tiers {COUNTRY_TIERS}")


@dataclass(frozen=True)
class SimulatedExperiment:
    """Observed experiment data plus the hidden ground truth.

    Attributes:
        data: What an analyst would see: user features, treatment, outcomes.
        truth: Potential-outcome probabilities and true effects, aligned with ``data``.
        config: The configuration that produced this experiment.
    """

    data: pd.DataFrame
    truth: pd.DataFrame
    config: SimulationConfig


def _baseline_logit(df: pd.DataFrame, engagement: np.ndarray) -> np.ndarray:
    return np.asarray(
        -2.2
        + 0.7 * engagement
        + 0.25 * (df["platform"] == "ios").to_numpy()
        - 0.20 * (df["channel"] == "paid_social").to_numpy()
        + 0.15 * np.log1p(df["offers_completed_90d"].to_numpy()),
        dtype=float,
    )


def _treatment_logit_shift(df: pd.DataFrame) -> np.ndarray:
    """True effect of the reward offer on the log-odds of converting."""
    return np.asarray(
        0.45 * (df["tenure_days"] < 30).to_numpy()
        + 0.30 * (df["sessions_7d"] <= 2).to_numpy()
        + 0.15 * (df["channel"] == "paid_social").to_numpy()
        - 0.60 * (df["offers_completed_90d"] >= 4).to_numpy(),
        dtype=float,
    )


def simulate_experiment(config: SimulationConfig | None = None) -> SimulatedExperiment:
    """Simulate one randomised reward-offer experiment.

    Args:
        config: Simulation parameters; defaults to :class:`SimulationConfig`.

    Returns:
        The observed data and the matching ground truth.
    """
    cfg = config or SimulationConfig()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_users

    engagement = rng.normal(size=n)
    tenure_days = np.clip(np.rint(rng.exponential(150.0, size=n)), 1, 1500).astype(int)
    df = pd.DataFrame(
        {
            "user_id": np.arange(n),
            "platform": rng.choice(PLATFORMS, size=n, p=[0.55, 0.35, 0.10]),
            "country_tier": rng.choice(COUNTRY_TIERS, size=n, p=[0.30, 0.40, 0.30]),
            "channel": rng.choice(CHANNELS, size=n, p=[0.50, 0.35, 0.15]),
            "tenure_days": tenure_days,
            "sessions_7d": rng.poisson(np.exp(1.0 + 0.6 * engagement)),
            "offers_completed_90d": rng.poisson(
                np.exp(-0.3 + 0.7 * engagement + 0.2 * (tenure_days > 90))
            ),
        }
    )

    # Engagement (ad) revenue has a persistent per-user level, observed twice:
    # once before the experiment (the CUPED covariate) and once during it.
    revenue_level = np.exp(-0.3 + 0.8 * engagement + 0.3 * (df["country_tier"] == 1).to_numpy())
    df["pre_revenue_14d"] = revenue_level * rng.gamma(6.0, 1 / 6, size=n)
    engagement_revenue = revenue_level * rng.gamma(6.0, 1 / 6, size=n)

    base = _baseline_logit(df, engagement)
    mu0 = expit(base)
    mu1 = expit(base + _treatment_logit_shift(df))

    treatment = (rng.random(n) < cfg.treatment_share).astype(int)
    p_convert = np.where(treatment == 1, mu1, mu0)
    converted = (rng.random(n) < p_convert).astype(int)

    payout = df["country_tier"].map(cfg.payout_by_tier).to_numpy(dtype=float)
    df["treatment"] = treatment
    df["converted"] = converted
    df["revenue_14d"] = payout * converted + engagement_revenue
    df["reward_paid"] = cfg.reward_cost * converted * treatment
    df["net_revenue_14d"] = df["revenue_14d"] - df["reward_paid"]

    tau = mu1 - mu0
    truth = pd.DataFrame(
        {
            "mu0": mu0,
            "mu1": mu1,
            "tau_conversion": tau,
            "tau_revenue": payout * tau,
            # E[net | T=1] - E[net | T=0] = payout*mu1 - cost*mu1 - payout*mu0
            "tau_profit": payout * tau - cfg.reward_cost * mu1,
            "payout": payout,
        },
        index=df.index,
    )

    if cfg.treated_logging_loss > 0:
        low_activity_treated = (treatment == 1) & (df["sessions_7d"].to_numpy() <= 1)
        dropped = low_activity_treated & (rng.random(n) < cfg.treated_logging_loss)
        df, truth = df.loc[~dropped], truth.loc[~dropped]

    return SimulatedExperiment(
        data=df.reset_index(drop=True), truth=truth.reset_index(drop=True), config=cfg
    )
