"""Check the off-policy estimator against ground truth over many simulated experiments.

For each seed, the profit-optimal policy (known in simulation) is evaluated against
"reward everyone" on a held-out half of the users with the inverse-propensity
estimator. If the estimator is unbiased and its standard error honest, the standardised
errors ``(estimate - truth) / se`` average about 0 and about 95% of the intervals
cover the truth.

Usage:
    python scripts/policy_estimator_calibration.py --n-seeds 150
"""

from __future__ import annotations

import argparse

import numpy as np

from uplift_lab.data import SimulationConfig, simulate_experiment
from uplift_lab.experiment import wilson_interval
from uplift_lab.uplift import ipw_policy_gain


def main() -> None:
    """Run the calibration check and print a summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=150)
    parser.add_argument("--n-users", type=int, default=200_000)
    args = parser.parse_args()

    z = np.empty(args.n_seeds)
    for i in range(args.n_seeds):
        sim = simulate_experiment(SimulationConfig(n_users=args.n_users, seed=i))
        df, truth = sim.data, sim.truth
        test = np.random.default_rng([i, 1]).random(len(df)) < 0.5
        tau_profit = truth["tau_profit"].to_numpy()[test]
        policy = tau_profit > 0
        value = ipw_policy_gain(
            df["net_revenue_14d"].to_numpy()[test],
            df["treatment"].to_numpy()[test],
            policy,
            np.ones_like(policy),
            propensity=sim.config.treatment_share,
            true_effect=tau_profit,
        )
        assert value.true_gain is not None
        z[i] = (value.gain - value.true_gain) / value.std_error

    covered = int((np.abs(z) < 1.959964).sum())
    low, high = wilson_interval(covered, z.size)
    print(f"experiments: {z.size}")
    print(f"mean standardised error: {z.mean():+.3f} (SE {z.std(ddof=1) / np.sqrt(z.size):.3f})")
    print(f"sd of standardised error: {z.std(ddof=1):.3f} (1.0 if the SE is honest)")
    print(f"95% CI coverage: {covered / z.size:.1%} (95% Wilson interval {low:.1%} to {high:.1%})")


if __name__ == "__main__":
    main()
