"""Re-run the simulated study on several seeds and tabulate the headline results.

One seed is one draw of the data. A finding is worth stating only if it holds across
draws, so this script reruns the study and prints a Markdown table for the README.

Usage:
    python scripts/seed_robustness.py --seeds 1 2 3 4 5
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

from uplift_lab.pipeline import StudyConfig, run_simulation_study


def _row(seed: int, r: dict[str, Any]) -> str:
    policies = {p["name"]: p for p in r["policy"]["values_per_user"]}
    vs = policies["Positive expected profit"]["vs_everyone"]
    net = r["ate"]["net_revenue_cuped"]
    models = r["uplift"]["models"]
    selected = r["uplift"]["selected_model"]
    best_by_truth = max(models, key=lambda m: models[m]["normalised_gain"])
    return (
        f"| {seed} "
        f"| {100 * r['ate']['revenue']['cuped']['variance_reduction']:.0f}% "
        f"| €{1000 * net['estimate']:+.0f} [{1000 * net['ci_low']:+.0f}, "
        f"{1000 * net['ci_high']:+.0f}] (true €{1000 * net['truth']:+.0f}) "
        f"| €{1000 * vs['gain']:.0f} [{1000 * vs['ci_low']:.0f}, {1000 * vs['ci_high']:.0f}] "
        f"(true €{1000 * vs['true_gain']:.0f}) "
        f"| {selected} ({100 * models[selected]['normalised_gain']:.0f}%) "
        f"| {best_by_truth} ({100 * models[best_by_truth]['normalised_gain']:.0f}%) "
        f"| {'yes' if r['srm']['broken']['mismatch'] else 'no'} |"
    )


def main() -> None:
    """Run the study per seed and print the table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--n-users", type=int, default=StudyConfig.n_users)
    args = parser.parse_args()

    print(
        "| Seed | CUPED variance cut | Blanket reward: net profit per 1,000 users "
        "| Profit targeting vs. blanket, per 1,000 users | Selected model (share of oracle "
        "gain) | Best model by truth | Logging bug caught |"
    )
    print("|---|---|---|---|---|---|---|")
    with tempfile.TemporaryDirectory() as tmp:
        for seed in args.seeds:
            # A/A and peeking simulations do not feed this table; keep them small.
            config = StudyConfig(n_users=args.n_users, seed=seed, aa_splits=20, peeking_sims=100)
            results = run_simulation_study(config, Path(tmp) / str(seed))
            print(_row(seed, results), flush=True)


if __name__ == "__main__":
    main()
