"""Render study results as a Markdown report.

Every number in the report is read from the results dictionary, so the report and
the code can never disagree. Sentences that state a conclusion are chosen by the
results (e.g. whether an interval excludes zero), not written in advance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def _pts(x: float, digits: int = 2) -> str:
    return f"{100 * x:+.{digits}f} pts"


def _eur(x: float, digits: int = 3) -> str:
    return f"€{x:+.{digits}f}"


def _ci(e: Mapping[str, Any], fmt: Any = _eur) -> str:
    return f"{fmt(e['estimate'])} [{fmt(e['ci_low'])}, {fmt(e['ci_high'])}]"


def _users(n: int | None) -> str:
    return "no look" if n is None else f"{n:,} users"


def _p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def _per_1000(v: Mapping[str, Any]) -> str:
    return f"€{1000 * v['gain']:.0f} [{1000 * v['ci_low']:.0f}, {1000 * v['ci_high']:.0f}]"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _significant(e: Mapping[str, Any]) -> bool:
    return bool(e["ci_low"] > 0 or e["ci_high"] < 0)


def render_simulation_report(r: Mapping[str, Any]) -> str:
    """Build the Markdown report for :func:`uplift_lab.pipeline.run_simulation_study`."""
    cfg, sim = r["config"], r["simulation"]
    ate, srm, aa, power, seq = r["ate"], r["srm"], r["aa"], r["power"], r["sequential"]
    up, pol = r["uplift"], r["policy"]
    rev = ate["revenue"]
    policies = {p["name"]: p for p in pol["values_per_user"]}
    profit_name = "Positive expected profit"
    profit = policies[profit_name]
    null_rows = {row["n_looks"]: row for row in seq["null"]}
    max_looks = max(null_rows)
    alt = seq["alternative_effect_0p04_sd"]
    net = ate["net_revenue_cuped"]

    out: list[str] = []
    add = out.append
    add("# Reward Uplift Lab: study report\n")
    add(
        f"Regenerate with `uplift-lab simulate --n-users {cfg['n_users']} --seed {cfg['seed']}`. "
        f"Data are **simulated** with known ground truth ({cfg['n_users']:,} users, "
        f"{_pct(sim['treatment_share'], 0)} randomised to a €{sim['reward_cost']:.2f} bonus "
        "reward), so every estimate below can be checked against the true answer. "
        "Money figures are EUR per user over the 14-day window unless stated.\n"
    )

    add("## Key findings\n")
    conv = ate["conversion"]
    conv_verdict = "lifts" if conv["ci_low"] > 0 else "does not clearly lift"
    if net["ci_high"] < 0:
        profit_verdict = "but loses money once the reward is paid"
    elif net["ci_low"] > 0:
        profit_verdict = "and stays profitable after the reward is paid"
    else:
        profit_verdict = "but its effect on profit is not distinguishable from zero"
    add(
        f"1. **Rewarding everyone {conv_verdict} conversion, {profit_verdict}.** "
        f"Conversion moves by {_pts(conv['estimate'])} "
        f"({_pct(ate['conversion_relative_lift']['estimate'])} relative; true "
        f"{_pts(conv['truth'])}); net revenue per user moves by {_ci(net)} (true "
        f"{_eur(net['truth'])}). The reward is also paid to users who would have converted "
        "anyway."
    )
    vs = profit["vs_everyone"]
    beats = "beats" if vs["ci_low"] > 0 else "is not clearly better than"
    add(
        f"2. **Profit-aware targeting {beats} rewarding everyone.** Rewarding only the "
        f"{_pct(profit['treated_share'], 0)} of users with positive expected profit earns "
        f"€{1000 * vs['gain']:.0f} more per 1,000 users than rewarding everyone "
        f"[{1000 * vs['ci_low']:.0f}, {1000 * vs['ci_high']:.0f}] (true difference "
        f"€{1000 * vs['true_gain']:.0f})."
    )
    add(
        f"3. **CUPED cuts the variance of the revenue estimate by "
        f"{_pct(rev['cuped']['variance_reduction'], 0)}** "
        f"(regression adjustment: {_pct(rev['regression_adjusted']['variance_reduction'], 0)}). "
        f"That is the precision of a test with "
        f"{_pct(1 / (1 - rev['cuped']['variance_reduction']) - 1, 0)} more users, for free."
    )
    add(
        f"4. **Peeking is dangerous.** Stopping at the first p < 0.05 over {max_looks} looks "
        f"gives a {_pct(null_rows[max_looks]['naive_peeking_rate'])} false-positive rate; "
        f"always-valid p-values keep it at {_pct(null_rows[max_looks]['msprt_rate'])}."
    )
    add(
        f"5. **A logging bug that drops {_pct(srm['logging_loss'], 0)} of low-activity treated "
        f"users is {'caught' if srm['broken']['mismatch'] else 'missed'} by the sample-ratio "
        f"check** (p = {srm['broken']['p_value']:.1e}; alarm threshold p < 0.001).\n"
    )

    add("## 1. Is the data trustworthy? Sample ratio mismatch\n")
    add(
        _table(
            ["Pipeline", "Treated", "Control", "Treated share", "p-value", "Mismatch?"],
            [
                [
                    name,
                    f"{s['n_treatment']:,}",
                    f"{s['n_control']:,}",
                    _pct(s["observed_treatment_share"], 2),
                    _p(s["p_value"]),
                    "**yes**" if s["mismatch"] else "no",
                ]
                for name, s in (("Healthy", srm["healthy"]), ("Broken logging", srm["broken"]))
            ],
        )
    )
    broken = srm["broken_conversion_effect"]
    direction = "overstates" if broken["estimate"] > broken["truth"] else "understates"
    add(
        f"\nThe broken pipeline drops treated users with low activity, who convert less "
        f"than average, so the treated arm looks healthier than it is. Here its conversion "
        f"estimate, {_pts(broken['estimate'])}, {direction} the true population effect of "
        f"{_pts(broken['truth'])}. In a real experiment there is no true value to compare "
        "with; the sample-ratio check is what raises the alarm, before anyone reads the "
        "effect. It uses a strict threshold (p < 0.001) because it runs on every "
        "experiment.\n"
    )

    add("## 2. Average effects\n")
    add(
        _table(
            ["Metric", "Estimator", "Estimate [95% CI]", "True value", "CI covers truth"],
            [
                [
                    "Conversion",
                    "Difference in means",
                    _ci(ate["conversion"], _pts),
                    _pts(ate["conversion"]["truth"]),
                    str(ate["conversion"]["ci_covers_truth"]),
                ],
                *[
                    [
                        "Revenue",
                        label,
                        _ci(rev[key]),
                        _eur(rev[key]["truth"]),
                        str(rev[key]["ci_covers_truth"]),
                    ]
                    for key, label in (
                        ("difference_in_means", "Difference in means"),
                        ("cuped", "CUPED (pre-period revenue)"),
                        ("regression_adjusted", "Regression adjustment (Lin 2013)"),
                    )
                ],
                [
                    "Net revenue (after reward)",
                    "CUPED",
                    _ci(net),
                    _eur(net["truth"]),
                    str(net["ci_covers_truth"]),
                ],
            ],
        )
    )
    add(
        "\n![Effect estimates with and without variance reduction]"
        "(figures/variance_reduction.png)\n"
    )
    add(
        "Adjusting for pre-experiment data narrows the interval without biasing it: the "
        "covariate is measured before randomisation, so it cannot differ between arms "
        "except by chance.\n"
    )

    add("## 3. Is the analysis calibrated? A/A tests\n")
    add(
        f"Control users were split at random {aa['difference_in_means']['n_splits']} times and "
        "analysed as if one half had been treated. With no real effect, about 5% of splits "
        "should be significant and p-values should be uniform.\n"
    )
    add(
        _table(
            ["Estimator", "False-positive rate", "95% CI", "Uniformity test p", "Calibrated"],
            [
                [
                    name,
                    _pct(a["false_positive_rate"]),
                    " to ".join(_pct(v) for v in a["false_positive_rate_ci"]),
                    f"{a['uniformity_p_value']:.2f}",
                    "yes" if a["calibrated"] else "**no**",
                ]
                for name, a in (
                    ("Difference in means", aa["difference_in_means"]),
                    ("CUPED", aa["cuped"]),
                )
            ],
        )
    )
    add("\n![A/A p-value distribution](figures/aa_pvalues.png)\n")

    add("## 4. Planning the next test\n")
    add(
        f"- Baseline conversion is {_pct(power['baseline_conversion'])}. Detecting a 1-point "
        f"lift (80% power, α = 0.05) needs {power['users_for_1pt_conversion_lift']:,} users; "
        f"a 0.5-point lift needs {power['users_for_0p5pt_conversion_lift']:,}, about four "
        "times as many, because sample size scales with 1 / effect².\n"
        f"- With this experiment's {cfg['n_users']:,} users the smallest detectable revenue "
        f"effect is €{power['revenue_mde_this_experiment']:.3f} per user; with CUPED it is "
        f"€{power['revenue_mde_with_cuped']:.3f}.\n"
    )

    add("## 5. Peeking and always-valid inference\n")
    add(
        f"{cfg['peeking_sims']:,} simulated null experiments per row (no true effect), "
        "analysed three ways:\n"
    )
    add(
        _table(
            ["Looks", "Test once at end", "Stop at first p < 0.05", "Always-valid (mSPRT)"],
            [
                [
                    str(k),
                    _pct(row["fixed_horizon_rate"]),
                    _pct(row["naive_peeking_rate"]),
                    _pct(row["msprt_rate"]),
                ]
                for k, row in sorted(null_rows.items())
            ],
        )
    )
    add("\n![False-positive rate by number of looks](figures/peeking.png)\n")
    add(
        f"Always-valid p-values are not free. With a true effect of 0.04 standard deviations "
        f"and 20 looks, the fixed-horizon test detects it {_pct(alt['fixed_horizon_rate'], 0)} "
        f"of the time and the mSPRT {_pct(alt['msprt_rate'], 0)}; when the mSPRT does detect "
        f"it, it stops after {_pct(alt['msprt_mean_stop_fraction'], 0)} of the planned sample "
        "on average. The trade is lower power for the freedom to monitor and stop early. "
        "Replaying this experiment's revenue in arrival order with 20 looks, the naive "
        f"test is first significant after {_users(seq['first_significant_n_users']['naive'])} "
        "and the always-valid test after "
        f"{_users(seq['first_significant_n_users']['always_valid'])}.\n"
    )

    add("## 6. Who responds to the reward? Heterogeneous effects\n")
    add(
        f"Four meta-learners were trained on {up['n_train']:,} users (a quarter of them held "
        f"back for model selection) and evaluated on {up['n_test']:,} test users. The Qini "
        "areas are all a real analysis can see; the last three columns use the hidden "
        "truth.\n"
    )
    add(
        _table(
            [
                "Model",
                "Validation Qini",
                "Test Qini",
                "PEHE (lower is better)",
                "Rank corr. with truth",
                "Share of oracle gain",
            ],
            [
                [
                    name,
                    f"{m['validation_qini_area']:.4f}",
                    f"{m['observed_qini_area']:.4f}",
                    f"{m['pehe']:.4f}",
                    f"{m['spearman']:.2f}",
                    _pct(m["normalised_gain"], 0),
                ]
                for name, m in sorted(
                    up["models"].items(), key=lambda kv: -kv[1]["observed_qini_area"]
                )
            ],
        )
    )
    add(
        f"\nOracle observed Qini area: {up['oracle_observed_qini_area']:.4f}. Selected model "
        f"(by {up['selected_by']}): **{up['selected_model']}**.\n"
    )
    add("![Observed Qini curves and true gain curves](figures/targeting_curves.png)\n")
    add("![Observed uplift by decile](figures/uplift_deciles.png)\n")
    by_truth = sorted(up["models"].items(), key=lambda kv: -kv[1]["normalised_gain"])
    add(
        "Ranked by the hidden truth: "
        + ", ".join(f"{n} ({_pct(m['normalised_gain'], 0)} of oracle gain)" for n, m in by_truth)
        + ". The T-learner fits each arm separately, so its two models make independent "
        "errors that do not cancel when subtracted; the X- and DR-learners model the effect "
        "itself. The observed Qini area separates the models less sharply than the truth "
        "does, because it is computed from noisy outcomes on a finite sample.\n"
    )
    segments = r["sql"]["segment_effects"]
    add("Segment view (from `src/uplift_lab/sql/segment_effects.sql`):\n")
    add(
        _table(
            ["Dimension", "Segment", "Control rate", "Treated rate", "Lift ± 1.96 SE"],
            [
                [
                    s["dimension"],
                    s["segment"],
                    _pct(s["rate_control"]),
                    _pct(s["rate_treatment"]),
                    f"{_pts(s['lift'])} ± {100 * 1.96 * s['lift_std_error']:.2f}",
                ]
                for s in segments
            ],
        )
    )
    add("")

    add("## 7. Who should get the reward? Profit-aware targeting\n")
    add(
        "Expected incremental profit per user is `payout × uplift − reward cost × P(convert "
        f"| rewarded)`. Uplift comes from the {pol['uplift_model']}, the treated conversion "
        f"probability from the {pol['cost_model']}. Each policy is evaluated on held-out "
        "randomised users by inverse-propensity weighting against a no-reward baseline, "
        "with CUPED-adjusted net revenue as the outcome. All figures are EUR per 1,000 "
        "users. The paired comparison against rewarding everyone is much tighter than "
        "either value alone, because both policies are scored on the same users.\n"
    )
    add(
        _table(
            [
                "Policy",
                "Users rewarded",
                "Gain vs. no rewards [95% CI]",
                "True",
                "Gain vs. rewarding everyone [95% CI]",
                "True",
            ],
            [
                [
                    p["name"],
                    _pct(p["treated_share"], 0),
                    _per_1000(p),
                    f"€{1000 * p['true_gain']:.0f}",
                    _per_1000(p["vs_everyone"]),
                    f"€{1000 * p['vs_everyone']['true_gain']:.0f}",
                ]
                for p in pol["values_per_user"]
            ],
        )
    )
    add("\n![Policy values](figures/policy_values.png)\n")
    uplift_only = next(p for p in pol["values_per_user"] if p["name"].startswith("Top "))
    add(
        f"Ranking by uplift alone (true €{1000 * uplift_only['true_gain']:.0f} per 1,000 "
        f"users) ignores two things: the reward is also paid to users who would have "
        "converted anyway, and payouts differ by country. Ranking by expected profit "
        f"captures both (true €{1000 * profit['true_gain']:.0f}). When the reward budget is "
        "capped, users are added in order of expected profit per euro of reward spend.\n"
    )

    add("## Limitations\n")
    add(
        "- The data are simulated. The simulator encodes plausible but invented behaviour; "
        "the methods are validated against it, not the business conclusions. "
        "`uplift-lab criteo` runs the observable parts on the real Criteo Uplift dataset.\n"
        "- Effects are measured over 14 days. Rewards may shift conversions in time or change "
        "long-term behaviour (novelty effects, reward dependence), which a short test misses.\n"
        "- Policy values are estimated on the same held-out sample, so their errors are "
        "correlated; differences between policies are more precise than each value alone.\n"
        "- Payout per conversion is treated as a known price per country tier, and the "
        "reward cost model (T-learner) is fitted on the same training users as the uplift "
        "model; errors in either move users across the profit threshold.\n"
        "- The selected uplift model is chosen on a validation split, so the reported test "
        "performance is not inflated by the choice; with four close candidates, which one "
        "wins can change with the seed.\n"
        "- The always-valid test assumes a normal approximation with a plug-in variance and "
        "a mixing variance chosen in advance; a badly chosen mixing variance costs power, "
        "not validity.\n"
    )
    return "\n".join(out)


def render_criteo_report(r: Mapping[str, Any]) -> str:
    """Build the Markdown report for :func:`uplift_lab.pipeline.run_criteo_study`."""
    srm, ate, lift, up = r["srm"], r["ate"], r["relative_lift"], r["uplift"]
    out = [
        "# Reward Uplift Lab: Criteo Uplift (real data)\n",
        f"{r['n_users']:,} users, outcome `{r['outcome']}`. Real data has no ground truth, "
        "so models are compared on held-out Qini areas and decile calibration only.\n",
        "## Sample ratio\n",
        f"Observed treated share {_pct(srm['observed_treatment_share'], 2)} vs designed "
        f"{_pct(srm['expected_treatment_share'], 0)} (p = {_p(srm['p_value'])}, "
        f"mismatch: {'**yes**' if srm['mismatch'] else 'no'}). The dataset documents its "
        "treatment share only approximately, so with millions of users a flag here can mean "
        "the true design share differs slightly from the documented one rather than a "
        "logging fault; check the observed share before drawing conclusions.\n",
        "## Average effect\n",
        f"Difference in `{r['outcome']}` rate: {_ci(ate, _pts)}; relative lift "
        f"{_ci(lift, _pct)}.\n",
        "## Uplift models\n",
        _table(
            ["Model", "Validation Qini", "Test Qini"],
            [
                [n, f"{m['validation_qini_area']:.5f}", f"{m['observed_qini_area']:.5f}"]
                for n, m in sorted(
                    up["models"].items(), key=lambda kv: -kv[1]["observed_qini_area"]
                )
            ],
        ),
        f"\nSelected model (by {up['selected_by']}): **{up['selected_model']}**.\n",
        "![Observed Qini curves](figures/criteo_qini.png)\n",
        "![Observed uplift by decile](figures/criteo_deciles.png)\n",
    ]
    return "\n".join(out)
