"""End-to-end studies: the simulated ground-truth study and the Criteo real-data study.

Each study returns a JSON-serialisable dictionary of results and writes its figures,
so the report can be regenerated from scratch with one command.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from uplift_lab import plots
from uplift_lab.data import SimulationConfig, feature_matrix, simulate_experiment
from uplift_lab.data.criteo import CRITEO_FEATURES
from uplift_lab.experiment import (
    cuped,
    difference_in_means,
    minimum_detectable_effect,
    regression_adjusted,
    relative_lift,
    required_sample_size_proportions,
    run_aa_tests,
    sample_ratio_mismatch,
    sequential_monitor,
    simulate_peeking,
)
from uplift_lab.experiment.sequential import PeekingResult
from uplift_lab.sql import run_query
from uplift_lab.uplift import (
    DRLearner,
    MetaLearner,
    SLearner,
    TLearner,
    XLearner,
    budgeted_profit_policy,
    expected_incremental_profit,
    ipw_policy_gain,
    qini_curve,
    top_fraction_policy,
    true_gain_curve,
    truth_metrics,
    uplift_by_bin,
)

log = logging.getLogger(__name__)

LEARNERS: dict[str, Callable[[], MetaLearner]] = {
    "S-learner": SLearner,
    "T-learner": TLearner,
    "X-learner": XLearner,
    "DR-learner": DRLearner,
}


@dataclass(frozen=True)
class StudyConfig:
    """Settings for the simulated study.

    Attributes:
        n_users: Users in the simulated experiment.
        seed: Seed for data, splits and simulations.
        test_share: Share of users held out to evaluate uplift models and policies.
        aa_splits: Number of A/A splits per estimator.
        peeking_looks: Numbers of interim looks to compare.
        peeking_sims: Simulated experiments per number of looks.
        logging_loss: Treated-user loss injected into the "broken" experiment.
        uplift_top_fraction: Share treated by the uplift-only policy.
        budget_share: Reward budget as a share of the treat-everyone spend.
    """

    n_users: int = 200_000
    seed: int = 7
    test_share: float = 0.5
    aa_splits: int = 500
    peeking_looks: tuple[int, ...] = (1, 2, 5, 10, 20, 50)
    peeking_sims: int = 4_000
    logging_loss: float = 0.10
    uplift_top_fraction: float = 0.30
    budget_share: float = 0.25


def _effect(estimate: Any, truth: float | None = None) -> dict[str, Any]:
    out: dict[str, Any] = estimate.to_dict()
    if truth is not None:
        out["truth"] = float(truth)
        out["ci_covers_truth"] = bool(out["ci_low"] <= truth <= out["ci_high"])
    return out


# Independent random streams for each purpose, so that changing one split never
# shifts the random numbers used by another (or by the simulator, seeded with ``seed``).
TEST_STREAM, VALIDATION_STREAM = 1, 2


def _split(n: int, share: float, seed: int, stream: int) -> NDArray[np.bool_]:
    rng = np.random.default_rng([seed, stream])
    return np.asarray(rng.random(n) < share)


@dataclass(frozen=True)
class FittedLearners:
    """Meta-learners fitted on one split and scored on a separate validation split.

    Attributes:
        learners: Fitted learners by name.
        validation_qini: Observed Qini area of each learner on the validation split.
        selected: Name of the learner with the highest validation Qini area.
    """

    learners: dict[str, MetaLearner]
    validation_qini: dict[str, float]
    selected: str


def fit_and_select(
    X: NDArray[np.float64],
    t: NDArray[np.int_],
    y: NDArray[np.float64],
    *,
    validation_share: float = 0.25,
    seed: int = 0,
) -> FittedLearners:
    """Fit every learner on part of the training data and pick one on the rest.

    Choosing the model on the same users it is later evaluated on would make the
    reported test performance optimistic, because the winner is partly the model that
    got lucky on that sample. Selection therefore uses a validation split carved out
    of the training data, and the test split is touched only for the final report.
    """
    validation = _split(len(y), validation_share, seed, VALIDATION_STREAM)
    fit = ~validation
    learners: dict[str, MetaLearner] = {}
    validation_qini: dict[str, float] = {}
    for name, factory in LEARNERS.items():
        log.info("fitting %s", name)
        learner = factory().fit(X[fit], t[fit], y[fit])
        score = learner.predict_uplift(X[validation])
        validation_qini[name] = qini_curve(y[validation], t[validation], score).area_over_random()
        learners[name] = learner
    selected = max(validation_qini, key=validation_qini.__getitem__)
    return FittedLearners(learners, validation_qini, selected)


def _peeking_record(result: PeekingResult) -> dict[str, Any]:
    record = asdict(result)
    record["rate_cis"] = {rule: list(ci) for rule, ci in result.rate_cis.items()}
    return record


def run_simulation_study(config: StudyConfig, out_dir: Path) -> dict[str, Any]:
    """Run the full simulated study and write figures to ``out_dir / 'figures'``."""
    figures = out_dir / "figures"
    sim = simulate_experiment(SimulationConfig(n_users=config.n_users, seed=config.seed))
    df, truth, cfg = sim.data, sim.truth, sim.config
    t = df["treatment"].to_numpy()
    results: dict[str, Any] = {"config": asdict(config), "simulation": asdict(cfg)}

    # 1. Data quality: sample ratio mismatch on a healthy and a broken pipeline.
    log.info("sample ratio checks")
    broken = simulate_experiment(
        SimulationConfig(
            n_users=config.n_users, seed=config.seed, treated_logging_loss=config.logging_loss
        )
    )
    results["srm"] = {
        "healthy": sample_ratio_mismatch(t, cfg.treatment_share).to_dict(),
        "broken": sample_ratio_mismatch(broken.data["treatment"], cfg.treatment_share).to_dict(),
        "broken_conversion_effect": _effect(
            difference_in_means(broken.data["converted"], broken.data["treatment"]),
            # The question the experiment asks is about the whole population.
            float(truth["tau_conversion"].mean()),
        ),
        "logging_loss": config.logging_loss,
    }

    # 2. Descriptive SQL.
    results["sql"] = {
        "arm_summary": run_query("arm_summary", {"experiment": df}).to_dict("records"),
        "segment_effects": run_query("segment_effects", {"experiment": df}).to_dict("records"),
    }

    # 3. Average effects, with and without variance reduction.
    log.info("average treatment effects")
    X_all = feature_matrix(df).to_numpy()
    true_conv = float(truth["tau_conversion"].mean())
    true_rev = float(truth["tau_revenue"].mean())
    true_profit = float(truth["tau_profit"].mean())
    revenue_cuped = cuped(df["revenue_14d"], t, df["pre_revenue_14d"])
    revenue_lin = regression_adjusted(df["revenue_14d"], t, X_all)
    net_cuped = cuped(df["net_revenue_14d"], t, df["pre_revenue_14d"])
    results["ate"] = {
        "conversion": _effect(difference_in_means(df["converted"], t), true_conv),
        "conversion_relative_lift": relative_lift(df["converted"], t).to_dict(),
        "revenue": {
            "difference_in_means": _effect(difference_in_means(df["revenue_14d"], t), true_rev),
            "cuped": {
                **_effect(revenue_cuped.effect, true_rev),
                "variance_reduction": revenue_cuped.variance_reduction,
                "theta": revenue_cuped.theta,
            },
            "regression_adjusted": {
                **_effect(revenue_lin.effect, true_rev),
                "variance_reduction": revenue_lin.variance_reduction,
            },
        },
        "net_revenue_cuped": {
            **_effect(net_cuped.effect, true_profit),
            "variance_reduction": net_cuped.variance_reduction,
        },
    }
    rev = results["ate"]["revenue"]
    plots.estimate_intervals(
        [
            {"label": "Difference in means", **rev["difference_in_means"]},
            {"label": "CUPED (pre-period revenue)", **rev["cuped"]},
            {"label": "Regression adjustment (all features)", **rev["regression_adjusted"]},
        ],
        true_rev,
        figures / "variance_reduction.png",
        xlabel="Effect on 14-day revenue (EUR per user)",
    )

    # 4. A/A calibration of both estimators on the control arm.
    log.info("A/A tests")
    control = t == 0
    y_ctrl = df.loc[control, "revenue_14d"].to_numpy()
    x_ctrl = df.loc[control, "pre_revenue_14d"].to_numpy()
    aa_dim = run_aa_tests(y_ctrl, n_splits=config.aa_splits, seed=config.seed)
    aa_cuped = run_aa_tests(
        y_ctrl,
        n_splits=config.aa_splits,
        seed=config.seed + 1,
        estimator=lambda y, tt: cuped(y, tt, x_ctrl).effect,
    )
    results["aa"] = {
        name: {
            "false_positive_rate": r.false_positive_rate,
            "false_positive_rate_ci": r.false_positive_rate_ci,
            "uniformity_p_value": r.uniformity_p_value,
            "calibrated": r.calibrated,
            "n_splits": int(r.p_values.size),
        }
        for name, r in (("difference_in_means", aa_dim), ("cuped", aa_cuped))
    }
    plots.aa_pvalue_ecdf(
        {"Difference in means": aa_dim.p_values, "CUPED": aa_cuped.p_values},
        figures / "aa_pvalues.png",
    )

    # 5. Planning: sample size and minimum detectable effect.
    baseline = float(df.loc[control, "converted"].mean())
    sd_rev = float(df["revenue_14d"].std(ddof=1))
    results["power"] = {
        "baseline_conversion": baseline,
        "users_for_1pt_conversion_lift": required_sample_size_proportions(baseline, 0.01),
        "users_for_0p5pt_conversion_lift": required_sample_size_proportions(baseline, 0.005),
        "revenue_sd": sd_rev,
        "revenue_mde_this_experiment": minimum_detectable_effect(sd_rev, len(df)),
        "revenue_mde_with_cuped": minimum_detectable_effect(
            sd_rev, len(df), variance_reduction=revenue_cuped.variance_reduction
        ),
    }

    # 6. Peeking: false positives under the null, power under an alternative.
    log.info("peeking simulations")
    peeking_rows = [
        _peeking_record(
            simulate_peeking(n_looks=k, n_sims=config.peeking_sims, seed=config.seed + k)
        )
        for k in config.peeking_looks
    ]
    power_alt = simulate_peeking(
        n_looks=20, n_sims=config.peeking_sims, effect=0.04, seed=config.seed
    )
    monitor = sequential_monitor(df["revenue_14d"], t, n_looks=20, tau2=(0.05 * sd_rev) ** 2)
    first = {
        rule: (
            int(monitor.loc[monitor[f"{rule}_significant"], "n_users"].min())
            if monitor[f"{rule}_significant"].any()
            else None
        )
        for rule in ("naive", "always_valid")
    }
    results["sequential"] = {
        "null": peeking_rows,
        "alternative_effect_0p04_sd": _peeking_record(power_alt),
        "monitor_revenue": monitor.to_dict("records"),
        "first_significant_n_users": first,
    }
    plots.peeking_rates(peeking_rows, 0.05, figures / "peeking.png")

    # 7. Heterogeneous effects: fit and select on training users, report on test users.
    log.info("uplift models")
    test = _split(len(df), config.test_share, config.seed, TEST_STREAM)
    train = ~test
    y_conv = df["converted"].to_numpy(dtype=float)
    fitted = fit_and_select(X_all[train], t[train], y_conv[train], seed=config.seed)
    best = fitted.selected
    preds = {name: m.predict_uplift(X_all[test]) for name, m in fitted.learners.items()}
    tau_test = truth.loc[test, "tau_conversion"].to_numpy()
    model_results, observed, gains = {}, {}, {}
    for name, tau_hat in preds.items():
        q = qini_curve(y_conv[test], t[test], tau_hat)
        g = true_gain_curve(tau_test, tau_hat)
        observed[name] = (q.fractions, q.values)
        gains[name] = (g.fractions, g.values)
        model_results[name] = {
            "validation_qini_area": fitted.validation_qini[name],
            "observed_qini_area": q.area_over_random(),
            **asdict(truth_metrics(tau_test, tau_hat)),
        }
    oracle = true_gain_curve(tau_test, tau_test)
    oracle_q = qini_curve(y_conv[test], t[test], tau_test)
    observed["oracle"] = (oracle_q.fractions, oracle_q.values)
    gains["oracle"] = (oracle.fractions, oracle.values)
    deciles = uplift_by_bin(y_conv[test], t[test], preds[best])
    results["uplift"] = {
        "models": model_results,
        "oracle_observed_qini_area": oracle_q.area_over_random(),
        "selected_model": best,
        "selected_by": "highest Qini area on a validation split of the training users",
        "deciles": deciles,
        "n_train": int(train.sum()),
        "n_test": int(test.sum()),
    }
    plots.targeting_curves(observed, gains, figures / "targeting_curves.png")
    plots.uplift_deciles(deciles, best, figures / "uplift_deciles.png")

    # 8. Profit-aware targeting, evaluated off-policy on held-out randomised data.
    log.info("policy evaluation")
    t_learner = fitted.learners["T-learner"]
    if not isinstance(t_learner, TLearner):  # pragma: no cover - guarded by LEARNERS
        raise TypeError("cost model must be a TLearner")
    _, mu1_hat = t_learner.predict_outcomes(X_all[test])
    payout = truth.loc[test, "payout"].to_numpy()  # payout per conversion is a known price
    tau_hat = preds[best]
    profit_hat = expected_incremental_profit(tau_hat, mu1_hat, payout, cfg.reward_cost)
    expected_cost = cfg.reward_cost * np.clip(mu1_hat, 0, 1)
    budget = config.budget_share * float(expected_cost.sum())

    # CUPED-adjust the outcome with theta from training users only: the pre-period
    # term has zero mean under every policy's weights, so it removes noise, not signal.
    train_rev = df.loc[train]
    theta = float(
        np.cov(train_rev["net_revenue_14d"], train_rev["pre_revenue_14d"])[0, 1]
        / train_rev["pre_revenue_14d"].var()
    )
    pre = df.loc[test, "pre_revenue_14d"].to_numpy()
    outcome = df.loc[test, "net_revenue_14d"].to_numpy() - theta * (pre - pre.mean())
    tau_profit = truth.loc[test, "tau_profit"].to_numpy()
    n_test = int(test.sum())
    policies = {
        "Reward everyone": np.ones(n_test, dtype=bool),
        f"Top {config.uplift_top_fraction:.0%} by uplift": top_fraction_policy(
            tau_hat, config.uplift_top_fraction
        ),
        "Positive expected profit": budgeted_profit_policy(profit_hat, expected_cost),
        f"Profit, budget {config.budget_share:.0%} of full spend": budgeted_profit_policy(
            profit_hat, expected_cost, budget
        ),
        "Oracle (true profit > 0)": tau_profit > 0,
    }
    everyone = policies["Reward everyone"]
    policy_rows = []
    for name, policy in policies.items():
        value = ipw_policy_gain(
            outcome,
            t[test],
            policy,
            propensity=cfg.treatment_share,
            name=name,
            true_effect=tau_profit,
        ).to_dict()
        row: dict[str, Any] = dict(value)
        # Paired comparison with the blanket policy: the shared noise cancels.
        versus = ipw_policy_gain(
            outcome,
            t[test],
            policy,
            everyone,
            propensity=cfg.treatment_share,
            name=name,
            true_effect=tau_profit,
        )
        row["vs_everyone"] = {
            "gain": versus.gain,
            "ci_low": versus.ci_low,
            "ci_high": versus.ci_high,
            "true_gain": versus.true_gain,
        }
        policy_rows.append(row)
    results["policy"] = {
        "values_per_user": policy_rows,
        "uplift_model": best,
        "cost_model": "T-learner mu1",
        "budget_eur": budget,
        "cuped_theta": theta,
    }
    plots.policy_values(policy_rows, figures / "policy_values.png", scale=1000)
    return results


def run_criteo_study(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    outcome: str = "visit",
    treatment_share: float = 0.85,
    test_share: float = 0.5,
    seed: int = 0,
) -> dict[str, Any]:
    """Run the observable parts of the study on the real Criteo Uplift data.

    Real data has no ground truth, so models are compared on held-out Qini areas and
    decile calibration only.
    """
    if outcome not in ("visit", "conversion"):
        raise ValueError("outcome must be 'visit' or 'conversion'")
    figures = out_dir / "figures"
    t = df["treatment"].to_numpy()
    y = df[outcome].to_numpy(dtype=float)
    X = df[list(CRITEO_FEATURES)].to_numpy(dtype=float)

    results: dict[str, Any] = {
        "n_users": len(df),
        "outcome": outcome,
        "srm": sample_ratio_mismatch(t, treatment_share).to_dict(),
        "ate": difference_in_means(y, t).to_dict(),
        "relative_lift": relative_lift(y, t).to_dict(),
    }
    test = _split(len(df), test_share, seed, TEST_STREAM)
    fitted = fit_and_select(X[~test], t[~test], y[~test], seed=seed)
    best = fitted.selected
    observed = {}
    models = {}
    for name, learner in fitted.learners.items():
        tau_hat = learner.predict_uplift(X[test])
        q = qini_curve(y[test], t[test], tau_hat)
        observed[name] = (q.fractions, q.values)
        models[name] = {
            "validation_qini_area": fitted.validation_qini[name],
            "observed_qini_area": q.area_over_random(),
        }
        if name == best:
            deciles = uplift_by_bin(y[test], t[test], tau_hat)
    results["uplift"] = {
        "models": models,
        "selected_model": best,
        "selected_by": "highest Qini area on a validation split of the training users",
        "deciles": deciles,
    }
    plots.qini_curves(
        observed, figures / "criteo_qini.png", title=f"Criteo: observed Qini curve ({outcome})"
    )
    plots.uplift_deciles(deciles, best, figures / "criteo_deciles.png")
    return results
