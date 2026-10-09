"""Static figures for the study report (matplotlib, PNG).

One visual system for every chart: a fixed categorical order, thin marks, recessive
grid and axes, direct labels where they help, and text in neutral ink rather than
series colours.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "font.size": 10,
        "legend.frameon": False,
        "legend.labelcolor": TEXT_PRIMARY,
        "lines.linewidth": 2.0,
    }
)


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _reference_line(ax: Axes, y: float, label: str) -> None:
    ax.axhline(y, color=TEXT_SECONDARY, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.annotate(
        label,
        xy=(0.42, y),
        xycoords=("axes fraction", "data"),
        xytext=(0, -4),
        textcoords="offset points",
        va="top",
        fontsize=9,
        color=TEXT_SECONDARY,
    )


def aa_pvalue_ecdf(p_values: Mapping[str, np.ndarray], path: Path) -> Path:
    """ECDF of A/A p-values per estimator against the uniform diagonal."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(
        [0, 1],
        [0, 1],
        color=TEXT_SECONDARY,
        linewidth=1.0,
        linestyle=(0, (4, 3)),
        label="Perfect calibration",
    )
    for color, (name, p) in zip(SERIES, p_values.items(), strict=False):
        x = np.sort(p)
        ax.step(x, np.arange(1, x.size + 1) / x.size, where="post", color=color, label=name)
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="p-value",
        ylabel="Share of A/A splits with a smaller p-value",
    )
    ax.set_title("A/A tests: p-value distribution against perfect calibration")
    ax.legend(loc="upper left")
    return _save(fig, path)


def estimate_intervals(
    estimates: Sequence[Mapping[str, Any]], truth: float, path: Path, *, xlabel: str
) -> Path:
    """Point estimates with 95% intervals for several estimators, and the true value."""
    fig, ax = plt.subplots(figsize=(6.8, 0.75 * len(estimates) + 1.6))
    labels = [str(e["label"]) for e in estimates]
    ys = np.arange(len(estimates))[::-1]
    for y, e in zip(ys, estimates, strict=True):
        ax.plot(
            [e["ci_low"], e["ci_high"]],
            [y, y],
            color=SERIES[0],
            linewidth=2.0,
            solid_capstyle="round",
        )
        ax.plot(
            e["estimate"],
            y,
            "o",
            color=SERIES[0],
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
        )
        width = e["ci_high"] - e["ci_low"]
        ax.annotate(
            f"±{width / 2:.3f}",
            xy=(e["ci_high"], y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=TEXT_SECONDARY,
        )
    ax.axvline(truth, color=TEXT_PRIMARY, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.annotate(
        "true effect",
        xy=(truth, ys.max() + 0.5),
        xytext=(4, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=9,
        color=TEXT_PRIMARY,
    )
    ax.set_yticks(ys, labels)
    ax.set_ylim(-0.6, ys.max() + 0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    ax.set_title("Effect estimate and 95% interval by estimator")
    return _save(fig, path)


def peeking_rates(rows: Sequence[Mapping[str, Any]], alpha: float, path: Path) -> Path:
    """False-positive rate by number of interim looks for each decision rule."""
    looks = np.array([r["n_looks"] for r in rows])
    rules = [
        ("naive_peeking", "Stop at first p < 0.05"),
        ("fixed_horizon", "Test once at the end"),
        ("msprt", "Always-valid p-value (mSPRT)"),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for color, (key, label) in zip(SERIES, rules, strict=False):
        rate = np.array([r[f"{key}_rate"] for r in rows])
        low = np.array([r["rate_cis"][key][0] for r in rows])
        high = np.array([r["rate_cis"][key][1] for r in rows])
        ax.fill_between(looks, low, high, color=color, alpha=0.15, linewidth=0)
        ax.plot(
            looks,
            rate,
            "o-",
            color=color,
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
        )
        ax.annotate(
            label,
            xy=(looks[-1], rate[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=TEXT_PRIMARY,
        )
    _reference_line(ax, alpha, f"α = {alpha:g}")
    ax.set_xscale("log")
    ax.set_xticks(looks, [str(k) for k in looks])
    ax.minorticks_off()
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set(
        xlabel="Number of looks at the data (log scale)",
        ylabel="False-positive rate (no true effect)",
    )
    ax.set_title("False-positive rate with no true effect, by decision rule")
    return _save(fig, path)


def _curve_panel(
    ax: Axes, curves: Mapping[str, tuple[np.ndarray, np.ndarray]], title: str, ylabel: str
) -> None:
    for color, (name, (x, y)) in zip(SERIES, curves.items(), strict=False):
        if name == "oracle":
            ax.plot(
                x,
                y,
                color=TEXT_PRIMARY,
                linewidth=1.2,
                linestyle=(0, (4, 3)),
                label="Oracle (true effects)",
            )
        else:
            ax.plot(x, y, color=color, label=name)
    end_value = next(iter(curves.values()))[1][-1]
    ax.plot(
        [0, 1],
        [0, end_value],
        color=TEXT_SECONDARY,
        linewidth=1.0,
        linestyle=(0, (1, 2)),
        label="Random targeting",
    )
    ax.set(
        xlim=(0, 1),
        xlabel="Share of users targeted (best-ranked first)",
        ylabel=ylabel,
        title=title,
    )
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


def targeting_curves(
    observed: Mapping[str, tuple[np.ndarray, np.ndarray]],
    true_gain: Mapping[str, tuple[np.ndarray, np.ndarray]],
    path: Path,
) -> Path:
    """Observed Qini curves next to noise-free true-gain curves, per model."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharex=True)
    _curve_panel(
        axes[0],
        observed,
        "What an analyst sees: observed Qini curve",
        "Incremental conversions per user",
    )
    _curve_panel(
        axes[1],
        true_gain,
        "Ground truth: true gain from targeting",
        "True incremental conversions per user",
    )
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, path)


def qini_curves(
    observed: Mapping[str, tuple[np.ndarray, np.ndarray]], path: Path, *, title: str
) -> Path:
    """Observed Qini curves only (for real data, where no ground truth exists)."""
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    _curve_panel(ax, observed, title, "Incremental outcomes per user")
    ax.legend(loc="lower right")
    return _save(fig, path)


def uplift_deciles(rows: Sequence[Mapping[str, float]], model: str, path: Path) -> Path:
    """Observed uplift per decile of predicted uplift, with predicted means overlaid."""
    bins = np.array([r["bin"] for r in rows])
    observed = np.array([r["observed_uplift"] for r in rows])
    se = np.array([r["std_error"] for r in rows])
    predicted = np.array([r["predicted_uplift"] for r in rows])
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(bins, observed, width=0.7, color=SERIES[0], label="Observed uplift (±95% CI)")
    ax.errorbar(
        bins, observed, yerr=1.96 * se, fmt="none", ecolor=TEXT_SECONDARY, elinewidth=1.0, capsize=3
    )
    ax.plot(
        bins,
        predicted,
        "o",
        color=SERIES[1],
        markersize=8,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
        label="Mean predicted uplift",
    )
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    ax.set_xticks(bins)
    ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=None))
    ax.set(xlabel="Decile of predicted uplift (1 = highest)", ylabel="Uplift in outcome rate")
    ax.set_title(f"{model}: observed vs. predicted uplift by decile")
    ax.legend(loc="upper right")
    return _save(fig, path)


def policy_values(rows: Sequence[Mapping[str, Any]], path: Path, *, scale: float) -> Path:
    """Estimated policy gains with 95% intervals and, if known, the true gains."""
    fig, ax = plt.subplots(figsize=(7.2, 0.7 * len(rows) + 1.6))
    ys = np.arange(len(rows))[::-1]
    for y, r in zip(ys, rows, strict=True):
        ax.plot(
            [r["ci_low"] * scale, r["ci_high"] * scale],
            [y, y],
            color=SERIES[0],
            linewidth=2.0,
            solid_capstyle="round",
        )
        ax.plot(
            r["gain"] * scale,
            y,
            "o",
            color=SERIES[0],
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            label="IPW estimate (95% CI)" if y == ys[0] else None,
        )
        if r.get("true_gain") is not None:
            ax.plot(
                r["true_gain"] * scale,
                y,
                "D",
                markersize=7,
                markerfacecolor="none",
                markeredgecolor=SERIES[1],
                markeredgewidth=1.8,
                label="True gain (simulation)" if y == ys[0] else None,
            )
    ax.axvline(0, color=TEXT_SECONDARY, linewidth=0.8)
    ax.set_yticks(ys, [f"{r['name']}  ({r['treated_share']:.0%} treated)" for r in rows])
    ax.set_ylim(-0.6, ys.max() + 0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Net profit vs. no rewards (EUR per 1,000 users)")
    ax.set_title("Net profit of each targeting policy vs. no rewards")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    return _save(fig, path)
