"""Sample ratio mismatch (SRM) check: the first test to run on any experiment.

If users were assigned 50/50 but the logged data is 50.6/49.4 on 200,000 users,
something between assignment and logging is broken (a crash, a redirect, a filter),
and every downstream metric is suspect. A chi-square goodness-of-fit test on arm
counts detects this. A strict threshold (p < 0.001) is standard, because the check is
run on every experiment and false alarms are costly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


@dataclass(frozen=True)
class SRMResult:
    """Outcome of a sample-ratio-mismatch test.

    Attributes:
        n_treatment: Observed users in the treatment arm.
        n_control: Observed users in the control arm.
        expected_treatment_share: Share the design intended for the treatment arm.
        observed_treatment_share: Share actually observed.
        chi2: Chi-square statistic.
        p_value: P-value of the goodness-of-fit test.
        mismatch: Whether ``p_value`` is below the alarm threshold.
    """

    n_treatment: int
    n_control: int
    expected_treatment_share: float
    observed_treatment_share: float
    chi2: float
    p_value: float
    mismatch: bool

    def to_dict(self) -> dict[str, float | int | bool]:
        """Return the result as a plain dictionary."""
        return asdict(self)


def sample_ratio_mismatch(
    t: ArrayLike, expected_treatment_share: float = 0.5, threshold: float = 0.001
) -> SRMResult:
    """Test whether observed arm sizes match the designed allocation.

    Args:
        t: Treatment indicator per user (1 = treated).
        expected_treatment_share: Designed probability of assignment to treatment.
        threshold: P-value below which a mismatch is declared.
    """
    t_arr = np.asarray(t)
    if not np.isin(t_arr, (0, 1)).all():
        raise ValueError("treatment must be binary (0/1)")
    if not 0.0 < expected_treatment_share < 1.0:
        raise ValueError("expected_treatment_share must be strictly between 0 and 1")
    n = t_arr.size
    n_t = int(t_arr.sum())
    observed = np.array([n_t, n - n_t])
    expected = n * np.array([expected_treatment_share, 1 - expected_treatment_share])
    chi2, p_value = stats.chisquare(observed, expected)
    return SRMResult(
        n_treatment=n_t,
        n_control=n - n_t,
        expected_treatment_share=expected_treatment_share,
        observed_treatment_share=n_t / n,
        chi2=float(chi2),
        p_value=float(p_value),
        mismatch=bool(p_value < threshold),
    )
