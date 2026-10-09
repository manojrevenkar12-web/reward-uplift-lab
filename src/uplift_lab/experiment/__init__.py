"""Trustworthy analysis of randomised experiments."""

from uplift_lab.experiment.aa import AAResult, run_aa_tests, wilson_interval
from uplift_lab.experiment.estimators import EffectEstimate, difference_in_means, relative_lift
from uplift_lab.experiment.power import (
    minimum_detectable_effect,
    required_sample_size_means,
    required_sample_size_proportions,
)
from uplift_lab.experiment.sequential import (
    PeekingResult,
    always_valid_p_values,
    msprt_log_lr,
    sequential_monitor,
    simulate_peeking,
)
from uplift_lab.experiment.srm import SRMResult, sample_ratio_mismatch
from uplift_lab.experiment.variance_reduction import AdjustedEstimate, cuped, regression_adjusted

__all__ = [
    "AAResult",
    "AdjustedEstimate",
    "EffectEstimate",
    "PeekingResult",
    "SRMResult",
    "always_valid_p_values",
    "cuped",
    "difference_in_means",
    "minimum_detectable_effect",
    "msprt_log_lr",
    "regression_adjusted",
    "relative_lift",
    "required_sample_size_means",
    "required_sample_size_proportions",
    "run_aa_tests",
    "sample_ratio_mismatch",
    "sequential_monitor",
    "simulate_peeking",
    "wilson_interval",
]
