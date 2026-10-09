"""Data sources: a ground-truth simulator and a loader for the public Criteo Uplift dataset."""

from uplift_lab.data.criteo import CRITEO_FEATURES, load_criteo
from uplift_lab.data.features import feature_matrix
from uplift_lab.data.simulate import SimulatedExperiment, SimulationConfig, simulate_experiment

__all__ = [
    "CRITEO_FEATURES",
    "SimulatedExperiment",
    "SimulationConfig",
    "feature_matrix",
    "load_criteo",
    "simulate_experiment",
]
