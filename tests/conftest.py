from __future__ import annotations

import pytest

from uplift_lab.data import SimulatedExperiment, SimulationConfig, simulate_experiment


@pytest.fixture(scope="session")
def experiment() -> SimulatedExperiment:
    """A mid-sized simulated experiment shared by tests that only read it."""
    return simulate_experiment(SimulationConfig(n_users=40_000, seed=11))
