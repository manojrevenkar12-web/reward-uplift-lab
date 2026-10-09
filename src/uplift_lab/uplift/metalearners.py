"""Meta-learners for the conditional average treatment effect (CATE).

A meta-learner turns ordinary supervised models into an estimate of how much the
treatment changes the outcome *for a user with features x*:
``tau(x) = E[Y | X=x, T=1] - E[Y | X=x, T=0]``. That is the quantity a targeting
policy needs; a plain response model only predicts who converts, not who converts
*because of* the reward.

Implemented learners (Künzel et al. 2019; Kennedy 2023):

* **S-learner**: one model with treatment as a feature. Simple, but regularisation
  can shrink small effects towards zero because the model may ignore the treatment.
* **T-learner**: separate models per arm; the effect is their difference. Errors of
  the two models do not cancel, so it is noisy when effects are small.
* **X-learner**: imputes individual effects with the opposite arm's model and
  regresses them on features; strong when one arm is much larger than the other.
* **DR-learner**: regresses a doubly robust pseudo-outcome built with cross-fitted
  nuisance models; the pseudo-outcome is unbiased for tau(x) if either the outcome
  models or the propensity are correct, which in a randomised test is guaranteed.

All learners take model *factories* so every fit gets a fresh, independent estimator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Protocol, Self, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold

FloatArray: TypeAlias = NDArray[np.float64]


class SupervisedModel(Protocol):
    """Minimal scikit-learn style estimator interface used by the meta-learners."""

    def fit(self, X: Any, y: Any) -> Any:
        """Fit the model."""
        ...

    def predict(self, X: Any) -> Any:
        """Predict outcomes."""
        ...


ModelFactory: TypeAlias = Callable[[], SupervisedModel]


def default_outcome_model() -> SupervisedModel:
    """Gradient-boosted classifier for binary outcomes, regularised for noisy targets."""
    model: SupervisedModel = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=200,
        l2_regularization=1.0,
        random_state=0,
    )
    return model


def default_effect_model() -> SupervisedModel:
    """Gradient-boosted regressor for (very noisy) imputed effects and pseudo-outcomes."""
    model: SupervisedModel = HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=500,
        l2_regularization=1.0,
        random_state=0,
    )
    return model


def predict_mean(model: SupervisedModel, X: ArrayLike) -> FloatArray:
    """Predict E[Y | X]: class-1 probability for classifiers, else the regression."""
    predict_proba = getattr(model, "predict_proba", None)
    if predict_proba is not None:
        return np.asarray(predict_proba(X)[:, 1], dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def _check_inputs(
    X: ArrayLike, t: ArrayLike, y: ArrayLike
) -> tuple[FloatArray, NDArray[np.bool_], FloatArray]:
    X_arr = np.asarray(X, dtype=float)
    t_arr = np.asarray(t)
    y_arr = np.asarray(y, dtype=float)
    if X_arr.ndim != 2 or X_arr.shape[0] != t_arr.size or t_arr.size != y_arr.size:
        raise ValueError("X must be 2-D with one row per element of t and y")
    if not np.isin(t_arr, (0, 1)).all():
        raise ValueError("treatment must be binary (0/1)")
    treated = t_arr.astype(bool)
    if treated.sum() < 10 or (~treated).sum() < 10:
        raise ValueError("each arm needs at least 10 users")
    return X_arr, treated, y_arr


class MetaLearner(ABC):
    """Common interface: fit on (X, t, y), then predict per-user uplift."""

    name: str = "meta_learner"

    def __init__(self) -> None:
        self._fitted = False

    @abstractmethod
    def _fit(self, X: FloatArray, treated: NDArray[np.bool_], y: FloatArray) -> None: ...

    @abstractmethod
    def _predict(self, X: FloatArray) -> FloatArray: ...

    def fit(self, X: ArrayLike, t: ArrayLike, y: ArrayLike) -> Self:
        """Fit the learner on features, binary treatment and outcome."""
        X_arr, treated, y_arr = _check_inputs(X, t, y)
        self._fit(X_arr, treated, y_arr)
        self._fitted = True
        return self

    def predict_uplift(self, X: ArrayLike) -> FloatArray:
        """Estimated treatment effect for each row of ``X``."""
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} must be fitted before predicting")
        return self._predict(np.asarray(X, dtype=float))


class SLearner(MetaLearner):
    """Single model ``f(x, t)``; uplift is ``f(x, 1) - f(x, 0)``."""

    name = "s_learner"

    def __init__(self, outcome_model: ModelFactory = default_outcome_model) -> None:
        super().__init__()
        self._factory = outcome_model

    def _fit(self, X: FloatArray, treated: NDArray[np.bool_], y: FloatArray) -> None:
        self._model = self._factory()
        self._model.fit(np.column_stack([X, treated.astype(float)]), y)

    def _predict(self, X: FloatArray) -> FloatArray:
        ones, zeros = np.ones((len(X), 1)), np.zeros((len(X), 1))
        return predict_mean(self._model, np.hstack([X, ones])) - predict_mean(
            self._model, np.hstack([X, zeros])
        )


class TLearner(MetaLearner):
    """Separate outcome models per arm; uplift is ``mu1(x) - mu0(x)``.

    Also exposes the per-arm predictions, which profit-aware targeting needs because
    a reward's cost depends on the treated conversion probability ``mu1(x)``.
    """

    name = "t_learner"

    def __init__(self, outcome_model: ModelFactory = default_outcome_model) -> None:
        super().__init__()
        self._factory = outcome_model

    def _fit(self, X: FloatArray, treated: NDArray[np.bool_], y: FloatArray) -> None:
        self._mu0 = self._factory()
        self._mu1 = self._factory()
        self._mu0.fit(X[~treated], y[~treated])
        self._mu1.fit(X[treated], y[treated])

    def predict_outcomes(self, X: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Predicted mean outcome under control and under treatment."""
        if not self._fitted:
            raise RuntimeError("TLearner must be fitted before predicting")
        X_arr = np.asarray(X, dtype=float)
        return predict_mean(self._mu0, X_arr), predict_mean(self._mu1, X_arr)

    def _predict(self, X: FloatArray) -> FloatArray:
        mu0, mu1 = self.predict_outcomes(X)
        return mu1 - mu0


class XLearner(MetaLearner):
    """X-learner with a propensity-weighted combination of the two effect models."""

    name = "x_learner"

    def __init__(
        self,
        outcome_model: ModelFactory = default_outcome_model,
        effect_model: ModelFactory = default_effect_model,
    ) -> None:
        super().__init__()
        self._outcome_factory = outcome_model
        self._effect_factory = effect_model

    def _fit(self, X: FloatArray, treated: NDArray[np.bool_], y: FloatArray) -> None:
        stage1 = TLearner(self._outcome_factory).fit(X, treated.astype(int), y)
        mu0, mu1 = stage1.predict_outcomes(X)
        d1 = y[treated] - mu0[treated]  # treated: observed minus imputed control
        d0 = mu1[~treated] - y[~treated]  # control: imputed treated minus observed
        self._tau1 = self._effect_factory()
        self._tau0 = self._effect_factory()
        self._tau1.fit(X[treated], d1)
        self._tau0.fit(X[~treated], d0)
        # Randomised experiment: the propensity is the constant assignment share.
        self._propensity = float(treated.mean())

    def _predict(self, X: FloatArray) -> FloatArray:
        g = self._propensity
        tau0 = np.asarray(self._tau0.predict(X), dtype=float)
        tau1 = np.asarray(self._tau1.predict(X), dtype=float)
        # Weight each arm's effect model by the *other* arm's share: the model trained
        # on the larger arm imputes with the better first-stage model.
        return np.asarray(g * tau0 + (1 - g) * tau1, dtype=float)


class DRLearner(MetaLearner):
    """Doubly robust learner with K-fold cross-fitting of the outcome models.

    The pseudo-outcome for each user is
    ``mu1 - mu0 + t (y - mu1) / e - (1 - t) (y - mu0) / (1 - e)``,
    with ``mu0``, ``mu1`` predicted by models that never saw that user, and ``e`` the
    assignment probability. Its conditional mean is exactly ``tau(x)``; a final
    regression on the features smooths it into a usable effect model.
    """

    name = "dr_learner"

    def __init__(
        self,
        outcome_model: ModelFactory = default_outcome_model,
        effect_model: ModelFactory = default_effect_model,
        n_folds: int = 5,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if n_folds < 2:
            raise ValueError("n_folds must be at least 2")
        self._outcome_factory = outcome_model
        self._effect_factory = effect_model
        self._n_folds = n_folds
        self._seed = seed

    def pseudo_outcomes(
        self, X: FloatArray, treated: NDArray[np.bool_], y: FloatArray
    ) -> FloatArray:
        """Cross-fitted doubly robust pseudo-outcomes (one per user)."""
        e = float(treated.mean())
        mu0, mu1 = np.empty_like(y), np.empty_like(y)
        folds = KFold(self._n_folds, shuffle=True, random_state=self._seed)
        for train, held_out in folds.split(X):
            nuisance = TLearner(self._outcome_factory).fit(
                X[train], treated[train].astype(int), y[train]
            )
            mu0[held_out], mu1[held_out] = nuisance.predict_outcomes(X[held_out])
        t = treated.astype(float)
        return np.asarray(
            mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e), dtype=float
        )

    def _fit(self, X: FloatArray, treated: NDArray[np.bool_], y: FloatArray) -> None:
        self._final = self._effect_factory()
        self._final.fit(X, self.pseudo_outcomes(X, treated, y))

    def _predict(self, X: FloatArray) -> FloatArray:
        return np.asarray(self._final.predict(X), dtype=float)
