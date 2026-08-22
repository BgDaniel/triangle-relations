"""Bottleneck-autoencoder reconstruction error, used as a dependence probe.

If three scalar quantities are functionally dependent (they lie on a 2D
surface within the 3D space of possible values), an autoencoder with a
2-dimensional latent bottleneck should be able to reconstruct them with much
smaller error than it can for three quantities that are locally independent
functions of the triangle's three degrees of freedom.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def reconstruction_error(
    X: np.ndarray,
    *,
    bottleneck: int = 2,
    hidden: int = 8,
    test_size: float = 0.3,
    n_restarts: int = 1,
    max_iter: int = 500,
    random_state: int | np.random.Generator | None = None,
) -> float:
    """Mean squared reconstruction error of a bottleneck autoencoder on ``X``.

    ``X`` is standardized (zero mean, unit variance per column) before
    training, so the returned error is in standardized units and comparable
    across different scalar triples regardless of their physical scale.
    Trains ``n_restarts`` networks with different random initializations and
    keeps the best (lowest test error) to reduce sensitivity to local minima.
    """
    rng = np.random.default_rng(random_state)
    X_train, X_test = train_test_split(
        X, test_size=test_size, random_state=int(rng.integers(2**32))
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    best_error = np.inf
    with warnings.catch_warnings():
        # Reaching max_iter without full convergence is expected: we cap
        # training cost deliberately and compare errors relatively (real vs.
        # shuffled-null), not against an absolute convergence criterion.
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for _ in range(n_restarts):
            model = MLPRegressor(
                hidden_layer_sizes=(hidden, bottleneck, hidden),
                activation="tanh",
                solver="adam",
                max_iter=max_iter,
                random_state=int(rng.integers(2**32)),
            )
            model.fit(X_train_s, X_train_s)
            pred = model.predict(X_test_s)
            error = float(np.mean((pred - X_test_s) ** 2))
            best_error = min(best_error, error)

    return best_error
