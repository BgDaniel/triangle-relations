"""Program 1: search for functional relations among triples of scalars.

A triangle has three degrees of freedom, so a relation among only three
derived scalar quantities (like Euler's d^2 = R^2 - 2Rr) is not expected
generically -- unlike a relation among four or more quantities, which is
guaranteed by dimension counting. This module searches all combinations of
three scalar quantities and flags those whose joint distribution collapses
onto a 2-dimensional surface far more than a permutation-shuffled null would
predict.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from joblib import Parallel, delayed

from triangle_relations.discovery.autoencoder import reconstruction_error


@dataclass
class RelationResult:
    names: tuple[str, str, str]
    real_error: float
    null_mean: float
    null_std: float
    z_score: float

    @property
    def ratio(self) -> float:
        """real_error / null_mean; smaller means a stronger relation."""
        return self.real_error / self.null_mean if self.null_mean > 0 else np.inf


def _shuffle_columns(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Independently permute each column, destroying joint structure while
    preserving each quantity's own marginal distribution exactly."""
    X_shuffled = np.empty_like(X)
    for j in range(X.shape[1]):
        X_shuffled[:, j] = rng.permutation(X[:, j])
    return X_shuffled


def _evaluate_triple(
    X: np.ndarray,
    names: tuple[str, str, str],
    *,
    n_shuffles: int,
    hidden: int,
    n_restarts: int,
    test_size: float,
    seed: int,
) -> RelationResult:
    rng = np.random.default_rng(seed)

    real_error = reconstruction_error(
        X,
        hidden=hidden,
        n_restarts=n_restarts,
        test_size=test_size,
        random_state=rng,
    )

    null_errors = []
    for _ in range(n_shuffles):
        X_shuffled = _shuffle_columns(X, rng)
        null_errors.append(
            reconstruction_error(
                X_shuffled,
                hidden=hidden,
                n_restarts=n_restarts,
                test_size=test_size,
                random_state=rng,
            )
        )
    null_mean = float(np.mean(null_errors))
    null_std = float(np.std(null_errors))
    z_score = (null_mean - real_error) / null_std if null_std > 0 else np.inf

    return RelationResult(
        names=names,
        real_error=real_error,
        null_mean=null_mean,
        null_std=null_std,
        z_score=z_score,
    )


def search_three_scalar_relations(
    names: list[str],
    data: np.ndarray,
    *,
    n_shuffles: int = 3,
    hidden: int = 8,
    n_restarts: int = 1,
    test_size: float = 0.3,
    min_std: float = 1e-9,
    n_jobs: int = 1,
    random_state: int | None = None,
) -> list[RelationResult]:
    """Search all combinations of three scalars in ``data`` for a hidden relation.

    Parameters
    ----------
    names, data:
        As returned by :func:`triangle_relations.discovery.sampling.build_scalar_dataset`.
    n_shuffles:
        Number of independent-column-shuffle nulls to average per triple.
    min_std:
        Columns with standard deviation below this (near-constant scalars,
        e.g. degenerate quantities) are skipped.
    n_jobs:
        Passed to :class:`joblib.Parallel`; -1 uses all cores.

    Returns
    -------
    Results sorted by ascending ``ratio`` (strongest candidate relation first).
    """
    valid_idx = [j for j in range(data.shape[1]) if np.std(data[:, j]) > min_std]
    if len(valid_idx) < len(names):
        dropped = sorted(set(range(len(names))) - set(valid_idx))
        print(
            "warning: skipping near-constant scalars: "
            + ", ".join(names[j] for j in dropped)
        )

    triples = list(combinations(valid_idx, 3))
    seed_seq = np.random.SeedSequence(random_state)
    seeds = seed_seq.generate_state(len(triples))

    results = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_triple)(
            data[:, [i, j, k]],
            (names[i], names[j], names[k]),
            n_shuffles=n_shuffles,
            hidden=hidden,
            n_restarts=n_restarts,
            test_size=test_size,
            seed=int(seed),
        )
        for (i, j, k), seed in zip(triples, seeds)
    )

    return sorted(results, key=lambda r: r.ratio)
