"""Worked example: apply the full Program 1 pipeline to a *known* relation.

Euler's relation, d^2 = R^2 - 2Rr (d = distance between incenter and
circumcenter), is a genuine functional dependency among exactly three
derived scalars. This script is a sanity check on the method itself: run the
autoencoder/permutation-null detector on (R, r, d), confirm it is flagged as
a strong candidate, fit an explicit polynomial relation, and check it against
Euler's formula. It also plots the raw 3D point cloud of (R, r, d) next to a
column-shuffled null, to make the "surface vs. volume" argument visible.

Run with:
    poetry run python -m triangle_relations.discovery.verify_euler_relation
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np

from triangle_relations.discovery.sampling import build_scalar_dataset
from triangle_relations.discovery.scalar_relations import (
    RelationResult,
    search_three_scalar_relations,
    shuffle_columns,
)
from triangle_relations.discovery.symbolic import PolynomialRelation, fit_polynomial_relation

logger = logging.getLogger(__name__)

#: The scalar triple to test: Euler's relation is a functional dependency
#: among exactly these three quantities.
SCALAR_NAMES = ["circumradius", "inradius", "dist_circumcenter__incenter"]

#: Number of random triangles to sample.
N_SAMPLES = 1000

#: Number of column-shuffled null datasets averaged per detection test.
N_SHUFFLES = 5

#: Autoencoder random restarts per training run (real data and each null).
N_RESTARTS = 2

#: Fixed seed, for reproducible sampling and null shuffles.
SEED = 0


def main() -> None:
    """Run the full pipeline on Euler's relation and report + plot the result."""
    rng = np.random.default_rng(SEED)
    logger.info("sampling random triangles and evaluating %s", SCALAR_NAMES)
    names, data = build_scalar_dataset(N_SAMPLES, rng, scalar_names=SCALAR_NAMES)

    _check_ground_truth(data)
    result = _run_detection(names, data)
    relation = _fit_symbolic_relation(names, data)
    _plot_surface_vs_volume(data, names, rng)

    logger.info(
        "summary: ratio=%.4f z=%.2f recovered relation (= 0): %s",
        result.ratio, result.z_score, relation.as_expr(),
    )


def _check_ground_truth(data: np.ndarray) -> None:
    """Log the residual of Euler's formula on our own geometry engine.

    This is independent of the discovery pipeline: it should already be
    (numerically) zero, since it is just re-evaluating a known formula on
    the quantities our :class:`Triangle` class computes.
    """
    R, r, d = data[:, 0], data[:, 1], data[:, 2]
    residual = d**2 - (R**2 - 2 * R * r)
    logger.info(
        "[ground truth] max |d^2 - (R^2 - 2Rr)| over samples: %.3e (should be ~0)",
        np.max(np.abs(residual)),
    )


def _run_detection(names: list[str], data: np.ndarray) -> RelationResult:
    """Run the autoencoder/permutation-null detector and log the result for (R, r, d)."""
    logger.info("[detection] running autoencoder vs. shuffled-null test on (R, r, d)...")
    results = search_three_scalar_relations(
        names, data, n_shuffles=N_SHUFFLES, n_restarts=N_RESTARTS, n_jobs=1, random_state=SEED
    )
    result = results[0]
    logger.info("  real reconstruction error : %.4g", result.real_error)
    logger.info("  null mean +- std          : %.4g +- %.2g", result.null_mean, result.null_std)
    logger.info("  z-score                   : %.2f", result.z_score)
    logger.info("  ratio (real / null_mean)  : %.4f  (small = strong relation)", result.ratio)
    return result


def _fit_symbolic_relation(names: list[str], data: np.ndarray) -> PolynomialRelation:
    """Fit and log the explicit degree-2 polynomial relation among ``names``."""
    logger.info("[symbolic fit] searching for a degree-2 polynomial relation...")
    relation = fit_polynomial_relation(data, tuple(names), max_degree=2)
    logger.info("  smallest/largest singular value ratio: %.2e", relation.singular_value_ratio)
    logger.info("  recovered relation (= 0): %s", relation.as_expr())
    return relation


def _trim_outliers(X: np.ndarray, q: float = 0.97) -> np.ndarray:
    """Return a boolean mask keeping rows within the ``q``-quantile on every column.

    Random-triangle sampling occasionally yields near-degenerate triangles
    with huge circumradius; a handful of such outliers would otherwise
    dominate the plot's axis ranges and hide the surface/volume shape in the
    typical-scale region.
    """
    thresholds = np.quantile(X, q, axis=0)
    return np.all(X <= thresholds, axis=1)


def _plot_surface_vs_volume(data: np.ndarray, names: list[str], rng: np.random.Generator) -> None:
    """Save (and show) a 3D scatter plot contrasting the real data against a shuffled null.

    The real (R, r, d) triple should visibly collapse onto a thin 2D
    surface, while the column-shuffled null should fill the full 3D volume.
    """
    shuffled = shuffle_columns(data, rng)

    data_plot = data[_trim_outliers(data)]
    shuffled_plot = shuffled[_trim_outliers(shuffled)]

    fig = plt.figure(figsize=(11, 5))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.scatter(data_plot[:, 0], data_plot[:, 1], data_plot[:, 2], s=6, alpha=0.6, color="tab:blue")
    ax1.set_xlabel(names[0])
    ax1.set_ylabel(names[1])
    ax1.set_zlabel(names[2])
    ax1.set_title("Real (R, r, d)\nconfined to a 2D surface")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.scatter(shuffled_plot[:, 0], shuffled_plot[:, 1], shuffled_plot[:, 2], s=6, alpha=0.6, color="tab:red")
    ax2.set_xlabel(names[0])
    ax2.set_ylabel(names[1])
    ax2.set_zlabel(names[2])
    ax2.set_title("Column-shuffled null\nfills the 3D volume")

    fig.suptitle("Euler's relation d^2 = R^2 - 2Rr as a 2D surface in (R, r, d)-space")
    fig.tight_layout()
    fig.savefig("euler_relation_check.png", dpi=150)
    logger.info("saved figure to euler_relation_check.png")
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
