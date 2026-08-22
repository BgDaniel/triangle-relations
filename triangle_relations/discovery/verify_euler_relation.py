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

import numpy as np

from triangle_relations.discovery.sampling import build_scalar_dataset
from triangle_relations.discovery.scalar_relations import (
    search_three_scalar_relations,
    shuffle_columns,
)
from triangle_relations.discovery.symbolic import fit_polynomial_relation

SCALAR_NAMES = ["circumradius", "inradius", "dist_circumcenter__incenter"]


def main() -> None:
    rng = np.random.default_rng(0)
    print(f"Sampling random triangles and evaluating {SCALAR_NAMES} ...")
    names, data = build_scalar_dataset(1000, rng, scalar_names=SCALAR_NAMES)
    R, r, d = data[:, 0], data[:, 1], data[:, 2]

    # Sanity check: Euler's relation should already hold exactly on our own
    # geometry engine, independent of anything the discovery pipeline does.
    residual = d**2 - (R**2 - 2 * R * r)
    print(
        f"\n[ground truth] max |d^2 - (R^2 - 2Rr)| over samples: "
        f"{np.max(np.abs(residual)):.3e}  (should be ~0)"
    )

    # --- Step 1: does the autoencoder/permutation-null test flag this triple? ---
    print("\n[detection] running autoencoder vs. shuffled-null test on (R, r, d)...")
    results = search_three_scalar_relations(
        names, data, n_shuffles=5, n_restarts=2, n_jobs=1, random_state=0
    )
    result = results[0]
    print(f"  real reconstruction error : {result.real_error:.4g}")
    print(f"  null mean +- std          : {result.null_mean:.4g} +- {result.null_std:.2g}")
    print(f"  z-score                   : {result.z_score:.2f}")
    print(f"  ratio (real / null_mean)  : {result.ratio:.4f}  (small = strong relation)")

    # --- Step 2: recover an explicit closed form ---
    print("\n[symbolic fit] searching for a degree-2 polynomial relation...")
    relation = fit_polynomial_relation(data, tuple(names), max_degree=2)
    print(f"  smallest/largest singular value ratio: {relation.singular_value_ratio:.2e}")
    print(f"  recovered relation (= 0): {relation.as_expr()}")

    # --- Step 3: plots -- real data (a surface) vs. shuffled null (a volume) ---
    _plot_surface_vs_volume(data, names, rng)


def _trim_outliers(X: np.ndarray, q: float = 0.97) -> np.ndarray:
    """Boolean mask keeping rows within the q-quantile on every column.

    Random-triangle sampling occasionally yields near-degenerate triangles
    with huge circumradius; a handful of such outliers would otherwise
    dominate the plot's axis ranges and hide the surface/volume shape in the
    typical-scale region.
    """
    thresholds = np.quantile(X, q, axis=0)
    return np.all(X <= thresholds, axis=1)


def _plot_surface_vs_volume(data: np.ndarray, names: list[str], rng: np.random.Generator) -> None:
    import matplotlib.pyplot as plt

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
    print("\nSaved figure to euler_relation_check.png")
    plt.show()


if __name__ == "__main__":
    main()
