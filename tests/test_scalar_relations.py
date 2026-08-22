import numpy as np

from triangle_relations.discovery.sampling import build_scalar_dataset
from triangle_relations.discovery.scalar_relations import search_three_scalar_relations
from triangle_relations.discovery.symbolic import fit_polynomial_relation


def test_discovery_pipeline_ranks_euler_relation_first():
    """End-to-end check: among a small candidate set of scalars, the pipeline
    should rank (circumradius, inradius, dist_circumcenter__incenter) -- i.e.
    R, r, d from Euler's relation d^2 = R^2 - 2Rr -- as the strongest triple.
    """
    rng = np.random.default_rng(0)
    scalars = ["circumradius", "inradius", "dist_circumcenter__incenter", "area"]
    names, data = build_scalar_dataset(300, rng, scalar_names=scalars)

    results = search_three_scalar_relations(
        names,
        data,
        n_shuffles=2,
        hidden=8,
        n_restarts=1,
        n_jobs=1,
        random_state=0,
    )

    assert set(results[0].names) == {
        "circumradius",
        "inradius",
        "dist_circumcenter__incenter",
    }
    # The Euler triple should reconstruct far better than its shuffled null.
    assert results[0].ratio < 0.5


def test_symbolic_fit_recovers_euler_relation():
    rng = np.random.default_rng(0)
    names, data = build_scalar_dataset(
        200, rng, scalar_names=["circumradius", "inradius", "dist_circumcenter__incenter"]
    )

    relation = fit_polynomial_relation(data, tuple(names), max_degree=2)

    # The fitted relation should vanish (near machine precision) on the data
    # that produced it, regardless of which particular scaling/sign SVD picks.
    residual = relation.residual(data)
    scale = np.max(np.abs(data)) ** 2
    assert np.max(np.abs(residual)) < 1e-6 * scale
