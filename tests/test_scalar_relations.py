import numpy as np

from triangle_relations.discovery.sampling import build_scalar_dataset
from triangle_relations.discovery.scalar_relations import search_three_scalar_relations


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
