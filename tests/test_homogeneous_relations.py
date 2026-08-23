import logging

import numpy as np
import pytest
import torch

from triangle_relations.discovery.homogeneous_relations import (
    HomogeneousRelationResult,
    embed_triple,
    log_euler_triple_rank,
    search_homogeneous_relations,
)
from triangle_relations.discovery.known_relations import is_euler_triple
from triangle_relations.discovery.shape_space import sample_shape_space
from triangle_relations.discovery.spherical_chart import (
    CANONICAL_POLE,
    forward_chart,
    inverse_chart_torch,
    orthonormal_frame,
)
from triangle_relations.discovery.sphere_autoencoder import sphere_reconstruction_error
from triangle_relations.geometry.triangle import Triangle


def test_scalar_degrees_are_correct_for_known_scalars():
    """Homogeneity degrees should match the theory: lengths=1, area=2."""
    assert Triangle.scalar_degree("area") == 2
    assert Triangle.scalar_degree("perimeter") == 1
    assert Triangle.scalar_degree("circumradius") == 1
    assert Triangle.scalar_degree("inradius") == 1
    # auto-registered pairwise point distance, added by
    # Triangle._register_pairwise_point_distances
    assert Triangle.scalar_degree("dist_circumcenter__incenter") == 1


def test_sample_shape_space_returns_nondegenerate_triangles():
    """Almost all requested lattice points should survive the degeneracy filter."""
    triangles = sample_shape_space(200)
    assert len(triangles) > 190
    assert all(t.area() > 0 for t in triangles)


def test_embed_triple_returns_unit_vectors_in_the_positive_octant():
    """The degree-equalized embedding should land on the unit sphere, all-positive."""
    triangles = sample_shape_space(100)
    points = embed_triple(
        triangles, ("circumradius", "inradius", "dist_circumcenter__incenter")
    )
    norms = np.linalg.norm(points, axis=1)
    assert norms == pytest.approx(1.0, abs=1e-6)
    assert np.all(points > 0)


def test_embed_triple_rejects_non_positive_degree_scalar(monkeypatch):
    """A degree-0 (or lower) scalar can't take a real d-th root, so it must be rejected."""
    monkeypatch.setitem(Triangle.SCALAR_DEGREES, "area", 0)
    triangles = sample_shape_space(50)
    with pytest.raises(ValueError):
        embed_triple(triangles, ("circumradius", "inradius", "area"))


def test_is_euler_triple_is_order_independent():
    assert is_euler_triple(("circumradius", "inradius", "dist_circumcenter__incenter"))
    assert is_euler_triple(("dist_circumcenter__incenter", "circumradius", "inradius"))
    assert not is_euler_triple(("circumradius", "inradius", "area"))


def test_log_euler_triple_rank_reports_correct_position(caplog):
    results = [
        HomogeneousRelationResult(names=("area", "perimeter", "circumradius"), degrees=(2, 1, 1), error=0.1),
        HomogeneousRelationResult(
            names=("circumradius", "inradius", "dist_circumcenter__incenter"),
            degrees=(1, 1, 1),
            error=0.3,
        ),
        HomogeneousRelationResult(names=("area", "circumradius", "inradius"), degrees=(2, 1, 1), error=0.9),
    ]
    with caplog.at_level(logging.INFO):
        log_euler_triple_rank(results)
    assert "#2 of 3" in caplog.text


def test_log_euler_triple_rank_handles_absence(caplog):
    results = [
        HomogeneousRelationResult(names=("area", "perimeter", "circumradius"), degrees=(2, 1, 1), error=0.1),
    ]
    with caplog.at_level(logging.INFO):
        log_euler_triple_rank(results)
    assert "not present" in caplog.text


def test_spherical_chart_round_trip():
    """Mapping a sphere point to chart coordinates and back should recover it exactly."""
    rng = np.random.default_rng(0)
    raw = rng.uniform(0.1, 1.0, size=(20, 3))
    points = raw / np.linalg.norm(raw, axis=1, keepdims=True)

    e1, e2 = orthonormal_frame(CANONICAL_POLE)
    chart = forward_chart(points, CANONICAL_POLE, e1, e2)

    recovered = inverse_chart_torch(
        torch.as_tensor(chart, dtype=torch.float32),
        torch.as_tensor(CANONICAL_POLE, dtype=torch.float32),
        torch.as_tensor(e1, dtype=torch.float32),
        torch.as_tensor(e2, dtype=torch.float32),
    ).numpy()

    assert recovered == pytest.approx(points, abs=1e-5)


def test_sphere_reconstruction_error_is_bounded():
    """Squared chordal error between two unit vectors is always in [0, 4]."""
    rng = np.random.default_rng(0)
    raw = rng.uniform(0.1, 1.0, size=(60, 3))
    points = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    error = sphere_reconstruction_error(
        points, hidden=6, n_restarts=1, test_size=0.3, n_epochs=50, lr=0.05, random_state=0
    )
    assert 0.0 <= error <= 4.0


def test_search_excludes_non_positive_degree_scalars(monkeypatch):
    """A scalar forced to degree 0 must never appear in any searched triple."""
    monkeypatch.setitem(Triangle.SCALAR_DEGREES, "area", 0)
    triangles = sample_shape_space(80)
    results = search_homogeneous_relations(
        triangles,
        ["circumradius", "inradius", "dist_circumcenter__incenter", "area"],
        hidden=6,
        n_restarts=1,
        test_size=0.3,
        n_epochs=50,
        lr=0.05,
        n_jobs=1,
        random_state=0,
        progress=False,
    )
    # with "area" excluded, only one combination of 3 remains: (R, r, OI)
    assert len(results) == 1
    assert all("area" not in r.names for r in results)


def test_homogeneous_search_ranks_euler_relation_first():
    """End-to-end check, mirroring test_scalar_relations.py's Program 1 test:
    among a small candidate set, (circumradius, inradius,
    dist_circumcenter__incenter) -- Euler's relation d^2 = R^2 - 2Rr -- should
    be flagged as by far the strongest candidate, with no permutation null.
    """
    triangles = sample_shape_space(300)
    scalars = ["circumradius", "inradius", "dist_circumcenter__incenter", "area"]

    results = search_homogeneous_relations(
        triangles,
        scalars,
        hidden=6,
        n_restarts=1,
        test_size=0.3,
        n_epochs=300,
        lr=0.05,
        n_jobs=1,
        random_state=0,
        progress=False,
    )

    assert set(results[0].names) == {
        "circumradius",
        "inradius",
        "dist_circumcenter__incenter",
    }
    # An exact relation should collapse to a near-zero reconstruction error;
    # empirically this stays below 0.0003 across seeds, so 0.001 has margin.
    assert results[0].error < 0.001
