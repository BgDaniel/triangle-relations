import numpy as np
import pytest

from triangle_relations import Triangle


def test_area_perimeter_3_4_5():
    t = Triangle((0, 0), (4, 0), (0, 3))
    assert t.area() == pytest.approx(6.0)
    assert t.perimeter() == pytest.approx(12.0)
    assert t.side_lengths() == pytest.approx((5.0, 3.0, 4.0))


def test_right_triangle_circumradius_is_half_hypotenuse():
    t = Triangle((0, 0), (4, 0), (0, 3))
    # hypotenuse is side 'a' (opposite vertex A), length 5
    assert t.circumradius() == pytest.approx(2.5)


def test_right_triangle_inradius():
    t = Triangle((0, 0), (4, 0), (0, 3))
    # for a right triangle, r = (leg1 + leg2 - hypotenuse) / 2
    assert t.inradius() == pytest.approx((3 + 4 - 5) / 2)


def test_equilateral_triangle_centers_coincide():
    t = Triangle((0, 1), (-np.sqrt(3) / 2, -0.5), (np.sqrt(3) / 2, -0.5))
    centroid = t.centroid()
    incenter = t.incenter()
    circumcenter = t.circumcenter()
    orthocenter = t.orthocenter()
    assert incenter == pytest.approx(centroid, abs=1e-9)
    assert circumcenter == pytest.approx(centroid, abs=1e-9)
    assert orthocenter == pytest.approx(centroid, abs=1e-9)
    assert t.inradius() == pytest.approx(t.circumradius() / 2)


def test_equilateral_triangle_steiner_foci_coincide_at_centroid():
    t = Triangle((0, 1), (-np.sqrt(3) / 2, -0.5), (np.sqrt(3) / 2, -0.5))
    f1, f2 = t.steiner_inellipse_foci()
    assert f1 == pytest.approx(t.centroid(), abs=1e-6)
    assert f2 == pytest.approx(t.centroid(), abs=1e-6)


@pytest.mark.parametrize("seed", range(20))
def test_euler_relation_holds_on_random_triangles(seed):
    rng = np.random.default_rng(seed)
    verts = rng.uniform(-1, 1, size=(3, 2))
    t = Triangle(*verts)
    if t.area() < 1e-6:
        pytest.skip("near-degenerate triangle")

    R = t.circumradius()
    r = t.inradius()
    d = np.linalg.norm(t.incenter() - t.circumcenter())

    assert d**2 == pytest.approx(R**2 - 2 * R * r, abs=1e-8)


def test_euler_line_collinearity_holds_on_random_triangles():
    rng = np.random.default_rng(1)
    verts = rng.uniform(-1, 1, size=(3, 2))
    t = Triangle(*verts)

    o = t.circumcenter()
    g = t.centroid()
    h = t.orthocenter()

    # Euler line: H = 3G - 2O
    assert h == pytest.approx(3 * g - 2 * o, abs=1e-8)


def test_pairwise_point_distances_are_registered_scalars():
    assert "dist_centroid__circumcenter" in Triangle.SCALARS
    assert "dist_circumcenter__incenter" in Triangle.SCALARS
    t = Triangle((0, 0), (4, 0), (0, 3))
    d = t.scalar("dist_circumcenter__incenter")
    expected = np.linalg.norm(t.circumcenter() - t.incenter())
    assert d == pytest.approx(expected)


def test_all_scalars_and_points_are_computable():
    t = Triangle((0.1, 0.2), (3.3, -1.1), (-2.0, 2.5))
    scalars = t.all_scalars()
    points = t.all_points()
    assert len(scalars) == len(Triangle.SCALARS)
    assert len(points) == len(Triangle.POINTS)
    assert all(np.isfinite(v) for v in scalars.values())
    assert all(np.all(np.isfinite(p)) for p in points.values())
