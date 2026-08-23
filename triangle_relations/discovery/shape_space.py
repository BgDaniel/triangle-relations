"""Sample the space of triangle *shapes* evenly, via an explicit chart.

A triangle up to translation, rotation, and uniform scaling ("shape alone")
is classically parametrized by a single complex number: normalize two of
its vertices to ``0`` and ``1`` by a (unique) orientation-preserving
similarity, and the third vertex lands at some
``zeta = (C - A) / (B - A) in C``. This *is* a chart on the 2-dimensional
space of triangle shapes (topologically a sphere, by a classical theorem of
Kendall (1984); see ``docs/discovering_triangle_relations.tex``, Section 6):
it covers every shape except the single degenerate configuration ``B == A``
(a single point, i.e. measure zero), and every ``zeta`` gives back a
concrete, evaluable triangle, ``Triangle((0, 0), (1, 0), (zeta.real, zeta.imag))``.

Sampling ``zeta`` uniformly on ``C`` would not evenly cover shape space --
the chart stretches near infinity -- so instead we place points evenly on
the unit sphere (a Fibonacci lattice, a standard low-discrepancy
construction) and push them through the *inverse* of this chart
(stereographic projection), giving deterministic, even coverage of shape
space without relying on random sampling to eventually fill it in.
"""

from __future__ import annotations

import logging

import numpy as np

from triangle_relations.geometry.triangle import Triangle

logger = logging.getLogger(__name__)

#: Below this ratio of (2*area) to (perimeter^2), a lattice-derived triangle
#: is considered too close to degenerate (near-collinear) and is dropped;
#: see triangle_relations.discovery.sampling._MIN_SHAPE_RATIO, the same
#: check used for i.i.d.-sampled triangles.
_MIN_SHAPE_RATIO = 1e-3


def _fibonacci_sphere(n: int) -> np.ndarray:
    """``n`` points spread evenly over the unit sphere ``S^2``, as an ``(n, 3)`` array.

    Standard Fibonacci-lattice construction: polar angle evenly spaced in
    ``cos(theta)`` (so the points are evenly spread by *area*, not by
    angle), azimuthal angle advanced by the golden angle each step. Never
    places a point exactly at either pole.
    """
    i = np.arange(n)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - (2.0 * i + 1.0) / n
    radius = np.sqrt(np.clip(1.0 - z**2, 0.0, None))
    theta = golden_angle * i
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.stack([x, y, z], axis=1)


def _stereographic_chart(sphere_points: np.ndarray) -> np.ndarray:
    """Project ``(n, 3)`` unit-sphere points to ``(n, 2)`` chart coordinates ``(u, v)``.

    Stereographic projection from the north pole ``(0, 0, 1)`` onto the
    ``z = 0`` plane: well-defined everywhere except at the pole itself
    (never hit by :func:`_fibonacci_sphere`, by construction).
    """
    x, y, z = sphere_points[:, 0], sphere_points[:, 1], sphere_points[:, 2]
    denom = 1.0 - z
    return np.stack([x / denom, y / denom], axis=1)


def _shape_ratio(u: float, v: float) -> float:
    """``2 * area / perimeter**2`` for ``Triangle((0, 0), (1, 0), (u, v))``, without building it.

    ``AB`` has fixed length 1, so this reduces to a direct function of
    ``(u, v)``; used to cheaply filter near-degenerate lattice points before
    constructing (and evaluating scalars on) the triangle itself.
    """
    area = abs(v) / 2.0
    perimeter = 1.0 + np.hypot(u, v) + np.hypot(u - 1.0, v)
    return (2.0 * area) / perimeter**2 if perimeter > 0 else 0.0


def sample_shape_space(n_samples: int) -> list[Triangle]:
    """Deterministically sample ``n_samples`` triangles evenly covering shape space.

    Unlike :func:`triangle_relations.discovery.sampling.random_triangle`,
    this takes no random generator: coverage comes from an even lattice on
    the shape sphere (see the module docstring), not from i.i.d. sampling,
    so the result is reproducible and does not rely on the sample size being
    "large enough" to fill in gaps. Each returned triangle has vertices
    ``(0, 0)``, ``(1, 0)``, and a third vertex determined by the lattice --
    the fixed base edge only fixes an (irrelevant, for homogeneous scalars)
    representative size and position for each shape, not the shape itself.

    A small number of lattice points landing extremely close to the
    degenerate (collinear) locus are dropped (logged at debug level) rather
    than resampled, since the lattice is deterministic; this trims at most a
    handful of points out of ``n_samples``.
    """
    sphere_points = _fibonacci_sphere(n_samples)
    chart = _stereographic_chart(sphere_points)

    triangles = []
    n_dropped = 0
    for u, v in chart:
        if _shape_ratio(u, v) <= _MIN_SHAPE_RATIO:
            n_dropped += 1
            continue
        triangles.append(Triangle((0.0, 0.0), (1.0, 0.0), (u, v)))

    if n_dropped:
        logger.debug("dropped %d near-degenerate lattice point(s) out of %d", n_dropped, n_samples)
    logger.info("sampled %d triangles evenly covering shape space", len(triangles))
    return triangles
