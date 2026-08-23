"""Visually inspect one candidate homogeneous triple's image on the target sphere.

A companion to the quantitative test in
:func:`~triangle_relations.discovery.homogeneous_relations.search_homogeneous_relations`:
that function reports a single reconstruction-error number per triple, which
is what the search ranks by, but a picture is a useful sanity check on a
promising candidate before trusting the number. This module samples shape
space, embeds a given triple the same way the search does (degree-equalized,
unit-normalized -- see :mod:`triangle_relations.discovery.homogeneous_relations`),
and projects that embedding down to a 2D plot via a coordinate chart, so it
can actually be drawn.

If the three scalars satisfy an exact homogeneous relation, the sampled
points should visibly collapse onto a thin 1-dimensional curve; if not, they
should fill an open 2-dimensional patch. This is exactly the dichotomy
Section 6 of ``docs/discovering_triangle_relations.tex`` describes for the
map ``Phi_bar: Sigma -> S^2_+``, made visible.

The chart used here is *not* fixed in advance (e.g. "always divide by the
third coordinate"): it is a stereographic projection whose pole is chosen
per triple, antipodal to that triple's own sampled points' mean direction,
so the chart's one excluded point sits as far as possible from the actual
data and distorts it as little as possible. This is the generic-chart
argument from Section 6's "Why a generic chart is safe" remark, applied
concretely: any reasonable chart would show the same qualitative picture
(curve vs. patch), this one is just chosen to look good for whichever
triple happens to be passed in.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np

from triangle_relations.discovery.homogeneous_relations import embed_triple
from triangle_relations.discovery.shape_space import sample_shape_space
from triangle_relations.discovery.spherical_chart import forward_chart, orthonormal_frame
from triangle_relations.geometry.triangle import Triangle

logger = logging.getLogger(__name__)


def _generic_chart(points: np.ndarray) -> np.ndarray:
    """Stereographically project unit vectors ``points`` (shape ``(n, 3)``) to 2D.

    The projection pole is the point antipodal to ``points``' own mean
    direction, so it (and the distortion that blows up near it) stays as
    far from the data as possible -- see the module docstring. Shares its
    underlying chart construction with
    :mod:`triangle_relations.discovery.sphere_autoencoder`, which instead
    uses a fixed pole -- see :mod:`triangle_relations.discovery.spherical_chart`.
    """
    mean_direction = points.mean(axis=0)
    mean_direction /= np.linalg.norm(mean_direction)
    pole = -mean_direction
    e1, e2 = orthonormal_frame(pole)
    return forward_chart(points, pole, e1, e2)


def plot_relation_image(
    names: tuple[str, str, str],
    *,
    n_samples: int = 4000,
    title: str | None = None,
) -> plt.Figure:
    """Plot the image of one candidate triple's degree-equalized embedding, via a generic chart.

    A curve means the three scalars satisfy an exact homogeneous relation;
    a filled 2D patch means they don't. See the module docstring.

    Parameters
    ----------
    names:
        The three scalar names to test (each must have positive
        homogeneity degree; see :attr:`Triangle.SCALAR_DEGREES`).
    n_samples:
        Number of triangles to sample, evenly, from shape space (see
        :func:`~triangle_relations.discovery.shape_space.sample_shape_space`).
    title:
        Plot title; defaults to a description built from ``names`` and
        their short symbols (see :attr:`Triangle.SCALAR_SYMBOLS`).

    Returns
    -------
    The matplotlib ``Figure``.
    """
    triangles = sample_shape_space(n_samples)
    points = embed_triple(triangles, names)
    chart = _generic_chart(points)

    symbols = ", ".join(Triangle.scalar_symbol(n) for n in names)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(chart[:, 0], chart[:, 1], s=4, alpha=0.5, color="tab:blue")
    ax.set_aspect("equal")
    ax.set_xlabel("chart coordinate u")
    ax.set_ylabel("chart coordinate v")
    ax.set_title(
        title
        or f"Image of ({symbols}) under a generic chart\n"
        "a curve means a relation; a filled patch means none"
    )
    logger.info("%s: plotted %d points", names, len(chart))
    return fig


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    logger.info("known relation -- Euler's (R, r, OI): expect a 1D curve")
    fig_euler = plot_relation_image(
        ("circumradius", "inradius", "dist_circumcenter__incenter"),
    )
    fig_euler.savefig("euler_relation_image.png", dpi=150)
    logger.info("saved euler_relation_image.png")

    logger.info("generic (unrelated) triple, for contrast: expect a filled 2D patch")
    fig_generic = plot_relation_image(
        ("inradius", "dist_circumcenter__incenter", "dist_centroid__orthocenter"),
    )
    fig_generic.savefig("generic_triple_image.png", dpi=150)
    logger.info("saved generic_triple_image.png")

    plt.show()
