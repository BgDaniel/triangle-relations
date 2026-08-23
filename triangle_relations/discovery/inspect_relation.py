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
Section 5 of ``docs/discovering_triangle_relations.tex`` describes for the
map ``Phi_bar: Sigma -> S^2_+``, made visible.

The chart used here is *not* fixed in advance (e.g. "always divide by the
third coordinate"): it is a stereographic projection whose pole is chosen
per triple, antipodal to that triple's own sampled points' mean direction,
so the chart's one excluded point sits as far as possible from the actual
data and distorts it as little as possible. This is the generic-chart
argument from Section 5's "Why a generic chart is safe" remark, applied
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
    logger.info("(%s): plotted %d points", symbols, len(chart))
    return fig


def plot_relation_images_comparison(
    names_a: tuple[str, str, str],
    names_b: tuple[str, str, str],
    *,
    n_samples: int = 4000,
    title_a: str | None = None,
    title_b: str | None = None,
) -> plt.Figure:
    """Plot two candidate triples' images side by side, on shared axes, for direct comparison.

    Like :func:`plot_relation_image`, but both triples are drawn in one
    figure with identical x/y limits (the combined extent of both point
    sets, in each triple's own chart coordinates), rather than each
    plot auto-scaling to its own data independently. That matters
    precisely when the two triples differ hugely in spread -- e.g.
    comparing a search's best candidate against its worst: with
    independent axes, both would look like "a small cluster" purely from
    auto-scaling, hiding just how much tighter one is. On a shared scale, a
    dramatically stronger candidate can end up looking like a single point
    next to the other's filled patch -- which is itself the honest,
    informative picture, not a plotting artifact.

    Parameters
    ----------
    names_a, names_b:
        The two triples to compare (each must have positive homogeneity
        degree; see :attr:`Triangle.SCALAR_DEGREES`).
    n_samples:
        Number of triangles to sample, evenly, from shape space, for each
        triple independently.
    title_a, title_b:
        Per-panel titles; default to a description built from the
        corresponding ``names`` and their short symbols.

    Returns
    -------
    The matplotlib ``Figure`` containing both panels.
    """
    triangles_a = sample_shape_space(n_samples)
    chart_a = _generic_chart(embed_triple(triangles_a, names_a))

    triangles_b = sample_shape_space(n_samples)
    chart_b = _generic_chart(embed_triple(triangles_b, names_b))

    all_u = np.concatenate([chart_a[:, 0], chart_b[:, 0]])
    all_v = np.concatenate([chart_a[:, 1], chart_b[:, 1]])
    u_pad = 0.05 * (all_u.max() - all_u.min()) or 1.0
    v_pad = 0.05 * (all_v.max() - all_v.min()) or 1.0
    xlim = (all_u.min() - u_pad, all_u.max() + u_pad)
    ylim = (all_v.min() - v_pad, all_v.max() + v_pad)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 5.5), constrained_layout=True)
    for ax, chart, names, title in (
        (ax_a, chart_a, names_a, title_a),
        (ax_b, chart_b, names_b, title_b),
    ):
        symbols = ", ".join(Triangle.scalar_symbol(n) for n in names)
        ax.scatter(chart[:, 0], chart[:, 1], s=4, alpha=0.5, color="tab:blue")
        ax.set_aspect("equal")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("chart coordinate u")
        ax.set_ylabel("chart coordinate v")
        ax.set_title(title or f"({symbols})", fontsize=10)

    fig.suptitle("Same chart-coordinate axes for both panels -- directly comparable spread")
    logger.info(
        "(%s) vs (%s): shared axes u in [%.3g, %.3g], v in [%.3g, %.3g]",
        ", ".join(Triangle.scalar_symbol(n) for n in names_a),
        ", ".join(Triangle.scalar_symbol(n) for n in names_b),
        *xlim, *ylim,
    )
    return fig


#: Current best and worst triples from a homogeneous search (see
#: scripts/discover_homogeneous_relations.py / plot_homogeneous_ranking.py's
#: "best"/"worst" log lines) -- update these whenever a new search changes
#: which triples sit at either extreme, so this script always inspects
#: today's actual best/worst rather than a stale example.
_BEST_TRIPLE = ("dist_centroid__circumcenter", "dist_centroid__orthocenter", "dist_circumcenter__orthocenter")
_WORST_TRIPLE = ("dist_incenter__orthocenter", "dist_incenter__steiner_focus_1", "dist_incenter__steiner_focus_2")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    logger.info("comparing current best and worst triples on shared axes")
    fig = plot_relation_images_comparison(
        _BEST_TRIPLE, _WORST_TRIPLE,
        title_a="best", title_b="worst",
    )
    fig.savefig("best_vs_worst_relation_image.png", dpi=150)
    logger.info("saved best_vs_worst_relation_image.png")

    plt.show()
