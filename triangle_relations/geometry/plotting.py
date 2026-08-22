"""Visualization for a :class:`~triangle_relations.geometry.triangle.Triangle`
and its derived objects.

:func:`plot_triangle` is normally called through
:meth:`Triangle.plot() <triangle_relations.geometry.triangle.Triangle.plot>`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from triangle_relations.geometry.triangle import Triangle

logger = logging.getLogger(__name__)


def plot_triangle(
    triangle: "Triangle",
    *,
    ax: "Axes | None" = None,
    show_centroid: bool = True,
    show_incenter: bool = True,
    show_circumcenter: bool = True,
    show_orthocenter: bool = True,
    show_steiner_foci: bool = True,
    show_incircle: bool = True,
    show_circumcircle: bool = True,
    show_steiner_inellipse: bool = True,
    show_euler_line: bool = True,
    labels: bool = True,
) -> "Axes":
    """Plot a triangle together with a selection of its derived objects.

    Parameters
    ----------
    triangle:
        The triangle to plot.
    ax:
        An existing matplotlib ``Axes`` to draw into; a new figure is
        created if omitted.
    show_centroid, show_incenter, show_circumcenter, show_orthocenter, show_steiner_foci:
        Whether to mark the corresponding derived point.
    show_incircle, show_circumcircle, show_steiner_inellipse:
        Whether to draw the corresponding circle/ellipse.
    show_euler_line:
        Whether to draw the line through the circumcenter and orthocenter.
    labels:
        Whether to annotate the vertices and derived points with short
        labels (A, B, C, G, I, O, H, F1, F2).

    Returns
    -------
    The matplotlib ``Axes`` used for the plot.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        logger.debug("no Axes supplied; creating a new figure")
        _, ax = plt.subplots(figsize=(6, 6))

    verts = triangle.vertices
    closed = np.vstack([verts, verts[0]])
    ax.plot(closed[:, 0], closed[:, 1], "-", color="black", linewidth=1.5)
    ax.scatter(verts[:, 0], verts[:, 1], color="black", zorder=5)
    if labels:
        for point, name in zip(verts, ("A", "B", "C")):
            ax.annotate(name, point, textcoords="offset points", xytext=(6, 6))

    points_to_plot: list[tuple[str, str, str]] = []
    if show_centroid:
        points_to_plot.append(("centroid", "G", "tab:blue"))
    if show_incenter:
        points_to_plot.append(("incenter", "I", "tab:green"))
    if show_circumcenter:
        points_to_plot.append(("circumcenter", "O", "tab:red"))
    if show_orthocenter:
        points_to_plot.append(("orthocenter", "H", "tab:purple"))
    if show_steiner_foci:
        points_to_plot.append(("steiner_focus_1", "F1", "tab:orange"))
        points_to_plot.append(("steiner_focus_2", "F2", "tab:orange"))

    for point_name, label, color in points_to_plot:
        p = triangle.point(point_name)
        ax.scatter(*p, color=color, zorder=5)
        if labels:
            ax.annotate(
                label, p, textcoords="offset points", xytext=(6, 6), color=color
            )

    if show_incircle:
        _plot_circle(ax, triangle.incenter(), triangle.inradius(), "tab:green")

    if show_circumcircle:
        _plot_circle(ax, triangle.circumcenter(), triangle.circumradius(), "tab:red")

    if show_steiner_inellipse:
        _plot_steiner_inellipse(ax, triangle, "tab:orange")

    if show_euler_line:
        o = triangle.circumcenter()
        h = triangle.orthocenter()
        direction = h - o
        extended = np.array([o - 0.3 * direction, h + 0.3 * direction])
        ax.plot(
            extended[:, 0], extended[:, 1], "--", color="gray", linewidth=1, zorder=1
        )

    ax.set_aspect("equal")
    logger.debug("plotted triangle with %d derived point(s) shown", len(points_to_plot))
    return ax


def _plot_circle(
    ax: "Axes", center: np.ndarray, radius: float, color: str, n_points: int = 200
) -> None:
    """Draw a circle of the given ``center`` and ``radius`` onto ``ax``."""
    theta = np.linspace(0, 2 * np.pi, n_points)
    circle = center + radius * np.column_stack([np.cos(theta), np.sin(theta)])
    ax.plot(circle[:, 0], circle[:, 1], color=color, linewidth=1, zorder=2)


def _plot_steiner_inellipse(
    ax: "Axes", triangle: "Triangle", color: str, n_points: int = 200
) -> None:
    """Draw the Steiner inellipse by mapping the reference equilateral
    triangle's incircle through :meth:`Triangle.affine_map_from_equilateral`."""
    M, t = triangle.affine_map_from_equilateral()
    r = 0.5  # inradius of the reference equilateral triangle (circumradius 1)
    theta = np.linspace(0, 2 * np.pi, n_points)
    unit_circle = r * np.column_stack([np.cos(theta), np.sin(theta)])
    ellipse = unit_circle @ M.T + t
    ax.plot(ellipse[:, 0], ellipse[:, 1], color=color, linewidth=1, zorder=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    import matplotlib.pyplot as plt

    from triangle_relations.geometry.triangle import Triangle

    sample = Triangle((0.5, 0.2), (4.0, 0.8), (1.5, 3.2))
    plot_triangle(sample)
    plt.title("Sample triangle with all derived objects")
    plt.show()
