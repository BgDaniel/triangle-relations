"""Visualization for a Triangle and its derived objects."""

from __future__ import annotations

import numpy as np


def plot_triangle(
    triangle,
    *,
    ax=None,
    show_centroid: bool = True,
    show_incenter: bool = True,
    show_circumcenter: bool = True,
    show_orthocenter: bool = False,
    show_steiner_foci: bool = False,
    show_incircle: bool = True,
    show_circumcircle: bool = True,
    show_steiner_inellipse: bool = True,
    show_euler_line: bool = False,
    labels: bool = True,
):
    """Plot a triangle together with a selection of its derived objects.

    Returns the matplotlib ``Axes`` used for the plot.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    verts = triangle.vertices
    closed = np.vstack([verts, verts[0]])
    ax.plot(closed[:, 0], closed[:, 1], "-", color="black", linewidth=1.5)
    ax.scatter(verts[:, 0], verts[:, 1], color="black", zorder=5)
    if labels:
        for point, name in zip(verts, ("A", "B", "C")):
            ax.annotate(name, point, textcoords="offset points", xytext=(6, 6))

    points_to_plot = []
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
    return ax


def _plot_circle(ax, center, radius, color, n_points: int = 200):
    theta = np.linspace(0, 2 * np.pi, n_points)
    circle = center + radius * np.column_stack([np.cos(theta), np.sin(theta)])
    ax.plot(circle[:, 0], circle[:, 1], color=color, linewidth=1, zorder=2)


def _plot_steiner_inellipse(ax, triangle, color, n_points: int = 200):
    M, t = triangle.affine_map_from_equilateral()
    r = 0.5  # inradius of the reference equilateral triangle (circumradius 1)
    theta = np.linspace(0, 2 * np.pi, n_points)
    unit_circle = r * np.column_stack([np.cos(theta), np.sin(theta)])
    ellipse = unit_circle @ M.T + t
    ax.plot(ellipse[:, 0], ellipse[:, 1], color=color, linewidth=1, zorder=2)
