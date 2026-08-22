"""Generate the figures embedded in docs/discovering_triangle_relations.tex.

Produces, into docs/figures/:
  - euler_surface.png: the physical Euler surface d = sqrt(R(R-2r)) over
    (R, r), with sampled real triangles overlaid to confirm they lie exactly
    on it.
  - euler_cone.png: the full algebraic variety {d^2 = R^2 - 2Rr}, both
    branches (R = r + sqrt(r^2+d^2) and R = r - sqrt(r^2+d^2)), showing the
    conical singularity at the origin and which branch is physical (R > 0).
  - euler_relation_check.png: the "surface vs. volume" point-cloud
    comparison from triangle_relations.discovery.verify_euler_relation,
    regenerated here for inclusion in the paper.

Run with:
    poetry run python docs/render_paper_figures.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from triangle_relations.discovery.sampling import build_scalar_dataset
from triangle_relations.discovery.scalar_relations import shuffle_columns
from triangle_relations.discovery.verify_euler_relation import (
    N_SAMPLES,
    SCALAR_NAMES,
    SEED,
    _trim_outliers,
)

logger = logging.getLogger(__name__)

FIGURES_DIR = Path(__file__).parent / "figures"


def render_euler_surface() -> None:
    """Plot the physical Euler surface with sampled triangles overlaid."""
    rng = np.random.default_rng(SEED)
    _, data = build_scalar_dataset(600, rng, scalar_names=SCALAR_NAMES)
    keep = _trim_outliers(data, q=0.85)
    R_pts, r_pts, d_pts = data[keep, 0], data[keep, 1], data[keep, 2]

    R_max = 1.05 * float(R_pts.max())
    R_grid = np.linspace(1e-3, R_max, 120)
    r_grid = np.linspace(1e-3, R_max / 2, 120)
    R_mesh, r_mesh = np.meshgrid(R_grid, r_grid)
    valid = r_mesh <= R_mesh / 2
    d_mesh = np.full_like(R_mesh, np.nan)
    d_mesh[valid] = np.sqrt(R_mesh[valid] * (R_mesh[valid] - 2 * r_mesh[valid]))

    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(
        R_mesh, r_mesh, d_mesh, cmap="viridis", alpha=0.55, linewidth=0, antialiased=True,
    )
    edge_R = R_grid
    ax.plot(edge_R, edge_R / 2, np.zeros_like(edge_R), color="crimson", linewidth=2,
            label="equilateral boundary ($r=R/2,\\,d=0$)")
    ax.scatter(R_pts, r_pts, d_pts, s=5, color="black", alpha=0.5, label="sampled triangles")
    ax.set_xlabel("$R$")
    ax.set_ylabel("$r$")
    ax.set_zlabel("$d$")
    ax.set_title("The physical Euler surface $d=\\sqrt{R(R-2r)}$")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out = FIGURES_DIR / "euler_surface.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    logger.info("wrote %s", out)


def render_euler_cone() -> None:
    """Plot both algebraic branches of d^2 = R^2 - 2Rr, showing the conical
    singularity at the origin and which branch is physically realized."""
    rho = 1.2
    n = 80
    r_lin = np.linspace(-rho, rho, n)
    d_lin = np.linspace(-rho, rho, n)
    r_mesh, d_mesh = np.meshgrid(r_lin, d_lin)
    radius = np.sqrt(r_mesh**2 + d_mesh**2)
    R_plus = r_mesh + radius
    R_minus = r_mesh - radius

    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(R_plus, r_mesh, d_mesh, color="seagreen", alpha=0.55, linewidth=0)
    ax.plot_surface(R_minus, r_mesh, d_mesh, color="lightgray", alpha=0.45, linewidth=0)
    ax.scatter([0], [0], [0], color="crimson", s=40, depthshade=False)
    ax.text(0, 0, 0.15, "singular vertex\n$(R,r,d)=(0,0,0)$", color="crimson", fontsize=8, ha="center")
    ax.set_xlabel("$R$")
    ax.set_ylabel("$r$")
    ax.set_zlabel("$d$")
    ax.set_title("Both branches of $d^2=R^2-2Rr$\nphysical: green ($R>0$)   unphysical: gray ($R\\leq 0$)")
    fig.tight_layout()
    out = FIGURES_DIR / "euler_cone.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    logger.info("wrote %s", out)


def render_surface_vs_volume() -> None:
    """Regenerate the real-vs-shuffled-null point cloud comparison."""
    rng = np.random.default_rng(SEED)
    names, data = build_scalar_dataset(N_SAMPLES, rng, scalar_names=SCALAR_NAMES)
    shuffled = shuffle_columns(data, rng)

    data_plot = data[_trim_outliers(data)]
    shuffled_plot = shuffled[_trim_outliers(shuffled)]

    fig = plt.figure(figsize=(11, 5))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.scatter(data_plot[:, 0], data_plot[:, 1], data_plot[:, 2], s=6, alpha=0.6, color="tab:blue")
    ax1.set_xlabel(names[0])
    ax1.set_ylabel(names[1])
    ax1.set_zlabel(names[2])
    ax1.set_title("Real $(R,r,d)$\nconfined to a 2D surface")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.scatter(shuffled_plot[:, 0], shuffled_plot[:, 1], shuffled_plot[:, 2], s=6, alpha=0.6, color="tab:red")
    ax2.set_xlabel(names[0])
    ax2.set_ylabel(names[1])
    ax2.set_zlabel(names[2])
    ax2.set_title("Column-shuffled null\nfills the 3D volume")

    fig.tight_layout()
    out = FIGURES_DIR / "euler_relation_check.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    logger.info("wrote %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    FIGURES_DIR.mkdir(exist_ok=True)
    render_euler_surface()
    render_euler_cone()
    render_surface_vs_volume()
