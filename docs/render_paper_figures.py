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

from triangle_relations.discovery.autoencoder import reconstruction_error
from triangle_relations.discovery.sampling import build_scalar_dataset
from triangle_relations.discovery.scalar_relations import shuffle_columns
from triangle_relations.discovery.verify_euler_relation import (
    N_SAMPLES,
    N_RESTARTS,
    SCALAR_NAMES,
    SEED,
    _trim_outliers,
)

logger = logging.getLogger(__name__)

FIGURES_DIR = Path(__file__).parent / "figures"


def render_euler_surface() -> None:
    """Plot the physical Euler surface with sampled triangles overlaid.

    Euler's relation is scale-invariant (a cone through the origin), so
    plotting it from R near 0 up to some R_max makes it look like a thin
    flared blade -- an artifact of including a huge range of scales in one
    plot, not of the surface itself. We instead zoom into a representative
    local patch (the 10th-90th percentile range of sampled R), which shows
    the surface's actual curvature and matches the paper's point that it is
    smooth *locally*, away from the origin.
    """
    rng = np.random.default_rng(SEED)
    _, data = build_scalar_dataset(800, rng, scalar_names=SCALAR_NAMES)
    keep = _trim_outliers(data, q=0.85)
    R_pts, r_pts, d_pts = data[keep, 0], data[keep, 1], data[keep, 2]

    R_lo, R_hi = np.quantile(R_pts, [0.1, 0.9])
    in_window = (R_pts >= R_lo) & (R_pts <= R_hi)
    R_pts, r_pts, d_pts = R_pts[in_window], r_pts[in_window], d_pts[in_window]

    # Parametrize r as a fraction of its (R-dependent) valid range [0, R/2],
    # so every grid point is automatically valid -- no ragged cutoff edge.
    R_grid = np.linspace(R_lo, R_hi, 100)
    frac_grid = np.linspace(0, 1, 100)
    R_mesh, frac_mesh = np.meshgrid(R_grid, frac_grid)
    r_mesh = frac_mesh * R_mesh / 2
    d_mesh = np.sqrt(R_mesh * (R_mesh - 2 * r_mesh))

    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(
        R_mesh, r_mesh, d_mesh, cmap="viridis", alpha=0.75, linewidth=0.1,
        edgecolor="gray", antialiased=True,
    )
    edge_R = R_grid
    ax.plot(edge_R, edge_R / 2, np.zeros_like(edge_R), color="crimson", linewidth=2.5,
            label="equilateral boundary ($r=R/2,\\,d=0$)")
    ax.scatter(R_pts, r_pts, d_pts, s=16, color="white", edgecolors="black",
               linewidth=0.6, depthshade=False, label="sampled triangles")
    ax.set_xlabel("$R$")
    ax.set_ylabel("$r$")
    ax.set_zlabel("$d$")
    ax.set_title("The physical Euler surface $d=\\sqrt{R(R-2r)}$\n(local patch, $R\\in[%.2f,%.2f]$)" % (R_lo, R_hi))
    ax.legend(loc="upper left", fontsize=8)
    ax.set_box_aspect((R_hi - R_lo, R_hi / 2, float(np.nanmax(d_mesh))))
    ax.view_init(elev=25, azim=-55)
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


def render_null_distribution(m: int = 20) -> None:
    """Plot the null reconstruction-loss distribution for (R, r, d) against
    the real loss, visualizing mu_null, sigma_null, and the z-score."""
    data_rng = np.random.default_rng(SEED)
    _, data = build_scalar_dataset(N_SAMPLES, data_rng, scalar_names=SCALAR_NAMES)

    # A fresh RNG stream for training/shuffling, independent of the one used
    # to sample triangles (matching verify_euler_relation's own separation
    # between data sampling and the detection step).
    train_rng = np.random.default_rng(SEED)
    L_real = reconstruction_error(data, n_restarts=N_RESTARTS, random_state=train_rng)
    null_losses = np.array([
        reconstruction_error(shuffle_columns(data, train_rng), n_restarts=N_RESTARTS, random_state=train_rng)
        for _ in range(m)
    ])
    mu_null = null_losses.mean()
    sigma_null = null_losses.std(ddof=1)
    z = (mu_null - L_real) / sigma_null

    rng_jitter = np.random.default_rng(0)
    y_jitter = rng_jitter.uniform(-0.15, 0.15, size=m)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.axvspan(mu_null - sigma_null, mu_null + sigma_null, color="tab:gray", alpha=0.2,
               label=r"$\mu_{\mathrm{null}}\pm\sigma_{\mathrm{null}}$")
    ax.axvline(mu_null, color="black", linestyle="--", linewidth=1, label=r"$\mu_{\mathrm{null}}$")
    ax.scatter(null_losses, y_jitter, color="tab:red", alpha=0.7, s=22,
               label=r"null losses $L_1',\dots,L_m'$")
    ax.axvline(L_real, color="tab:blue", linewidth=2.2, label=r"$L_{\mathrm{real}}$")
    ax.scatter([L_real], [0], color="tab:blue", s=40, zorder=5)

    ax.set_yticks([])
    ax.set_xlabel("reconstruction loss")
    ax.set_ylim(-0.5, 0.5)
    ax.set_title(f"Null distribution for $(R,r,d)$: $z={z:.1f}$")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    out = FIGURES_DIR / "null_distribution.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    logger.info("wrote %s (L_real=%.4g, mu_null=%.4g, sigma_null=%.4g, z=%.2f)",
                out, L_real, mu_null, sigma_null, z)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    FIGURES_DIR.mkdir(exist_ok=True)
    render_euler_surface()
    render_euler_cone()
    render_surface_vs_volume()
    render_null_distribution()
