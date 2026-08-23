"""A bottleneck autoencoder trained to minimize the sphere's own (pulled-back) distance.

Companion to :mod:`triangle_relations.discovery.autoencoder` (Program 1's
detector, trained via scikit-learn's plain Euclidean MSE) for Program 1b:
the target data here lives on a sphere (see
:mod:`triangle_relations.discovery.homogeneous_relations`), and the
question is whether it collapses onto a curve. But scikit-learn's
``MLPRegressor`` cannot minimize anything except ordinary Euclidean squared
error, and Euclidean distance in a flat chart is *not* the sphere's own
distance (a stereographic chart distorts distances, worse near its
excluded pole). This module represents the encoder's input and the
decoder's output as 2D stereographic-chart coordinates (see
:mod:`triangle_relations.discovery.spherical_chart`), but maps the
decoder's output back through the (differentiable) inverse chart *before*
computing the loss -- so the quantity gradient descent actually minimizes
is chordal distance on the sphere between the reconstructed point and the
real one, not a chart-space proxy for it. This needs PyTorch (autograd
through the inverse-chart map); scikit-learn has no custom-loss hook.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch import nn

from triangle_relations.discovery.spherical_chart import (
    CANONICAL_POLE,
    forward_chart,
    inverse_chart_torch,
    orthonormal_frame,
)

logger = logging.getLogger(__name__)

#: The maximum possible value of :func:`sphere_reconstruction_error`: squared
#: chordal distance between two unit vectors, ``||x - y||**2``, is maximized
#: at antipodal points (``y = -x``), where it equals ``||2x||**2 = 4``. Since
#: every triple's embedding is a unit vector by construction (see
#: :mod:`triangle_relations.discovery.homogeneous_relations`), this bound is
#: the same for every triple, with no per-triple calibration needed.
MAX_SPHERE_ERROR: float = 4.0


class _ChartAutoencoder(nn.Module):
    """A ``(2, hidden, bottleneck, hidden, 2)`` MLP, tanh-activated like Program 1's."""

    def __init__(self, hidden: int, bottleneck: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, bottleneck), nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden), nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def forward(self, chart_xy: torch.Tensor) -> torch.Tensor:
        """Encode then decode ``chart_xy`` (shape ``(n, 2)``), returning reconstructed chart coordinates."""
        return self.decoder(self.encoder(chart_xy))


def sphere_reconstruction_error(
    points: np.ndarray,
    *,
    pole: np.ndarray | None = None,
    bottleneck: int = 1,
    hidden: int = 8,
    test_size: float = 0.3,
    n_restarts: int = 1,
    n_epochs: int = 1500,
    lr: float = 0.02,
    random_state: int | np.random.Generator | None = None,
) -> float:
    """Held-out mean squared *sphere* (chordal) reconstruction error of a chart-based bottleneck autoencoder.

    ``points`` are unit vectors on a sphere (shape ``(n, 3)``, e.g. from
    :func:`~triangle_relations.discovery.homogeneous_relations.embed_triple`).
    Encoder input and decoder output are 2D coordinates in a stereographic
    chart of that sphere (see
    :mod:`triangle_relations.discovery.spherical_chart`); the loss pulls the
    decoder's output back through the (differentiable) inverse chart before
    comparing it to the true 3D point, so what is actually minimized is
    chordal distance on the sphere, not Euclidean distance in the chart --
    see the module docstring.

    Parameters
    ----------
    points:
        Unit vectors on the target sphere, shape ``(n, 3)``.
    pole:
        Stereographic-projection pole for the chart; defaults to
        :data:`~triangle_relations.discovery.spherical_chart.CANONICAL_POLE`,
        a fixed choice outside the positive octant every candidate triple's
        image lives in, so the chart itself introduces no per-triple
        calibration.
    bottleneck, hidden:
        Autoencoder topology: ``(2, hidden, bottleneck, hidden, 2)``.
    test_size:
        Fraction of ``points`` held out to measure reconstruction error.
    n_restarts:
        Number of independent random initializations to train; the lowest
        resulting test error is returned.
    n_epochs:
        Full-batch gradient-descent steps per training run.
    lr:
        Adam learning rate.
    random_state:
        Seed or :class:`numpy.random.Generator` controlling the train/test
        split, network initialization, and training.

    Returns
    -------
    The best (lowest) held-out mean squared chordal error across
    ``n_restarts`` runs, bounded in ``[0, 4]`` for any input (both endpoints
    of a unit-vector difference), regardless of the scalars behind
    ``points`` -- no per-triple calibration needed.
    """
    rng = np.random.default_rng(random_state)
    pole = CANONICAL_POLE if pole is None else pole
    e1, e2 = orthonormal_frame(pole)

    n_test = max(1, int(round(len(points) * test_size)))
    perm = rng.permutation(len(points))
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    chart = forward_chart(points, pole, e1, e2)
    chart_train_t = torch.as_tensor(chart[train_idx], dtype=torch.float32)
    chart_test_t = torch.as_tensor(chart[test_idx], dtype=torch.float32)
    true_train_t = torch.as_tensor(points[train_idx], dtype=torch.float32)
    true_test_t = torch.as_tensor(points[test_idx], dtype=torch.float32)
    pole_t = torch.as_tensor(pole, dtype=torch.float32)
    e1_t = torch.as_tensor(e1, dtype=torch.float32)
    e2_t = torch.as_tensor(e2, dtype=torch.float32)

    best_error = np.inf
    for restart in range(n_restarts):
        torch.manual_seed(int(rng.integers(2**32)))
        model = _ChartAutoencoder(hidden=hidden, bottleneck=bottleneck)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for _ in range(n_epochs):
            optimizer.zero_grad()
            pred_sphere = inverse_chart_torch(model(chart_train_t), pole_t, e1_t, e2_t)
            loss = torch.mean(torch.sum((pred_sphere - true_train_t) ** 2, dim=1))
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred_sphere_test = inverse_chart_torch(model(chart_test_t), pole_t, e1_t, e2_t)
            test_error = torch.mean(torch.sum((pred_sphere_test - true_test_t) ** 2, dim=1)).item()

        logger.debug("restart %d/%d: sphere reconstruction error = %.4g", restart + 1, n_restarts, test_error)
        best_error = min(best_error, test_error)

    return best_error
