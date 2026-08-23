"""Stereographic charts on a unit sphere, shared by the target-sphere detector and its plot.

A stereographic chart projects the sphere (minus one excluded point, its
"pole") onto a 2D plane. Used two ways in this project, both starting from
the same :func:`orthonormal_frame` / :func:`forward_chart` pair:

* :mod:`triangle_relations.discovery.sphere_autoencoder` uses a *fixed*
  pole (the same one for every candidate triple, chosen outside the
  positive octant every positive-degree scalar triple's image lives in --
  see :attr:`CANONICAL_POLE`), so the chart itself introduces no
  per-triple calibration; its :func:`inverse_chart_torch` is the
  differentiable half, used to pull the sphere's own distance back through
  the chart as the actual training loss.
* :mod:`triangle_relations.discovery.inspect_relation` instead picks a pole
  *adapted* to whichever triple is being plotted (antipodal to its own
  data's mean direction), purely to minimize visual distortion for that one
  plot -- a different, but equally "generic" in the sense of Section 5 of
  the theory doc, choice of the same underlying chart.
"""

from __future__ import annotations

import numpy as np
import torch

#: A fixed stereographic pole for the target-sphere detector: outside the
#: positive octant (negative in every coordinate), so it is never close to
#: -- and its chart distortion never affects -- the image of any triple of
#: positive, positive-degree scalars (see
#: triangle_relations.discovery.homogeneous_relations), which always lies
#: entirely within that octant.
CANONICAL_POLE: np.ndarray = -np.ones(3) / np.sqrt(3.0)


def orthonormal_frame(pole: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """An orthonormal basis ``(e1, e2)`` for the plane orthogonal to ``pole``.

    ``pole`` must be a unit vector. The basis itself is arbitrary (any
    rotation of it works equally well); a fixed reference vector is
    Gram-Schmidt'd off ``pole`` to build it deterministically.
    """
    reference = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(reference, pole)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    e1 = reference - np.dot(reference, pole) * pole
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(pole, e1)
    return e1, e2


def forward_chart(
    points: np.ndarray, pole: np.ndarray, e1: np.ndarray, e2: np.ndarray
) -> np.ndarray:
    """Stereographically project unit vectors ``points`` (shape ``(n, 3)``) to chart coordinates ``(n, 2)``.

    Projects from ``pole``: undefined only at ``pole`` itself, which none of
    this project's data ever reaches exactly (see the module docstring).
    """
    a = points @ e1
    b = points @ e2
    c = points @ pole
    denom = 1.0 - c
    return np.stack([a / denom, b / denom], axis=1)


def inverse_chart_torch(
    uv: torch.Tensor, pole: torch.Tensor, e1: torch.Tensor, e2: torch.Tensor
) -> torch.Tensor:
    """Differentiable inverse of :func:`forward_chart`: chart coordinates back to the unit sphere.

    Smooth and well-defined for every finite ``uv`` (the pole itself is only
    approached as ``|uv| -> infinity``, never hit), so this never needs a
    domain check. Used as the differentiable half of the pulled-back sphere
    metric in :mod:`triangle_relations.discovery.sphere_autoencoder`: mapping
    a decoder's chart-coordinate output back to the sphere before comparing
    it to the true target point is what makes "sphere distance," not raw
    chart-space Euclidean distance, the quantity actually being minimized.
    """
    u, v = uv[:, 0], uv[:, 1]
    r2 = u * u + v * v
    a = 2.0 * u / (1.0 + r2)
    b = 2.0 * v / (1.0 + r2)
    c = (r2 - 1.0) / (1.0 + r2)
    return a[:, None] * e1 + b[:, None] * e2 + c[:, None] * pole
