"""Random triangle generation and bulk scalar-quantity sampling."""

from __future__ import annotations

import logging

import numpy as np

from triangle_relations.geometry.triangle import Triangle

logger = logging.getLogger(__name__)

#: Below this ratio of (2*area) to (perimeter^2), a triangle is considered too
#: close to degenerate (near-collinear vertices) and is resampled. This keeps
#: quantities like circumradius/inradius from blowing up or losing precision.
_MIN_SHAPE_RATIO = 1e-3


def random_triangle(rng: np.random.Generator, scale: float = 1.0) -> Triangle:
    """Draw one triangle with vertices uniform in a ``scale``-sized square.

    Degenerate (near-collinear) triangles are rejected and resampled so that
    derived quantities such as circumradius and inradius stay well-behaved.

    Parameters
    ----------
    rng:
        A NumPy random number generator used for vertex sampling.
    scale:
        Vertices are drawn uniformly from ``[-scale, scale]`` in each
        coordinate.

    Returns
    -------
    A non-degenerate :class:`Triangle`.
    """
    n_rejected = 0
    while True:
        verts = rng.uniform(-scale, scale, size=(3, 2))
        triangle = Triangle(*verts)
        perimeter = triangle.perimeter()
        if perimeter == 0:
            n_rejected += 1
            continue
        shape_ratio = (2.0 * triangle.area()) / (perimeter**2)
        if shape_ratio > _MIN_SHAPE_RATIO:
            if n_rejected:
                logger.debug("rejected %d near-degenerate triangle(s) before accepting one", n_rejected)
            return triangle
        n_rejected += 1


def build_scalar_dataset(
    n_samples: int,
    rng: np.random.Generator,
    scalar_names: list[str] | None = None,
    scale: float = 1.0,
) -> tuple[list[str], np.ndarray]:
    """Sample ``n_samples`` random triangles and evaluate scalar quantities on each.

    All requested triangles are drawn once and every requested scalar is
    evaluated on the same set of triangles, so different 3-scalar
    combinations drawn from the result remain directly comparable.

    Parameters
    ----------
    n_samples:
        Number of random triangles to draw.
    rng:
        A NumPy random number generator used for sampling.
    scalar_names:
        Names of the scalars to evaluate (must be keys of
        :attr:`Triangle.SCALARS`). Defaults to every registered scalar.
    scale:
        Passed to :func:`random_triangle`.

    Returns
    -------
    A tuple ``(names, data)`` where ``data`` has shape
    ``(n_samples, len(names))`` and ``data[i, j]`` is scalar ``names[j]``
    evaluated on the i-th sampled triangle.
    """
    names = list(scalar_names) if scalar_names is not None else sorted(Triangle.SCALARS)
    logger.info("sampling %d random triangles for %d scalar(s)", n_samples, len(names))
    data = np.empty((n_samples, len(names)), dtype=float)
    for i in range(n_samples):
        triangle = random_triangle(rng, scale=scale)
        for j, name in enumerate(names):
            data[i, j] = triangle.scalar(name)
    logger.debug("finished sampling scalar dataset of shape %s", data.shape)
    return names, data
