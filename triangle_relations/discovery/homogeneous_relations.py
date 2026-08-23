"""Program 1b: search for *homogeneous* functional relations among triples of scalars.

A variant of Program 1 (:mod:`triangle_relations.discovery.scalar_relations`)
restricted to relations that are *homogeneous*, i.e. dimensionally
consistent under uniform scaling of the triangle (every term of the
relation scales the same way) -- which covers essentially every relation of
practical interest, Euler's included (``d^2 = R^2 - 2Rr`` is homogeneous of
degree 2 throughout). The full mathematical construction is written up in
``docs/discovering_triangle_relations.tex``, Section 5; this is a summary of
what the code below actually does.

Restricting to homogeneous relations lets scale be quotiented out
completely: a scalar ``f`` of homogeneity degree ``d`` (see
:attr:`Triangle.SCALAR_DEGREES`) satisfies ``f(T)**(1/d)`` scales *linearly*
with the triangle regardless of ``d``, so three such degree-equalized
scalars ``f1, f2, f3`` combine into a single point on the unit sphere,
``Phi(T) = normalize(f1(T)**(1/d1), f2(T)**(1/d2), f3(T)**(1/d3))``, that
depends only on the triangle's *shape*, not its size. Since shape space is
itself (topologically) a sphere -- see
:mod:`triangle_relations.discovery.shape_space` -- both the domain (shape)
and the target (``Phi``) are 2-dimensional and live inside a fixed, bounded
ambient space (points on a unit sphere), which is what removes the need for
a permutation null: unlike Program 1's raw scalar triples, whose scale and
spread differ arbitrarily from one triple to the next, ``Phi(T)`` for
*every* triple is confined to the same ``[-1, 1]^3`` unit sphere, so a fixed
reconstruction-error scale is already comparable across triples with no
per-triple calibration.

Whether ``(f1, f2, f3)`` satisfy an exact homogeneous relation is then a
question about ``Phi`` alone: generically, the 2-dimensional shape sphere
maps onto an *open* patch of the 2-dimensional target sphere (a local
diffeomorphism), and no further compression is possible; an exact relation
forces the image to collapse onto a 1-dimensional curve. We test this the
same way Program 1 tests its own dichotomy -- train a bottleneck
autoencoder and measure held-out reconstruction error -- but with a
bottleneck of size 1 (matching the target's own intrinsic dimension of 2,
same as Program 1's size-2 bottleneck matches its target's intrinsic
dimension of 3) and with no shuffled null: small error means the image did
collapse to a curve (strong evidence of a relation); error staying bounded
away from 0 means it didn't.

Concretely, the encoder/decoder operate on a 2D coordinate *chart* of the
target sphere, not its raw 3D ambient embedding -- necessary to write down
network input/output at all, since a manifold can't be handed to code
directly, and matching the target's intrinsic dimension, unlike the 3
ambient coordinates (which are redundant: any 2 of the 3 already determine
the third, on the sphere). The loss is not plain Euclidean distance in that
chart, though -- a stereographic chart distorts distances, worse away from
its excluded pole. Instead the decoder's chart-coordinate output is mapped
back through the (differentiable) *inverse* chart before being compared to
the true point, so what gradient descent actually minimizes is chordal
distance on the sphere itself. See
:mod:`triangle_relations.discovery.sphere_autoencoder` (this needs
PyTorch's autograd -- scikit-learn's ``MLPRegressor``, which Program 1
uses, has no custom-loss hook) and
:mod:`triangle_relations.discovery.spherical_chart` for the chart
construction, shared with :mod:`triangle_relations.discovery.inspect_relation`.

Two scope notes:

* This only applies to scalars with a well-defined *positive* homogeneity
  degree that are positive on every sampled triangle (needed for the real
  ``d``-th root above). Degree-0 (scale-invariant) scalars, like angles, are
  excluded -- see :func:`search_homogeneous_relations`.
* Turning a point of shape space into an actual triangle to evaluate scalars
  on requires a coordinate *chart* on shape space too, for the same reason
  as above; :mod:`triangle_relations.discovery.shape_space` picks one (a
  complex cross-ratio). Any chart -- on shape space or on the target sphere
  -- omits a lower-dimensional locus (a single point, for a stereographic
  chart); a *generic* choice of pole is safe because a relation-free
  triple's image is open (it covers a full neighborhood of the target
  sphere), so it can only meet that lower-dimensional excluded locus in a
  measure-zero way -- and if it somehow didn't (the image landed entirely
  inside the excluded locus), that would itself be a different discovered
  relation (the image confined to a lower-dimensional subset), not a
  failure of the method. See Section 5 of the theory doc for the fuller
  version of this remark.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from joblib import Parallel, delayed

from triangle_relations.discovery._parallel import joblib_progress
from triangle_relations.discovery.known_relations import is_euler_triple
from triangle_relations.discovery.sampling import evaluate_scalars
from triangle_relations.discovery.sphere_autoencoder import sphere_reconstruction_error
from triangle_relations.geometry.triangle import Triangle

logger = logging.getLogger(__name__)

#: Minimum number of sampled triangles that must survive the positivity
#: filter (see :func:`_degree_equalized_embedding`) for a triple to be
#: evaluated at all, below which it is skipped and logged rather than
#: trained on too little data.
_MIN_VALID_ROWS = 10


@dataclass
class HomogeneousRelationResult:
    """Outcome of testing one candidate triple of homogeneous scalars for a hidden relation.

    Unlike :class:`~triangle_relations.discovery.scalar_relations.RelationResult`,
    there is no null distribution here -- see the module docstring for why
    none is needed.

    Attributes
    ----------
    names:
        The three scalar names that were tested.
    degrees:
        Their homogeneity degrees (see :attr:`Triangle.SCALAR_DEGREES`).
    error:
        Held-out mean squared *sphere* (chordal) reconstruction error from
        :func:`~triangle_relations.discovery.sphere_autoencoder.sphere_reconstruction_error`,
        a bottleneck-1 autoencoder trained -- via a differentiable
        stereographic chart -- to directly minimize chordal distance on the
        target sphere, not a chart-space or standardized proxy for it (see
        that module's docstring for why this needed a custom PyTorch loss
        rather than Program 1's scikit-learn detector). I.e. how well the
        image on the target sphere compresses onto a *curve*. Bounded in
        ``[0, 4]`` for every triple regardless of its units or typical
        magnitude (both endpoints of a unit-vector difference), so it is
        comparable directly across triples and against a fixed threshold,
        with no per-triple calibration. Small error is strong evidence of a
        relation.
    """

    names: tuple[str, str, str]
    degrees: tuple[int, int, int]
    error: float


def _degree_equalized_embedding(
    data: np.ndarray, degrees: tuple[int, int, int]
) -> np.ndarray | None:
    """Map one triple's raw scalar columns to unit vectors on the target sphere.

    ``data`` has shape ``(n, 3)``. Rows where any column is non-positive are
    dropped (a real ``d``-th root needs a positive base); this should not
    happen for any currently-registered scalar on a non-degenerate
    triangle, but is checked defensively rather than assumed. Returns
    ``None`` (skip this triple) if fewer than :data:`_MIN_VALID_ROWS` rows
    survive.
    """
    valid = np.all(data > 0, axis=1)
    n_dropped = len(data) - int(valid.sum())
    if n_dropped:
        logger.debug("dropped %d/%d non-positive sample(s) for this triple", n_dropped, len(data))
    if valid.sum() < _MIN_VALID_ROWS:
        return None

    data = data[valid]
    equalized = np.stack([data[:, i] ** (1.0 / degrees[i]) for i in range(3)], axis=1)
    norms = np.linalg.norm(equalized, axis=1, keepdims=True)
    return equalized / norms


def _evaluate_triple(
    data: np.ndarray,
    names: tuple[str, str, str],
    degrees: tuple[int, int, int],
    *,
    hidden: int,
    n_restarts: int,
    test_size: float,
    n_epochs: int,
    lr: float,
    seed: int,
) -> HomogeneousRelationResult | None:
    """Run the bottleneck-1 detection test on one triple's degree-equalized embedding."""
    embedding = _degree_equalized_embedding(data, degrees)
    if embedding is None:
        logger.warning("%s: skipped, too few sampled triangles had all-positive values", names)
        return None

    error = sphere_reconstruction_error(
        embedding,
        bottleneck=1,
        hidden=hidden,
        n_restarts=n_restarts,
        test_size=test_size,
        n_epochs=n_epochs,
        lr=lr,
        random_state=seed,
    )
    logger.debug("%s: error=%.4g", names, error)
    return HomogeneousRelationResult(names=names, degrees=degrees, error=error)


def search_homogeneous_relations(
    triangles: list[Triangle],
    scalar_names: list[str] | None = None,
    *,
    hidden: int = 8,
    n_restarts: int = 1,
    test_size: float = 0.3,
    n_epochs: int = 1500,
    lr: float = 0.02,
    n_jobs: int = 1,
    random_state: int | None = None,
    progress: bool = True,
) -> list[HomogeneousRelationResult]:
    """Search all combinations of three *homogeneous* scalars for a hidden relation.

    Parameters
    ----------
    triangles:
        Triangles to evaluate scalars on -- typically
        :func:`~triangle_relations.discovery.shape_space.sample_shape_space`,
        for the even shape-space coverage this method is built around (see
        the module docstring), though any triangle list works.
    scalar_names:
        Names of scalars to search among (must be keys of
        :attr:`Triangle.SCALARS`). Defaults to every registered scalar.
        Scalars of degree <= 0 (e.g. angles) are always excluded (see the
        module docstring) and logged.
    hidden, n_restarts, test_size, n_epochs, lr:
        Passed to
        :func:`~triangle_relations.discovery.sphere_autoencoder.sphere_reconstruction_error`.
        Lowering ``n_epochs`` trades detection sharpness for speed -- useful
        for a quick first pass over a large search space, or in tests.
    n_jobs:
        Passed to :class:`joblib.Parallel`; ``-1`` uses all cores.
    random_state:
        Seed for the per-triple random seed sequence, for reproducibility.
    progress:
        Whether to display a tqdm progress bar over the combinations searched.

    Returns
    -------
    Results sorted by ascending ``error`` (strongest candidate relation first).
    """
    names, data = evaluate_scalars(triangles, scalar_names)
    degrees = [Triangle.scalar_degree(name) for name in names]

    valid_idx = [j for j, d in enumerate(degrees) if d > 0]
    if len(valid_idx) < len(names):
        dropped = [names[j] for j in range(len(names)) if degrees[j] <= 0]
        logger.info(
            "excluding %d non-positive-degree scalar(s) from the homogeneous search: %s",
            len(dropped), ", ".join(dropped),
        )

    triples = list(combinations(valid_idx, 3))
    logger.info(
        "searching %d homogeneous combination(s) of 3 scalars out of %d", len(triples), len(valid_idx)
    )
    seed_seq = np.random.SeedSequence(random_state)
    seeds = seed_seq.generate_state(len(triples))

    jobs = (
        delayed(_evaluate_triple)(
            data[:, [i, j, k]],
            (names[i], names[j], names[k]),
            (degrees[i], degrees[j], degrees[k]),
            hidden=hidden,
            n_restarts=n_restarts,
            test_size=test_size,
            n_epochs=n_epochs,
            lr=lr,
            seed=int(seed),
        )
        for (i, j, k), seed in zip(triples, seeds)
    )

    if progress:
        with joblib_progress(len(triples), desc="Searching homogeneous scalar triples"):
            results = Parallel(n_jobs=n_jobs)(jobs)
    else:
        results = Parallel(n_jobs=n_jobs)(jobs)

    results = [r for r in results if r is not None]
    logger.info("finished searching %d triple(s)", len(results))
    return sorted(results, key=lambda r: r.error)


def embed_triple(triangles: list[Triangle], names: tuple[str, str, str]) -> np.ndarray:
    """Degree-equalized, unit-normalized embedding of one scalar triple on ``triangles``.

    Public entry point to the same embedding :func:`search_homogeneous_relations`
    uses internally per candidate triple, for direct inspection -- see
    :mod:`triangle_relations.discovery.inspect_relation`.

    Returns
    -------
    An ``(n, 3)`` array of unit vectors on the target sphere (see the module
    docstring), one row per triangle that survived the positivity filter.

    Raises
    ------
    ValueError
        If any of the three scalars has non-positive homogeneity degree
        (see :attr:`Triangle.SCALAR_DEGREES`), or fewer than
        :data:`_MIN_VALID_ROWS` triangles survive the positivity filter.
    """
    _, data = evaluate_scalars(triangles, list(names))
    degrees = tuple(Triangle.scalar_degree(name) for name in names)
    if any(d <= 0 for d in degrees):
        raise ValueError(
            f"{names}: all three scalars must have positive homogeneity degree, got {degrees}"
        )
    embedding = _degree_equalized_embedding(data, degrees)
    if embedding is None:
        raise ValueError(f"{names}: too few sampled triangles had all-positive values")
    return embedding


def log_euler_triple_rank(results: list[HomogeneousRelationResult]) -> None:
    """Log the Euler triple's rank position by image-collapse error, if present in ``results``."""
    if not any(is_euler_triple(r.names) for r in results):
        logger.info("Euler triple (R, r, OI) is not present in this result set")
        return
    rank = next(i for i, r in enumerate(results) if is_euler_triple(r.names)) + 1
    logger.info("Euler triple (R, r, OI) ranks #%d of %d by image-collapse error", rank, len(results))
