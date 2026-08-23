"""Search for *homogeneous* functional relations among triples of triangle scalars.

Program 1b: like scripts/discover_scalar_relations.py, but restricted to
relations that are homogeneous under uniform scaling of the triangle (which
covers essentially every relation of practical interest -- see
triangle_relations.discovery.homogeneous_relations and Section 6 of
docs/discovering_triangle_relations.tex). In exchange, no permutation null
is needed at all: triangles are sampled once, evenly, over the space of
triangle *shapes* (triangle_relations.discovery.shape_space), and each
candidate triple is scored by a single held-out reconstruction error with no
shuffling or per-triple calibration -- the main practical benefit over
Program 1, whose null needs many shuffles (and hence many retrained
networks) per triple to be estimated precisely (see the "A note on
N_SHUFFLES" section of the README).

This is meant to be run directly (e.g. from within an IDE), not from the
command line: edit the configuration constants below and run the script.

Run with:
    poetry run python scripts/discover_homogeneous_relations.py
"""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path

from triangle_relations.discovery.homogeneous_relations import (
    HomogeneousRelationResult,
    log_euler_triple_rank,
    search_homogeneous_relations,
)
from triangle_relations.discovery.shape_space import sample_shape_space
from triangle_relations.geometry.triangle import Triangle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration -- edit these directly and re-run.
# ---------------------------------------------------------------------------

#: Number of triangles to sample, evenly, over shape space.
N_SAMPLES = 2000

#: Autoencoder hidden-layer width (topology is (HIDDEN, 1, HIDDEN)).
HIDDEN = 8

#: Autoencoder random restarts per training run.
N_RESTARTS = 1

#: Held-out fraction used to measure autoencoder reconstruction error.
TEST_SIZE = 0.3

#: Number of top-ranked triples to report.
TOP = 10

#: Passed to joblib.Parallel; -1 uses all available cores.
N_JOBS = -1

#: Seed for the per-triple random seed sequence (network init / train-test
#: split only -- the shape-space sample itself is deterministic).
SEED = 0

#: Subset of scalar names to search (must be keys of Triangle.SCALARS), or
#: None to search every registered scalar with positive homogeneity degree.
SCALAR_NAMES: list[str] | None = None

#: Environment variable holding the directory where output files (e.g. the
#: ranking CSV) are written; see the README for how to set it permanently.
OUTPUT_DIR_ENV_VAR = "PATH_TO_OUTPUT_FOLDER"

#: Optional path to write the full ranking as CSV, or None to skip.
_output_dir = os.environ.get(OUTPUT_DIR_ENV_VAR)
OUTPUT_CSV: str | None = (
    str(Path(_output_dir) / "homogeneous_ranking.csv") if _output_dir else None
)


def main() -> None:
    """Sample shape space, search all homogeneous scalar triples, and report the ranking."""
    if OUTPUT_CSV:
        logger.info("writing ranking to %s (from $%s)", OUTPUT_CSV, OUTPUT_DIR_ENV_VAR)
    else:
        logger.info(
            "$%s is not set (or OUTPUT_CSV was overridden to None); skipping CSV output",
            OUTPUT_DIR_ENV_VAR,
        )

    if SCALAR_NAMES is None:
        logger.info("available scalars (%d): %s", len(Triangle.SCALARS), sorted(Triangle.SCALARS))

    triangles = sample_shape_space(N_SAMPLES)

    results = search_homogeneous_relations(
        triangles,
        SCALAR_NAMES,
        hidden=HIDDEN,
        n_restarts=N_RESTARTS,
        test_size=TEST_SIZE,
        n_jobs=N_JOBS,
        random_state=SEED,
    )

    _log_ranking(results[:TOP])
    log_euler_triple_rank(results)

    if OUTPUT_CSV:
        _write_csv(results, OUTPUT_CSV)


def _log_ranking(results: list[HomogeneousRelationResult]) -> None:
    """Log the ranked results as an aligned table (strongest candidate first)."""
    header = f"{'rank':>4}  {'triple':60}  {'degrees':>10}  {'error':>10}"
    logger.info(header)
    logger.info("-" * len(header))
    for rank, r in enumerate(results, start=1):
        logger.info(
            "%4d  %-60s  %10s  %10.4g",
            rank, ", ".join(r.names), "/".join(str(d) for d in r.degrees), r.error,
        )


def _write_csv(results: list[HomogeneousRelationResult], path: str) -> None:
    """Write the full ranking to ``path`` as CSV, creating parent directories as needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name_1", "name_2", "name_3", "degree_1", "degree_2", "degree_3", "error"])
        for r in results:
            writer.writerow([*r.names, *r.degrees, r.error])
    logger.info("wrote full results (%d triples) to %s", len(results), path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
