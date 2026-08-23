"""Search for functional relations among triples of triangle-derived scalars.

This is meant to be run directly (e.g. from within an IDE), not from the
command line: edit the configuration constants below and run the script.

Run with:
    poetry run python scripts/discover_scalar_relations.py
"""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from triangle_relations.discovery.ranking_plot import (
    log_euler_triple_rank,
    plot_ratio_ranking,
    plot_z_score_ranking,
)
from triangle_relations.discovery.sampling import build_scalar_dataset
from triangle_relations.discovery.scalar_relations import RelationResult, search_three_scalar_relations
from triangle_relations.geometry.triangle import Triangle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration -- edit these directly and re-run.
# ---------------------------------------------------------------------------

#: Number of random triangles to sample.
N_SAMPLES = 5000

#: Random triangle vertices are drawn uniformly from [-SCALE, SCALE]^2.
SCALE = 1.0

#: Autoencoder hidden-layer width (topology is (HIDDEN, 2, HIDDEN)).
HIDDEN = 8

#: Autoencoder random restarts per training run (real data and each null).
N_RESTARTS = 1

#: Number of independent column-shuffled null datasets averaged per triple.
N_SHUFFLES = 500

#: Held-out fraction used to measure autoencoder reconstruction error.
TEST_SIZE = 0.3

#: Number of top-ranked triples to report.
TOP = 10

#: Passed to joblib.Parallel; -1 uses all available cores.
N_JOBS = -1

#: Seed for reproducible sampling and null shuffles.
SEED = 0

#: Subset of scalar names to search (must be keys of Triangle.SCALARS), or
#: None to search every registered scalar. The full search is combinatorial
#: (C(n, 3) triples) and can be slow; restrict this list for a quick pass.
SCALAR_NAMES: list[str] | None = None

#: Environment variable holding the directory where output files (e.g. the
#: ranking CSV) are written. Set this once in your environment rather than
#: hardcoding a machine-specific path here -- see the README for how to set
#: it permanently in VS Code.
OUTPUT_DIR_ENV_VAR = "PATH_TO_OUTPUT_FOLDER"

#: Optional path to write the full ranking as CSV, or None to skip. Derived
#: from the OUTPUT_DIR_ENV_VAR environment variable if it's set (as
#: "<OUTPUT_DIR>/ranking.csv"); edit directly to override, or set the env
#: var to None here to disable regardless of the environment.
_output_dir = os.environ.get(OUTPUT_DIR_ENV_VAR)
OUTPUT_CSV: str | None = str(Path(_output_dir) / "ranking.csv") if _output_dir else None

#: Whether to plot the z-score and ratio rankings (see triangle_relations
#: .discovery.ranking_plot) after a successful run.
PLOT_RANKING: bool = True


def main() -> None:
    """Sample random triangles, search all scalar triples, and report the ranking."""
    if OUTPUT_CSV:
        logger.info("writing ranking to %s (from $%s)", OUTPUT_CSV, OUTPUT_DIR_ENV_VAR)
    else:
        logger.info(
            "$%s is not set (or OUTPUT_CSV was overridden to None); skipping CSV output",
            OUTPUT_DIR_ENV_VAR,
        )

    rng = np.random.default_rng(SEED)

    if SCALAR_NAMES is None:
        logger.info("available scalars (%d): %s", len(Triangle.SCALARS), sorted(Triangle.SCALARS))

    logger.info("sampling %d random triangles...", N_SAMPLES)
    names, data = build_scalar_dataset(N_SAMPLES, rng, scalar_names=SCALAR_NAMES, scale=SCALE)

    n_triples = len(names) * (len(names) - 1) * (len(names) - 2) // 6
    logger.info("searching %d combinations of 3 scalars out of %d...", n_triples, len(names))

    results = search_three_scalar_relations(
        names,
        data,
        n_shuffles=N_SHUFFLES,
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

    if PLOT_RANKING:
        plot_z_score_ranking(results, top=TOP)
        plot_ratio_ranking(results, top=TOP)
        plt.show()


def _log_ranking(results: list[RelationResult]) -> None:
    """Log the ranked results as an aligned table (strongest candidate first)."""
    header = f"{'rank':>4}  {'triple':60}  {'real_err':>10}  {'null_mean':>10}  {'z_score':>8}  {'ratio':>8}"
    logger.info(header)
    logger.info("-" * len(header))
    for rank, r in enumerate(results, start=1):
        logger.info(
            "%4d  %-60s  %10.4g  %10.4g  %8.2f  %8.3f",
            rank, ", ".join(r.names), r.real_error, r.null_mean, r.z_score, r.ratio,
        )


def _write_csv(results: list[RelationResult], path: str) -> None:
    """Write the full ranking to ``path`` as CSV, creating parent directories as needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name_1", "name_2", "name_3", "real_error", "null_mean", "null_std", "z_score", "ratio"])
        for r in results:
            writer.writerow([*r.names, r.real_error, r.null_mean, r.null_std, r.z_score, r.ratio])
    logger.info("wrote full results (%d triples) to %s", len(results), path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
