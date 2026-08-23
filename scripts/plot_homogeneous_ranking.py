"""Plot the ranking from a saved discover_homogeneous_relations.py CSV.

This is meant to be run directly (e.g. from within an IDE): edit CSV_PATH
below to point at a ranking CSV (written when discover_homogeneous_relations.py
is run with OUTPUT_CSV set) and run this script.

For Program 1's z-score/ratio rankings, see scripts/plot_ranking.py instead.
Unlike that script, there is only one plot here: Program 1b reports a single
score per triple (no z-score/ratio split, and no null-based relative-sigma
companion metric -- see triangle_relations.discovery.ranking_plot's module
docstring).

Run with:
    poetry run python scripts/plot_homogeneous_ranking.py
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt

from triangle_relations.discovery.homogeneous_relations import log_euler_triple_rank
from triangle_relations.discovery.ranking_plot import (
    load_homogeneous_ranking_csv,
    plot_homogeneous_ranking,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration -- edit these directly and re-run.
# ---------------------------------------------------------------------------

#: Same environment variable discover_homogeneous_relations.py reads its
#: output directory from, so both scripts point at the same CSV by default.
OUTPUT_DIR_ENV_VAR = "PATH_TO_OUTPUT_FOLDER"

#: Path to the ranking CSV written by scripts/discover_homogeneous_relations.py.
#: Derived from the OUTPUT_DIR_ENV_VAR environment variable if it's set;
#: edit directly to override.
_output_dir = os.environ.get(OUTPUT_DIR_ENV_VAR)
CSV_PATH: str = (
    str(Path(_output_dir) / "homogeneous_ranking.csv")
    if _output_dir
    else "output/homogeneous_ranking.csv"
)

#: Number of top-ranked triples to show.
TOP = 20


def main() -> None:
    """Load CSV_PATH and plot its top TOP candidate triples by (sphere-collapse) error."""
    logger.info("reading ranking from %s", CSV_PATH)
    results = load_homogeneous_ranking_csv(CSV_PATH)
    log_euler_triple_rank(results)
    plot_homogeneous_ranking(results, top=TOP)
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
