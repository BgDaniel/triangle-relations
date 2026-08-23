"""Plot the z-score ranking from a saved discover_scalar_relations.py CSV.

This is meant to be run directly (e.g. from within an IDE): edit CSV_PATH
below to point at a ranking CSV (written when discover_scalar_relations.py
is run with OUTPUT_CSV set) and run this script.

Run with:
    poetry run python scripts/plot_ranking.py
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt

from triangle_relations.discovery.ranking_plot import plot_ranking_from_csv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration -- edit these directly and re-run.
# ---------------------------------------------------------------------------

#: Same environment variable discover_scalar_relations.py reads its output
#: directory from, so both scripts point at the same ranking.csv by default.
OUTPUT_DIR_ENV_VAR = "PATH_TO_OUTPUT_FOLDER"

#: Path to the ranking CSV written by scripts/discover_scalar_relations.py.
#: Derived from the OUTPUT_DIR_ENV_VAR environment variable if it's set;
#: edit directly to override.
_output_dir = os.environ.get(OUTPUT_DIR_ENV_VAR)
CSV_PATH: str = str(Path(_output_dir) / "ranking.csv") if _output_dir else "output/ranking.csv"

#: Number of top-ranked triples to show.
TOP = 20


def main() -> None:
    """Load CSV_PATH and plot its top TOP candidate triples by z-score."""
    logger.info("reading ranking from %s", CSV_PATH)
    plot_ranking_from_csv(CSV_PATH, top=TOP)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
