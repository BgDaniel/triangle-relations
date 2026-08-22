"""Plot a sample triangle together with all of its derived objects.

This is meant to be run directly (e.g. from within an IDE): it has no
command-line arguments, edit ``SAMPLE_TRIANGLE`` below to plot a different
triangle.

Run with:
    poetry run python scripts/plot_sample_triangle.py
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt

from triangle_relations import Triangle

logger = logging.getLogger(__name__)

#: The three vertices of the triangle to plot.
SAMPLE_TRIANGLE = Triangle((0.5, 0.2), (4.0, 0.8), (1.5, 3.2))


def main() -> None:
    """Plot :data:`SAMPLE_TRIANGLE` with every derived object shown, and display it."""
    logger.info("plotting sample triangle with vertices:\n%s", SAMPLE_TRIANGLE.vertices)
    SAMPLE_TRIANGLE.plot()
    plt.title("Sample triangle with all derived objects")
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
