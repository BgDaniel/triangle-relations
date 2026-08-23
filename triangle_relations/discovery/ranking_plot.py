"""Plot the ranking of candidate scalar triples produced by Program 1.

:func:`plot_ranking` draws a horizontal bar chart of z-scores for the
top-ranked triples returned by
:func:`~triangle_relations.discovery.scalar_relations.search_three_scalar_relations`.
:func:`load_ranking_csv` reconstructs that same list of results from a CSV
file previously written by ``scripts/discover_scalar_relations.py``, so a
completed search can be re-plotted later without rerunning it; see
``scripts/plot_ranking.py`` for a ready-to-run script that does exactly this.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from triangle_relations.discovery.scalar_relations import RelationResult

if TYPE_CHECKING:
    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)


def load_ranking_csv(path: str | Path) -> list[RelationResult]:
    """Load a ranking CSV written by ``scripts/discover_scalar_relations.py``.

    Parameters
    ----------
    path:
        CSV path with columns ``name_1, name_2, name_3, real_error,
        null_mean, null_std, z_score, ratio`` (the ``ratio`` column is
        ignored on load, since it is a derived property of
        :class:`RelationResult`).

    Returns
    -------
    Results sorted by ascending ``ratio`` (strongest candidate first), as
    :func:`~triangle_relations.discovery.scalar_relations.search_three_scalar_relations`
    would return them.
    """
    results = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            results.append(
                RelationResult(
                    names=(row["name_1"], row["name_2"], row["name_3"]),
                    real_error=float(row["real_error"]),
                    null_mean=float(row["null_mean"]),
                    null_std=float(row["null_std"]),
                    z_score=float(row["z_score"]),
                )
            )
    logger.info("loaded %d result(s) from %s", len(results), path)
    return sorted(results, key=lambda r: r.ratio)


def plot_ranking(
    results: list[RelationResult],
    *,
    top: int = 20,
    ax: "Axes | None" = None,
) -> "Axes":
    """Plot a horizontal bar chart of z-scores for the top-ranked triples.

    Parameters
    ----------
    results:
        Results as returned by
        :func:`~triangle_relations.discovery.scalar_relations.search_three_scalar_relations`
        or :func:`load_ranking_csv`, in any order: this function sorts them
        by descending ``z_score`` itself (independently of whatever order
        they arrived in, e.g. the ``ratio``-based order
        ``search_three_scalar_relations`` returns) before selecting and
        plotting the top ``top``, so the plotted order always matches what
        it's plotting.
    top:
        Maximum number of triples to show.
    ax:
        An existing matplotlib ``Axes`` to draw into; a new figure is
        created if omitted.

    Returns
    -------
    The matplotlib ``Axes`` used for the plot.
    """
    if not results:
        raise ValueError("no results to plot")

    shown = sorted(results, key=lambda r: r.z_score, reverse=True)[:top]
    labels = [", ".join(r.names) for r in shown]
    z_scores = [r.z_score for r in shown]

    if ax is None:
        logger.debug("no Axes supplied; creating a new figure")
        _, ax = plt.subplots(figsize=(9, 0.4 * len(shown) + 1.5))

    y = range(len(shown))
    colors = ["tab:green" if z >= 0 else "tab:red" for z in z_scores]
    ax.barh(y, z_scores, color=colors)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()  # strongest candidate (first in the list) on top
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("z-score")
    ax.set_title(f"Top {len(shown)} of {len(results)} candidate triples by z-score")
    ax.figure.tight_layout()
    return ax


def plot_ranking_from_csv(
    path: str | Path,
    *,
    top: int = 20,
    ax: "Axes | None" = None,
) -> "Axes":
    """Convenience wrapper: :func:`load_ranking_csv` then :func:`plot_ranking`."""
    return plot_ranking(load_ranking_csv(path), top=top, ax=ax)
