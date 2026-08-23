"""Plot the ranking of candidate scalar triples produced by Program 1.

:func:`plot_ranking` draws two stacked, x-axis-aligned bar-chart panels for
the top-``top`` candidate triples by z-score: the z-score itself on top, and
the *relative* null standard deviation (``null_std / null_mean``) for those
same triples, in the same order, below. Since
``z = (null_mean - real_error) / null_std``, a small ``null_std`` alone can
inflate a z-score without the real/null gap actually being large; triples
whose relative sigma falls below :data:`SMALL_RELATIVE_SIGMA_THRESHOLD` are
flagged in orange in both panels, as a visual cue to treat that z-score with
more caution (e.g. by checking ``ratio`` instead, from the CSV or the log
table) rather than take it at face value.

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
from triangle_relations.geometry.triangle import Triangle

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

#: Relative null standard deviation (null_std / null_mean) below this is
#: flagged: a small denominator can inflate a triple's z-score even without
#: a strong real/null gap, so its z-score alone is less trustworthy.
SMALL_RELATIVE_SIGMA_THRESHOLD = 0.15


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


def _symbol_label(names: tuple[str, str, str]) -> str:
    """Render a scalar triple as its short-symbol form, e.g. ``(R, r, OI)``."""
    return "(" + ", ".join(Triangle.scalar_symbol(n) for n in names) + ")"


def _relative_sigma(r: RelationResult) -> float:
    """``null_std / null_mean``, or 0.0 if ``null_mean`` is non-positive (near-degenerate)."""
    return r.null_std / r.null_mean if r.null_mean > 0 else 0.0


def plot_ranking(results: list[RelationResult], *, top: int = 20) -> "Figure":
    """Plot the top triples by z-score, with their relative null std alongside.

    Two stacked panels share the same x-axis: the same top-``top`` triples,
    in the same z-score-ranked order, so a given row shows both numbers for
    the same triple. Scalar names are abbreviated to their short symbols
    (see :attr:`~triangle_relations.geometry.triangle.Triangle.SCALAR_SYMBOLS`),
    e.g. ``(R, r, OI)`` for ``(circumradius, inradius, dist_circumcenter__incenter)``.

    Parameters
    ----------
    results:
        Results as returned by
        :func:`~triangle_relations.discovery.scalar_relations.search_three_scalar_relations`
        or :func:`load_ranking_csv`, in any order.
    top:
        Maximum number of triples to show.

    Returns
    -------
    The matplotlib ``Figure`` containing both panels.
    """
    if not results:
        raise ValueError("no results to plot")

    shown = sorted(results, key=lambda r: r.z_score, reverse=True)[:top]
    labels = [_symbol_label(r.names) for r in shown]
    z_scores = [r.z_score for r in shown]
    rel_sigmas = [_relative_sigma(r) for r in shown]
    flagged = [s < SMALL_RELATIVE_SIGMA_THRESHOLD for s in rel_sigmas]
    colors = ["tab:orange" if f else "tab:blue" for f in flagged]
    logger.info(
        "%d of the top %d triples have relative null std below %.2f (z-score may be inflated)",
        sum(flagged), len(shown), SMALL_RELATIVE_SIGMA_THRESHOLD,
    )

    figsize = (max(8.0, 0.55 * len(shown) + 2.0), 7.0)
    fig, (ax_z, ax_sigma) = plt.subplots(
        2, 1, figsize=figsize, sharex=True, constrained_layout=True,
    )

    x = range(len(shown))
    ax_z.bar(x, z_scores, color=colors)
    ax_z.axhline(0, color="black", linewidth=0.8)
    ax_z.set_ylabel("z-score")
    ax_z.set_title(f"Top {len(shown)} of {len(results)} by z-score", fontsize=10)
    ax_z.tick_params(labelbottom=False)  # x labels drawn once, on the shared bottom panel

    ax_sigma.bar(x, rel_sigmas, color=colors)
    ax_sigma.axhline(
        SMALL_RELATIVE_SIGMA_THRESHOLD, color="gray", linewidth=0.8, linestyle="--",
    )
    ax_sigma.set_ylabel("relative null std\n(null_std / null_mean)")
    ax_sigma.set_title("Same triples: how tight was the null estimate?", fontsize=10)

    ax_sigma.set_xticks(list(x))
    ax_sigma.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)

    # constrained_layout (unlike tight_layout) reserves space for the
    # suptitle automatically, so it doesn't overlap the panels below it.
    fig.suptitle(
        f"Candidate scalar triples ranked by z-score\n"
        f"(orange = relative null std < {SMALL_RELATIVE_SIGMA_THRESHOLD}: "
        f"treat this z-score with caution, e.g. check ratio instead)"
    )
    return fig


def plot_ranking_from_csv(path: str | Path, *, top: int = 20) -> "Figure":
    """Convenience wrapper: :func:`load_ranking_csv` then :func:`plot_ranking`."""
    return plot_ranking(load_ranking_csv(path), top=top)
