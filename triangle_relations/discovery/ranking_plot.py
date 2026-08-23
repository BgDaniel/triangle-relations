"""Plot the ranking of candidate scalar triples produced by Program 1.

Two separate figures, both with the same two-panel structure: the ranking
metric on top, and the *relative* null standard deviation
(``null_std / null_mean``) for those same triples, in the same order, below
(since ``z = (null_mean - real_error) / null_std``, a small ``null_std``
alone can inflate a z-score without the real/null gap actually being large --
see :func:`plot_z_score_ranking`'s docstring for why this matters, and the
module-level note on shuffle counts below for why it matters *more* than you
might expect).

* :func:`plot_z_score_ranking` ranks by z-score, bigger bar = stronger evidence.
* :func:`plot_ratio_ranking` ranks by ratio (``real_error / null_mean``),
  smaller bar = stronger evidence (a reference line at 1.0 marks "no better
  than the shuffled null").

In both figures, a triple that makes the top-``top`` list under *both*
z-score and ratio, or that is the classical triple behind Euler's relation
(:data:`EULER_TRIPLE_NAMES`), is drawn in red instead of blue; the Euler
triple additionally gets a small "Euler" label above its bar, since a triple
can be red for either reason and the label disambiguates which. The Euler
triple is always included (even if it falls outside the top ``top``), so
it's visible as a reference point in every plot; :func:`log_euler_triple_rank`
logs its rank position under both metrics regardless of whether it's plotted.

A note on ``n_shuffles`` (how many null repeats each triple's
``null_std``/``null_mean`` are estimated from): the standard error of a
sample standard deviation estimated from :math:`n` points is approximately
:math:`\\sigma / \\sqrt{2(n-1)}` -- so at :math:`n=3` the estimate of
``null_std`` itself carries roughly 50% relative uncertainty, and even at
:math:`n=30` (a common default here) it's still roughly 13%. Reaching 10%
takes about 50 shuffles, and 5% takes about 200. Since z-score divides by
this noisy estimate, a small ``n_shuffles`` genuinely can produce
misleadingly large or small z-scores, independent of `top`, which is
exactly what the second (relative-null-std) panel is for: it's a cheap,
per-triple sanity check on how much to trust the z-score panel above it, no
matter how many shuffles were used to compute it. If you need a
quantitatively trustworthy z-score, the cheapest way to get one is a
two-stage search: rank the full candidate set once with a small
``n_shuffles`` (using ``ratio``, which doesn't depend on this noisy estimate
at all, as the primary signal), then rerun just the top handful of
candidates with a much larger ``n_shuffles`` (50-200+) to get a properly
precise z-score for those finalists.

:func:`load_ranking_csv` reconstructs a list of results from a CSV file
previously written by ``scripts/discover_scalar_relations.py``, so a
completed search can be re-plotted later without rerunning it; see
``scripts/plot_ranking.py`` for a ready-to-run script that does exactly this.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import matplotlib.pyplot as plt

from triangle_relations.discovery.known_relations import EULER_TRIPLE_NAMES, is_euler_triple
from triangle_relations.discovery.scalar_relations import RelationResult
from triangle_relations.geometry.triangle import Triangle

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

#: Relative null standard deviation (null_std / null_mean) below this is
#: flagged in the log message: a small denominator can inflate a triple's
#: z-score even without a strong real/null gap. See the module docstring
#: for how this connects to the number of shuffles used to estimate it.
SMALL_RELATIVE_SIGMA_THRESHOLD = 0.15

_HIGHLIGHT_COLOR = "red"
_DEFAULT_COLOR = "tab:blue"


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


def _is_euler_triple(r: RelationResult) -> bool:
    return is_euler_triple(r.names)


def _top_name_sets(results: list[RelationResult], top: int) -> tuple[set[frozenset], set[frozenset]]:
    """The ``top``-``top`` triples' name-sets, ranked by z-score and by ratio respectively."""
    by_z = sorted(results, key=lambda r: r.z_score, reverse=True)[:top]
    by_ratio = sorted(results, key=lambda r: r.ratio)[:top]
    return {frozenset(r.names) for r in by_z}, {frozenset(r.names) for r in by_ratio}


def _select_with_euler_reference(
    results: list[RelationResult], shown: list[RelationResult],
) -> tuple[list[RelationResult], bool]:
    """Append the Euler triple to ``shown`` if it exists in ``results`` but isn't already shown.

    Returns the (possibly extended) list and whether an append happened, so
    callers can note it in their title.
    """
    if any(_is_euler_triple(r) for r in shown):
        return shown, False
    euler_result = next((r for r in results if _is_euler_triple(r)), None)
    if euler_result is None:
        return shown, False
    return [*shown, euler_result], True


def _highlight_colors(shown: list[RelationResult], in_both: set[frozenset]) -> list[str]:
    return [
        _HIGHLIGHT_COLOR if (frozenset(r.names) in in_both or _is_euler_triple(r)) else _DEFAULT_COLOR
        for r in shown
    ]


def _label_euler_bar(ax: "Axes", shown: list[RelationResult], x: range, heights: list[float]) -> None:
    """Annotate the Euler triple's bar with a small "Euler" label, if present in ``shown``."""
    for i, r in enumerate(shown):
        if _is_euler_triple(r):
            ax.annotate(
                "Euler", (x[i], heights[i]), textcoords="offset points", xytext=(0, 4),
                ha="center", fontsize=7, color="black",
            )


def log_euler_triple_rank(results: list[RelationResult]) -> None:
    """Log the Euler triple's rank position under both metrics, if present in ``results``.

    Useful independently of plotting, to see at a glance how the known
    reference relation fares in a given search.
    """
    if not any(_is_euler_triple(r) for r in results):
        logger.info("Euler triple (R, r, OI) is not present in this result set")
        return
    by_z = sorted(results, key=lambda r: r.z_score, reverse=True)
    by_ratio = sorted(results, key=lambda r: r.ratio)
    rank_z = next(i for i, r in enumerate(by_z) if _is_euler_triple(r)) + 1
    rank_ratio = next(i for i, r in enumerate(by_ratio) if _is_euler_triple(r)) + 1
    logger.info(
        "Euler triple (R, r, OI) ranks #%d of %d by z-score, #%d of %d by ratio",
        rank_z, len(results), rank_ratio, len(results),
    )


def _two_panel_ranking(
    results: list[RelationResult],
    *,
    top: int,
    sort_key: Callable[[RelationResult], float],
    sort_reverse: bool,
    value_fn: Callable[[RelationResult], float],
    value_ylabel: str,
    metric_name: str,
    reference_line: float,
) -> "Figure":
    """Shared implementation behind :func:`plot_z_score_ranking` and :func:`plot_ratio_ranking`.

    Draws two stacked, x-axis-aligned panels: ``value_fn`` (ranked by
    ``sort_key``) on top, and the same triples' relative null std below.
    """
    if not results:
        raise ValueError("no results to plot")

    shown = sorted(results, key=sort_key, reverse=sort_reverse)[:top]
    shown, euler_appended = _select_with_euler_reference(results, shown)
    by_z_top, by_ratio_top = _top_name_sets(results, top)
    in_both = by_z_top & by_ratio_top

    labels = [_symbol_label(r.names) for r in shown]
    values = [value_fn(r) for r in shown]
    rel_sigmas = [_relative_sigma(r) for r in shown]
    colors = _highlight_colors(shown, in_both)

    flagged = [s < SMALL_RELATIVE_SIGMA_THRESHOLD for s in rel_sigmas]
    logger.info(
        "%d of %d shown triples (ranked by %s) have relative null std below %.2f "
        "(their z-score may be inflated)",
        sum(flagged), len(shown), metric_name, SMALL_RELATIVE_SIGMA_THRESHOLD,
    )

    figsize = (max(8.0, 0.55 * len(shown) + 2.0), 7.0)
    fig, (ax_top, ax_sigma) = plt.subplots(
        2, 1, figsize=figsize, sharex=True, constrained_layout=True,
    )

    x = range(len(shown))
    ax_top.bar(x, values, color=colors)
    ax_top.axhline(reference_line, color="black", linewidth=0.8)
    ax_top.set_ylabel(value_ylabel)
    title_suffix = " (+ Euler reference)" if euler_appended else ""
    ax_top.set_title(f"Top {len(shown)} of {len(results)} by {metric_name}{title_suffix}", fontsize=10)
    ax_top.tick_params(labelbottom=False)  # x labels drawn once, on the shared bottom panel
    _label_euler_bar(ax_top, shown, x, values)

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
        f"Candidate scalar triples ranked by {metric_name}\n"
        f"(red = top-ranked by both z-score and ratio, or the reference Euler triple)"
    )
    return fig


def plot_z_score_ranking(results: list[RelationResult], *, top: int = 20) -> "Figure":
    """Plot the top triples by z-score, with their relative null std alongside.

    Two stacked panels share the same x-axis: the same top-``top`` triples,
    in the same z-score-ranked order, so a given column shows both numbers
    for the same triple. Scalar names are abbreviated to their short symbols
    (see :attr:`~triangle_relations.geometry.triangle.Triangle.SCALAR_SYMBOLS`),
    e.g. ``(R, r, OI)`` for ``(circumradius, inradius, dist_circumcenter__incenter)``.
    Red bars are triples ranked in the top ``top`` by *both* z-score and
    ratio, or the reference Euler triple (labeled); see the module docstring.

    Parameters
    ----------
    results:
        Results as returned by
        :func:`~triangle_relations.discovery.scalar_relations.search_three_scalar_relations`
        or :func:`load_ranking_csv`, in any order.
    top:
        Maximum number of triples to show (the Euler triple is always shown
        in addition, if present in ``results``).

    Returns
    -------
    The matplotlib ``Figure`` containing both panels.
    """
    return _two_panel_ranking(
        results, top=top,
        sort_key=lambda r: r.z_score, sort_reverse=True,
        value_fn=lambda r: r.z_score, value_ylabel="z-score",
        metric_name="z-score", reference_line=0.0,
    )


def plot_ratio_ranking(results: list[RelationResult], *, top: int = 20) -> "Figure":
    """Plot the top triples by ratio (``real_error / null_mean``), with their relative null std alongside.

    Bars plot the raw ratio, ascending (lowest, i.e. strongest, first), with
    a reference line at 1.0 -- a ratio of 1 means the autoencoder does no
    better on the real data than on the shuffled null, so smaller bars are
    stronger candidates here, unlike the z-score panel where bigger is
    stronger. The bottom panel is the same relative-null-std sanity check,
    for these (ratio-ranked) triples. Red bars are triples ranked in the top
    ``top`` by *both* ratio and z-score, or the reference Euler triple
    (labeled); see the module docstring.

    Parameters
    ----------
    results:
        Results as returned by
        :func:`~triangle_relations.discovery.scalar_relations.search_three_scalar_relations`
        or :func:`load_ranking_csv`, in any order.
    top:
        Maximum number of triples to show (the Euler triple is always shown
        in addition, if present in ``results``).

    Returns
    -------
    The matplotlib ``Figure`` containing both panels.
    """
    return _two_panel_ranking(
        results, top=top,
        sort_key=lambda r: r.ratio, sort_reverse=False,
        value_fn=lambda r: r.ratio,
        value_ylabel="ratio  (real_error / null_mean);\nlower = stronger",
        metric_name="ratio", reference_line=1.0,
    )


def load_and_plot_z_score_ranking(path: str | Path, *, top: int = 20) -> "Figure":
    """Convenience wrapper: :func:`load_ranking_csv` then :func:`plot_z_score_ranking`."""
    return plot_z_score_ranking(load_ranking_csv(path), top=top)


def load_and_plot_ratio_ranking(path: str | Path, *, top: int = 20) -> "Figure":
    """Convenience wrapper: :func:`load_ranking_csv` then :func:`plot_ratio_ranking`."""
    return plot_ratio_ranking(load_ranking_csv(path), top=top)
