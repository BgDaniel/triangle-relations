"""Render docs/theory.pdf from plain text/mathtext, as a fallback PDF.

docs/theory.tex is the authoritative source and should be compiled with a
real LaTeX engine (e.g. ``pdflatex theory.tex`` via MiKTeX/TeX Live) for a
properly typeset document; this script exists only to produce a readable PDF
on machines without a LaTeX installation, using matplotlib's built-in
mathtext renderer (no external dependencies beyond matplotlib).

Run with:
    poetry run python docs/render_theory_pdf.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PAGE_SIZE = (8.5, 11)
MARGIN_TOP = 0.95
MARGIN_LEFT = 0.08
LINE_HEIGHT = 0.030
WRAP_WIDTH = 92

FONT_SIZES = {"title": 18, "heading": 13, "para": 10.5, "eq": 12}


def new_page():
    fig = plt.figure(figsize=PAGE_SIZE)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis("off")
    return fig, ax, MARGIN_TOP


def draw_title(ax, y, text):
    ax.text(0.5, y, text, ha="center", va="top", fontsize=FONT_SIZES["title"], weight="bold")
    return y - 2.2 * LINE_HEIGHT


def draw_heading(ax, y, text):
    ax.text(MARGIN_LEFT, y, text, ha="left", va="top", fontsize=FONT_SIZES["heading"], weight="bold")
    return y - 1.6 * LINE_HEIGHT


def draw_para(ax, y, text):
    for line in textwrap.wrap(text, WRAP_WIDTH):
        ax.text(MARGIN_LEFT, y, line, ha="left", va="top", fontsize=FONT_SIZES["para"])
        y -= LINE_HEIGHT
    return y - 0.5 * LINE_HEIGHT


def draw_eq(ax, y, mathtext):
    ax.text(0.5, y, f"${mathtext}$", ha="center", va="top", fontsize=FONT_SIZES["eq"])
    return y - 1.6 * LINE_HEIGHT


def draw_list(ax, y, items):
    for item in items:
        wrapped = textwrap.wrap(item, WRAP_WIDTH - 3)
        ax.text(MARGIN_LEFT + 0.02, y, "• " + wrapped[0], ha="left", va="top", fontsize=FONT_SIZES["para"])
        y -= LINE_HEIGHT
        for cont in wrapped[1:]:
            ax.text(MARGIN_LEFT + 0.05, y, cont, ha="left", va="top", fontsize=FONT_SIZES["para"])
            y -= LINE_HEIGHT
    return y - 0.5 * LINE_HEIGHT


def build_pdf(out_path: Path) -> None:
    with PdfPages(out_path) as pdf:
        # ---- Page 1 ----
        fig, ax, y = new_page()
        y = draw_title(ax, y, "Discovering Hidden Relations\nAmong Triangle Invariants")
        y -= LINE_HEIGHT
        y = draw_heading(ax, y, "1. Setup: the moduli space of triangles")
        y = draw_para(
            ax, y,
            "A triangle is specified by three points in the plane: six real parameters. "
            "Every quantity we consider (area, perimeter, circumradius R, inradius r, "
            "distances between derived points, ...) is built purely from the vertex "
            "coordinates and is invariant under rigid motions: translating or rotating "
            "the triangle changes none of them. Translations (2 parameters) and rotations "
            "(1 parameter) form a 3-dimensional group acting on the 6-dimensional space of "
            "vertex triples; quotienting it out leaves a 3-dimensional moduli space M of "
            "triangle shapes up to congruence (reflection is a further discrete symmetry "
            "and does not change this dimension count). Concretely, M can be "
            "coordinatized by the three side lengths (a, b, c), subject to the triangle "
            "inequality.",
        )
        y = draw_para(
            ax, y,
            "Every derived scalar quantity is a smooth function f : M -> R. Choosing k "
            "such quantities gives a smooth map F = (f_1, ..., f_k) : M -> R^k.",
        )
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Page 2 ----
        fig, ax, y = new_page()
        y = draw_heading(ax, y, "2. Why relations among four or more quantities are not interesting")
        y = draw_para(
            ax, y,
            "If k > dim(M) = 3, a generic smooth map from a 3-manifold into R^k has, as "
            "its image, an (at most) 3-dimensional submanifold of R^k -- embedding a "
            "3-dimensional space into a higher-dimensional ambient space produces k-3 "
            "constraint equations for free, regardless of which k functions were chosen, "
            "as long as they are not specially degenerate. This image is cut out locally "
            "by k-3 independent equations G_i(f_1, ..., f_k) = 0. So finding some relation "
            "among four or more triangle invariants is guaranteed by dimension counting "
            "alone -- it says nothing specific about triangle geometry.",
        )
        y = draw_heading(ax, y, "3. Why a relation among exactly three quantities is special")
        y = draw_para(
            ax, y,
            "If k = dim(M) = 3 exactly, a generic map F : M -> R^3 is, at a generic "
            "point, a local diffeomorphism: its Jacobian DF has full rank 3, and the "
            "triple (f1, f2, f3) sweeps out a full 3-dimensional open region of R^3 -- a "
            "solid blob, not a surface. An exact relation G(f1, f2, f3) = 0 holding "
            "everywhere is equivalent to DF having rank <= 2 everywhere: the image "
            "collapses onto a 2-dimensional surface. That is not generic -- it is a "
            "genuine algebraic coincidence, exactly the kind of fact that constitutes a "
            "classical theorem. Euler's relation",
        )
        y = draw_eq(ax, y, r"d^2 = R^2 - 2Rr")
        y = draw_para(
            ax, y,
            "(d = distance between incenter and circumcenter) is exactly a statement that "
            "the Jacobian of (R, r, d) has rank <= 2 everywhere, collapsing three "
            "ostensibly independent quantities onto a single surface. This is the precise "
            "sense in which searching for relations among exactly three derived scalars "
            "-- and not four -- is the search for something structurally new.",
        )
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Page 3 ----
        fig, ax, y = new_page()
        y = draw_heading(ax, y, "4. Detecting rank-deficiency with a bottleneck autoencoder")
        y = draw_para(
            ax, y,
            "Given a candidate triple (f1, f2, f3), we want a numerical test for whether "
            "the induced map F has full rank 3 generically (data fills a 3D region) or is "
            "everywhere rank <= 2 (data confined to a 2D surface). We sample many random "
            "triangles and evaluate (f1, f2, f3) on each, producing a point cloud X in R^3.",
        )
        y = draw_para(
            ax, y,
            "An autoencoder with a 2-dimensional latent bottleneck -- encoder e: R^3 -> "
            "R^2, decoder d: R^2 -> R^3, trained to minimize E||d(e(x)) - x||^2 -- is "
            "exactly searching for the best rank-2 (nonlinear) approximation to the data:",
        )
        y = draw_list(
            ax, y,
            [
                "If the data truly lies on a 2D surface (an exact relation holds), a "
                "2-dimensional latent code is not actually a bottleneck for it -- a "
                "sufficiently expressive encoder/decoder can drive reconstruction error "
                "to near zero: the encoder learns a local chart of the surface, the "
                "decoder its inverse.",
                "If the data genuinely fills an open 3D region, no continuous map "
                "factoring through a 2D intermediate space can reconstruct it without "
                "loss: an entire dimension of variation is discarded, so reconstruction "
                "error is bounded away from zero, on the order of the data's own extent "
                "in that direction.",
            ],
        )
        y = draw_para(
            ax, y,
            "This gives a clean signature: near-zero reconstruction error with a "
            "2-dimensional bottleneck is evidence for a relation among the three chosen "
            "quantities; error comparable to the data's own scale is consistent with the "
            "generic, full-rank, no-relation case.",
        )
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Page 4 ----
        fig, ax, y = new_page()
        y = draw_heading(ax, y, "5. Calibrating “near zero” with a permutation null")
        y = draw_para(
            ax, y,
            "An absolute error threshold is not meaningful across different triples: "
            "they differ in units, scale, and marginal shape, and since all derived "
            "scalars are ultimately functions of the same three triangle parameters, some "
            "triples are mildly correlated even without an exact relation. We need a "
            "triple-specific baseline for “no relation”.",
        )
        y = draw_para(
            ax, y,
            "We construct a null sample X' by independently permuting each of the three "
            "columns of X. This preserves each quantity's own marginal distribution while "
            "completely destroying the joint dependency between them. Any real functional "
            "relation is annihilated by this shuffle. Training the same autoencoder on X' "
            "(repeated a few times) estimates the error expected for three quantities with "
            "these marginals but no shared structure. Comparing the real error against this "
            "null's mean and standard deviation (z-score, or ratio) yields a self-"
            "calibrated significance measure, robust to units and distribution shape. A "
            "small ratio is the fingerprint of an unexpected relation.",
        )
        y = draw_heading(ax, y, "6. From detection to an explicit formula")
        y = draw_para(
            ax, y,
            "The autoencoder step is a screening tool: it flags which triples are "
            "suspicious, not what the relation says. Once flagged, we build a dictionary "
            "of monomials in (f1, f2, f3) up to degree D, evaluate them on the sampled "
            "data to form a design matrix Phi (rescaling each column to comparable "
            "variance), and look for a linear combination of its columns that vanishes on "
            "every sample -- the right singular vector of the smallest singular value.",
        )
        y = draw_para(
            ax, y,
            "Applied to (R, r, d) at degree 2, this recovers -- up to overall scale and "
            "sign -- Euler's relation on every sampled triangle, found purely by sampling "
            "random triangles and asking a null-space question, with no prior knowledge "
            "of the formula.",
        )
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Page 5 ----
        fig, ax, y = new_page()
        y = draw_heading(ax, y, "7. The full pipeline")
        y = draw_list(
            ax, y,
            [
                "Enumerate candidate triples of derived scalar quantities.",
                "For each triple, sample random triangles once, train a 2-bottleneck "
                "autoencoder, and compare its held-out reconstruction error to a "
                "column-shuffled null (several repeats).",
                "Rank triples by how much better the real data compresses than its null "
                "counterpart (small ratio = strong candidate).",
                "For top candidates, fit a low-degree polynomial null-space relation and "
                "validate that it holds (residual near zero) on fresh samples.",
            ],
        )
        y = draw_para(
            ax, y,
            "triangle_relations/discovery/verify_euler_relation.py runs this exact "
            "pipeline on (R, r, d) as a worked example, confirming it rediscovers Euler's "
            "relation end to end.",
        )
        pdf.savefig(fig)
        plt.close(fig)


if __name__ == "__main__":
    build_pdf(Path(__file__).parent / "theory.pdf")
    print("Wrote docs/theory.pdf")
