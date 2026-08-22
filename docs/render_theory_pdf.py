"""Render docs/theory.pdf as a professionally typeset, journal-style document.

docs/theory.tex is the authoritative LaTeX source and should be compiled
with a real engine (e.g. ``pdflatex theory.tex`` via MiKTeX/TeX Live) for the
canonical version. This script renders the same content as a standalone PDF
using only matplotlib -- no LaTeX installation required -- by using
matplotlib's bundled STIX fonts (a serif family designed to match classic
math-journal typesetting, both for body text and for mathtext equations) and
a small hand-rolled text-layout engine that performs real word-justification
by measuring glyph advance widths.

Run with:
    poetry run python docs/render_theory_pdf.py
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextToPath

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page geometry (A4) and typography constants.
# ---------------------------------------------------------------------------

PAGE_WIDTH_IN = 8.27
PAGE_HEIGHT_IN = 11.69
PT_PER_IN = 72.0
PAGE_WIDTH_PT = PAGE_WIDTH_IN * PT_PER_IN
PAGE_HEIGHT_PT = PAGE_HEIGHT_IN * PT_PER_IN

MARGIN_IN = 1.0
CONTENT_WIDTH_PT = (PAGE_WIDTH_IN - 2 * MARGIN_IN) * PT_PER_IN
LEFT_FRAC = MARGIN_IN / PAGE_WIDTH_IN
RIGHT_FRAC = 1.0 - LEFT_FRAC
TOP_Y_FRAC = 1.0 - MARGIN_IN / PAGE_HEIGHT_IN
BOTTOM_Y_FRAC = MARGIN_IN / PAGE_HEIGHT_IN

TITLE_SIZE = 19.0
BYLINE_SIZE = 12.0
HEADING_SIZE = 13.0
BODY_SIZE = 10.5
EQUATION_SIZE = 14.0
ABSTRACT_SIZE = 10.0
FOOTER_SIZE = 9.0
HEADER_SIZE = 8.0

LINE_HEIGHT_FACTOR = 1.45
PARA_GAP_FACTOR = 0.9  # extra blank space after a paragraph, in body line-heights
ABSTRACT_INSET_IN = 0.4

DOC_TITLE = "Discovering Hidden Relations Among Triangle Invariants"

_DATA_PATH = Path(matplotlib.get_data_path())
PROP_REGULAR = FontProperties(fname=str(_DATA_PATH / "fonts/ttf/STIXGeneral.ttf"))
PROP_BOLD = FontProperties(fname=str(_DATA_PATH / "fonts/ttf/STIXGeneralBol.ttf"))
_T2P = TextToPath()

# matplotlib's mathtext also draws in the STIX family, so equations blend
# with the surrounding body text as they would in a real LaTeX document.
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


@lru_cache(maxsize=None)
def _measure(word: str, fontsize: float, bold: bool = False) -> float:
    """Return the advance width, in points, of ``word`` set at ``fontsize``.

    Cached because the same words (especially common ones) are measured
    repeatedly while wrapping and justifying paragraphs.
    """
    if word == "":
        return 0.0
    prop = (PROP_BOLD if bold else PROP_REGULAR).copy()
    prop.set_size(fontsize)
    width, _height, _descent = _T2P.get_text_width_height_descent(word, prop, ismath=False)
    return width


def _line_height_frac(fontsize: float) -> float:
    """Vertical spacing (in y-axis fraction) for one line of the given font size."""
    return (fontsize * LINE_HEIGHT_FACTOR) / PAGE_HEIGHT_PT


class _Document:
    """A minimal flowing-text PDF builder: tracks a page cursor and starts
    new pages automatically as content overflows, so section content does
    not need to be manually split into pages by hand."""

    def __init__(self, pdf: PdfPages) -> None:
        self._pdf = pdf
        self._page_num = 0
        self._fig = None
        self._ax = None
        self._y = 0.0
        self._new_page()

    def _new_page(self) -> None:
        if self._fig is not None:
            self._finish_page()
        self._page_num += 1
        self._fig = plt.figure(figsize=(PAGE_WIDTH_IN, PAGE_HEIGHT_IN))
        self._ax = self._fig.add_axes((0, 0, 1, 1))
        self._ax.axis("off")
        self._ax.set_xlim(0, 1)
        self._ax.set_ylim(0, 1)
        self._y = TOP_Y_FRAC
        if self._page_num > 1:
            self._ax.text(
                0.5, 1.0 - 0.55 / PAGE_HEIGHT_IN, DOC_TITLE,
                transform=self._ax.transAxes, ha="center", va="top",
                fontsize=HEADER_SIZE, color="0.45", fontproperties=PROP_REGULAR,
            )

    def _finish_page(self) -> None:
        self._ax.text(
            0.5, BOTTOM_Y_FRAC / 2, str(self._page_num),
            transform=self._ax.transAxes, ha="center", va="center",
            fontsize=FOOTER_SIZE, color="0.45", fontproperties=PROP_REGULAR,
        )
        self._pdf.savefig(self._fig)
        plt.close(self._fig)

    def close(self) -> None:
        self._finish_page()

    def _ensure_space(self, needed_frac: float) -> None:
        if self._y - needed_frac < BOTTOM_Y_FRAC:
            self._new_page()

    # -- content blocks ---------------------------------------------------

    def title_block(self, title: str, byline: str) -> None:
        """Draw the title, byline, and a horizontal rule under them."""
        self._y = draw_paragraph(
            self._ax, self._y, title, fontsize=TITLE_SIZE, bold=True,
            x_left=LEFT_FRAC, width_pt=CONTENT_WIDTH_PT, justify=False, center=True,
        )
        self._y -= 0.3 * _line_height_frac(BYLINE_SIZE)
        self._y = draw_paragraph(
            self._ax, self._y, byline, fontsize=BYLINE_SIZE, bold=False,
            x_left=LEFT_FRAC, width_pt=CONTENT_WIDTH_PT, justify=False, center=True,
        )
        self._y -= 0.6 * _line_height_frac(BODY_SIZE)
        self._ax.plot([LEFT_FRAC, RIGHT_FRAC], [self._y, self._y], color="black", linewidth=0.8, transform=self._ax.transAxes)
        self._y -= 1.1 * _line_height_frac(BODY_SIZE)

    def abstract(self, text: str, keywords: str) -> None:
        """Draw an inset 'Abstract' block, followed by a 'Keywords' line."""
        inset = ABSTRACT_INSET_IN / PAGE_WIDTH_IN
        x_left = LEFT_FRAC + inset
        width_pt = CONTENT_WIDTH_PT - 2 * ABSTRACT_INSET_IN * PT_PER_IN

        self._y = draw_paragraph(
            self._ax, self._y, "Abstract", fontsize=HEADING_SIZE, bold=True,
            x_left=x_left, width_pt=width_pt, justify=False, center=True,
        )
        self._y -= 0.3 * _line_height_frac(ABSTRACT_SIZE)
        self._y = draw_paragraph(
            self._ax, self._y, text, fontsize=ABSTRACT_SIZE, bold=False,
            x_left=x_left, width_pt=width_pt, justify=True,
        )
        self._y -= 0.3 * _line_height_frac(ABSTRACT_SIZE)
        self._y = draw_paragraph(
            self._ax, self._y, f"Keywords: {keywords}", fontsize=ABSTRACT_SIZE, bold=False,
            x_left=x_left, width_pt=width_pt, justify=False,
        )
        self._y -= 1.2 * _line_height_frac(BODY_SIZE)

    def heading(self, text: str) -> None:
        self._ensure_space(3 * _line_height_frac(HEADING_SIZE))
        self._y = draw_paragraph(
            self._ax, self._y, text, fontsize=HEADING_SIZE, bold=True,
            x_left=LEFT_FRAC, width_pt=CONTENT_WIDTH_PT, justify=False,
        )
        self._y -= 0.4 * _line_height_frac(BODY_SIZE)

    def paragraph(self, text: str) -> None:
        # Reserve at least two lines' worth of space so a paragraph never
        # starts with a single orphan line at the very bottom of a page.
        self._ensure_space(2 * _line_height_frac(BODY_SIZE))
        self._y = draw_paragraph(
            self._ax, self._y, text, fontsize=BODY_SIZE, bold=False,
            x_left=LEFT_FRAC, width_pt=CONTENT_WIDTH_PT, justify=True,
        )
        self._y -= PARA_GAP_FACTOR * _line_height_frac(BODY_SIZE)

    def equation(self, mathtext: str) -> None:
        self._ensure_space(2.2 * _line_height_frac(EQUATION_SIZE))
        self._y -= 0.3 * _line_height_frac(EQUATION_SIZE)
        self._ax.text(
            0.5, self._y, f"${mathtext}$", transform=self._ax.transAxes,
            ha="center", va="top", fontsize=EQUATION_SIZE,
        )
        self._y -= 1.7 * _line_height_frac(EQUATION_SIZE)

    def list_items(self, items: list[str], indent_in: float = 0.22) -> None:
        indent_pt = indent_in * PT_PER_IN
        for item in items:
            self._ensure_space(2 * _line_height_frac(BODY_SIZE))
            self._ax.text(
                LEFT_FRAC, self._y, "•", transform=self._ax.transAxes,
                ha="left", va="top", fontsize=BODY_SIZE, fontproperties=PROP_REGULAR,
            )
            self._y = draw_paragraph(
                self._ax, self._y, item, fontsize=BODY_SIZE, bold=False,
                x_left=LEFT_FRAC + indent_pt / PAGE_WIDTH_PT,
                width_pt=CONTENT_WIDTH_PT - indent_pt, justify=True,
            )
        self._y -= PARA_GAP_FACTOR * _line_height_frac(BODY_SIZE)


def draw_paragraph(
    ax,
    y: float,
    text: str,
    *,
    fontsize: float,
    bold: bool,
    x_left: float,
    width_pt: float,
    justify: bool,
    center: bool = False,
) -> float:
    """Wrap ``text`` to ``width_pt`` and draw it starting at height ``y``.

    Parameters
    ----------
    ax:
        The (0-1 fraction, axis-off) Axes to draw into.
    y:
        Top of the paragraph, in y-axis fraction.
    text:
        The paragraph text (plain, will be whitespace-split and re-wrapped).
    fontsize:
        Font size in points.
    bold:
        Whether to use the bold STIX weight.
    x_left:
        Left edge of the text column, in x-axis fraction.
    width_pt:
        Column width, in points.
    justify:
        If True, stretch inter-word spacing so every line but the last fills
        the column exactly; the last line (and any single-word line) is left
        unstretched.
    center:
        If True, center each line instead of left/justified alignment
        (used for the title and headings); mutually exclusive with justify.

    Returns
    -------
    The new y position (in y-axis fraction) after the paragraph.
    """
    words = text.split()
    lines: list[list[str]] = []
    current: list[str] = []
    current_width = 0.0
    space_width = _measure(" ", fontsize, bold)

    for word in words:
        word_width = _measure(word, fontsize, bold)
        projected = current_width + (space_width if current else 0.0) + word_width
        if current and projected > width_pt:
            lines.append(current)
            current, current_width = [word], word_width
        else:
            current.append(word)
            current_width = projected if current else word_width

    if current:
        lines.append(current)

    for i, line_words in enumerate(lines):
        is_last = i == len(lines) - 1
        if center:
            _draw_line_centered(ax, y, line_words, fontsize, bold)
        else:
            _draw_line(
                ax, x_left, y, line_words, fontsize, bold, width_pt,
                justify=justify and not is_last and len(line_words) > 1,
            )
        y -= _line_height_frac(fontsize)

    return y


def _draw_line(
    ax, x_left: float, y: float, words: list[str], fontsize: float, bold: bool,
    width_pt: float, *, justify: bool,
) -> None:
    """Draw one line of already-wrapped words, optionally justified to ``width_pt``."""
    word_widths = [_measure(w, fontsize, bold) for w in words]
    total_width = sum(word_widths)
    n_gaps = len(words) - 1
    space_width = _measure(" ", fontsize, bold)
    gap = (width_pt - total_width) / n_gaps if justify and n_gaps > 0 else space_width

    x_pt = 0.0
    prop = PROP_BOLD if bold else PROP_REGULAR
    for word, word_width in zip(words, word_widths):
        x_frac = x_left + x_pt / PAGE_WIDTH_PT
        ax.text(x_frac, y, word, transform=ax.transAxes, ha="left", va="top", fontsize=fontsize, fontproperties=prop)
        x_pt += word_width + gap


def _draw_line_centered(ax, y: float, words: list[str], fontsize: float, bold: bool) -> None:
    """Draw one line of words centered on the page (used for titles/headings)."""
    line_text = " ".join(words)
    prop = PROP_BOLD if bold else PROP_REGULAR
    ax.text(0.5, y, line_text, transform=ax.transAxes, ha="center", va="top", fontsize=fontsize, fontproperties=prop)


# ---------------------------------------------------------------------------
# Document content.
# ---------------------------------------------------------------------------

ABSTRACT_TEXT = (
    "We describe a numerical method for discovering previously unknown functional "
    "relations among scalar quantities derived from a triangle, in the spirit of "
    "Euler's classical relation between the circumradius, the inradius, and the "
    "distance between the incenter and circumcenter. Because a triangle has three "
    "degrees of freedom up to rigid motion, a relation among four or more derived "
    "scalars is guaranteed by dimension counting alone and is therefore "
    "uninformative; a relation among exactly three scalars, by contrast, is a "
    "genuine and generically unexpected algebraic coincidence. We detect such "
    "coincidences by training a bottleneck autoencoder on sampled triangle data and "
    "comparing its reconstruction error against a column-permuted null, then "
    "recover an explicit closed form for flagged candidates via a monomial "
    "null-space fit. Applied to the circumradius, inradius, and incenter-to-"
    "circumcenter distance, the method rediscovers Euler's relation end to end, "
    "with no prior knowledge of the formula."
)
KEYWORDS = "triangle geometry; Euler's relation; functional dependence; autoencoders; symbolic regression"


def build_pdf(out_path: Path) -> None:
    """Render the full theory document to ``out_path``."""
    logger.info("rendering %s", out_path)
    with PdfPages(out_path) as pdf:
        doc = _Document(pdf)

        doc.title_block(DOC_TITLE, "triangle-relations project")
        doc.abstract(ABSTRACT_TEXT, KEYWORDS)

        doc.heading("1. Setup: the moduli space of triangles")
        doc.paragraph(
            "A triangle is specified by three points in the plane: six real "
            "parameters. Every quantity we consider (area, perimeter, circumradius "
            "R, inradius r, distances between derived points, and so on) is built "
            "purely from the vertex coordinates and is invariant under rigid "
            "motions: translating or rotating the triangle changes none of them. "
            "Translations (2 parameters) and rotations (1 parameter) form a "
            "3-dimensional group acting on the 6-dimensional space of vertex "
            "triples; quotienting it out leaves a 3-dimensional moduli space M of "
            "triangle shapes up to congruence (reflection is a further discrete "
            "symmetry and does not change this dimension count). Concretely, M can "
            "be coordinatized by the three side lengths (a, b, c), subject to the "
            "triangle inequality."
        )
        doc.paragraph(
            "Every derived scalar quantity is a smooth function f : M -> R. "
            "Choosing k such quantities gives a smooth map "
            "F = (f_1, ..., f_k) : M -> R^k."
        )

        doc.heading("2. Why relations among four or more quantities are not interesting")
        doc.paragraph(
            "If k > dim(M) = 3, a generic smooth map from a 3-manifold into R^k has, "
            "as its image, an (at most) 3-dimensional submanifold of R^k -- "
            "embedding a 3-dimensional space into a higher-dimensional ambient space "
            "produces k - 3 constraint equations for free, regardless of which k "
            "functions were chosen, as long as they are not specially degenerate. "
            "This image is cut out locally by k - 3 independent equations "
            "G_i(f_1, ..., f_k) = 0. So finding some relation among four or more "
            "triangle invariants is guaranteed by dimension counting alone -- it "
            "says nothing specific about triangle geometry."
        )

        doc.heading("3. Why a relation among exactly three quantities is special")
        doc.paragraph(
            "If k = dim(M) = 3 exactly, a generic map F : M -> R^3 is, at a generic "
            "point, a local diffeomorphism: its Jacobian DF has full rank 3, and the "
            "triple (f1, f2, f3) sweeps out a full 3-dimensional open region of R^3 "
            "-- a solid blob, not a surface. An exact relation G(f1, f2, f3) = 0 "
            "holding everywhere is equivalent to DF having rank at most 2 "
            "everywhere: the image collapses onto a 2-dimensional surface. That is "
            "not generic -- it is a genuine algebraic coincidence, exactly the kind "
            "of fact that constitutes a classical theorem. Euler's relation"
        )
        doc.equation(r"d^2 = R^2 - 2Rr")
        doc.paragraph(
            "(d being the distance between the incenter and circumcenter) is "
            "exactly a statement that the Jacobian of (R, r, d) has rank at most 2 "
            "everywhere, collapsing three ostensibly independent quantities onto a "
            "single surface. This is the precise sense in which searching for "
            "relations among exactly three derived scalars -- and not four -- is "
            "the search for something structurally new."
        )

        doc.heading("4. Detecting rank-deficiency with a bottleneck autoencoder")
        doc.paragraph(
            "Given a candidate triple (f1, f2, f3), we want a numerical test for "
            "whether the induced map F has full rank 3 generically (data fills a 3D "
            "region) or is everywhere rank at most 2 (data confined to a 2D "
            "surface). We sample many random triangles and evaluate (f1, f2, f3) on "
            "each, producing a point cloud X in R^3."
        )
        doc.paragraph(
            "An autoencoder with a 2-dimensional latent bottleneck -- an encoder "
            "e: R^3 -> R^2 and a decoder d: R^2 -> R^3, trained to minimize the "
            "expected squared reconstruction error -- is exactly searching for the "
            "best rank-2, nonlinear approximation to the data:"
        )
        doc.list_items([
            "If the data truly lies on a 2D surface (an exact relation holds), a "
            "2-dimensional latent code is not actually a bottleneck for it -- a "
            "sufficiently expressive encoder/decoder can drive reconstruction error "
            "to near zero: the encoder learns a local chart of the surface, and the "
            "decoder learns its inverse.",
            "If the data genuinely fills an open 3D region, no continuous map "
            "factoring through a 2D intermediate space can reconstruct it without "
            "loss: an entire dimension of variation is necessarily discarded, so "
            "reconstruction error is bounded away from zero, on the order of the "
            "data's own extent in the discarded direction.",
        ])
        doc.paragraph(
            "This gives a clean signature: near-zero reconstruction error with a "
            "2-dimensional bottleneck is evidence for a relation among the three "
            "chosen quantities; error comparable to the data's own scale is "
            "consistent with the generic, full-rank, no-relation case."
        )

        doc.heading("5. Calibrating “near zero” with a permutation null")
        doc.paragraph(
            "An absolute error threshold is not meaningful across different "
            "triples: they differ in units, scale, and marginal shape, and since "
            "all derived scalars are ultimately functions of the same three "
            "triangle parameters, some triples are mildly correlated even without "
            "an exact relation. We need a triple-specific baseline for “no "
            "relation.”"
        )
        doc.paragraph(
            "We construct a null sample X' by independently permuting each of the "
            "three columns of X. This preserves each quantity's own marginal "
            "distribution while completely destroying the joint dependency between "
            "them. Any real functional relation is annihilated by this shuffle. "
            "Training the same autoencoder on X' (repeated a few times) estimates "
            "the error expected for three quantities with these marginals but no "
            "shared structure. Comparing the real error against this null's mean "
            "and standard deviation, via a z-score or their ratio, yields a "
            "self-calibrated significance measure that is robust to units and "
            "distribution shape. A small ratio is the fingerprint of an unexpected "
            "relation."
        )

        doc.heading("6. From detection to an explicit formula")
        doc.paragraph(
            "The autoencoder step is a screening tool: it flags which triples are "
            "suspicious, not what the relation says. Once a triple is flagged, we "
            "build a dictionary of monomials in (f1, f2, f3) up to a chosen degree "
            "D, evaluate them on the sampled data to form a design matrix Phi "
            "(rescaling each column to comparable variance), and look for a linear "
            "combination of its columns that vanishes on every sample -- the right "
            "singular vector belonging to the smallest singular value of Phi."
        )
        doc.paragraph(
            "Applied to (R, r, d) at degree 2, this recovers -- up to overall scale "
            "and sign, exactly the ambiguity inherent in reading off a null vector "
            "-- Euler's relation on every sampled triangle, found purely by "
            "sampling random triangles and asking a null-space question, with no "
            "prior knowledge of the formula. A very small trailing singular value "
            "(relative to the largest) is itself independent confirmation that an "
            "exact low-degree polynomial relation exists, complementary to the "
            "autoencoder's nonlinear, degree-agnostic detector."
        )

        doc.heading("7. The full pipeline")
        doc.list_items([
            "Enumerate candidate triples of derived scalar quantities.",
            "For each triple, sample random triangles once, train a 2-bottleneck "
            "autoencoder, and compare its held-out reconstruction error to a "
            "column-shuffled null (several repeats).",
            "Rank triples by how much better the real data compresses than its "
            "null counterpart (a small ratio indicates a strong candidate).",
            "For top candidates, fit a low-degree polynomial null-space relation "
            "and validate that it holds (residual near zero) on fresh samples.",
        ])
        doc.paragraph(
            "The triangle_relations.discovery.verify_euler_relation module runs "
            "this exact pipeline on (R, r, d) as a worked example, confirming it "
            "rediscovers Euler's relation end to end."
        )

        doc.close()
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    build_pdf(Path(__file__).parent / "theory.pdf")
