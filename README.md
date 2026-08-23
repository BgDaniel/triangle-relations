# triangle-relations

A Python-based computational geometry project for discovering and analyzing hidden relationships among triangle-derived quantities, combining numerical methods, linear algebra, geometric coincidence detection, and machine learning such as autoencoders.

## Setup

```
poetry install
```

## The `Triangle` class

```python
from triangle_relations import Triangle

t = Triangle((0, 0), (4, 0), (0, 3))
t.area()               # 6.0
t.circumradius()
t.inradius()
t.incenter()
t.circumcenter()
t.steiner_inellipse_foci()

t.all_scalars()         # dict of every registered scalar quantity
t.all_points()          # dict of every registered derived point

t.plot()   # every derived object is shown by default; pass show_x=False to hide one
```

Run `poetry run python scripts/plot_sample_triangle.py` for a standalone demo on a sample triangle.

Derived points (`Triangle.POINTS`): centroid, circumcenter, incenter, orthocenter,
and the two foci of the Steiner inellipse (via Marden's theorem).

Derived scalars (`Triangle.SCALARS`): area, perimeter, semiperimeter, circumradius,
inradius, side lengths, angles, plus the pairwise distance between *every*
registered point (auto-generated, so adding a new point automatically adds new
candidate scalars for relation discovery).

## Program 1 — discovering relations among triples of scalars

A triangle has three degrees of freedom, so a functional relation among only
*three* derived scalars (like Euler's `d^2 = R^2 - 2Rr`) is not expected
generically, unlike a relation among four or more quantities (guaranteed by
dimension counting). `scripts/discover_scalar_relations.py` searches all
`C(n, 3)` combinations of scalars and flags the ones whose joint distribution
collapses onto a 2D surface far more than a permutation-shuffled null predicts:

1. Sample `N` random triangles once; evaluate every scalar on the same sample.
2. For each triple, train a bottleneck autoencoder (3 → hidden → **2** → hidden → 3)
   and measure held-out reconstruction error.
3. Compare against a null built by independently shuffling each column
   (destroys joint structure, preserves each quantity's own marginal
   distribution) and repeating the same training procedure.
4. Rank triples by `real_error / null_mean` — a low ratio means the
   autoencoder compresses the real data far better than it compresses the
   shuffled control, i.e. a strong candidate functional relation.

This is meant to be run from an IDE rather than the command line: open
`scripts/discover_scalar_relations.py` and edit the configuration constants
at the top (`N_SAMPLES`, `SCALAR_NAMES`, `N_JOBS`, `OUTPUT_CSV`, ...), then run:

```
poetry run python scripts/discover_scalar_relations.py
```

`SCALAR_NAMES` restricts the search space (the full search is combinatorial
and can be slow, so a `tqdm` progress bar tracks how many of the `C(n, 3)`
combinations have been processed so far); `OUTPUT_CSV` optionally dumps the
full ranking to a file.

`OUTPUT_CSV` and `scripts/plot_ranking.py`'s `CSV_PATH` are both derived
automatically from the `PATH_TO_OUTPUT_FOLDER` environment variable
(as `<PATH_TO_OUTPUT_FOLDER>/ranking.csv`) if it's set, rather than
a path hardcoded in the script — set it once in your environment (see
"Setting an environment variable permanently in VS Code" below) and both
scripts pick it up automatically. If it's unset, `OUTPUT_CSV` defaults to
`None` (no CSV written) and `CSV_PATH` defaults to `output/ranking.csv`.

By default (`PLOT_RANKING = True`) two figures are shown automatically once
the search finishes:

* **z-score plot**: two stacked, x-axis-aligned panels for the top `TOP`
  triples by z-score — the z-score itself on top, and — for those exact
  same triples, in the same order (combination names labeled once, on the
  shared bottom axis) — their *relative* null standard deviation
  (`null_std / null_mean`) below. Since `z = (null_mean - real_error) /
  null_std`, a small `null_std` alone can inflate a z-score without the
  real/null gap actually being large, so seeing it alongside the z-score is
  a sanity check on how much to trust it.
* **ratio plot**: the same two-panel layout, but ranked by ratio
  (`real_error / null_mean`, ascending — smaller means stronger here, with a
  reference line at 1.0 marking "no better than the shuffled null") with the
  same relative null std sanity-check panel below it.

In both plots, a triple ranked in the top `TOP` by *both* z-score and ratio
— or the classical triple behind Euler's relation, `(R, r, OI)`
(`circumradius, inradius, dist_circumcenter__incenter`; always shown for
reference even if it falls outside `TOP`, and labeled "Euler" on its bar) —
is drawn in red instead of blue. Scalar names throughout are abbreviated to
short symbols (e.g. `R` = circumradius, `r` = inradius, `OI` = distance
between circumcenter and incenter — see
`Triangle.SCALAR_SYMBOLS`/`Triangle.scalar_symbol`) so labels stay compact
even for long triples. `discover_scalar_relations.py` and `plot_ranking.py`
both also log the Euler triple's rank position under each metric (e.g.
`Euler triple (R, r, OI) ranks #3 of 120 by z-score, #1 of 120 by ratio`),
whether or not it happens to fall inside `TOP`.

To re-plot a previously saved ranking later without rerunning the search, run:

```
poetry run python scripts/plot_ranking.py
```

(`triangle_relations.discovery.ranking_plot.plot_z_score_ranking` and
`plot_ratio_ranking` are the underlying functions, usable directly on a
`list[RelationResult]` — e.g. one loaded via `load_ranking_csv` — too.)

#### A note on `N_SHUFFLES`

`null_std` (and hence both the z-score and the relative-null-std panel
above) is a *sample* standard deviation estimated from only `N_SHUFFLES`
independent shuffles per triple, so it is itself a noisy estimate: the
standard error of a sample standard deviation from `n` points is roughly
`sigma / sqrt(2*(n-1))`. At `N_SHUFFLES = 3` that's about 50% relative
uncertainty on `null_std` itself — the earlier default of 3 was picked for
runtime, not because it was statistically adequate, and it wasn't. Even
`N_SHUFFLES = 30` is still around 13%; reaching 10% needs roughly 50
shuffles, and 5% needs roughly 200. Since z-score divides by this noisy
`null_std`, a small `N_SHUFFLES` can produce a misleadingly large *or*
small z-score for the same underlying triple. `ratio` (`real_error /
null_mean`) doesn't have this problem, since it only depends on
`null_mean`, not `null_std` — the cheapest practical fix is a two-stage
search: screen the full combinatorial space once with a small
`N_SHUFFLES` using `ratio` as the primary signal (the relative-null-std
panel flags triples whose z-score shouldn't be trusted yet), then rerun
just the resulting shortlist with a much larger `N_SHUFFLES` (50-200+, via
`SCALAR_NAMES` restricted to the shortlist's scalars) to get precise
z-scores for the finalists.

<details>
<summary>Setting <code>PATH_TO_OUTPUT_FOLDER</code> permanently in VS Code</summary>

**Option A — Windows user environment variable** (works in any terminal, not
just VS Code; requires restarting VS Code once afterward to pick it up):
Windows Settings → search "environment variables" → *Edit environment
variables for your account* → *New...* → name `PATH_TO_OUTPUT_FOLDER`,
value e.g. `C:\Projects\triangle-relations\output`. Equivalently, in
PowerShell: `[Environment]::SetEnvironmentVariable("PATH_TO_OUTPUT_FOLDER", "C:\Projects\triangle-relations\output", "User")`.

**Option B — a `.env` file in the workspace root**: create `.env` next to
`pyproject.toml` with a line like
`PATH_TO_OUTPUT_FOLDER=C:\Projects\triangle-relations\output`.
VS Code's Python extension loads this automatically for the integrated
terminal and for debugging (the `python.envFile` setting, default
`${workspaceFolder}/.env`) — no restart needed, just reopen the terminal.
Since it's project-scoped, add `.env` to `.gitignore` unless you want to
commit a shared default for the team.

</details>

The autoencoder is used purely as a *detector*: it only tells you whether a
triple of quantities is suspiciously dependent, not what the relation
actually is (the trained network is not an interpretable formula). Going
from "there's a relation" to an explicit closed form like
`d^2 = R^2 - 2Rr` is a separate problem (symbolic regression) that this
project does not currently attempt.

### Worked example: rediscovering Euler's relation

`triangle_relations/discovery/verify_euler_relation.py` runs the detection
pipeline on `(circumradius, inradius, dist_circumcenter__incenter)` as a
validation that the method actually recovers a *known* theorem: it checks
the detection ratio/z-score, and saves a 3D scatter plot contrasting the
real data (confined to a thin 2D surface) against the shuffled null (filling
the 3D volume).

```
poetry run python -m triangle_relations.discovery.verify_euler_relation
```

### Theory

`docs/discovering_triangle_relations.tex` explains the reasoning in more
depth: why a triangle's three degrees of freedom make a relation among
*four* derived scalars guaranteed by dimension counting but a relation among
exactly *three* a genuine (and rare) algebraic coincidence, and how the
autoencoder + shuffled-null steps fit together to detect one — including a
mathematically precise treatment of the detector itself (a sufficiency and
a necessity proposition, with proof sketches, characterizing exactly when
reconstruction error goes to zero vs. stays bounded away from it). It's a
standard `amsart` (AMS journal article) document with a proper abstract,
MSC classification, and keywords, set in `newtxtext`/`newtxmath` (a
Times-like math/text pairing).
`docs/discovering_triangle_relations.pdf` is the compiled PDF, checked into
the repo so it's viewable without a LaTeX install. To recompile it after
editing the `.tex` (e.g. on Overleaf, or locally with any LaTeX
distribution):

```
pdflatex discovering_triangle_relations.tex
```

Section 6 of the `.tex` source, "A scale-free reformulation via shape
space," derives Program 1b below in full — including *why* a coordinate
chart is unavoidable to make any of this numerically concrete, and why a
generic choice of chart can't silently hide a relation from the search (a
Remark titled "Why a generic chart is safe").

## Program 1b — homogeneous relations, without a null

A second, complementary search, restricted to relations that are
*homogeneous* under uniform scaling of the triangle (every term scales the
same way — true of essentially every relation of practical interest,
Euler's included: `d^2`, `R^2`, `Rr` are all degree 2). In exchange for that
restriction, it needs **no permutation null at all**, which is what makes
Program 1's `N_SHUFFLES` expensive to push up (see above): triangles are
sampled once, evenly, over the 2-dimensional space of triangle *shapes*
(forgetting size — a classical theorem of Kendall identifies this space
with a sphere), and each candidate triple is scored by a single held-out
reconstruction error that is already comparable across triples with no
per-triple calibration, since it's computed on data that always lives on
the same fixed unit sphere regardless of which scalars are chosen.

Concretely: a scalar `f` of homogeneity degree `d` (`0` for angles, `1` for
lengths/distances/`R`/`r`/perimeter, `2` for area — see
`Triangle.SCALAR_DEGREES`) satisfies `f(T)**(1/d)` scales *linearly* with
the triangle regardless of `d`; combining three such degree-equalized,
positive scalars into a unit vector gives a point that depends only on
shape, not size. A relation shows up as this map collapsing its image onto
a curve rather than covering an open patch of the target sphere — detected
the same way as Program 1 (a bottleneck autoencoder), but with a bottleneck
of size 1 instead of 2, and no null. Degree-0 (scale-invariant) scalars,
like angles, can't take part (`f**(1/0)` is undefined) and are excluded from
this search automatically; Program 1 still covers them.

The autoencoder's input and output are 2D coordinates in a chart of the
target sphere, not its raw 3D embedding — matching the target's actual
(2-dimensional) intrinsic dimension. Naive Euclidean distance *in that
chart* is not a meaningful error, though: a stereographic chart distorts
distances, worse away from its excluded point. So the loss instead maps the
decoder's chart-coordinate output back through the (differentiable)
*inverse* chart before comparing it to the true point — what gradient
descent actually minimizes is the sphere's own chordal distance, not a
chart-space stand-in for it. `scikit-learn`'s `MLPRegressor` (which Program
1 uses) has no hook for a custom loss like this, so this part
(`triangle_relations.discovery.sphere_autoencoder`) is a small hand-written
PyTorch training loop instead — the one place this project needs PyTorch
rather than scikit-learn.

This is meant to be run from an IDE, same as Program 1: open
`scripts/discover_homogeneous_relations.py`, edit the configuration
constants at the top, and run:

```
poetry run python scripts/discover_homogeneous_relations.py
```

It logs the ranked triples (smallest `error` = strongest candidate) and,
like `scripts/discover_scalar_relations.py`, the Euler triple's rank
position, optionally writes a CSV (`homogeneous_ranking.csv`, next to
Program 1's `ranking.csv`, both under `PATH_TO_OUTPUT_FOLDER`), and by
default (`PLOT_RANKING = True`) plots the ranking once the search finishes
(`triangle_relations.discovery.ranking_plot.plot_homogeneous_ranking`) —
a single bar chart, since there's only one score per triple here, unlike
Program 1's two-plot z-score/ratio split. Red bars mark the reference Euler
triple, as in Program 1's plots.

To re-plot a previously saved ranking later without rerunning the search,
run:

```
poetry run python scripts/plot_homogeneous_ranking.py
```

(the Program-1b counterpart to `scripts/plot_ranking.py`, which plots
Program 1's own CSV; the two are independent scripts, each reading its own
CSV path).

Program 1b shares search/progress-bar plumbing
(`triangle_relations.discovery._parallel`) with Program 1, rather than
duplicating it, but has its own detector core
(`triangle_relations.discovery.sphere_autoencoder`, `.spherical_chart`) —
Program 1's `triangle_relations.discovery.autoencoder` can't be reused here
since it always minimizes plain Euclidean error, not the chart-aware sphere
metric this needs. `triangle_relations.discovery.shape_space` (sampling)
and `.homogeneous_relations` (search/scoring) are new too.

### Inspecting a candidate visually

`triangle_relations/discovery/inspect_relation.py` is a visual companion to
the reconstruction-error number above: given one candidate triple, it plots
the image of its degree-equalized embedding under a chart chosen generically
for that triple's own data (see the "Why a generic chart is safe" remark in
the theory doc). An exact relation should be visible directly as the sampled
points collapsing onto a thin 1D curve; no relation should look like a
filled 2D patch.

```
poetry run python -m triangle_relations.discovery.inspect_relation
```

The `__main__` block runs both cases for comparison: Euler's triple
(collapses to a curve) against an unrelated triple (fills a 2D patch),
saving `euler_relation_image.png` and `generic_triple_image.png`. Call
`plot_relation_image(names)` directly to inspect any other candidate.

## Program 2 — incidence relations between derived points (planned)

Not yet implemented: searching combinations of three or more derived points
for geometric incidence conditions such as collinearity.

## Tests

```
poetry run pytest
```
