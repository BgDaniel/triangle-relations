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

By default (`PLOT_RANKING = True`) a horizontal bar chart of z-scores for
the top `TOP` triples is shown automatically once the search finishes. To
re-plot a previously saved ranking later without rerunning the search, run:

```
poetry run python scripts/plot_ranking.py
```

(`triangle_relations.discovery.ranking_plot.plot_ranking` and
`load_ranking_csv` are the underlying functions, usable directly too.)

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

## Program 2 — incidence relations between derived points (planned)

Not yet implemented: searching combinations of three or more derived points
for geometric incidence conditions such as collinearity.

## Tests

```
poetry run pytest
```
