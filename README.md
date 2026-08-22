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
and can be slow); `OUTPUT_CSV` optionally dumps the full ranking to a file.

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
