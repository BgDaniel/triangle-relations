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

```
poetry run python scripts/discover_scalar_relations.py --n-samples 1500 --top 15
```

Useful flags: `--scalars a,b,c,...` to restrict the search space (the full
search is combinatorial and can be slow), `--n-jobs -1` to parallelize across
cores (default), `--output results.csv` to dump the full ranking.

Once a candidate triple looks promising, fit an explicit polynomial relation:

```python
from triangle_relations.discovery.sampling import build_scalar_dataset
from triangle_relations.discovery.symbolic import fit_polynomial_relation
import numpy as np

rng = np.random.default_rng(0)
names, data = build_scalar_dataset(
    500, rng, scalar_names=["circumradius", "inradius", "dist_circumcenter__incenter"]
)
relation = fit_polynomial_relation(data, tuple(names), max_degree=2)
print(relation.as_expr())   # -> recovers Euler's relation up to sign/scale
```

This fits a linear combination of monomials (up to `max_degree`) that
vanishes on the sampled data, via the smallest singular vector of the
(normalized) monomial design matrix — exactly how Euler's relation shows up
as a linear dependency among `{R^2, Rr, d^2}`.

### Worked example: rediscovering Euler's relation

`triangle_relations/discovery/verify_euler_relation.py` runs the entire
pipeline above on `(circumradius, inradius, dist_circumcenter__incenter)` as
a validation that the method actually recovers a *known* theorem: it checks
the detection ratio/z-score, fits the polynomial relation (recovering
`d^2 = R^2 - 2Rr` up to sign/scale), and saves a 3D scatter plot contrasting
the real data (confined to a thin 2D surface) against the shuffled null
(filling the 3D volume).

```
poetry run python -m triangle_relations.discovery.verify_euler_relation
```

### Theory

`docs/theory.tex` explains the reasoning in more depth: why a triangle's
three degrees of freedom make a relation among *four* derived scalars
guaranteed by dimension counting but a relation among exactly *three* a
genuine (and rare) algebraic coincidence, and how the autoencoder + shuffled
null + polynomial null-space steps each fit into detecting and confirming
one. Compile it with a LaTeX distribution (e.g. `pdflatex theory.tex`) for a
properly typeset PDF. `docs/theory.pdf` is a plain fallback rendering (via
`docs/render_theory_pdf.py`, using matplotlib only) for reading without a
LaTeX install; regenerate it with:

```
poetry run python docs/render_theory_pdf.py
```

## Program 2 — incidence relations between derived points (planned)

Not yet implemented: searching combinations of three or more derived points
for geometric incidence conditions such as collinearity.

## Tests

```
poetry run pytest
```
