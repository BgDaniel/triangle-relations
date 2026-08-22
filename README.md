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

t.plot(show_orthocenter=True, show_steiner_foci=True, show_euler_line=True)
```

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

## Program 2 — incidence relations between derived points (planned)

Not yet implemented: searching combinations of three or more derived points
for geometric incidence conditions such as collinearity.

## Tests

```
poetry run pytest
```
