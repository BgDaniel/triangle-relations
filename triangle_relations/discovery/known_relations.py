"""Reference relations shared by both discovery pipelines and their plots/logs.

Kept separate from any one pipeline so both Program 1
(:mod:`triangle_relations.discovery.scalar_relations`) and Program 1b
(:mod:`triangle_relations.discovery.homogeneous_relations`) -- and
:mod:`triangle_relations.discovery.ranking_plot` -- can check a result
against the same known triple without duplicating it.
"""

from __future__ import annotations

#: The scalar triple behind Euler's classical relation, d^2 = R^2 - 2Rr,
#: where d is the circumcenter-incenter distance (OI in symbol form):
#: circumradius (R), inradius (r), and dist_circumcenter__incenter (OI).
#: A frozenset, so membership doesn't depend on name order.
EULER_TRIPLE_NAMES: frozenset[str] = frozenset(
    {"circumradius", "inradius", "dist_circumcenter__incenter"}
)


def is_euler_triple(names: tuple[str, str, str]) -> bool:
    """Whether ``names`` is the Euler triple, regardless of order."""
    return frozenset(names) == EULER_TRIPLE_NAMES
