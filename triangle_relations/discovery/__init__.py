"""Discovering functional relations among triangle-derived scalars.

Two pipelines, sharing the bottleneck-autoencoder detector
(:mod:`triangle_relations.discovery.autoencoder`) and search plumbing
(:mod:`triangle_relations.discovery._parallel`):

* Program 1 (:mod:`triangle_relations.discovery.scalar_relations`): general,
  detects any functional relation, calibrated against a permutation null.
* Program 1b (:mod:`triangle_relations.discovery.homogeneous_relations`,
  sampling via :mod:`triangle_relations.discovery.shape_space`): restricted
  to *homogeneous* relations, needs no null (see Section 5 of
  ``docs/discovering_triangle_relations.tex``), so it can afford much more
  precise per-triple error estimates for the same cost.

Both pipelines are purely detectors: they flag *whether* a triple is
suspiciously dependent, not what the relation is. Recovering an explicit
closed form (symbolic regression) is a separate problem this project does
not currently attempt; see the README.
"""
