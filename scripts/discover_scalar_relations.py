#!/usr/bin/env python
"""CLI: search for functional relations among triples of triangle scalars.

Example
-------
    poetry run python scripts/discover_scalar_relations.py --n-samples 1500 --top 15
"""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from triangle_relations.discovery.sampling import build_scalar_dataset
from triangle_relations.discovery.scalar_relations import search_three_scalar_relations
from triangle_relations.geometry.triangle import Triangle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=1500)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--n-restarts", type=int, default=1)
    parser.add_argument("--n-shuffles", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scalars",
        type=str,
        default=None,
        help="comma-separated subset of scalar names to search (default: all)",
    )
    parser.add_argument("--output", type=str, default=None, help="optional CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    scalar_names = args.scalars.split(",") if args.scalars else None
    if scalar_names is None:
        print(f"Available scalars ({len(Triangle.SCALARS)}): {sorted(Triangle.SCALARS)}\n")

    print(f"Sampling {args.n_samples} random triangles...")
    names, data = build_scalar_dataset(
        args.n_samples, rng, scalar_names=scalar_names, scale=args.scale
    )

    n_triples = len(names) * (len(names) - 1) * (len(names) - 2) // 6
    print(f"Searching {n_triples} combinations of 3 scalars out of {len(names)}...\n")

    results = search_three_scalar_relations(
        names,
        data,
        n_shuffles=args.n_shuffles,
        hidden=args.hidden,
        n_restarts=args.n_restarts,
        test_size=args.test_size,
        n_jobs=args.n_jobs,
        random_state=args.seed,
    )

    header = f"{'rank':>4}  {'triple':60}  {'real_err':>10}  {'null_mean':>10}  {'z_score':>8}  {'ratio':>8}"
    print(header)
    print("-" * len(header))
    for rank, r in enumerate(results[: args.top], start=1):
        print(
            f"{rank:>4}  {', '.join(r.names):60}  {r.real_error:10.4g}  "
            f"{r.null_mean:10.4g}  {r.z_score:8.2f}  {r.ratio:8.3f}"
        )

    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name_1", "name_2", "name_3", "real_error", "null_mean", "null_std", "z_score", "ratio"])
            for r in results:
                writer.writerow([*r.names, r.real_error, r.null_mean, r.null_std, r.z_score, r.ratio])
        print(f"\nWrote full results ({len(results)} triples) to {args.output}")


if __name__ == "__main__":
    sys.exit(main())
