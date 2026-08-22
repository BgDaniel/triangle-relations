"""Fit an explicit polynomial relation for a promising scalar triple.

Given samples of three scalars that a discovery run flagged as suspiciously
dependent, this builds a dictionary of monomials up to a chosen degree and
looks for a linear combination of them that vanishes on the data (a near-null
vector of the monomial design matrix, found via SVD). This is exactly how
Euler's relation d^2 - R^2 + 2Rr = 0 shows up: it is a linear relation among
the degree-2 monomials {R^2, Rr, Rd, r^2, rd, d^2, ...}.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp


@dataclass
class PolynomialRelation:
    variables: tuple[sp.Symbol, ...]
    exponents: list[tuple[int, ...]]
    coefficients: np.ndarray
    singular_value_ratio: float  # smallest singular value / largest; ~0 means a real relation

    def as_expr(self, *, rationalize: bool = True, tol: float = 1e-3) -> sp.Expr:
        expr = sp.Integer(0)
        coeffs = self.coefficients / np.max(np.abs(self.coefficients))
        for coeff, exps in zip(coeffs, self.exponents):
            if abs(coeff) < tol:
                continue
            c = sp.nsimplify(coeff, rational=True, tolerance=tol) if rationalize else coeff
            term = c
            for var, e in zip(self.variables, exps):
                if e:
                    term *= var**e
            expr += term
        return sp.expand(expr)

    def residual(self, data: np.ndarray) -> np.ndarray:
        """Evaluate the (unnormalized) polynomial on raw samples; near zero
        everywhere confirms the relation."""
        values = np.zeros(data.shape[0])
        for coeff, exps in zip(self.coefficients, self.exponents):
            term = coeff * np.ones(data.shape[0])
            for col, e in enumerate(exps):
                if e:
                    term = term * data[:, col] ** e
            values += term
        return values


def fit_polynomial_relation(
    data: np.ndarray,
    names: tuple[str, ...],
    *,
    max_degree: int = 2,
) -> PolynomialRelation:
    """Search for a polynomial relation (up to ``max_degree``) among columns of ``data``.

    ``data`` has shape ``(n_samples, n_vars)`` with ``n_vars == len(names)``.
    """
    n_vars = data.shape[1]
    exponents = [
        exps
        for degree in range(max_degree + 1)
        for exps in _exponents_of_degree(n_vars, degree)
    ]

    monomials = np.column_stack(
        [_evaluate_monomial(data, exps) for exps in exponents]
    )
    # Normalize each monomial column so the SVD isn't dominated by scale
    # differences between e.g. constant/linear and quadratic terms.
    col_scales = np.std(monomials, axis=0)
    col_scales[col_scales == 0] = 1.0
    normalized = monomials / col_scales

    _, singular_values, vt = np.linalg.svd(normalized, full_matrices=False)
    coefficients = vt[-1] / col_scales
    ratio = float(singular_values[-1] / singular_values[0])

    variables = sp.symbols(names)
    return PolynomialRelation(
        variables=variables,
        exponents=exponents,
        coefficients=coefficients,
        singular_value_ratio=ratio,
    )


def _exponents_of_degree(n_vars: int, degree: int):
    """All exponent tuples of length ``n_vars`` summing to exactly ``degree``."""
    if n_vars == 1:
        yield (degree,)
        return
    for k in range(degree + 1):
        for rest in _exponents_of_degree(n_vars - 1, degree - k):
            yield (k,) + rest


def _evaluate_monomial(data: np.ndarray, exponents: tuple[int, ...]) -> np.ndarray:
    result = np.ones(data.shape[0])
    for col, e in enumerate(exponents):
        if e:
            result = result * data[:, col] ** e
    return result
