"""Fit an explicit polynomial relation for a promising scalar triple.

Given samples of three scalars that a discovery run flagged as suspiciously
dependent, this builds a dictionary of monomials up to a chosen degree and
looks for a linear combination of them that vanishes on the data (a near-null
vector of the monomial design matrix, found via SVD). This is exactly how
Euler's relation d^2 - R^2 + 2Rr = 0 shows up: it is a linear relation among
the degree-2 monomials {R^2, Rr, Rd, r^2, rd, d^2, ...}.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import sympy as sp

logger = logging.getLogger(__name__)


@dataclass
class PolynomialRelation:
    """A candidate polynomial relation ``sum(coefficients[i] * monomial[i]) = 0``.

    Attributes
    ----------
    variables:
        SymPy symbols for each input column, in column order.
    exponents:
        One exponent tuple per monomial (parallel to ``coefficients``); e.g.
        ``(2, 0, 1)`` for three variables means ``variables[0]**2 * variables[2]``.
    coefficients:
        The fitted coefficient of each monomial.
    singular_value_ratio:
        Smallest singular value divided by the largest, from the SVD used to
        fit this relation. Close to zero indicates a genuine near-exact
        polynomial relation of the chosen degree; not small means no such
        relation was found and the returned coefficients are not meaningful.
    """

    variables: tuple[sp.Symbol, ...]
    exponents: list[tuple[int, ...]]
    coefficients: np.ndarray
    singular_value_ratio: float

    def as_expr(self, *, rationalize: bool = True, tol: float = 1e-3) -> sp.Expr:
        """Render the relation as a SymPy expression (implicitly ``== 0``).

        Parameters
        ----------
        rationalize:
            If ``True``, snap near-rational coefficients to exact rationals
            (via :func:`sympy.nsimplify`) for a readable closed form, e.g.
            ``-1/2`` instead of ``-0.4999999``.
        tol:
            Coefficients smaller than ``tol`` (after normalizing the largest
            coefficient to 1) are dropped as noise; also used as the
            rationalization tolerance.

        Returns
        -------
        A SymPy expression that should be (approximately) zero for every
        sample if this relation is genuine.
        """
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
        """Evaluate the (unnormalized) polynomial on raw samples.

        Parameters
        ----------
        data:
            Sample matrix of shape ``(n_samples, len(variables))``, in the
            same column order used to fit this relation.

        Returns
        -------
        One residual value per sample; values near zero everywhere confirm
        the relation holds on that data.
        """
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

    Builds every monomial in the input variables up to ``max_degree``,
    evaluates them on ``data``, and takes the right singular vector of the
    smallest singular value of the (column-normalized) monomial design
    matrix as the candidate relation's coefficients.

    Parameters
    ----------
    data:
        Sample matrix of shape ``(n_samples, n_vars)`` with
        ``n_vars == len(names)``.
    names:
        Variable names, used to build the returned SymPy symbols.
    max_degree:
        Highest total monomial degree to include in the search dictionary.

    Returns
    -------
    The best-fit :class:`PolynomialRelation` of the requested degree.
    """
    n_vars = data.shape[1]
    exponents = [
        exps
        for degree in range(max_degree + 1)
        for exps in _exponents_of_degree(n_vars, degree)
    ]
    logger.info(
        "fitting a degree-%d polynomial relation among %s (%d monomials)",
        max_degree, names, len(exponents),
    )

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
    logger.info("smallest/largest singular value ratio = %.3e", ratio)

    variables = sp.symbols(names)
    return PolynomialRelation(
        variables=variables,
        exponents=exponents,
        coefficients=coefficients,
        singular_value_ratio=ratio,
    )


def _exponents_of_degree(n_vars: int, degree: int) -> Iterator[tuple[int, ...]]:
    """Yield every exponent tuple of length ``n_vars`` summing to exactly ``degree``."""
    if n_vars == 1:
        yield (degree,)
        return
    for k in range(degree + 1):
        for rest in _exponents_of_degree(n_vars - 1, degree - k):
            yield (k,) + rest


def _evaluate_monomial(data: np.ndarray, exponents: tuple[int, ...]) -> np.ndarray:
    """Evaluate a single monomial (given by its exponent tuple) on every row of ``data``."""
    result = np.ones(data.shape[0])
    for col, e in enumerate(exponents):
        if e:
            result = result * data[:, col] ** e
    return result
