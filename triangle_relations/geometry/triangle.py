"""Triangle class with derived points, scalar quantities, and plotting.

Points (vectors) and scalars are exposed through two registries,
``Triangle.POINTS`` and ``Triangle.SCALARS``, mapping a name to a callable
that takes a ``Triangle`` and returns either a 2-vector (points) or a float
(scalars). All pairwise distances between registered points are added to
``Triangle.SCALARS`` automatically, so adding a new point automatically
enlarges the set of scalar quantities available for relation discovery.
"""

from __future__ import annotations

from functools import cached_property
from itertools import combinations
from typing import Callable

import numpy as np


class Triangle:
    """A triangle defined by three points in the plane.

    Parameters
    ----------
    p1, p2, p3:
        The three vertices, each any array-like of length 2. Stored as
        ``vertices[0], vertices[1], vertices[2]`` corresponding to A, B, C.
    """

    def __init__(self, p1, p2, p3):
        vertices = np.array([p1, p2, p3], dtype=float)
        if vertices.shape != (3, 2):
            raise ValueError(f"expected three 2D points, got shape {vertices.shape}")
        self.vertices = vertices

    @property
    def a_vertex(self) -> np.ndarray:
        return self.vertices[0]

    @property
    def b_vertex(self) -> np.ndarray:
        return self.vertices[1]

    @property
    def c_vertex(self) -> np.ndarray:
        return self.vertices[2]

    # ------------------------------------------------------------------
    # Basic quantities
    # ------------------------------------------------------------------

    @cached_property
    def _side_lengths(self) -> tuple[float, float, float]:
        A, B, C = self.vertices
        a = np.linalg.norm(B - C)
        b = np.linalg.norm(C - A)
        c = np.linalg.norm(A - B)
        return a, b, c

    def side_lengths(self) -> tuple[float, float, float]:
        """Side lengths (a, b, c) opposite vertices (A, B, C)."""
        return self._side_lengths

    def side_a(self) -> float:
        return self.side_lengths()[0]

    def side_b(self) -> float:
        return self.side_lengths()[1]

    def side_c(self) -> float:
        return self.side_lengths()[2]

    @cached_property
    def _area(self) -> float:
        A, B, C = self.vertices
        return 0.5 * abs(
            A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1])
        )

    def area(self) -> float:
        return self._area

    def perimeter(self) -> float:
        return sum(self.side_lengths())

    def semiperimeter(self) -> float:
        return self.perimeter() / 2.0

    @cached_property
    def _angles(self) -> tuple[float, float, float]:
        a, b, c = self.side_lengths()

        def angle(opposite, adj1, adj2):
            cos_theta = (adj1**2 + adj2**2 - opposite**2) / (2 * adj1 * adj2)
            return np.arccos(np.clip(cos_theta, -1.0, 1.0))

        A = angle(a, b, c)
        B = angle(b, a, c)
        C = angle(c, a, b)
        return A, B, C

    def angles(self) -> tuple[float, float, float]:
        """Interior angles (A, B, C) in radians, opposite sides (a, b, c)."""
        return self._angles

    def angle_A(self) -> float:
        return self.angles()[0]

    def angle_B(self) -> float:
        return self.angles()[1]

    def angle_C(self) -> float:
        return self.angles()[2]

    def circumradius(self) -> float:
        a, b, c = self.side_lengths()
        area = self.area()
        return (a * b * c) / (4.0 * area)

    def inradius(self) -> float:
        return self.area() / self.semiperimeter()

    # ------------------------------------------------------------------
    # Derived points
    # ------------------------------------------------------------------

    @cached_property
    def _centroid(self) -> np.ndarray:
        return self.vertices.mean(axis=0)

    def centroid(self) -> np.ndarray:
        return self._centroid

    @cached_property
    def _incenter(self) -> np.ndarray:
        a, b, c = self.side_lengths()
        A, B, C = self.vertices
        return (a * A + b * B + c * C) / (a + b + c)

    def incenter(self) -> np.ndarray:
        return self._incenter

    @cached_property
    def _circumcenter(self) -> np.ndarray:
        A, B, C = self.vertices
        ax, ay = A
        bx, by = B
        cx, cy = C
        d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        a2 = ax**2 + ay**2
        b2 = bx**2 + by**2
        c2 = cx**2 + cy**2
        ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
        uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
        return np.array([ux, uy])

    def circumcenter(self) -> np.ndarray:
        return self._circumcenter

    @cached_property
    def _orthocenter(self) -> np.ndarray:
        A, B, C = self.vertices
        # Altitude from A: (P - A) . (C - B) = 0
        # Altitude from B: (P - B) . (C - A) = 0
        d1 = C - B
        d2 = C - A
        M = np.array([d1, d2])
        rhs = np.array([np.dot(d1, A), np.dot(d2, B)])
        return np.linalg.solve(M, rhs)

    def orthocenter(self) -> np.ndarray:
        """Orthocenter, computed as the intersection of two altitudes.

        Deliberately independent of the circumcenter/centroid formulas so
        that the Euler-line collinearity (O, G, H) is a genuine, testable
        relation rather than one baked in by construction.
        """
        return self._orthocenter

    @cached_property
    def _steiner_inellipse_foci(self) -> tuple[np.ndarray, np.ndarray]:
        z = self.vertices[:, 0] + 1j * self.vertices[:, 1]
        z1, z2, z3 = z
        s = z1 + z2 + z3
        disc = z1**2 + z2**2 + z3**2 - z1 * z2 - z2 * z3 - z3 * z1
        root = np.sqrt(disc)
        f1 = (s + root) / 3.0
        f2 = (s - root) / 3.0
        return np.array([f1.real, f1.imag]), np.array([f2.real, f2.imag])

    def steiner_inellipse_foci(self) -> tuple[np.ndarray, np.ndarray]:
        """The two foci of the Steiner inellipse, via Marden's theorem.

        The vertices are viewed as complex numbers z1, z2, z3; the foci are
        the roots of the derivative of p(z) = (z - z1)(z - z2)(z - z3).
        """
        return self._steiner_inellipse_foci

    def steiner_focus_1(self) -> np.ndarray:
        return self.steiner_inellipse_foci()[0]

    def steiner_focus_2(self) -> np.ndarray:
        return self.steiner_inellipse_foci()[1]

    def affine_map_from_equilateral(self) -> tuple[np.ndarray, np.ndarray]:
        """Affine map (M, t) sending a fixed equilateral triangle to this one.

        Useful for drawing the Steiner inellipse: the image of the
        equilateral triangle's incircle under this map is exactly the
        Steiner inellipse of ``self``.
        """
        E = np.array(
            [
                [0.0, 1.0],
                [-np.sqrt(3.0) / 2.0, -0.5],
                [np.sqrt(3.0) / 2.0, -0.5],
            ]
        )
        E_hom = np.hstack([E, np.ones((3, 1))])  # (3, 3), rows = [Ex, Ey, 1]
        P = self.vertices  # (3, 2)
        # P.T = X @ E_hom.T for X = [M | t] (2x3), i.e. E_hom @ X.T = P
        X = np.linalg.solve(E_hom, P).T
        M, t = X[:, :2], X[:, 2]
        return M, t

    # ------------------------------------------------------------------
    # Registries
    # ------------------------------------------------------------------

    POINTS: dict[str, Callable[["Triangle"], np.ndarray]] = {
        "centroid": lambda t: t.centroid(),
        "circumcenter": lambda t: t.circumcenter(),
        "incenter": lambda t: t.incenter(),
        "orthocenter": lambda t: t.orthocenter(),
        "steiner_focus_1": lambda t: t.steiner_focus_1(),
        "steiner_focus_2": lambda t: t.steiner_focus_2(),
    }

    SCALARS: dict[str, Callable[["Triangle"], float]] = {
        "area": lambda t: t.area(),
        "perimeter": lambda t: t.perimeter(),
        "semiperimeter": lambda t: t.semiperimeter(),
        "circumradius": lambda t: t.circumradius(),
        "inradius": lambda t: t.inradius(),
        "side_a": lambda t: t.side_a(),
        "side_b": lambda t: t.side_b(),
        "side_c": lambda t: t.side_c(),
        "angle_A": lambda t: t.angle_A(),
        "angle_B": lambda t: t.angle_B(),
        "angle_C": lambda t: t.angle_C(),
    }

    @classmethod
    def _register_pairwise_point_distances(cls) -> None:
        for name1, name2 in combinations(sorted(cls.POINTS), 2):
            key = f"dist_{name1}__{name2}"
            if key in cls.SCALARS:
                continue

            def make(n1=name1, n2=name2):
                return lambda t: float(
                    np.linalg.norm(t.point(n1) - t.point(n2))
                )

            cls.SCALARS[key] = make()

    def point(self, name: str) -> np.ndarray:
        return self.POINTS[name](self)

    def scalar(self, name: str) -> float:
        return float(self.SCALARS[name](self))

    def all_scalars(self) -> dict[str, float]:
        return {name: self.scalar(name) for name in self.SCALARS}

    def all_points(self) -> dict[str, np.ndarray]:
        return {name: self.point(name) for name in self.POINTS}

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(self, **kwargs):
        from triangle_relations.geometry.plotting import plot_triangle

        return plot_triangle(self, **kwargs)


Triangle._register_pairwise_point_distances()
