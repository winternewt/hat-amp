"""Penrose Robinson-triangle tiling generation.

This module ports the Penrose subdivision used by the third-party
percolation scripts into the package's numpy-based API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

TriangleKind = Literal["thin", "thick"]

PHI = (math.sqrt(5.0) + 1.0) / 2.0


@dataclass(frozen=True)
class PenroseTriangle:
    """A Robinson triangle used in the Penrose substitution."""

    shape: TriangleKind
    v1: np.ndarray
    v2: np.ndarray
    v3: np.ndarray

    def subdivide(self) -> list["PenroseTriangle"]:
        """Return the Penrose subdivision of this triangle."""
        if self.shape == "thin":
            p1 = self.v1 + (self.v2 - self.v1) / PHI
            return [
                PenroseTriangle("thin", self.v3, p1, self.v2),
                PenroseTriangle("thick", p1, self.v3, self.v1),
            ]

        p2 = self.v2 + (self.v1 - self.v2) / PHI
        p3 = self.v2 + (self.v3 - self.v2) / PHI
        return [
            PenroseTriangle("thick", p3, self.v3, self.v1),
            PenroseTriangle("thick", p2, p3, self.v2),
            PenroseTriangle("thin", p3, p2, self.v1),
        ]

    def get_vertices(self) -> np.ndarray:
        """Return triangle vertices as a ``(3, 2)`` float64 array."""
        return np.array([self.v1, self.v2, self.v3], dtype=np.float64)

    def get_edges(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return the two percolation edges used by the reference scripts."""
        return [(self.v1, self.v3), (self.v1, self.v2)]


@dataclass
class PenroseTiling:
    """A Penrose tiling generated from a reflected thin-triangle star."""

    divisions: int = 4
    base: int = 5
    scale: float = 200.0
    triangles: list[PenroseTriangle] = field(default_factory=list)

    def create_initial_tiles(self) -> None:
        """Create the initial reflected thin-triangle star."""
        initial_scale = self.scale * 0.5
        triangles: list[PenroseTriangle] = []

        for i in range(self.base * 2):
            angle_2 = (2 * i - 1) * math.pi / (self.base * 2)
            angle_3 = (2 * i + 1) * math.pi / (self.base * 2)
            v2 = np.array(
                [initial_scale * math.cos(angle_2), initial_scale * math.sin(angle_2)],
                dtype=np.float64,
            )
            v3 = np.array(
                [initial_scale * math.cos(angle_3), initial_scale * math.sin(angle_3)],
                dtype=np.float64,
            )

            if i % 2 == 0:
                v2, v3 = v3, v2

            triangles.append(PenroseTriangle("thin", np.zeros(2, dtype=np.float64), v2, v3))

        self.triangles = triangles

    def subdivide_all(self) -> None:
        """Apply all configured subdivision rounds."""
        for _ in range(self.divisions):
            self.triangles = [
                new_triangle
                for triangle in self.triangles
                for new_triangle in triangle.subdivide()
            ]

    def make_tiling(self) -> None:
        """Generate the tiling in-place."""
        self.create_initial_tiles()
        self.subdivide_all()

    def polygons(self) -> list[np.ndarray]:
        """Return all triangles as polygon arrays."""
        return [triangle.get_vertices() for triangle in self.triangles]


def generate_penrose_tiling(
    divisions: int = 4,
    base: int = 5,
    scale: float = 200.0,
) -> PenroseTiling:
    """Generate and return a Penrose tiling."""
    tiling = PenroseTiling(divisions=divisions, base=base, scale=scale)
    tiling.make_tiling()
    return tiling
