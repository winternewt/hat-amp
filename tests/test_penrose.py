"""Tests for Penrose tiling generation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from hat_amp.penrose import PHI, PenroseTriangle, generate_penrose_tiling


@pytest.mark.parametrize(
    "divisions, expected_count",
    [(0, 10), (1, 20), (2, 50), (3, 130)],
)
def test_penrose_triangle_counts(divisions: int, expected_count: int) -> None:
    """Subdivision counts should follow the Robinson triangle substitution."""
    tiling = generate_penrose_tiling(divisions=divisions, base=5, scale=200.0)
    assert len(tiling.triangles) == expected_count


def test_thin_triangle_subdivision_uses_golden_ratio() -> None:
    """The first thin subdivision point lies 1/phi along v1 -> v2."""
    triangle = PenroseTriangle(
        "thin",
        np.array([0.0, 0.0]),
        np.array([PHI, 0.0]),
        np.array([0.0, 1.0]),
    )
    thin, thick = triangle.subdivide()

    np.testing.assert_allclose(thin.v2, np.array([1.0, 0.0]))
    np.testing.assert_allclose(thick.v1, np.array([1.0, 0.0]))


def _reference_vertices(triangle: Any) -> np.ndarray:
    return np.array(
        [
            [triangle.v1.real, triangle.v1.imag],
            [triangle.v2.real, triangle.v2.imag],
            [triangle.v3.real, triangle.v3.imag],
        ],
        dtype=np.float64,
    )


@pytest.mark.reference
@pytest.mark.parametrize("divisions", [0, 1, 2, 3])
def test_penrose_matches_reference(
    reference_module: Any,
    divisions: int,
) -> None:
    """Generated Penrose triangles should match the reference numerically."""
    reference = reference_module("penrose_tiling_generator.py")
    ref_tiling = reference.PenroseTiling(divisions=divisions, base=5, scale=200)
    ref_tiling.make_tiling()

    ours = generate_penrose_tiling(divisions=divisions, base=5, scale=200.0)
    assert len(ours.triangles) == len(ref_tiling.triangles)

    for ours_triangle, ref_triangle in zip(ours.triangles, ref_tiling.triangles, strict=True):
        assert ours_triangle.shape == ref_triangle.shape
        np.testing.assert_allclose(
            ours_triangle.get_vertices(),
            _reference_vertices(ref_triangle),
            atol=1e-10,
            rtol=0.0,
        )
