"""Tests for percolation utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from hat_amp.graph import VertexGraph
from hat_amp.percolation import (
    BoundarySets,
    Criterion,
    WeightedQuickUnionUF,
    extrapolate_pc,
    run_bond_trials,
    run_site_trials,
)


def _square_graph() -> VertexGraph:
    nodes = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    neighbors = [
        np.array([1, 2], dtype=np.int32),
        np.array([0, 3], dtype=np.int32),
        np.array([0, 3], dtype=np.int32),
        np.array([1, 2], dtype=np.int32),
    ]
    edges = np.array([[0, 1], [0, 2], [1, 3], [2, 3]], dtype=np.int32)
    return VertexGraph(nodes=nodes, neighbors=neighbors, edges=edges)


def _square_boundaries() -> BoundarySets:
    return BoundarySets(
        top=np.array([2, 3], dtype=np.int32),
        bottom=np.array([0, 1], dtype=np.int32),
        left=np.array([0, 2], dtype=np.int32),
        right=np.array([1, 3], dtype=np.int32),
    )


def test_weighted_quick_union_connectivity() -> None:
    """Union-find should connect transitive components."""
    uf = WeightedQuickUnionUF(5)
    uf.union(0, 1)
    uf.union(1, 2)

    assert uf.connected(0, 2)
    assert not uf.connected(0, 3)


def test_seeded_site_trials_are_reproducible() -> None:
    """A fixed seed should produce stable site-percolation trial arrays."""
    graph = _square_graph()
    boundaries = _square_boundaries()

    first = run_site_trials(
        graph,
        boundaries,
        trials=8,
        seed=123,
        criterion=Criterion.UNION,
    )
    second = run_site_trials(
        graph,
        boundaries,
        trials=8,
        seed=123,
        criterion=Criterion.UNION,
    )

    np.testing.assert_array_equal(first, second)
    assert np.all((first > 0.0) & (first <= 1.0))


def test_seeded_bond_trials_are_reproducible() -> None:
    """A fixed seed should produce stable bond-percolation trial arrays."""
    graph = _square_graph()
    boundaries = _square_boundaries()

    first = run_bond_trials(
        graph.nodes,
        graph.edges,
        boundaries,
        trials=8,
        seed=456,
        criterion=Criterion.INTERSECTION,
    )
    second = run_bond_trials(
        graph.nodes,
        graph.edges,
        boundaries,
        trials=8,
        seed=456,
        criterion=Criterion.INTERSECTION,
    )

    np.testing.assert_array_equal(first, second)
    assert np.all((first > 0.0) & (first <= 1.0))


def test_extrapolate_pc_returns_known_linear_intercept() -> None:
    """Synthetic data following y = 0.5 + 0.25*x should fit p_c = 0.5."""
    L_values = np.array([4.0, 8.0, 16.0, 32.0], dtype=np.float64)
    x = L_values ** (-3.0 / 4.0)
    means_i = 0.5 + 0.25 * x
    means_u = 0.5 + 0.10 * x
    trials_i = [np.array([mean - 0.01, mean, mean + 0.01]) for mean in means_i]
    trials_u = [np.array([mean - 0.01, mean, mean + 0.01]) for mean in means_u]

    result = extrapolate_pc(L_values, trials_i, trials_u)

    assert result.intersection.pc == pytest.approx(0.5, abs=1e-12)
    assert result.union.pc == pytest.approx(0.5, abs=1e-12)


@pytest.mark.reference
def test_extrapolate_pc_matches_reference(reference_module: Any) -> None:
    """Finite-size extrapolation should match the reference implementation."""
    reference_module("hat_graph_builder.py")
    reference = reference_module("percolation.py")
    L_values = [8.0, 16.0, 32.0, 64.0]
    trials_i = [
        np.array([0.58, 0.60, 0.62]),
        np.array([0.55, 0.56, 0.57]),
        np.array([0.53, 0.535, 0.54]),
        np.array([0.515, 0.52, 0.525]),
    ]
    trials_u = [
        np.array([0.48, 0.50, 0.52]),
        np.array([0.49, 0.50, 0.51]),
        np.array([0.495, 0.50, 0.505]),
        np.array([0.498, 0.50, 0.502]),
    ]

    ours = extrapolate_pc(L_values, trials_i, trials_u)
    theirs = reference.extrapolate_pc_raw(L_values, trials_i, trials_u)

    assert ours.intersection.pc == pytest.approx(theirs["I"]["pc"], abs=1e-12)
    assert ours.union.pc == pytest.approx(theirs["U"]["pc"], abs=1e-12)
    assert ours.average.pc == pytest.approx(theirs["A"]["pc"], abs=1e-12)
