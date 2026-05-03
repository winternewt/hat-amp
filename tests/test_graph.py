"""Tests for tiling graph construction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from hat_amp.graph import build_dual_graph, build_vertex_graph, crop_square, neighbors_to_csr
from hat_amp.tiling import generate_patch_tiling


def test_vertex_graph_builds_edges_for_single_square() -> None:
    """A single square polygon should produce four nodes and four edges."""
    square = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=np.float64,
    )
    graph = build_vertex_graph([square])

    assert graph.nodes.shape == (4, 2)
    assert graph.edges.shape == (4, 2)
    assert sorted(len(neighbors) for neighbors in graph.neighbors) == [2, 2, 2, 2]


def test_dual_graph_connects_tiles_sharing_an_edge() -> None:
    """Two squares sharing a full edge should be adjacent in the dual graph."""
    left = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    right = np.array([[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]])
    graph = build_dual_graph([left, right])

    np.testing.assert_array_equal(graph.edges, np.array([[0, 1]], dtype=np.int32))
    np.testing.assert_array_equal(graph.neighbors[0], np.array([1], dtype=np.int32))
    np.testing.assert_array_equal(graph.neighbors[1], np.array([0], dtype=np.int32))


def test_crop_square_extracts_boundaries() -> None:
    """Square crops should retain induced edges and side boundary sets."""
    polygons = [
        np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]),
        np.array([[2.0, 0.0], [4.0, 0.0], [4.0, 2.0], [2.0, 2.0]]),
    ]
    graph = build_vertex_graph(polygons)
    cropped = crop_square(graph, L=4.0, center=(2.0, 1.0), boundary_thickness=1.0)

    assert cropped.node_count == 6
    assert cropped.edge_count == 7
    assert len(cropped.left_boundary_nodes) == 2
    assert len(cropped.right_boundary_nodes) == 2
    assert len(cropped.top_boundary_nodes) == 3
    assert len(cropped.bottom_boundary_nodes) == 3


def test_neighbors_to_csr_roundtrip_lengths() -> None:
    """CSR starts should encode adjacency-list lengths."""
    neighbors = [
        np.array([1, 2], dtype=np.int32),
        np.array([0], dtype=np.int32),
        np.array([0], dtype=np.int32),
    ]
    arr, starts = neighbors_to_csr(neighbors)

    np.testing.assert_array_equal(arr, np.array([1, 2, 0, 0], dtype=np.int32))
    np.testing.assert_array_equal(starts, np.array([0, 2, 3, 4], dtype=np.int32))


def _reference_patch(reference_hat_generator: Any, level: int) -> Any:
    tiles = [
        reference_hat_generator.H_init(),
        reference_hat_generator.T_init(),
        reference_hat_generator.P_init(),
        reference_hat_generator.F_init(),
    ]
    patch = None
    for _ in range(level):
        patch = reference_hat_generator.constructPatch(*tiles)
        tiles = reference_hat_generator.constructMetatiles(patch)
    assert patch is not None
    return patch


@pytest.mark.reference
@pytest.mark.parametrize("level", [1, 2])
def test_vertex_graph_matches_reference_counts(
    reference_module: Any,
    level: int,
) -> None:
    """Vertex graph node and edge counts should match the reference builder."""
    reference_hat = reference_module("hat_generator.py")
    reference_graph = reference_module("hat_graph_builder.py")
    patch = _reference_patch(reference_hat, level)
    ref_nodes, ref_neighbors = reference_graph.build_neighbor_graph_fast(
        patch,
        level=level + 1,
    )

    ours = build_vertex_graph(generate_patch_tiling(level))
    assert ours.nodes.shape[0] == ref_nodes.shape[0]
    assert ours.edges.shape[0] == sum(len(n) for n in ref_neighbors) // 2


@pytest.mark.reference
@pytest.mark.parametrize("level", [1, 2])
def test_dual_graph_matches_reference_counts(
    reference_module: Any,
    level: int,
) -> None:
    """Dual graph node and edge counts should match the reference builder."""
    reference_hat = reference_module("hat_generator.py")
    reference_dual = reference_module("hat_dual_graph_builder.py")
    patch = _reference_patch(reference_hat, level)
    ref_nodes, ref_neighbors, ref_polygons = reference_dual.build_tile_graph(
        patch,
        level=level + 1,
    )

    ours = build_dual_graph(generate_patch_tiling(level))
    assert ours.nodes.shape[0] == ref_nodes.shape[0]
    assert len(ours.polygons) == len(ref_polygons)
    assert ours.edges.shape[0] == sum(len(n) for n in ref_neighbors) // 2
