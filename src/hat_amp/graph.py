"""Graph construction utilities for polygon tilings."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.spatial import KDTree


@dataclass(frozen=True)
class VertexGraph:
    """A graph whose nodes are merged polygon vertices."""

    nodes: np.ndarray
    neighbors: list[np.ndarray]
    edges: np.ndarray


@dataclass(frozen=True)
class DualGraph:
    """A tile-adjacency graph whose nodes are polygon centroids."""

    centroids: np.ndarray
    neighbors: list[np.ndarray]
    polygons: list[np.ndarray]
    edges: np.ndarray

    @property
    def nodes(self) -> np.ndarray:
        """Alias centroids as nodes for generic graph consumers."""
        return self.centroids


@dataclass(frozen=True)
class CroppedGraph:
    """A square-window induced subgraph and its boundary node sets."""

    L_value: float
    nodes: np.ndarray
    neighbors: list[np.ndarray]
    edges: np.ndarray
    top_boundary_nodes: np.ndarray
    bottom_boundary_nodes: np.ndarray
    left_boundary_nodes: np.ndarray
    right_boundary_nodes: np.ndarray
    center: np.ndarray

    @property
    def node_count(self) -> int:
        """Number of nodes in the square crop."""
        return int(self.nodes.shape[0])

    @property
    def edge_count(self) -> int:
        """Number of undirected edges in the square crop."""
        return int(self.edges.shape[0])


class _GraphLike(Protocol):
    nodes: np.ndarray
    neighbors: list[np.ndarray]


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int64)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = int(self.parent[x])
        return x

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if self.size[root_a] < self.size[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        self.size[root_a] += self.size[root_b]


def _as_polygons(polygons: list[np.ndarray]) -> list[np.ndarray]:
    return [np.asarray(polygon, dtype=np.float64) for polygon in polygons]


def _neighbors_from_edges(node_count: int, edges: set[tuple[int, int]]) -> list[np.ndarray]:
    neighbors: list[list[int]] = [[] for _ in range(node_count)]
    for i, j in sorted(edges):
        neighbors[i].append(j)
        neighbors[j].append(i)
    return [np.array(node_neighbors, dtype=np.int32) for node_neighbors in neighbors]


def build_vertex_graph(polygons: list[np.ndarray], tol: float = 1e-5) -> VertexGraph:
    """Build a vertex graph by merging coincident polygon vertices."""
    normalized = _as_polygons(polygons)
    if not normalized:
        empty_edges = np.empty((0, 2), dtype=np.int32)
        return VertexGraph(np.empty((0, 2), dtype=np.float64), [], empty_edges)

    nodes = np.vstack(normalized)
    union_find = _UnionFind(len(nodes))
    tree = KDTree(nodes)
    for i, j in tree.query_pairs(r=tol):
        union_find.union(i, j)

    roots = np.array([union_find.find(i) for i in range(len(nodes))], dtype=np.int64)
    unique_roots = np.unique(roots)
    root_to_idx = {root: idx for idx, root in enumerate(unique_roots)}
    node_to_unique = np.array([root_to_idx[root] for root in roots], dtype=np.int32)
    unique_nodes = nodes[unique_roots]

    edges: set[tuple[int, int]] = set()
    node_offset = 0
    for polygon in normalized:
        n_vertices = polygon.shape[0]
        for i in range(n_vertices):
            idx1 = int(node_to_unique[node_offset + i])
            idx2 = int(node_to_unique[node_offset + ((i + 1) % n_vertices)])
            if idx1 != idx2:
                edges.add((min(idx1, idx2), max(idx1, idx2)))
        node_offset += n_vertices

    edge_array = np.array(sorted(edges), dtype=np.int32).reshape((-1, 2))
    neighbors = _neighbors_from_edges(len(unique_nodes), edges)
    return VertexGraph(nodes=unique_nodes, neighbors=neighbors, edges=edge_array)


def build_dual_graph(polygons: list[np.ndarray], tol: float = 1e-5) -> DualGraph:
    """Build a tile-adjacency graph from polygons sharing full edges."""
    normalized = _as_polygons(polygons)
    if not normalized:
        empty_edges = np.empty((0, 2), dtype=np.int32)
        return DualGraph(np.empty((0, 2), dtype=np.float64), [], [], empty_edges)

    centroids = np.array([polygon.mean(axis=0) for polygon in normalized], dtype=np.float64)
    all_vertices: list[np.ndarray] = []
    tile_ids: list[int] = []
    for tile_id, polygon in enumerate(normalized):
        for vertex in polygon:
            all_vertices.append(vertex)
            tile_ids.append(tile_id)

    vertex_array = np.array(all_vertices, dtype=np.float64)
    tile_id_array = np.array(tile_ids, dtype=np.int32)
    tree = KDTree(vertex_array)
    shared_vertex_count: defaultdict[tuple[int, int], int] = defaultdict(int)

    for i, j in tree.query_pairs(r=tol):
        tile_i = int(tile_id_array[i])
        tile_j = int(tile_id_array[j])
        if tile_i != tile_j:
            shared_vertex_count[(min(tile_i, tile_j), max(tile_i, tile_j))] += 1

    edges = {
        (tile_i, tile_j)
        for (tile_i, tile_j), count in shared_vertex_count.items()
        if count >= 2
    }
    edge_array = np.array(sorted(edges), dtype=np.int32).reshape((-1, 2))
    neighbors = _neighbors_from_edges(len(normalized), edges)
    return DualGraph(
        centroids=centroids,
        neighbors=neighbors,
        polygons=normalized,
        edges=edge_array,
    )


def neighbors_to_csr(neighbors: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Convert adjacency lists to CSR-style neighbor arrays."""
    starts = np.zeros(len(neighbors) + 1, dtype=np.int32)
    total = 0
    for i, node_neighbors in enumerate(neighbors):
        starts[i] = total
        total += len(node_neighbors)
    starts[-1] = total

    neighbor_array = np.empty(total, dtype=np.int32)
    offset = 0
    for node_neighbors in neighbors:
        end = offset + len(node_neighbors)
        neighbor_array[offset:end] = node_neighbors
        offset = end
    return neighbor_array, starts


def crop_square(
    graph: _GraphLike,
    L: float,
    center: tuple[float, float] | np.ndarray | None = None,
    boundary_thickness: float = 1.0,
) -> CroppedGraph:
    """Return the induced graph inside a square frame."""
    nodes = np.asarray(graph.nodes, dtype=np.float64)
    if center is None:
        if len(nodes) == 0:
            center_arr = np.zeros(2, dtype=np.float64)
        else:
            center_arr = np.array(
                [
                    (nodes[:, 0].min() + nodes[:, 0].max()) / 2.0,
                    (nodes[:, 1].min() + nodes[:, 1].max()) / 2.0,
                ],
                dtype=np.float64,
            )
    else:
        center_arr = np.asarray(center, dtype=np.float64)

    x_min, x_max = center_arr[0] - L / 2.0, center_arr[0] + L / 2.0
    y_min, y_max = center_arr[1] - L / 2.0, center_arr[1] + L / 2.0

    inside_mask = (
        (nodes[:, 0] >= x_min)
        & (nodes[:, 0] <= x_max)
        & (nodes[:, 1] >= y_min)
        & (nodes[:, 1] <= y_max)
    )
    inside_indices = np.where(inside_mask)[0]
    cropped_nodes = nodes[inside_indices]

    original_to_new = {int(orig): new for new, orig in enumerate(inside_indices)}
    edge_set: set[tuple[int, int]] = set()
    cropped_neighbors_lists: list[list[int]] = [[] for _ in range(len(inside_indices))]

    for new_i, original_i in enumerate(inside_indices):
        for original_j in graph.neighbors[int(original_i)]:
            new_j = original_to_new.get(int(original_j))
            if new_j is not None:
                cropped_neighbors_lists[new_i].append(new_j)
                edge_set.add((min(new_i, new_j), max(new_i, new_j)))

    cropped_neighbors = [
        np.array(sorted(set(node_neighbors)), dtype=np.int32)
        for node_neighbors in cropped_neighbors_lists
    ]
    cropped_edges = np.array(sorted(edge_set), dtype=np.int32).reshape((-1, 2))

    top_mask = cropped_nodes[:, 1] >= y_max - boundary_thickness
    bottom_mask = cropped_nodes[:, 1] <= y_min + boundary_thickness
    left_mask = cropped_nodes[:, 0] <= x_min + boundary_thickness
    right_mask = cropped_nodes[:, 0] >= x_max - boundary_thickness

    return CroppedGraph(
        L_value=float(L),
        nodes=cropped_nodes,
        neighbors=cropped_neighbors,
        edges=cropped_edges,
        top_boundary_nodes=np.where(top_mask)[0].astype(np.int32),
        bottom_boundary_nodes=np.where(bottom_mask)[0].astype(np.int32),
        left_boundary_nodes=np.where(left_mask)[0].astype(np.int32),
        right_boundary_nodes=np.where(right_mask)[0].astype(np.int32),
        center=center_arr,
    )
