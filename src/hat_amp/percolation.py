"""Site and bond percolation utilities for finite tiling graphs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np
from scipy import stats

from hat_amp.graph import CroppedGraph, neighbors_to_csr


class Criterion(str, Enum):
    """Two-channel crossing criteria."""

    INTERSECTION = "I"
    UNION = "U"


class _GraphLike(Protocol):
    nodes: np.ndarray
    neighbors: list[np.ndarray]


@dataclass(frozen=True)
class BoundarySets:
    """Boundary node indices for square-frame crossing tests."""

    top: np.ndarray
    bottom: np.ndarray
    left: np.ndarray
    right: np.ndarray

    @classmethod
    def from_cropped_graph(cls, graph: CroppedGraph) -> "BoundarySets":
        """Create boundary sets from a cropped square graph."""
        return cls(
            top=graph.top_boundary_nodes,
            bottom=graph.bottom_boundary_nodes,
            left=graph.left_boundary_nodes,
            right=graph.right_boundary_nodes,
        )


@dataclass(frozen=True)
class FitResult:
    """Weighted finite-size scaling fit for one crossing statistic."""

    pc: float
    amplitude: float
    pc_std: float
    amplitude_std: float
    pc_ci: tuple[float, float]


@dataclass(frozen=True)
class ExtrapolationResult:
    """Extrapolated p_c fits for intersection, union, and average curves."""

    intersection: FitResult
    union: FitResult
    average: FitResult


class WeightedQuickUnionUF:
    """Weighted union-find with path compression."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, p: int) -> int:
        """Return the canonical root for ``p``."""
        while p != self.parent[p]:
            self.parent[p] = self.parent[self.parent[p]]
            p = self.parent[p]
        return p

    def connected(self, p: int, q: int) -> bool:
        """Return whether ``p`` and ``q`` are in the same component."""
        return self.find(p) == self.find(q)

    def union(self, p: int, q: int) -> None:
        """Join the components containing ``p`` and ``q``."""
        root_p = self.find(p)
        root_q = self.find(q)
        if root_p == root_q:
            return
        if self.size[root_p] < self.size[root_q]:
            self.parent[root_p] = root_q
            self.size[root_q] += self.size[root_p]
        else:
            self.parent[root_q] = root_p
            self.size[root_p] += self.size[root_q]


class SitePercolation:
    """Incremental site percolation simulation on an undirected graph."""

    def __init__(
        self,
        graph: _GraphLike,
        boundaries: BoundarySets,
        criterion: Criterion = Criterion.INTERSECTION,
    ) -> None:
        self.node_count = int(len(graph.nodes))
        self.neighbors_arr, self.neighbor_starts = neighbors_to_csr(graph.neighbors)
        self.top_set = set(int(i) for i in boundaries.top)
        self.bottom_set = set(int(i) for i in boundaries.bottom)
        self.left_set = set(int(i) for i in boundaries.left)
        self.right_set = set(int(i) for i in boundaries.right)
        self.criterion = criterion
        self.sites = np.zeros(self.node_count, dtype=bool)
        self.uf_tb = WeightedQuickUnionUF(self.node_count + 2)
        self.uf_lr = WeightedQuickUnionUF(self.node_count + 2)
        self.virtual_top = self.node_count
        self.virtual_bottom = self.node_count + 1
        self.virtual_left = self.node_count
        self.virtual_right = self.node_count + 1
        self.open_sites = 0

    def open_site(self, idx: int) -> None:
        """Open one site and connect it to already-open neighbors."""
        if self.sites[idx]:
            return
        self.sites[idx] = True
        self.open_sites += 1
        if idx in self.top_set:
            self.uf_tb.union(self.virtual_top, idx)
        if idx in self.bottom_set:
            self.uf_tb.union(self.virtual_bottom, idx)
        if idx in self.left_set:
            self.uf_lr.union(self.virtual_left, idx)
        if idx in self.right_set:
            self.uf_lr.union(self.virtual_right, idx)

        start = int(self.neighbor_starts[idx])
        end = int(self.neighbor_starts[idx + 1])
        for neighbor in self.neighbors_arr[start:end]:
            neighbor_idx = int(neighbor)
            if self.sites[neighbor_idx]:
                self.uf_tb.union(idx, neighbor_idx)
                self.uf_lr.union(idx, neighbor_idx)

    def percolates(self) -> bool:
        """Return whether the requested crossing criterion is satisfied."""
        top_bottom = self.uf_tb.connected(self.virtual_top, self.virtual_bottom)
        left_right = self.uf_lr.connected(self.virtual_left, self.virtual_right)
        if self.criterion == Criterion.INTERSECTION:
            return top_bottom and left_right
        return top_bottom or left_right


class BondPercolation:
    """Incremental bond percolation simulation on an undirected edge set."""

    def __init__(
        self,
        nodes: np.ndarray,
        edges: np.ndarray,
        boundaries: BoundarySets,
        criterion: Criterion = Criterion.INTERSECTION,
    ) -> None:
        self.node_count = int(len(nodes))
        self.edges = np.asarray(edges, dtype=np.int32)
        self.edge_count = int(len(self.edges))
        self.criterion = criterion
        self.uf_tb = WeightedQuickUnionUF(self.node_count + 2)
        self.uf_lr = WeightedQuickUnionUF(self.node_count + 2)
        self.virtual_top = self.node_count
        self.virtual_bottom = self.node_count + 1
        self.virtual_left = self.node_count
        self.virtual_right = self.node_count + 1
        self.open_bonds = 0

        for node in boundaries.top:
            self.uf_tb.union(self.virtual_top, int(node))
        for node in boundaries.bottom:
            self.uf_tb.union(self.virtual_bottom, int(node))
        for node in boundaries.left:
            self.uf_lr.union(self.virtual_left, int(node))
        for node in boundaries.right:
            self.uf_lr.union(self.virtual_right, int(node))

    def open_bond(self, idx: int) -> None:
        """Open one bond and connect its endpoints."""
        self.open_bonds += 1
        u, v = self.edges[idx]
        self.uf_tb.union(int(u), int(v))
        self.uf_lr.union(int(u), int(v))

    def percolates(self) -> bool:
        """Return whether the requested crossing criterion is satisfied."""
        top_bottom = self.uf_tb.connected(self.virtual_top, self.virtual_bottom)
        left_right = self.uf_lr.connected(self.virtual_left, self.virtual_right)
        if self.criterion == Criterion.INTERSECTION:
            return top_bottom and left_right
        return top_bottom or left_right


def run_site_trials(
    graph: _GraphLike,
    boundaries: BoundarySets,
    trials: int,
    seed: int | None = None,
    criterion: Criterion = Criterion.INTERSECTION,
) -> np.ndarray:
    """Run site-percolation trials and return threshold fractions."""
    rng = np.random.default_rng(seed)
    results = np.empty(trials, dtype=np.float64)
    node_count = int(len(graph.nodes))
    if node_count == 0:
        return np.zeros(trials, dtype=np.float64)

    for trial_idx in range(trials):
        sim = SitePercolation(graph, boundaries, criterion=criterion)
        for site in rng.permutation(node_count):
            sim.open_site(int(site))
            if sim.percolates():
                break
        results[trial_idx] = sim.open_sites / node_count
    return results


def run_bond_trials(
    nodes: np.ndarray,
    edges: np.ndarray,
    boundaries: BoundarySets,
    trials: int,
    seed: int | None = None,
    criterion: Criterion = Criterion.INTERSECTION,
) -> np.ndarray:
    """Run bond-percolation trials and return threshold fractions."""
    rng = np.random.default_rng(seed)
    edge_array = np.asarray(edges, dtype=np.int32)
    edge_count = int(len(edge_array))
    results = np.empty(trials, dtype=np.float64)
    if edge_count == 0:
        return np.zeros(trials, dtype=np.float64)

    for trial_idx in range(trials):
        sim = BondPercolation(nodes, edge_array, boundaries, criterion=criterion)
        for edge in rng.permutation(edge_count):
            sim.open_bond(int(edge))
            if sim.percolates():
                break
        results[trial_idx] = sim.open_bonds / edge_count
    return results


def _wls_fit(
    x: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    confidence: float,
) -> FitResult:
    safe_sigma = np.where(sigma > 0.0, sigma, np.finfo(np.float64).eps)
    weights = 1.0 / safe_sigma**2
    design = np.column_stack([np.ones_like(x), x])
    weighted_design = design * weights[:, None]
    xtwx = design.T @ weighted_design
    xtwy = design.T @ (weights * y)
    coefficients = np.linalg.solve(xtwx, xtwy)
    covariance = np.linalg.inv(xtwx)

    pc_hat = float(coefficients[0])
    amplitude_hat = float(coefficients[1])
    pc_std = float(np.sqrt(covariance[0, 0]))
    amplitude_std = float(np.sqrt(covariance[1, 1]))
    dof = max(len(x) - 2, 1)
    t_crit = float(stats.t.ppf((1.0 + confidence) / 2.0, df=dof))
    pc_ci = (pc_hat - t_crit * pc_std, pc_hat + t_crit * pc_std)

    return FitResult(
        pc=pc_hat,
        amplitude=amplitude_hat,
        pc_std=pc_std,
        amplitude_std=amplitude_std,
        pc_ci=pc_ci,
    )


def extrapolate_pc(
    L_values: list[float] | np.ndarray,
    trials_intersection: list[np.ndarray],
    trials_union: list[np.ndarray],
    nu: float = 4.0 / 3.0,
    confidence: float = 0.95,
) -> ExtrapolationResult:
    """Extrapolate p_c using the reference finite-size scaling law."""
    L = np.asarray(L_values, dtype=np.float64)
    x = L ** (-1.0 / nu)

    means_i: list[float] = []
    means_u: list[float] = []
    means_a: list[float] = []
    sigma_i: list[float] = []
    sigma_u: list[float] = []
    sigma_a: list[float] = []

    for result_i, result_u in zip(trials_intersection, trials_union, strict=True):
        r_i = np.asarray(result_i, dtype=np.float64)
        r_u = np.asarray(result_u, dtype=np.float64)
        trial_count = len(r_i)
        ddof = 1 if trial_count > 1 else 0

        mean_i = float(r_i.mean())
        mean_u = float(r_u.mean())
        std_i = float(r_i.std(ddof=ddof))
        std_u = float(r_u.std(ddof=ddof))
        covariance = float(np.cov(r_i, r_u, ddof=ddof)[0, 1]) if trial_count > 1 else 0.0

        means_i.append(mean_i)
        means_u.append(mean_u)
        means_a.append(0.5 * (mean_i + mean_u))
        sigma_i.append(std_i / np.sqrt(trial_count))
        sigma_u.append(std_u / np.sqrt(trial_count))
        sigma_a.append(
            0.5
            * np.sqrt(
                std_i**2 / trial_count
                + std_u**2 / trial_count
                + 2.0 * covariance / trial_count
            )
        )

    return ExtrapolationResult(
        intersection=_wls_fit(x, np.array(means_i), np.array(sigma_i), confidence),
        union=_wls_fit(x, np.array(means_u), np.array(sigma_u), confidence),
        average=_wls_fit(x, np.array(means_a), np.array(sigma_a), confidence),
    )
