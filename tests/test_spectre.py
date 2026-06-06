"""Tests for the Spectre chiral aperiodic monotile tiling module."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hat_amp.spectre import (
    GOLD_VERTEX_IDX,
    SPECTRE_OUTLINE,
    add_gold_vertex,
    expand_spectres,
    generate_spectre_tiling,
    generate_spectre_tiling_labeled,
    strip_gold_vertex,
    _build_spectre_base,
    _build_supertiles,
)


# ---------------------------------------------------------------------------
# Tile outline geometry
# ---------------------------------------------------------------------------

def test_spectre_outline_vertex_count():
    assert len(SPECTRE_OUTLINE) == 14


def test_spectre_outline_finite():
    for v in SPECTRE_OUTLINE:
        assert np.all(np.isfinite(v))


def test_gold_vertex_collinear():
    """Edges V9→V10 and V10→V11 must be parallel (gold vertex is collinear)."""
    v9  = SPECTRE_OUTLINE[9]
    v10 = SPECTRE_OUTLINE[GOLD_VERTEX_IDX]
    v11 = SPECTRE_OUTLINE[11]
    e1 = v10 - v9
    e2 = v11 - v10
    cross = e1[0] * e2[1] - e1[1] * e2[0]
    assert abs(cross) < 1e-10, "V10 is not collinear (gold vertex check failed)"


def test_all_edges_unit_length():
    """All 14 edges of SPECTRE_OUTLINE must be unit length.

    Regression: V4 and V5 were previously computed as (1.5+HR3, HR3-0.5)
    instead of (1.5+HR3, 1.5-HR3), pinching those edges to length ~0.732.
    """
    pts = np.array(SPECTRE_OUTLINE)
    for i in range(len(pts)):
        edge = pts[(i + 1) % len(pts)] - pts[i]
        length = float(np.linalg.norm(edge))
        assert abs(length - 1.0) < 1e-9, (
            f"Edge {i}→{(i+1)%len(pts)} has length {length:.8f}, expected 1.0"
        )


def test_v4_v5_y_coordinates():
    """V4 and V5 must have y = 1.5 - sqrt(3)/2 ≈ 0.6340, not 0.3660.

    Regression: symbolic expression _HR3-0.5 gives 0.366 (wrong);
    correct expression is 1.5-_HR3 = 0.634.
    """
    _HR3 = math.sqrt(3) / 2
    expected_y = 1.5 - _HR3
    assert abs(SPECTRE_OUTLINE[4][1] - expected_y) < 1e-12, (
        f"V4 y = {SPECTRE_OUTLINE[4][1]:.8f}, expected {expected_y:.8f}"
    )
    assert abs(SPECTRE_OUTLINE[5][1] - expected_y) < 1e-12, (
        f"V5 y = {SPECTRE_OUTLINE[5][1]:.8f}, expected {expected_y:.8f}"
    )


# ---------------------------------------------------------------------------
# Tile count progression (S_{n+1} = 7 S_n + M_n, M_n here = polygon count
# contributed by each Mystic compound at that level)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("level,expected", [
    (0, 1),    # single Delta leaf
    (1, 9),    # 7 S-leaves + 1 Gamma(2 leaves)
    (2, 71),   # 7*9 + 8  (Gamma level-1 yields 6 S-leaves + 1 Gamma(2))
])
def test_tile_count(level, expected):
    polygons = generate_spectre_tiling(level)
    assert len(polygons) == expected


def test_tile_count_level3():
    # Level 3: 7 * 71 + M_2, where M_2 = 6*9 + 8 = 62
    polygons = generate_spectre_tiling(3)
    assert len(polygons) == 7 * 71 + 62


# ---------------------------------------------------------------------------
# Polygon shape
# ---------------------------------------------------------------------------

def test_polygon_shape_level1():
    polygons = generate_spectre_tiling(1)
    for p in polygons:
        assert p.shape == (14, 2), f"Expected (14, 2), got {p.shape}"
        assert np.all(np.isfinite(p))


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def test_labels_level0():
    polygons, labels = generate_spectre_tiling_labeled(0)
    assert labels == ["S"]


def test_labels_level1():
    polygons, labels = generate_spectre_tiling_labeled(1)
    assert len(polygons) == 9
    assert labels.count("S") == 7
    assert labels.count("M") == 2


def test_labels_only_s_and_m():
    _, labels = generate_spectre_tiling_labeled(2)
    assert set(labels).issubset({"S", "M"})


# ---------------------------------------------------------------------------
# Gold vertex helpers
# ---------------------------------------------------------------------------

def test_strip_gold_vertex():
    polys = generate_spectre_tiling(1)
    stripped = strip_gold_vertex(polys)
    for p in stripped:
        assert p.shape == (13, 2)


def test_add_gold_vertex_roundtrip():
    polys = generate_spectre_tiling(1)
    stripped = strip_gold_vertex(polys)
    restored = add_gold_vertex(stripped)
    for orig, rest in zip(polys, restored):
        assert rest.shape == (14, 2)
        # Gold vertex should be near the midpoint of V9 and V11 in the original
        np.testing.assert_allclose(rest[GOLD_VERTEX_IDX], orig[GOLD_VERTEX_IDX], atol=1e-10)


# ---------------------------------------------------------------------------
# Base system sanity
# ---------------------------------------------------------------------------

def test_base_system_keys():
    sys_ = _build_spectre_base()
    expected = {"Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Phi", "Psi", "Gamma"}
    assert set(sys_.keys()) == expected


def test_gamma_has_two_children():
    from hat_amp.spectre import SpectreNode
    sys_ = _build_spectre_base()
    gamma = sys_["Gamma"]
    assert isinstance(gamma, SpectreNode)
    assert len(gamma.children) == 2


def test_supertile_has_eight_children():
    from hat_amp.spectre import SpectreNode
    sys_ = _build_spectre_base()
    sys1 = _build_supertiles(sys_)
    delta1 = sys1["Delta"]
    assert isinstance(delta1, SpectreNode)
    # Delta rule has no 'null' — exactly 8 children
    assert len(delta1.children) == 8


def test_gamma_supertile_has_seven_children():
    from hat_amp.spectre import SpectreNode
    sys_ = _build_spectre_base()
    sys1 = _build_supertiles(sys_)
    gamma1 = sys1["Gamma"]
    assert isinstance(gamma1, SpectreNode)
    # Gamma rule has one 'null' — 7 children
    assert len(gamma1.children) == 7


# ---------------------------------------------------------------------------
# Graph integration (requires scipy)
# ---------------------------------------------------------------------------

pytest.importorskip("scipy", reason="scipy required for graph tests")


def test_vertex_graph_level1():
    from hat_amp.graph import build_vertex_graph
    polygons = generate_spectre_tiling(1)
    graph = build_vertex_graph(polygons)
    assert graph.nodes.shape[1] == 2
    assert len(graph.edges) > 0


def test_vertex_graph_bipartite_vs_natural():
    """Bipartite (14-vertex) graph has more nodes than stripped (13-vertex) graph."""
    from hat_amp.graph import build_vertex_graph
    polygons = generate_spectre_tiling(2)
    g_full    = build_vertex_graph(polygons)
    g_stripped = build_vertex_graph(strip_gold_vertex(polygons))
    assert g_full.nodes.shape[0] > g_stripped.nodes.shape[0]
