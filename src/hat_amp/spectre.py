"""Spectre chiral aperiodic monotile tiling via the 9-tile substitution system.

Port of Craig Kaplan's spectre app (https://cs.uwaterloo.ca/~csk/spectre/).
Nine tile types: eight Spectre variants (Delta, Theta, Lambda, Xi, Pi, Sigma,
Phi, Psi) plus the Mystic compound (Gamma) which pairs two Spectres at 30°.

Each returned polygon has 14 vertices.  Vertex index 10 is the collinear
"gold vertex" (edges 9→10 and 10→11 are both leftward), which makes the vertex
graph bipartite per Singh & Flicker (PRB 109 L220303).  Call
``strip_gold_vertex`` to obtain 13-vertex polygons for natural graph analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from hat_amp.geometry import (
    IDENTITY,
    apply_pt,
    compose,
    from_flat,
    pt,
    rotation,
    translation,
)

_HR3 = math.sqrt(3) / 2  # 0.8660254037844386

# ---------------------------------------------------------------------------
# Spectre tile — 14 Cartesian vertices.
# V10 is the gold vertex: edges V9→V10 and V10→V11 are both (−1, 0).
# ---------------------------------------------------------------------------
SPECTRE_OUTLINE: list[np.ndarray] = [
    pt(0.0,                       0.0),               # 0
    pt(1.0,                       0.0),               # 1
    pt(1.5,                      -_HR3),              # 2
    pt(1.5 + _HR3,                0.5 - _HR3),        # 3
    pt(1.5 + _HR3,                1.5 - _HR3),        # 4
    pt(2.5 + _HR3,                1.5 - _HR3),        # 5
    pt(3.0 + _HR3,                1.5),               # 6
    pt(3.0,                       2.0),               # 7
    pt(3.0 - _HR3,                1.5),               # 8
    pt(2.5 - _HR3,                1.5 + _HR3),        # 9
    pt(1.5 - _HR3,                1.5 + _HR3),        # 10  ← gold vertex
    pt(0.5 - _HR3,                1.5 + _HR3),        # 11
    pt(-_HR3,                     1.5),               # 12
    pt(0.0,                       1.0),               # 13
]

# Four "key" (quad) vertices used for supertile edge-matching.
SPECTRE_KEYS: list[np.ndarray] = [
    SPECTRE_OUTLINE[3],
    SPECTRE_OUTLINE[5],
    SPECTRE_OUTLINE[7],
    SPECTRE_OUTLINE[11],
]

#: Index of the gold (collinear) vertex within SPECTRE_OUTLINE.
GOLD_VERTEX_IDX: int = 10


# ---------------------------------------------------------------------------
# Tile node types
# ---------------------------------------------------------------------------

@dataclass
class SpectreTile:
    """Leaf tile.  Labels 'Gamma1'/'Gamma2' mark the two halves of a Mystic."""

    label: str
    shape: list[np.ndarray] = field(default_factory=lambda: list(SPECTRE_OUTLINE))
    quad: list[np.ndarray]  = field(default_factory=lambda: list(SPECTRE_KEYS))


@dataclass
class SpectreNode:
    """Container node — a placed collection of child tiles or sub-nodes."""

    children: list = field(default_factory=list)   # [(transform, SpectreTile|SpectreNode)]
    quad: list[np.ndarray] = field(default_factory=list)

    def add_child(self, T: np.ndarray, geom: "SpectreTile | SpectreNode") -> None:
        self.children.append((T, geom))


# ---------------------------------------------------------------------------
# Base-level tile system
# ---------------------------------------------------------------------------

def _build_spectre_base() -> dict[str, "SpectreTile | SpectreNode"]:
    """Return the nine base-level tile objects (level 0)."""
    sys_: dict[str, SpectreTile | SpectreNode] = {}
    for lab in ("Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Phi", "Psi"):
        sys_[lab] = SpectreTile(label=lab)

    # Gamma = Mystic: two Spectres, the second translated to V8 and rotated π/6.
    gamma: SpectreNode = SpectreNode(quad=list(SPECTRE_KEYS))
    gamma.add_child(IDENTITY.copy(), SpectreTile("Gamma1"))
    v8 = SPECTRE_OUTLINE[8]
    gamma.add_child(
        compose(translation(v8[0], v8[1]), rotation(math.pi / 6.0)),
        SpectreTile("Gamma2"),
    )
    sys_["Gamma"] = gamma
    return sys_


# ---------------------------------------------------------------------------
# Supertile substitution
# ---------------------------------------------------------------------------

_SUPER_RULES: dict[str, list[str | None]] = {
    "Gamma":  ["Pi",  "Delta", None,  "Theta", "Sigma", "Xi",     "Phi",    "Gamma"],
    "Delta":  ["Xi",  "Delta", "Xi",  "Phi",   "Sigma", "Pi",     "Phi",    "Gamma"],
    "Theta":  ["Psi", "Delta", "Pi",  "Phi",   "Sigma", "Pi",     "Phi",    "Gamma"],
    "Lambda": ["Psi", "Delta", "Xi",  "Phi",   "Sigma", "Pi",     "Phi",    "Gamma"],
    "Xi":     ["Psi", "Delta", "Pi",  "Phi",   "Sigma", "Psi",    "Phi",    "Gamma"],
    "Pi":     ["Psi", "Delta", "Xi",  "Phi",   "Sigma", "Psi",    "Phi",    "Gamma"],
    "Sigma":  ["Xi",  "Delta", "Xi",  "Phi",   "Sigma", "Pi",     "Lambda", "Gamma"],
    "Phi":    ["Psi", "Delta", "Psi", "Phi",   "Sigma", "Pi",     "Phi",    "Gamma"],
    "Psi":    ["Psi", "Delta", "Psi", "Phi",   "Sigma", "Psi",    "Phi",    "Gamma"],
}


def _compute_supertile_transforms(
    quad: list[np.ndarray],
) -> list[np.ndarray]:
    """Return the eight placement transforms for one inflation step.

    Port of the transform-building loop in Kaplan's ``buildSupertiles``.
    Each transform T[i] places the i-th sub-tile within a supertile so
    consecutive tiles share an edge (matched via the quad key-points).
    A final x-reflection (R) is applied to all transforms.
    """
    R = from_flat([-1.0, 0.0, 0.0, 0.0, 1.0, 0.0])   # x-reflection

    # (angle_deg, from_quad_idx, to_quad_idx)
    t_rules = [
        (60,   3, 1),
        (0,    2, 0),
        (60,   3, 1),
        (60,   3, 1),
        (0,    2, 0),
        (60,   3, 1),
        (-120, 3, 3),
    ]

    Ts: list[np.ndarray] = [IDENTITY.copy()]
    total_ang = 0.0
    rot = IDENTITY.copy()
    tquad = list(quad)

    for ang, from_idx, to_idx in t_rules:
        total_ang += ang
        if ang != 0:
            rot = rotation(math.radians(total_ang))
            tquad = [apply_pt(rot, q) for q in quad]
        dest = apply_pt(Ts[-1], quad[from_idx])
        src  = tquad[to_idx]
        ttt  = translation(dest[0] - src[0], dest[1] - src[1])
        Ts.append(compose(ttt, rot))

    return [compose(R, T) for T in Ts]


def _build_supertiles(
    sys_: dict[str, "SpectreTile | SpectreNode"],
) -> dict[str, "SpectreNode"]:
    """Apply one level of supertile substitution and return the new system."""
    quad = sys_["Delta"].quad
    Ts   = _compute_supertile_transforms(quad)

    super_quad = [
        apply_pt(Ts[6], quad[2]),
        apply_pt(Ts[5], quad[1]),
        apply_pt(Ts[3], quad[2]),
        apply_pt(Ts[0], quad[1]),
    ]

    result: dict[str, SpectreNode] = {}
    for lab, subs in _SUPER_RULES.items():
        node = SpectreNode(quad=super_quad)
        for idx, sub_lab in enumerate(subs):
            if sub_lab is not None:
                node.add_child(Ts[idx], sys_[sub_lab])
        result[lab] = node
    return result


# ---------------------------------------------------------------------------
# Polygon collection (tree walk)
# ---------------------------------------------------------------------------

def _collect_spectres(
    geom: "SpectreTile | SpectreNode",
    transform: np.ndarray,
    polygons: list[np.ndarray],
    labels: list[str],
) -> None:
    if isinstance(geom, SpectreTile):
        pts = np.array(SPECTRE_OUTLINE, dtype=np.float64)
        transformed = (pts @ transform[:2, :2].T) + transform[:2, 2]
        polygons.append(transformed)
        labels.append("M" if geom.label in ("Gamma1", "Gamma2") else "S")
    else:  # SpectreNode
        for T, child in geom.children:
            _collect_spectres(child, compose(transform, T), polygons, labels)


def expand_spectres(
    geom: "SpectreTile | SpectreNode",
    transform: np.ndarray | None = None,
) -> tuple[list[np.ndarray], list[str]]:
    """Walk the substitution tree and collect all leaf Spectre polygons.

    Returns
    -------
    polygons : list of (14, 2) arrays
    labels   : list of ``'S'`` (standalone Spectre) or ``'M'`` (Mystic component)
    """
    if transform is None:
        transform = IDENTITY.copy()
    polygons: list[np.ndarray] = []
    labels: list[str] = []
    _collect_spectres(geom, transform, polygons, labels)
    return polygons, labels


# ---------------------------------------------------------------------------
# Gold-vertex helpers
# ---------------------------------------------------------------------------

def strip_gold_vertex(polygons: list[np.ndarray]) -> list[np.ndarray]:
    """Remove the gold vertex (index 10) from each 14-vertex polygon.

    Returns 13-vertex polygons for natural (non-bipartite) vertex-graph
    analysis.  Singh & Flicker (PRB 109 L220303) re-insert this vertex to
    restore bipartiteness; ``add_gold_vertex`` reverses this operation.
    """
    return [
        np.delete(p, GOLD_VERTEX_IDX, axis=0) if p.shape[0] == 14 else p
        for p in polygons
    ]


def add_gold_vertex(polygons: list[np.ndarray]) -> list[np.ndarray]:
    """Insert the gold vertex back into each 13-vertex polygon.

    The gold vertex lies at the midpoint of the doubled edge
    V9→(V10)→V11 (all at the same y-coordinate), restoring the
    14-vertex bipartite form.
    """
    result = []
    for p in polygons:
        if p.shape[0] == 13:
            # Gold vertex sits between what is now index 9 and 10
            v9  = p[9]
            v10 = p[10]   # was V11 before stripping
            gold = 0.5 * (v9 + v10)
            p = np.insert(p, 10, gold, axis=0)
        result.append(p)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_spectre_tiling(level: int) -> list[np.ndarray]:
    """Generate a Spectre tiling at the given inflation level.

    Starts from the ``Delta`` supertile and inflates *level* times.

    Returns
    -------
    list of (14, 2) ndarray
        Every Spectre polygon (Spectre variants and both halves of Mystic
        compounds).  Polygon count follows
        S_{n+1} = 7·S_n + M_n, M_{n+1} = 6·S_n + M_n with S_0 = 1, M_0 = 2.
    """
    sys_ = _build_spectre_base()
    for _ in range(level):
        sys_ = _build_supertiles(sys_)
    polygons, _ = expand_spectres(sys_["Delta"])
    return polygons


def generate_spectre_tiling_labeled(
    level: int,
) -> tuple[list[np.ndarray], list[str]]:
    """Generate a Spectre tiling with per-polygon chirality/type labels.

    Returns
    -------
    polygons : list of (14, 2) ndarray
    labels   : list of ``'S'`` or ``'M'`` (Mystic-interior component)
    """
    sys_ = _build_spectre_base()
    for _ in range(level):
        sys_ = _build_supertiles(sys_)
    return expand_spectres(sys_["Delta"])
