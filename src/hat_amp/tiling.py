"""Hat monotile tiling generation via hierarchical substitution.

Faithful port of Kaplan's hatviz (``hat.js``).  The substitution system
uses four metatile types (H, T, P, F) whose outlines and children are
defined in terms of affine-transformed copies of themselves.  Calling
``generate_tiling(level)`` inflates the system *level* times and returns
every hat polygon as an (N, 2) numpy array of Cartesian vertex coords.
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
    hex_pt,
    intersect,
    match_two,
    pt,
    rot_about,
    rotation,
    translation,
)

# ---------------------------------------------------------------------------
# Hat outline — 13 vertices in hex coordinates
# ---------------------------------------------------------------------------
HAT_OUTLINE: list[np.ndarray] = [
    hex_pt(0, 0),   hex_pt(-1, -1), hex_pt(0, -2),
    hex_pt(2, -2),  hex_pt(2, -1),  hex_pt(4, -2),
    hex_pt(5, -1),  hex_pt(4, 0),   hex_pt(3, 0),
    hex_pt(2, 2),   hex_pt(0, 3),   hex_pt(0, 2),
    hex_pt(-1, 2),
]

_HR3 = math.sqrt(3) / 2  # half root-3


# ---------------------------------------------------------------------------
# Symbolic tiling nodes
# ---------------------------------------------------------------------------

@dataclass
class ChildPlacement:
    """A child geometry placed by an affine transform."""

    T: np.ndarray          # affine transform placing this child
    geom: "HatTile | MetaTile"


@dataclass
class HatTile:
    """Leaf node — a single hat tile with a colour label."""
    label: str
    shape: list[np.ndarray] = field(default_factory=lambda: list(HAT_OUTLINE))


@dataclass
class MetaTile:
    """Internal node — a group of transformed children + an outline."""
    shape: list[np.ndarray]
    width: float
    children: list[ChildPlacement] = field(default_factory=list)

    def add_child(self, T: np.ndarray, geom: "HatTile | MetaTile") -> None:
        self.children.append(ChildPlacement(T=T, geom=geom))

    def eval_child(self, n: int, i: int) -> np.ndarray:
        """Evaluate the *i*-th outline vertex of child *n* in parent frame."""
        return apply_pt(self.children[n].T, self.children[n].geom.shape[i])

    def recentre(self) -> None:
        """Translate outline + children so centroid is at the origin."""
        cx = sum(p[0] for p in self.shape) / len(self.shape)
        cy = sum(p[1] for p in self.shape) / len(self.shape)
        tr = np.array([-cx, -cy], dtype=np.float64)
        self.shape = [p + tr for p in self.shape]
        M = translation(-cx, -cy)
        for ch in self.children:
            ch.T = compose(M, ch.T)


Patch = MetaTile


@dataclass(frozen=True)
class MetatileSet:
    """Compact symbolic state for one inflation level."""

    H: MetaTile
    T: MetaTile
    P: MetaTile
    F: MetaTile


# ---------------------------------------------------------------------------
# Singleton hat tile objects (one per colour label)
# ---------------------------------------------------------------------------
_H1_hat = HatTile(label="H1")
_H_hat  = HatTile(label="H")
_T_hat  = HatTile(label="T")
_P_hat  = HatTile(label="P")
_F_hat  = HatTile(label="F")


# ---------------------------------------------------------------------------
# Initial metatile definitions  (level-1)
# ---------------------------------------------------------------------------

def _build_H_init() -> MetaTile:
    H_outline = [
        pt(0, 0), pt(4, 0), pt(4.5, _HR3),
        pt(2.5, 5 * _HR3), pt(1.5, 5 * _HR3), pt(-0.5, _HR3),
    ]
    meta = MetaTile(shape=H_outline, width=2)

    meta.add_child(
        match_two(HAT_OUTLINE[5], HAT_OUTLINE[7], H_outline[5], H_outline[0]),
        _H_hat,
    )
    meta.add_child(
        match_two(HAT_OUTLINE[9], HAT_OUTLINE[11], H_outline[1], H_outline[2]),
        _H_hat,
    )
    meta.add_child(
        match_two(HAT_OUTLINE[5], HAT_OUTLINE[7], H_outline[3], H_outline[4]),
        _H_hat,
    )
    # The H1 (reflected) hat — uses explicit matrix from Kaplan
    meta.add_child(
        compose(
            translation(2.5, _HR3),
            compose(
                from_flat([-0.5, -_HR3, 0, _HR3, -0.5, 0]),
                from_flat([0.5, 0, 0, 0, -0.5, 0]),
            ),
        ),
        _H1_hat,
    )
    return meta


def _build_T_init() -> MetaTile:
    T_outline = [pt(0, 0), pt(3, 0), pt(1.5, 3 * _HR3)]
    meta = MetaTile(shape=T_outline, width=2)
    meta.add_child(from_flat([0.5, 0, 0.5, 0, 0.5, _HR3]), _T_hat)
    return meta


def _build_P_init() -> MetaTile:
    P_outline = [pt(0, 0), pt(4, 0), pt(3, 2 * _HR3), pt(-1, 2 * _HR3)]
    meta = MetaTile(shape=P_outline, width=2)
    meta.add_child(from_flat([0.5, 0, 1.5, 0, 0.5, _HR3]), _P_hat)
    meta.add_child(
        compose(
            translation(0, 2 * _HR3),
            compose(
                from_flat([0.5, _HR3, 0, -_HR3, 0.5, 0]),
                from_flat([0.5, 0.0, 0.0, 0.0, 0.5, 0.0]),
            ),
        ),
        _P_hat,
    )
    return meta


def _build_F_init() -> MetaTile:
    F_outline = [
        pt(0, 0), pt(3, 0), pt(3.5, _HR3),
        pt(3, 2 * _HR3), pt(-1, 2 * _HR3),
    ]
    meta = MetaTile(shape=F_outline, width=2)
    meta.add_child(from_flat([0.5, 0, 1.5, 0, 0.5, _HR3]), _F_hat)
    meta.add_child(
        compose(
            translation(0, 2 * _HR3),
            compose(
                from_flat([0.5, _HR3, 0, -_HR3, 0.5, 0]),
                from_flat([0.5, 0.0, 0.0, 0.0, 0.5, 0.0]),
            ),
        ),
        _F_hat,
    )
    return meta


# ---------------------------------------------------------------------------
# constructPatch — Kaplan's 28 substitution rules
# ---------------------------------------------------------------------------

def _construct_patch(
    H: MetaTile,
    T: MetaTile,
    P: MetaTile,
    F: MetaTile,
) -> Patch:
    """Apply the 28 placement rules to build a larger patch.

    The rules list is copied verbatim from Kaplan's ``constructPatch``.
    """
    rules: list[list] = [
        ["H"],
        [0, 0, "P", 2],
        [1, 0, "H", 2],
        [2, 0, "P", 2],
        [3, 0, "H", 2],
        [4, 4, "P", 2],
        [0, 4, "F", 3],
        [2, 4, "F", 3],
        [4, 1, 3, 2, "F", 0],
        [8, 3, "H", 0],
        [9, 2, "P", 0],
        [10, 2, "H", 0],
        [11, 4, "P", 2],
        [12, 0, "H", 2],
        [13, 0, "F", 3],
        [14, 2, "F", 1],
        [15, 3, "H", 4],
        [8, 2, "F", 1],
        [17, 3, "H", 0],
        [18, 2, "P", 0],
        [19, 2, "H", 2],
        [20, 4, "F", 3],
        [20, 0, "P", 2],
        [22, 0, "H", 2],
        [23, 4, "F", 3],
        [23, 0, "F", 3],
        [16, 0, "P", 2],
        [9, 4, 0, 2, "T", 2],
        [4, 0, "F", 3],
    ]

    shapes = {"H": H, "T": T, "P": P, "F": F}
    ret = MetaTile(shape=[], width=H.width)

    for r in rules:
        if len(r) == 1:
            # First rule — place the seed metatile at identity
            ret.add_child(IDENTITY.copy(), shapes[r[0]])
        elif len(r) == 4:
            # 4-element rule: [child_idx, vertex_idx, shape_name, new_vertex_idx]
            ch_idx, v_idx, shape_name, nv_idx = r
            poly = ret.children[ch_idx].geom.shape
            T_mat = ret.children[ch_idx].T
            P_pt = apply_pt(T_mat, poly[(v_idx + 1) % len(poly)])
            Q_pt = apply_pt(T_mat, poly[v_idx])
            nshp = shapes[shape_name]
            npoly = nshp.shape
            ret.add_child(
                match_two(
                    npoly[nv_idx],
                    npoly[(nv_idx + 1) % len(npoly)],
                    P_pt,
                    Q_pt,
                ),
                nshp,
            )
        else:
            # 6-element rule: [ch_P_idx, v_P, ch_Q_idx, v_Q, shape_name, nv_idx]
            ch_P_idx, v_P, ch_Q_idx, v_Q, shape_name, nv_idx = r
            chP = ret.children[ch_P_idx]
            chQ = ret.children[ch_Q_idx]
            P_pt = apply_pt(chQ.T, chQ.geom.shape[v_Q])
            Q_pt = apply_pt(chP.T, chP.geom.shape[v_P])
            nshp = shapes[shape_name]
            npoly = nshp.shape
            ret.add_child(
                match_two(
                    npoly[nv_idx],
                    npoly[(nv_idx + 1) % len(npoly)],
                    P_pt,
                    Q_pt,
                ),
                nshp,
            )

    return ret


# ---------------------------------------------------------------------------
# constructMetatiles — extract new H/T/P/F from patch
# ---------------------------------------------------------------------------

def _construct_metatiles(
    patch: Patch,
) -> tuple[MetaTile, MetaTile, MetaTile, MetaTile]:
    """Extract the next-level H, T, P, F metatiles from *patch*."""

    bps1 = patch.eval_child(8, 2)
    bps2 = patch.eval_child(21, 2)
    rbps = apply_pt(rot_about(bps1, -2.0 * math.pi / 3.0), bps2)

    p72 = patch.eval_child(7, 2)
    p252 = patch.eval_child(25, 2)

    llc = intersect(bps1, rbps, patch.eval_child(6, 2), p72)
    w = patch.eval_child(6, 2) - llc

    # --- new H outline (6 vertices) ---
    new_H_outline: list[np.ndarray] = [llc, bps1]
    w = apply_pt(rotation(-math.pi / 3), w)
    new_H_outline.append(new_H_outline[1] + w)
    new_H_outline.append(patch.eval_child(14, 2))
    w = apply_pt(rotation(-math.pi / 3), w)
    new_H_outline.append(new_H_outline[3] - w)
    new_H_outline.append(patch.eval_child(6, 2))

    new_H = MetaTile(shape=new_H_outline, width=patch.width * 2)
    for ch_idx in [0, 9, 16, 27, 26, 6, 1, 8, 10, 15]:
        new_H.add_child(
            patch.children[ch_idx].T.copy(),
            patch.children[ch_idx].geom,
        )

    # --- new P outline (4 vertices) ---
    new_P_outline = [p72, p72 + (bps1 - llc), bps1, llc]
    new_P = MetaTile(shape=new_P_outline, width=patch.width * 2)
    for ch_idx in [7, 2, 3, 4, 28]:
        new_P.add_child(
            patch.children[ch_idx].T.copy(),
            patch.children[ch_idx].geom,
        )

    # --- new F outline (5 vertices) ---
    new_F_outline = [
        bps2,
        patch.eval_child(24, 2),
        patch.eval_child(25, 0),
        p252,
        p252 + (llc - bps1),
    ]
    new_F = MetaTile(shape=new_F_outline, width=patch.width * 2)
    for ch_idx in [21, 20, 22, 23, 24, 25]:
        new_F.add_child(
            patch.children[ch_idx].T.copy(),
            patch.children[ch_idx].geom,
        )

    # --- new T outline (3 vertices) ---
    AAA = new_H_outline[2]
    BBB = new_H_outline[1] + (new_H_outline[4] - new_H_outline[5])
    CCC = apply_pt(rot_about(BBB, -math.pi / 3), AAA)
    new_T_outline = [BBB, CCC, AAA]
    new_T = MetaTile(shape=new_T_outline, width=patch.width * 2)
    new_T.add_child(
        patch.children[11].T.copy(),
        patch.children[11].geom,
    )

    # Recentre all metatiles
    new_H.recentre()
    new_P.recentre()
    new_F.recentre()
    new_T.recentre()

    return new_H, new_T, new_P, new_F


# ---------------------------------------------------------------------------
# Recursive hat extraction
# ---------------------------------------------------------------------------

def _collect_hats(
    geom: "HatTile | MetaTile",
    transform: np.ndarray,
    out: list[np.ndarray],
) -> None:
    """Walk the tree, collecting transformed hat polygons at the leaves."""
    if isinstance(geom, HatTile):
        pts = np.array(HAT_OUTLINE, dtype=np.float64)   # (13, 2)
        transformed = (pts @ transform[:2, :2].T) + transform[:2, 2]
        out.append(transformed)
    elif isinstance(geom, MetaTile):
        for child in geom.children:
            _collect_hats(child.geom, compose(transform, child.T), out)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def initial_metatiles() -> MetatileSet:
    """Return the initial symbolic H/T/P/F metatile set."""
    return MetatileSet(
        H=_build_H_init(),
        T=_build_T_init(),
        P=_build_P_init(),
        F=_build_F_init(),
    )


def construct_patch(tiles: MetatileSet) -> Patch:
    """Build Kaplan's full intermediate substitution patch for *tiles*."""
    return _construct_patch(tiles.H, tiles.T, tiles.P, tiles.F)


def extract_metatiles(patch: Patch) -> MetatileSet:
    """Extract the next symbolic H/T/P/F metatile set from *patch*."""
    H, T, P, F = _construct_metatiles(patch)
    return MetatileSet(H=H, T=T, P=P, F=F)


def inflate_metatiles(tiles: MetatileSet) -> MetatileSet:
    """Inflate one symbolic metatile generation."""
    return extract_metatiles(construct_patch(tiles))


def expand_hats(geom: HatTile | MetaTile) -> list[np.ndarray]:
    """Expand a symbolic geometry tree into atomic hat polygons."""
    hats: list[np.ndarray] = []
    _collect_hats(geom, IDENTITY.copy(), hats)
    return hats


def generate_tiling(level: int) -> list[np.ndarray]:
    """Generate hat tiling at given inflation level.

    Starting from the four base metatiles (H, T, P, F), the substitution
    system is applied *level* times.  The H metatile is then expanded and
    all hat polygons are returned.

    Returns
    -------
    list[np.ndarray]
        List of hat polygons, each an (13, 2) array of vertex coordinates.
    """
    tiles = initial_metatiles()

    for _ in range(level):
        tiles = inflate_metatiles(tiles)

    return expand_hats(tiles.H)


def generate_patch_tiling(level: int) -> list[np.ndarray]:
    """Generate hats from the full intermediate substitution patch.

    Kaplan's ``constructPatch`` first builds a larger arrangement of old
    metatiles, then ``constructMetatiles`` extracts the next H/T/P/F metatiles
    from it.  Percolation workflows use that larger patch directly as a wider
    sampling window.  ``level`` matches the third-party runner's inflation
    depth and must be at least 1.
    """
    if level < 1:
        msg = "Patch tiling level must be at least 1"
        raise ValueError(msg)

    tiles = initial_metatiles()

    patch: Patch | None = None
    for _ in range(level):
        patch = construct_patch(tiles)
        tiles = extract_metatiles(patch)

    if patch is None:
        msg = "Patch tiling level must be at least 1"
        raise ValueError(msg)

    return expand_hats(patch)
