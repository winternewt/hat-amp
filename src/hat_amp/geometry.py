"""2D affine transform primitives for hat tiling geometry.

Port of Kaplan's hatviz/geometry.js.  Affine transforms are 3x3 numpy
matrices (homogeneous coordinates); points are length-2 numpy arrays.
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SQRT3_OVER_2 = math.sqrt(3) / 2

IDENTITY: np.ndarray = np.eye(3, dtype=np.float64)

# ---------------------------------------------------------------------------
# Point helpers
# ---------------------------------------------------------------------------

def pt(x: float, y: float) -> np.ndarray:
    """Create a 2D point as a numpy array."""
    return np.array([x, y], dtype=np.float64)


def hex_pt(x: float, y: float) -> np.ndarray:
    """Convert hex coordinates to Cartesian.

    ``hex_pt(x, y)`` -> ``(x + 0.5*y, (sqrt(3)/2)*y)``
    """
    return np.array([x + 0.5 * y, _SQRT3_OVER_2 * y], dtype=np.float64)


# ---------------------------------------------------------------------------
# Affine transform construction  (3x3 homogeneous matrices)
# ---------------------------------------------------------------------------

def translation(tx: float, ty: float) -> np.ndarray:
    """Return a 3x3 affine translation matrix."""
    m = np.eye(3, dtype=np.float64)
    m[0, 2] = tx
    m[1, 2] = ty
    return m


def rotation(angle: float) -> np.ndarray:
    """Return a 3x3 affine rotation matrix (angle in radians)."""
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def rot_about(p: np.ndarray, angle: float) -> np.ndarray:
    """Rotation by *angle* radians about point *p*."""
    return compose(translation(p[0], p[1]),
                   compose(rotation(angle),
                           translation(-p[0], -p[1])))


# ---------------------------------------------------------------------------
# Affine algebra
# ---------------------------------------------------------------------------

def compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compose two affine transforms: ``a @ b``."""
    return a @ b


def inv(t: np.ndarray) -> np.ndarray:
    """Invert an affine transform."""
    return np.linalg.inv(t)


def apply_pt(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Apply affine transform *t* to a single 2D point *p*."""
    return (t[:2, :2] @ p) + t[:2, 2]


def apply_pts(t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply affine transform *t* to an (N, 2) array of points."""
    return (pts @ t[:2, :2].T) + t[:2, 2]


# ---------------------------------------------------------------------------
# Segment matching (core of the substitution system)
# ---------------------------------------------------------------------------

def _match_seg(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return the affine transform that maps the unit segment (0,0)->(1,0)
    to the segment *p* -> *q*.

    This mirrors Kaplan's ``matchSeg`` which maps [0,1] to p->q:
        [qx-px, py-qy, px,
         qy-py, qx-px, py]
    converted to a 3x3 matrix.
    """
    dx = q[0] - p[0]
    dy = q[1] - p[1]
    return np.array([
        [ dx, p[1] - q[1], p[0]],
        [ dy,  dx,          p[1]],
        [0.0, 0.0,          1.0],
    ], dtype=np.float64)


def match_two(
    p1: np.ndarray,
    q1: np.ndarray,
    p2: np.ndarray,
    q2: np.ndarray,
) -> np.ndarray:
    """Return the affine transform mapping segment *p1*->*q1* onto *p2*->*q2*.

    Equivalent to ``matchSeg(p2,q2) @ inv(matchSeg(p1,q1))``.
    """
    return compose(_match_seg(p2, q2), inv(_match_seg(p1, q1)))


# ---------------------------------------------------------------------------
# Line intersection (used in constructMetatiles)
# ---------------------------------------------------------------------------

def intersect(
    p1: np.ndarray, q1: np.ndarray,
    p2: np.ndarray, q2: np.ndarray,
) -> np.ndarray:
    """Intersect two lines defined by segments p1->q1 and p2->q2."""
    d = (q2[1] - p2[1]) * (q1[0] - p1[0]) - (q2[0] - p2[0]) * (q1[1] - p1[1])
    ua = ((q2[0] - p2[0]) * (p1[1] - p2[1]) - (q2[1] - p2[1]) * (p1[0] - p2[0])) / d
    return pt(p1[0] + ua * (q1[0] - p1[0]), p1[1] + ua * (q1[1] - p1[1]))


# ---------------------------------------------------------------------------
# Convenience: build a 3x3 affine from the 6-element flat representation
# used in Kaplan's JS code:  [a, b, tx, c, d, ty]
# ---------------------------------------------------------------------------

def from_flat(flat: list[float] | tuple[float, ...]) -> np.ndarray:
    """Convert Kaplan's 6-element ``[a, b, tx, c, d, ty]`` to a 3x3 matrix."""
    a, b, tx, c, d, ty = flat
    return np.array([
        [a, b, tx],
        [c, d, ty],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
