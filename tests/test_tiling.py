"""Tests for hat tiling generation.

Hat counts at each substitution level are verified against Kaplan's
reference JS implementation (hatviz).  The counts follow the recurrence
whose characteristic sequence is 2, 5, 13, 34, 89, ... (squares thereof
give 4, 25, 169, 1156, 7921).
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np
import pytest

from hat_amp.tiling import (
    construct_patch,
    expand_hats,
    extract_metatiles,
    generate_patch_tiling,
    generate_tiling,
    inflate_metatiles,
    initial_metatiles,
)


@pytest.mark.parametrize(
    "level, expected_count",
    [
        (0, 4),
        (1, 25),
        (2, 169),
        (3, 1156),
    ],
    ids=["level-0", "level-1", "level-2", "level-3"],
)
def test_hat_count(level: int, expected_count: int) -> None:
    """Hat count at level *level* should be *expected_count*."""
    hats = generate_tiling(level)
    assert len(hats) == expected_count, (
        f"Level {level}: expected {expected_count} hats, got {len(hats)}"
    )


def test_hat_polygon_shape() -> None:
    """Each hat polygon must have 13 vertices in 2D."""
    hats = generate_tiling(0)
    for i, hat in enumerate(hats):
        assert isinstance(hat, np.ndarray), f"Hat {i} is not an ndarray"
        assert hat.shape == (13, 2), f"Hat {i} shape is {hat.shape}, expected (13, 2)"


def test_hat_polygons_finite() -> None:
    """All vertex coordinates must be finite."""
    hats = generate_tiling(2)
    for i, hat in enumerate(hats):
        assert np.all(np.isfinite(hat)), f"Hat {i} contains non-finite values"


def _assert_polygon_lists_close(
    ours: list[np.ndarray],
    theirs: list[np.ndarray],
    atol: float,
) -> None:
    assert len(ours) == len(theirs)
    for ours_polygon, theirs_polygon in zip(ours, theirs, strict=True):
        np.testing.assert_allclose(ours_polygon, theirs_polygon, atol=atol, rtol=0.0)


def test_symbolic_inflation_matches_explicit_patch_extraction() -> None:
    """Public symbolic helpers should preserve the Kaplan two-stage pipeline."""
    tiles = initial_metatiles()
    via_inflate = inflate_metatiles(tiles)
    via_patch = extract_metatiles(construct_patch(tiles))

    for inflated, extracted in [
        (via_inflate.H, via_patch.H),
        (via_inflate.T, via_patch.T),
        (via_inflate.P, via_patch.P),
        (via_inflate.F, via_patch.F),
    ]:
        _assert_polygon_lists_close(
            expand_hats(inflated),
            expand_hats(extracted),
            atol=1e-12,
        )


def _third_party_collect_hats(
    third_party_hat_generator: ModuleType,
    geom: Any,
    transform: list[float],
    out: list[np.ndarray],
) -> None:
    """Collect atomic hat polygons from the vendored implementation."""
    if hasattr(geom, "children"):
        for child in geom.children:
            _third_party_collect_hats(
                third_party_hat_generator,
                child["geom"],
                third_party_hat_generator.mul(transform, child["T"]),
                out,
            )
        return

    polygon = np.array(
        [
            [
                third_party_hat_generator.transPt(transform, p)["x"],
                third_party_hat_generator.transPt(transform, p)["y"],
            ]
            for p in geom.shape
        ],
        dtype=np.float64,
    )
    out.append(polygon)


def _third_party_tiles(
    third_party_hat_generator: ModuleType,
    level: int,
) -> tuple[list[Any], Any | None]:
    """Build third-party tiles and final full patch for an inflation level."""
    tiles = [
        third_party_hat_generator.H_init(),
        third_party_hat_generator.T_init(),
        third_party_hat_generator.P_init(),
        third_party_hat_generator.F_init(),
    ]

    patch: Any | None = None
    for _ in range(level):
        patch = third_party_hat_generator.constructPatch(*tiles)
        tiles = third_party_hat_generator.constructMetatiles(patch)

    return tiles, patch


def _third_party_h_tiling(
    third_party_hat_generator: ModuleType,
    level: int,
) -> list[np.ndarray]:
    """Generate the third-party extracted H metatile output."""
    tiles, _ = _third_party_tiles(third_party_hat_generator, level)
    hats: list[np.ndarray] = []
    _third_party_collect_hats(
        third_party_hat_generator,
        tiles[0],
        list(third_party_hat_generator.ident),
        hats,
    )
    return hats


def _third_party_patch_tiling(
    third_party_hat_generator: ModuleType,
    level: int,
) -> list[np.ndarray]:
    """Generate the third-party full intermediate constructPatch output."""
    _, patch = _third_party_tiles(third_party_hat_generator, level)
    assert patch is not None

    hats: list[np.ndarray] = []
    _third_party_collect_hats(
        third_party_hat_generator,
        patch,
        list(third_party_hat_generator.ident),
        hats,
    )
    return hats


@pytest.mark.parametrize("level", [0, 1, 2, 3], ids=["level-0", "level-1", "level-2", "level-3"])
@pytest.mark.reference
def test_h_metatile_matches_third_party_generator(
    reference_module: Any,
    level: int,
) -> None:
    """The extracted H metatile should match the vendored implementation."""
    third_party_hat_generator = reference_module("hat_generator.py")
    ours = generate_tiling(level)
    theirs = _third_party_h_tiling(third_party_hat_generator, level)

    _assert_polygon_lists_close(ours, theirs, atol=1e-10)


@pytest.mark.parametrize("level", [1, 2, 3, 4], ids=["level-1", "level-2", "level-3", "level-4"])
@pytest.mark.reference
def test_full_patch_matches_third_party_generator(
    reference_module: Any,
    level: int,
) -> None:
    """The larger constructPatch output should match third-party percolation logic."""
    third_party_hat_generator = reference_module("hat_generator.py")
    ours = generate_patch_tiling(level)
    theirs = _third_party_patch_tiling(third_party_hat_generator, level)

    _assert_polygon_lists_close(ours, theirs, atol=1e-9)
