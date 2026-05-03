"""Tests for SVG and PNG visualization helpers."""

from __future__ import annotations

from xml.etree import ElementTree

import numpy as np

from hat_amp.viz import render_patch_svg, render_single_tile_svg, render_svg, save_png, save_svg

SVG_NS = "{http://www.w3.org/2000/svg}"


def test_render_svg_is_well_formed_and_counts_polygons() -> None:
    """SVG output should parse and contain one polygon per input polygon."""
    polygons = [
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        np.array([[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]]),
    ]
    svg = render_svg(polygons, title="two triangles")
    root = ElementTree.fromstring(svg)
    polygon_elements = root.findall(f".//{SVG_NS}polygon")

    assert root.tag == f"{SVG_NS}svg"
    assert len(polygon_elements) == 2


def test_render_single_tile_svg_has_one_polygon() -> None:
    """A single tile render should contain exactly one polygon."""
    root = ElementTree.fromstring(render_single_tile_svg())
    polygon_elements = root.findall(f".//{SVG_NS}polygon")

    assert len(polygon_elements) == 1


def test_render_patch_svg_has_expected_hat_count() -> None:
    """Hat level 1 renders should include 25 hat polygons."""
    root = ElementTree.fromstring(render_patch_svg(1, source="hat"))
    polygon_elements = root.findall(f".//{SVG_NS}polygon")

    assert len(polygon_elements) == 25


def test_save_svg_writes_file(tmp_path) -> None:
    """SVG files should be written as UTF-8 text."""
    output = save_svg(render_single_tile_svg(), tmp_path / "tile.svg")

    assert output.exists()
    assert output.read_text(encoding="utf-8").startswith("<svg")


def test_save_png_writes_file(tmp_path) -> None:
    """PNG export should create a non-empty file."""
    output = save_png(render_single_tile_svg(), tmp_path / "tile.png")

    assert output.exists()
    assert output.stat().st_size > 0
