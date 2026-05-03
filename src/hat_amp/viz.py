"""SVG and PNG rendering helpers for tiling patches."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from xml.sax.saxutils import escape

import numpy as np

from hat_amp.penrose import generate_penrose_tiling
from hat_amp.tiling import HAT_OUTLINE, generate_patch_tiling, generate_tiling

TileSource = Literal["hat", "hat-patch", "penrose"]


def _normalize_polygons(polygons: list[np.ndarray]) -> list[np.ndarray]:
    return [np.asarray(polygon, dtype=np.float64) for polygon in polygons]


def _bounds(polygons: list[np.ndarray], padding: float) -> tuple[float, float, float, float]:
    if not polygons:
        return (0.0, 0.0, 1.0, 1.0)
    points = np.vstack(polygons)
    min_x = float(points[:, 0].min() - padding)
    min_y = float(points[:, 1].min() - padding)
    max_x = float(points[:, 0].max() + padding)
    max_y = float(points[:, 1].max() + padding)
    return (min_x, min_y, max_x - min_x, max_y - min_y)


def _svg_points(
    polygon: np.ndarray,
    viewbox: tuple[float, float, float, float],
) -> str:
    min_x, min_y, _width, height = viewbox
    max_y = min_y + height
    return " ".join(f"{x - min_x:.12g},{max_y - y:.12g}" for x, y in polygon)


def render_svg(
    polygons: list[np.ndarray],
    *,
    stroke: str = "#222222",
    fill: str = "none",
    stroke_width: float = 0.03,
    width: int = 800,
    height: int = 800,
    viewbox: tuple[float, float, float, float] | None = None,
    padding: float = 1.0,
    title: str | None = None,
) -> str:
    """Render polygons to an SVG document string."""
    normalized = _normalize_polygons(polygons)
    vb = _bounds(normalized, padding) if viewbox is None else viewbox
    title_element = f"<title>{escape(title)}</title>\n" if title else ""

    polygon_elements: list[str] = []
    for polygon in normalized:
        point_text = _svg_points(polygon, vb)
        polygon_elements.append(
            (
                f'<polygon points="{point_text}" fill="{escape(fill)}" '
                f'stroke="{escape(stroke)}" stroke-width="{stroke_width:.12g}" />'
            )
        )

    return (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="0 0 {vb[2]:.12g} {vb[3]:.12g}">\n'
        f"{title_element}"
        + "\n".join(polygon_elements)
        + "\n</svg>\n"
    )


def render_single_tile_svg(
    label: str = "H",
    *,
    stroke: str = "#222222",
    fill: str = "none",
    width: int = 800,
    height: int = 800,
) -> str:
    """Render a single tile outline as SVG.

    Currently ``label`` is accepted for future tile variants; all labels render
    the hat outline.
    """
    return render_svg(
        [np.array(HAT_OUTLINE, dtype=np.float64)],
        stroke=stroke,
        fill=fill,
        width=width,
        height=height,
        title=f"single tile {label}",
    )


def render_patch_svg(
    level: int,
    *,
    source: TileSource = "hat",
    stroke: str = "#222222",
    fill: str = "none",
    width: int = 800,
    height: int = 800,
) -> str:
    """Render a hat, hat-patch, or Penrose patch at the requested level."""
    if source == "hat":
        polygons = generate_tiling(level)
    elif source == "hat-patch":
        polygons = generate_patch_tiling(level)
    elif source == "penrose":
        polygons = generate_penrose_tiling(divisions=level).polygons()
    else:
        msg = f"Unsupported tile source: {source}"
        raise ValueError(msg)
    return render_svg(polygons, stroke=stroke, fill=fill, width=width, height=height)


def save_svg(svg_string: str, path: str | Path) -> Path:
    """Write an SVG string to disk and return the path."""
    output = Path(path)
    output.write_text(svg_string, encoding="utf-8")
    return output


def save_png(
    svg_or_polygons: str | list[np.ndarray],
    path: str | Path,
    *,
    dpi: int = 96,
) -> Path:
    """Render SVG or polygons to a PNG file via optional CairoSVG."""
    import cairosvg

    svg_string = (
        render_svg(svg_or_polygons)
        if isinstance(svg_or_polygons, list)
        else svg_or_polygons
    )
    output = Path(path)
    cairosvg.svg2png(bytestring=svg_string.encode("utf-8"), write_to=str(output), dpi=dpi)
    return output
