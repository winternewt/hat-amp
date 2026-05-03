# hat-amp API

`hat-amp` provides NumPy-based geometry tools for hat monotile and Penrose
tilings, graph construction, square-window cropping, percolation experiments,
result persistence, and SVG/PNG rendering.

The package name is `hat-amp`; Python imports use `hat_amp`.

```python
from hat_amp.tiling import generate_tiling

hats = generate_tiling(level=3)
```

## Installation Extras

The core package depends on `numpy` and `pydantic`.

Graph and percolation helpers need SciPy:

```bash
uv add "hat-amp[graph]"
```

PNG export needs CairoSVG:

```bash
uv add "hat-amp[viz]"
```

## Tiling

Import from `hat_amp.tiling`:

```python
from hat_amp.tiling import (
    ChildPlacement,
    HatTile,
    MetaTile,
    MetatileSet,
    Patch,
    construct_patch,
    expand_hats,
    extract_metatiles,
    generate_patch_tiling,
    generate_tiling,
    inflate_metatiles,
    initial_metatiles,
)
```

### `generate_tiling(level) -> list[np.ndarray]`

Returns atomic hat polygons from the extracted `H` metatile after `level`
inflation steps. Each polygon is a `(13, 2)` `float64` array.

| Level | Hats |
| ---: | ---: |
| 0 | 4 |
| 1 | 25 |
| 2 | 169 |
| 3 | 1,156 |
| 4 | 7,921 |

### `generate_patch_tiling(level) -> list[np.ndarray]`

Returns atomic hat polygons from the full intermediate Kaplan
`construct_patch(...)` staging patch. `level` must be at least `1`.

| Level | Patch Hats |
| ---: | ---: |
| 1 | 77 |
| 2 | 526 |
| 3 | 3,603 |
| 4 | 24,693 |

Use this for wider finite-window workflows and reference parity with the
third-party percolation scripts.

### Symbolic Pipeline

`MetatileSet` stores the symbolic `H`, `T`, `P`, and `F` metatiles for one
inflation level.

```python
from hat_amp.tiling import construct_patch, expand_hats, inflate_metatiles, initial_metatiles

tiles = initial_metatiles()
for _ in range(3):
    tiles = inflate_metatiles(tiles)

h_hats = expand_hats(tiles.H)
patch_hats = expand_hats(construct_patch(tiles))
```

The explicit two-stage form is:

```python
patch = construct_patch(tiles)
next_tiles = extract_metatiles(patch)
```

`inflate_metatiles(tiles)` is equivalent to
`extract_metatiles(construct_patch(tiles))`.

### Data Model

`HatTile` is a leaf node with:

- `label: str`
- `shape: list[np.ndarray]`

`ChildPlacement` stores a child object and its placement transform:

- `T: np.ndarray`, a `3x3` homogeneous affine transform
- `geom: HatTile | MetaTile`

`MetaTile` is an internal symbolic node:

- `shape: list[np.ndarray]`
- `width: float`
- `children: list[ChildPlacement]`
- `add_child(T, geom) -> None`
- `eval_child(n, i) -> np.ndarray`
- `recentre() -> None`

`Patch` is an alias for `MetaTile` used to name the full intermediate Kaplan
staging patch.

Coordinates are Cartesian `float64`. The hat outline is defined internally in
hex coordinates and converted by:

```python
hex_pt(x, y) = (x + 0.5 * y, (sqrt(3) / 2) * y)
```

## Penrose

Import from `hat_amp.penrose`:

```python
from hat_amp.penrose import PHI, PenroseTiling, PenroseTriangle, generate_penrose_tiling
```

### `generate_penrose_tiling(divisions=4, base=5, scale=200.0) -> PenroseTiling`

Builds a Robinson-triangle Penrose tiling from a reflected thin-triangle star.

```python
from hat_amp.penrose import generate_penrose_tiling

tiling = generate_penrose_tiling(divisions=4, base=5, scale=200.0)
triangles = tiling.triangles
polygons = tiling.polygons()
```

`PenroseTriangle` fields:

- `shape: Literal["thin", "thick"]`
- `v1: np.ndarray`
- `v2: np.ndarray`
- `v3: np.ndarray`

Methods:

- `subdivide() -> list[PenroseTriangle]`
- `get_vertices() -> np.ndarray`, shape `(3, 2)`
- `get_edges() -> list[tuple[np.ndarray, np.ndarray]]`

`PenroseTiling` fields:

- `divisions: int`
- `base: int`
- `scale: float`
- `triangles: list[PenroseTriangle]`

Methods:

- `create_initial_tiles() -> None`
- `subdivide_all() -> None`
- `make_tiling() -> None`
- `polygons() -> list[np.ndarray]`

## Graphs

Import from `hat_amp.graph`:

```python
from hat_amp.graph import (
    CroppedGraph,
    DualGraph,
    VertexGraph,
    build_dual_graph,
    build_vertex_graph,
    crop_square,
    neighbors_to_csr,
)
```

### `build_vertex_graph(polygons, tol=1e-5) -> VertexGraph`

Builds a graph by merging coincident polygon vertices with a SciPy `KDTree`.
Edges follow each polygon boundary.

```python
from hat_amp.graph import build_vertex_graph
from hat_amp.tiling import generate_patch_tiling

polygons = generate_patch_tiling(level=2)
graph = build_vertex_graph(polygons)
```

`VertexGraph` fields:

- `nodes: np.ndarray`, shape `(N, 2)`
- `neighbors: list[np.ndarray]`
- `edges: np.ndarray`, shape `(E, 2)`

### `build_dual_graph(polygons, tol=1e-5) -> DualGraph`

Builds a tile-adjacency graph. Each polygon centroid is a node; two tiles are
adjacent when they share at least two coincident vertices, i.e. a full edge.

`DualGraph` fields:

- `centroids: np.ndarray`
- `nodes: np.ndarray`, alias for `centroids`
- `neighbors: list[np.ndarray]`
- `polygons: list[np.ndarray]`
- `edges: np.ndarray`

### `crop_square(graph, L, center=None, boundary_thickness=1.0) -> CroppedGraph`

Returns the induced subgraph inside a square frame. If `center` is omitted, the
center is derived from the graph bounding box.

```python
from hat_amp.graph import crop_square
from hat_amp.percolation import BoundarySets

cropped = crop_square(graph, L=100.0, boundary_thickness=1.0)
boundaries = BoundarySets.from_cropped_graph(cropped)
```

`CroppedGraph` fields:

- `L_value: float`
- `nodes: np.ndarray`
- `neighbors: list[np.ndarray]`
- `edges: np.ndarray`
- `top_boundary_nodes: np.ndarray`
- `bottom_boundary_nodes: np.ndarray`
- `left_boundary_nodes: np.ndarray`
- `right_boundary_nodes: np.ndarray`
- `center: np.ndarray`
- `node_count: int`
- `edge_count: int`

### `neighbors_to_csr(neighbors) -> tuple[np.ndarray, np.ndarray]`

Converts adjacency lists to flat neighbor and start-offset arrays for fast
percolation iteration.

## Percolation

Import from `hat_amp.percolation`:

```python
from hat_amp.percolation import (
    BondPercolation,
    BoundarySets,
    Criterion,
    ExtrapolationResult,
    FitResult,
    SitePercolation,
    WeightedQuickUnionUF,
    extrapolate_pc,
    run_bond_trials,
    run_site_trials,
)
```

`Criterion.INTERSECTION` means both top-bottom and left-right channels must
connect. `Criterion.UNION` means either channel may connect.

### Site Trials

```python
from hat_amp.graph import build_vertex_graph, crop_square
from hat_amp.percolation import BoundarySets, Criterion, run_site_trials
from hat_amp.tiling import generate_patch_tiling

graph = build_vertex_graph(generate_patch_tiling(level=3))
cropped = crop_square(graph, L=100.0)
boundaries = BoundarySets.from_cropped_graph(cropped)

thresholds = run_site_trials(
    cropped,
    boundaries,
    trials=200,
    seed=123,
    criterion=Criterion.INTERSECTION,
)
```

`run_site_trials(...)` returns a `float64` array of threshold fractions, one per
trial.

### Bond Trials

```python
from hat_amp.percolation import Criterion, run_bond_trials

thresholds = run_bond_trials(
    cropped.nodes,
    cropped.edges,
    boundaries,
    trials=200,
    seed=123,
    criterion=Criterion.UNION,
)
```

### Finite-Size Extrapolation

```python
from hat_amp.percolation import extrapolate_pc

fit = extrapolate_pc(
    L_values=[50.0, 75.0, 100.0],
    trials_intersection=raw_I,
    trials_union=raw_U,
    nu=4.0 / 3.0,
    confidence=0.95,
)

print(fit.intersection.pc)
print(fit.union.pc_ci)
print(fit.average.amplitude)
```

`ExtrapolationResult` contains:

- `intersection: FitResult`
- `union: FitResult`
- `average: FitResult`

`FitResult` contains:

- `pc: float`
- `amplitude: float`
- `pc_std: float`
- `amplitude_std: float`
- `pc_ci: tuple[float, float]`

## Results

Import from `hat_amp.results`:

```python
from hat_amp.results import PercolationResults
```

`PercolationResults` is a pydantic v2 model that stores metadata plus raw trial
arrays and saves them to `.npz`.

Fields:

- `tiling_type: str`
- `seed: int | None`
- `trials: int`
- `L_values: np.ndarray`
- `raw_SI: list[np.ndarray]`
- `raw_SU: list[np.ndarray]`
- `raw_BI: list[np.ndarray] | None`
- `raw_BU: list[np.ndarray] | None`
- `extra_meta: dict[str, Any]`
- `timestamp: str`

Properties and methods:

- `has_bond -> bool`
- `n_L -> int`
- `means_stds() -> tuple[np.ndarray, ...]`
- `save(path) -> Path`
- `PercolationResults.load(path) -> PercolationResults`

Example:

```python
import numpy as np

from hat_amp.results import PercolationResults

res = PercolationResults(
    tiling_type="hat-vertex",
    seed=123,
    trials=200,
    L_values=np.array([50.0, 75.0, 100.0]),
    raw_SI=raw_I,
    raw_SU=raw_U,
    extra_meta={"level": 3},
)

res.save("hat_vertex_results.npz")
loaded = PercolationResults.load("hat_vertex_results.npz")
```

## Visualization

Import from `hat_amp.viz`:

```python
from hat_amp.viz import (
    render_patch_svg,
    render_single_tile_svg,
    render_svg,
    save_png,
    save_svg,
)
```

### `render_svg(polygons, ...) -> str`

Renders a list of polygon arrays to an SVG document string.

```python
from hat_amp.tiling import generate_tiling
from hat_amp.viz import render_svg, save_svg

svg = render_svg(generate_tiling(level=1), fill="none", stroke="#222222")
save_svg(svg, "hat_level_1.svg")
```

Options include:

- `stroke: str`
- `fill: str`
- `stroke_width: float`
- `width: int`
- `height: int`
- `viewbox: tuple[float, float, float, float] | None`
- `padding: float`
- `title: str | None`

### `render_single_tile_svg(label="H") -> str`

Renders one hat outline as SVG.

### `render_patch_svg(level, source="hat") -> str`

Convenience renderer for generated patches.

Supported sources:

- `"hat"`: `generate_tiling(level)`
- `"hat-patch"`: `generate_patch_tiling(level)`
- `"penrose"`: `generate_penrose_tiling(divisions=level).polygons()`

### `save_png(svg_or_polygons, path, dpi=96) -> Path`

Writes PNG output via optional CairoSVG. The first argument can be an SVG string
or a list of polygon arrays.

```python
from hat_amp.viz import render_patch_svg, save_png

svg = render_patch_svg(2, source="penrose")
save_png(svg, "penrose_level_2.png")
```

## End-to-End Example

```python
from hat_amp.graph import build_vertex_graph, crop_square
from hat_amp.percolation import BoundarySets, Criterion, run_site_trials
from hat_amp.tiling import generate_patch_tiling
from hat_amp.viz import render_svg, save_svg

polygons = generate_patch_tiling(level=3)
graph = build_vertex_graph(polygons)
cropped = crop_square(graph, L=100.0)
boundaries = BoundarySets.from_cropped_graph(cropped)

thresholds = run_site_trials(
    cropped,
    boundaries,
    trials=100,
    seed=42,
    criterion=Criterion.INTERSECTION,
)

svg = render_svg(polygons)
save_svg(svg, "hat_patch_level_3.svg")
```

## Validation

The test suite checks:

- hat `H` metatile generation against the upstream third-party reference
- full hat patch generation against the upstream reference
- Penrose subdivision coordinates against the upstream reference
- vertex and dual graph counts against upstream graph builders
- WLS percolation extrapolation against the upstream implementation
- SVG well-formedness and PNG creation

Reference tests download source files from:

```text
https://raw.githubusercontent.com/aaryashBharadwaj/Aperiodic-Monotile-Percolation/Version_3_Final/
```

They are skipped by default. Enable them with:

```bash
uv run pytest --run-reference
```

or:

```bash
HAT_AMP_RUN_REFERENCE=1 uv run pytest
```
