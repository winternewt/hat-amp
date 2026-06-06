# hat-amp

Python library for aperiodic monotile tilings and percolation experiments.
Supports Hat (Tile 1,0) and Spectre (Tile 1,1) families.

## Commands

```bash
uv sync                          # install all deps (run after clone or pyproject change)
uv run pytest                    # full test suite
uv run pytest tests/test_spectre.py -v   # just Spectre tests
uv run ruff check src/ tests/    # lint
uv run ruff format src/ tests/   # format
```

No separate build step — `hat-amp` is installed editable via `uv sync`.

## Project layout

```
src/hat_amp/
  geometry.py   — 2D affine transform primitives (pt, rotation, translation, match_two, …)
  tiling.py     — Hat monotile: HatTile / MetaTile hierarchy, 4-metatile (H/T/P/F) substitution
  spectre.py    — Spectre monotile: SpectreTile / SpectreNode, 9-type substitution (Delta…Gamma)
  penrose.py    — Penrose Robinson-triangle tiling
  graph.py      — VertexGraph / DualGraph / CroppedGraph builders (KDTree vertex merging)
  percolation.py — Site/bond percolation, union-find, finite-size extrapolation
  viz.py        — SVG/PNG rendering; render_patch_svg(level, source=…)
  results.py    — PercolationResults Pydantic model, .npz persistence
```

## Spectre specifics

- `generate_spectre_tiling(level)` → `list[ndarray(14,2)]`
- `generate_spectre_tiling_labeled(level)` → `(polygons, labels)` where labels ∈ `{'S', 'M'}`
  - `'S'` = standalone right-handed Spectre
  - `'M'` = component of a Mystic compound (two Spectres fused at 30°, plays "anti-spectre" role)
- Polygons default to **14 vertices** — vertex index 10 is the Singh-Flicker **gold vertex**
  (collinear: edges 9→10 and 10→11 are both leftward), which makes the vertex graph bipartite.
- `strip_gold_vertex(polygons)` → 13-vertex polygons for natural (non-bipartite) graph analysis
- `add_gold_vertex(polygons)` → restores 14-vertex form
- 9-tile substitution system: Gamma (Mystic) + Delta/Theta/Lambda/Xi/Pi/Sigma/Phi/Psi
- Tile count recursion (polygon count from expanding one S or M supertile):
  S_{n+1} = 7·S_n + M_n,  M_{n+1} = 6·S_n + M_n,  S_0=1, M_0=2

## Percolation workflow

```python
from hat_amp.spectre import generate_spectre_tiling, strip_gold_vertex
from hat_amp.graph import build_vertex_graph, crop_square
from hat_amp.percolation import BoundarySets, run_site_trials, Criterion

polygons = generate_spectre_tiling(level=3)          # natural graph
# polygons = strip_gold_vertex(polygons)             # 13-vertex non-bipartite
graph    = build_vertex_graph(polygons)
cropped  = crop_square(graph, L=60)
bounds   = BoundarySets.from_cropped_graph(cropped)
thresholds = run_site_trials(cropped, bounds, trials=200, criterion=Criterion.INTERSECTION)
```

## Visualization

```python
from hat_amp.viz import render_patch_svg, save_svg
svg = render_patch_svg(2, source="spectre")   # S tiles blue, M tiles red
save_svg(svg, "spectre_patch.svg")
# other sources: "hat", "hat-patch", "penrose"
```

## Reference tests

Third-party reference comparisons are skipped by default:

```bash
uv run pytest --run-reference    # or HAT_AMP_RUN_REFERENCE=1 uv run pytest
```

## Key references

- Hat: Smith-Myers-Kaplan-Goodman-Strauss arXiv:2303.10798; Kaplan hatviz JS
- Spectre: Smith-Myers-Kaplan-Goodman-Strauss arXiv:2305.17743; Kaplan https://cs.uwaterloo.ca/~csk/spectre/
- Exact dimer / gold vertex: Singh & Flicker PRB 109 L220303 (arXiv:2309.14447)
- Percolation template: Bhola-Biswas-Islam-Damle PRX 12 021058; Bhola-Damle arXiv:2311.05634
