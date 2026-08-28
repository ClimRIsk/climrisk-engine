"""
Slope, aspect, TPI, and D8 flow-direction analysis from a DEM grid.

All functions operate on 2D lists of floats (elevations in metres).
NaN cells are treated as nodata and are skipped in neighbourhood
calculations where possible.

References
----------
- Horn (1981): slope/aspect via 3×3 finite-difference kernel
- Weiss (2001): Topographic Position Index (TPI)
- O'Callaghan & Mark (1984): D8 flow direction algorithm
- Tarboton (1997): upstream area accumulation
"""

from __future__ import annotations

import math
from typing import Optional

# ---------------------------------------------------------------------------
# Slope and aspect
# ---------------------------------------------------------------------------

def slope_degrees(dem: list[list[float]], cell_size_m: float) -> list[list[float]]:
    """
    Compute slope angle (degrees) at every cell using Horn's 3×3 kernel.

    Boundary cells get NaN.  NaN elevations in the neighbourhood are
    replaced with the centre cell's elevation (flat extrapolation).
    """
    rows, cols = len(dem), len(dem[0])
    out = [[float("nan")] * cols for _ in range(rows)]
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            z = dem[r][c]
            if math.isnan(z):
                continue
            def _e(rr, cc):
                v = dem[rr][cc]
                return z if math.isnan(v) else v
            # Horn kernel
            dz_dx = ((_e(r-1,c+1) + 2*_e(r,c+1) + _e(r+1,c+1))
                   - (_e(r-1,c-1) + 2*_e(r,c-1) + _e(r+1,c-1))) / (8 * cell_size_m)
            dz_dy = ((_e(r+1,c-1) + 2*_e(r+1,c) + _e(r+1,c+1))
                   - (_e(r-1,c-1) + 2*_e(r-1,c) + _e(r-1,c+1))) / (8 * cell_size_m)
            out[r][c] = math.degrees(math.atan(math.sqrt(dz_dx**2 + dz_dy**2)))
    return out


def aspect_degrees(dem: list[list[float]], cell_size_m: float) -> list[list[float]]:
    """
    Compute aspect (degrees from north, 0–360, clockwise) using Horn's kernel.
    Flat areas return NaN.
    """
    rows, cols = len(dem), len(dem[0])
    out = [[float("nan")] * cols for _ in range(rows)]
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            z = dem[r][c]
            if math.isnan(z):
                continue
            def _e(rr, cc):
                v = dem[rr][cc]
                return z if math.isnan(v) else v
            dz_dx = ((_e(r-1,c+1) + 2*_e(r,c+1) + _e(r+1,c+1))
                   - (_e(r-1,c-1) + 2*_e(r,c-1) + _e(r+1,c-1))) / (8 * cell_size_m)
            dz_dy = ((_e(r+1,c-1) + 2*_e(r+1,c) + _e(r+1,c+1))
                   - (_e(r-1,c-1) + 2*_e(r-1,c) + _e(r-1,c+1))) / (8 * cell_size_m)
            if dz_dx == 0 and dz_dy == 0:
                out[r][c] = float("nan")
            else:
                a = 180 - math.degrees(math.atan2(dz_dy, -dz_dx)) + 90
                out[r][c] = a % 360
    return out


# ---------------------------------------------------------------------------
# Topographic Position Index (TPI)
# ---------------------------------------------------------------------------

def tpi(dem: list[list[float]], window_cells: int = 5) -> list[list[float]]:
    """
    Topographic Position Index = elevation − mean of neighbourhood.

    Positive TPI → hill/ridge (asset above surroundings)
    Negative TPI → valley/depression (asset below surroundings)
    Near zero   → mid slope or flat terrain

    window_cells : radius of neighbourhood (cells) — default 5 → 11×11 window
    """
    rows, cols = len(dem), len(dem[0])
    out = [[float("nan")] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            z = dem[r][c]
            if math.isnan(z):
                continue
            vals = []
            for dr in range(-window_cells, window_cells + 1):
                for dc in range(-window_cells, window_cells + 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < rows and 0 <= cc < cols and not math.isnan(dem[rr][cc]):
                        vals.append(dem[rr][cc])
            if vals:
                out[r][c] = z - (sum(vals) / len(vals))
    return out


def terrain_position_label(tpi_val: float, slope_deg: float) -> str:
    """
    Classify terrain position from TPI and slope (Weiss 2001 scheme).

    Returns one of: 'ridge', 'upper slope', 'mid slope', 'footslope', 'valley'
    """
    if math.isnan(tpi_val) or math.isnan(slope_deg):
        return "unknown"
    if tpi_val > 10 and slope_deg < 6:
        return "plateau/ridge"
    if tpi_val > 10:
        return "ridge"
    if tpi_val > 3:
        return "upper slope"
    if tpi_val < -10 and slope_deg < 6:
        return "valley bottom"
    if tpi_val < -10:
        return "valley"
    if tpi_val < -3:
        return "footslope"
    if slope_deg < 6:
        return "flat"
    return "mid slope"


# ---------------------------------------------------------------------------
# D8 flow direction and upstream accumulation
# ---------------------------------------------------------------------------

# D8 direction offsets (row_delta, col_delta) for 8 neighbours
# Ordered: E, SE, S, SW, W, NW, N, NE
_D8_OFFSETS = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]

def flow_direction_d8(dem: list[list[float]]) -> list[list[int]]:
    """
    Compute D8 flow direction for each cell.

    Returns a 2D grid of direction indices (0–7, from _D8_OFFSETS),
    or -1 for sinks/boundaries/nodata.
    """
    rows, cols = len(dem), len(dem[0])
    out = [[-1] * cols for _ in range(rows)]
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            z = dem[r][c]
            if math.isnan(z):
                continue
            best_dir, best_drop = -1, 0.0
            for d, (dr, dc) in enumerate(_D8_OFFSETS):
                nz = dem[r + dr][c + dc]
                if math.isnan(nz):
                    continue
                drop = z - nz
                if drop > best_drop:
                    best_drop = drop
                    best_dir = d
            out[r][c] = best_dir
    return out


def upstream_area_cells(
    flow_dir: list[list[int]],
    target_row: int,
    target_col: int,
) -> set[tuple[int, int]]:
    """
    BFS/DFS upstream: find all cells that drain into (target_row, target_col)
    following D8 flow direction.

    Returns set of (row, col) pairs — includes the target cell itself.
    """
    rows, cols = len(flow_dir), len(flow_dir[0])

    # Reverse lookup: which cells flow *into* (r, c)?
    def flows_into(r: int, c: int) -> list[tuple[int, int]]:
        upstreams = []
        for d, (dr, dc) in enumerate(_D8_OFFSETS):
            pr, pc = r - dr, c - dc  # potential upstream neighbour
            if 0 <= pr < rows and 0 <= pc < cols:
                if flow_dir[pr][pc] == d:
                    upstreams.append((pr, pc))
        return upstreams

    visited: set[tuple[int, int]] = set()
    stack = [(target_row, target_col)]
    while stack:
        cell = stack.pop()
        if cell in visited:
            continue
        visited.add(cell)
        stack.extend(flows_into(*cell))
    return visited


def upstream_area_km2(
    flow_dir:    list[list[int]],
    target_row:  int,
    target_col:  int,
    cell_size_m: float,
) -> float:
    """Area (km²) of the drainage basin upstream of the target cell."""
    cells = upstream_area_cells(flow_dir, target_row, target_col)
    area_m2 = len(cells) * (cell_size_m ** 2)
    return area_m2 / 1_000_000.0


# ---------------------------------------------------------------------------
# Upslope statistics (for debris flow source characterisation)
# ---------------------------------------------------------------------------

def upslope_slope_stats(
    slope_grid:  list[list[float]],
    upstream_cells: set[tuple[int, int]],
    exclude_cell: tuple[int, int],
) -> dict:
    """
    Mean, max, and >25° fraction of slope angles in the upstream catchment.

    Excludes the asset cell itself.  Returns dict with keys:
    mean_deg, max_deg, pct_over_25, pct_over_35, n_cells
    """
    vals = []
    for (r, c) in upstream_cells:
        if (r, c) == exclude_cell:
            continue
        v = slope_grid[r][c]
        if not math.isnan(v):
            vals.append(v)
    if not vals:
        return {"mean_deg": float("nan"), "max_deg": float("nan"),
                "pct_over_25": 0.0, "pct_over_35": 0.0, "n_cells": 0}
    return {
        "mean_deg":    sum(vals) / len(vals),
        "max_deg":     max(vals),
        "pct_over_25": 100 * sum(1 for v in vals if v > 25) / len(vals),
        "pct_over_35": 100 * sum(1 for v in vals if v > 35) / len(vals),
        "n_cells":     len(vals),
    }


def upslope_elevation_range(
    dem: list[list[float]],
    upstream_cells: set[tuple[int, int]],
    asset_elev_m: float,
) -> float:
    """
    Maximum elevation difference between any upstream cell and the asset
    (metres).  Represents the maximum potential energy head for debris flows.
    """
    max_upstream = asset_elev_m
    for (r, c) in upstream_cells:
        v = dem[r][c]
        if not math.isnan(v) and v > max_upstream:
            max_upstream = v
    return max_upstream - asset_elev_m
