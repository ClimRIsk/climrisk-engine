"""
Open-Topo-Data connector — SRTM 30m elevation grid.

Fetches elevation data from https://api.opentopodata.org (free, no key)
using the SRTM30m dataset (NASA Shuttle Radar Topography Mission, ~30m
horizontal resolution, global coverage 60°S–60°N).

Usage
-----
    from cri.connectors.open_topo_data import get_elevation_grid, ElevationGrid

    grid = get_elevation_grid(lat=25.7741, lon=-80.1842, radius_km=3.0, spacing_m=200)
    # grid.elevations[row][col] — 2D array, grid.cell_size_m, grid.asset_row/col
    print(f"Asset elevation: {grid.asset_elevation_m:.1f} m")
    print(f"Grid shape: {grid.rows} × {grid.cols}")

API limits
----------
- 100 locations per call, no API key required
- Rate limit: ~1 request per second
- Response time: ~1–3 s per 100-point batch
- Data source: SRTM v3 (NASA / USGS), void-filled
- Horizontal accuracy: ±20 m
- Vertical accuracy: ±16 m (90th percentile, open terrain)

Caching
-------
Elevation grids are cached in-memory for 24 hours (terrain doesn't change).
Cache key: (lat, lon, radius_km, spacing_m) rounded to 4 decimal places.
"""

from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional
import urllib.request
import urllib.parse
import json

log = logging.getLogger(__name__)

_TOPO_API = "https://api.opentopodata.org/v1/srtm30m"
_CALL_INTERVAL = 1.1          # seconds between API calls (respect rate limit)
_TIMEOUT       = 20           # seconds per HTTP call
_BATCH_SIZE    = 100          # max locations per API call
_NODATA        = -32768       # SRTM nodata sentinel

_last_call_time: float = 0.0

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ElevationGrid:
    """
    2D grid of SRTM elevations centred on an asset location.

    Coordinate system
    -----------------
    Row 0 = northernmost strip (highest latitude)
    Col 0 = westernmost strip (lowest longitude)
    Asset location = (asset_row, asset_col) — the centre cell.
    """
    lat:       float           # asset latitude
    lon:       float           # asset longitude
    radius_km: float           # radius of coverage
    cell_size_m: float         # horizontal spacing between cells (metres)
    rows:      int             # total rows
    cols:      int             # total cols
    asset_row: int             # row index of asset centre
    asset_col: int             # col index of asset centre

    # 2D elevation array (metres ASL), NaN where SRTM has no data
    elevations: list[list[float]] = field(default_factory=list)

    @property
    def asset_elevation_m(self) -> float:
        return self.elevations[self.asset_row][self.asset_col]

    def elevation_at(self, row: int, col: int) -> float:
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.elevations[row][col]
        return float("nan")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _metres_per_degree_lat(lat: float) -> float:
    """Approximate metres per degree of latitude (nearly constant)."""
    return 111_132.0 - 559.8 * math.cos(2 * math.radians(lat)) + 1.175 * math.cos(4 * math.radians(lat))


def _metres_per_degree_lon(lat: float) -> float:
    """Approximate metres per degree of longitude at given latitude."""
    return 111_412.0 * math.cos(math.radians(lat)) - 93.5 * math.cos(3 * math.radians(lat))


def _throttle() -> None:
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < _CALL_INTERVAL:
        time.sleep(_CALL_INTERVAL - elapsed)
    _last_call_time = time.time()


def _batch_elevations(locations: list[tuple[float, float]]) -> list[float]:
    """
    Fetch elevations for up to BATCH_SIZE (lat, lon) pairs.
    Returns list of elevations in same order; NaN for nodata/failure.
    """
    _throttle()
    loc_str = "|".join(f"{lat},{lon}" for lat, lon in locations)
    url = f"{_TOPO_API}?locations={urllib.parse.quote(loc_str, safe=',|.')}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CRI-terrain/1.0"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read())
        results = data.get("results", [])
        out = []
        for r in results:
            el = r.get("elevation")
            if el is None or el == _NODATA:
                out.append(float("nan"))
            else:
                out.append(float(el))
        return out
    except Exception as exc:
        log.warning("Open-Topo-Data batch failed: %s", exc)
        return [float("nan")] * len(locations)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# In-memory cache: key → ElevationGrid
_GRID_CACHE: dict[tuple, ElevationGrid] = {}

def get_elevation_grid(
    lat:        float,
    lon:        float,
    radius_km:  float = 3.0,
    spacing_m:  float = 200.0,
) -> ElevationGrid:
    """
    Fetch a 2D SRTM elevation grid centred on (lat, lon).

    Parameters
    ----------
    lat        : Asset latitude (decimal degrees)
    lon        : Asset longitude (decimal degrees)
    radius_km  : Half-width of grid in km (default 3 km → 6×6 km coverage)
    spacing_m  : Cell spacing in metres (default 200 m → 31×31 grid for 3km radius)

    Returns
    -------
    ElevationGrid with .elevations, .asset_elevation_m, etc.
    """
    cache_key = (round(lat, 4), round(lon, 4), round(radius_km, 2), round(spacing_m, 0))
    if cache_key in _GRID_CACHE:
        log.debug("Terrain grid cache hit for (%.4f, %.4f)", lat, lon)
        return _GRID_CACHE[cache_key]

    # Compute grid dimensions
    m_per_deg_lat = _metres_per_degree_lat(lat)
    m_per_deg_lon = _metres_per_degree_lon(lat)
    delta_lat = spacing_m / m_per_deg_lat   # degrees per cell (latitude)
    delta_lon = spacing_m / m_per_deg_lon   # degrees per cell (longitude)

    n_cells = int(math.ceil(radius_km * 1000 / spacing_m))  # cells from centre to edge
    half = n_cells

    rows = 2 * half + 1
    cols = 2 * half + 1
    asset_row = half
    asset_col = half

    # Build list of (lat, lon) for every grid cell — row-major
    locations: list[tuple[float, float]] = []
    for r in range(rows):
        for c in range(cols):
            cell_lat = lat + (half - r) * delta_lat  # row 0 = northmost
            cell_lon = lon + (c - half) * delta_lon  # col 0 = westmost
            locations.append((cell_lat, cell_lon))

    log.info("Fetching SRTM grid %d×%d (%d pts) at (%.4f, %.4f) r=%.1fkm s=%.0fm",
             rows, cols, len(locations), lat, lon, radius_km, spacing_m)

    # Batch into ≤100-point chunks
    all_elevs: list[float] = []
    for i in range(0, len(locations), _BATCH_SIZE):
        batch = locations[i : i + _BATCH_SIZE]
        all_elevs.extend(_batch_elevations(batch))
        if i + _BATCH_SIZE < len(locations):
            time.sleep(0.1)  # brief pause between batches

    # Reshape into 2D
    elevations: list[list[float]] = []
    idx = 0
    for r in range(rows):
        row_data = []
        for c in range(cols):
            row_data.append(all_elevs[idx])
            idx += 1
        elevations.append(row_data)

    grid = ElevationGrid(
        lat=lat, lon=lon,
        radius_km=radius_km,
        cell_size_m=spacing_m,
        rows=rows, cols=cols,
        asset_row=asset_row, asset_col=asset_col,
        elevations=elevations,
    )
    _GRID_CACHE[cache_key] = grid
    return grid


def get_elevation(lat: float, lon: float) -> Optional[float]:
    """Single-point elevation lookup. Returns metres ASL or None on failure."""
    result = _batch_elevations([(lat, lon)])
    v = result[0]
    return None if math.isnan(v) else v
