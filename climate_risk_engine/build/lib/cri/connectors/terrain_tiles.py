"""
Terrain Tile connector — AWS Mapzen/Amazon terrain tiles at ~10 m resolution.

URL scheme
----------
https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png

Terrarium RGB encoding
----------------------
    elevation_m = (R * 256 + G + B / 256) - 32768

At zoom level 14 the tile resolution is ~9.5 m/pixel at the equator, scaling
as cos(lat) in the east-west direction — roughly 9–11 m depending on latitude.
This is ~100× finer than point-sampling Open-Topo-Data at 100 m spacing and
gives sufficient cell density for proper D8 watershed delineation.

Public API, no authentication required.

Usage
-----
    from cri.connectors.terrain_tiles import get_elevation_grid, ElevationGrid

    grid = get_elevation_grid(lat=17.6868, lon=83.2185, radius_km=2.5)
    # grid.elevations  → 2-D numpy array (rows × cols), metres
    # grid.cell_size_m → actual pixel size in metres (lat direction)
    # grid.asset_row, grid.asset_col → index of the asset within the grid
"""

from __future__ import annotations

import io
import logging
import math
import time
from dataclasses import dataclass
from typing import Optional
from urllib.request import urlopen, Request

import numpy as np

try:
    from PIL import Image as PILImage
except ImportError as exc:  # pragma: no cover
    raise ImportError("Pillow is required: pip install Pillow") from exc

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TILE_URL   = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
_TILE_SIZE  = 256          # pixels per tile side
_ZOOM       = 14           # default zoom — ~9.5 m/px at equator
_TIMEOUT    = 15           # seconds per HTTP request
_RETRY      = 2            # retries on transient failure

# In-memory tile cache keyed by (z, x, y) → numpy uint8 (256, 256, 3)
_TILE_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


# ---------------------------------------------------------------------------
# Data class (same interface as open_topo_data.ElevationGrid)
# ---------------------------------------------------------------------------

@dataclass
class ElevationGrid:
    lat:        float          # asset latitude
    lon:        float          # asset longitude
    radius_km:  float
    cell_size_m: float         # pixel size in metres (lat direction)
    rows:       int
    cols:       int
    asset_row:  int
    asset_col:  int
    elevations: np.ndarray     # shape (rows, cols), dtype float64, metres


# ---------------------------------------------------------------------------
# Tile coordinate math (Web Mercator / Slippy Map)
# ---------------------------------------------------------------------------

def _lat_lon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    """Return (tile_x, tile_y) for the tile containing (lat, lon)."""
    n = 2 ** zoom
    tx = int((lon + 180.0) / 360.0 * n)
    ty_f = (1.0 - math.log(math.tan(math.radians(lat))
                           + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0
    ty = int(ty_f * n)
    # clamp
    tx = max(0, min(n - 1, tx))
    ty = max(0, min(n - 1, ty))
    return tx, ty


def _tile_top_left(tx: int, ty: int, zoom: int) -> tuple[float, float]:
    """Return (lat, lon) of the TOP-LEFT corner of tile (tx, ty)."""
    n = 2 ** zoom
    lon = tx / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * ty / n)))
    lat = math.degrees(lat_rad)
    return lat, lon


def _tile_pixel_size_m(lat: float, zoom: int) -> tuple[float, float]:
    """
    Return (pixel_size_lat_m, pixel_size_lon_m) at the given latitude.

    Both are approximately equal (tiles are square in Web Mercator pixels),
    but east-west ground distance shrinks with cos(lat).
    """
    n_tiles   = 2 ** zoom
    circ_m    = 2 * math.pi * 6_378_137.0      # Earth equatorial circumference
    lon_m_per_pixel = circ_m / (n_tiles * _TILE_SIZE)  # at equator
    lat_m_per_pixel = lon_m_per_pixel                  # square in Mercator
    lon_m_per_pixel_at_lat = lon_m_per_pixel * math.cos(math.radians(lat))
    return lat_m_per_pixel, lon_m_per_pixel_at_lat


# ---------------------------------------------------------------------------
# HTTP fetch with retry and in-memory cache
# ---------------------------------------------------------------------------

def _fetch_tile(z: int, x: int, y: int) -> Optional[np.ndarray]:
    """
    Fetch terrarium tile (z, x, y) and return uint8 RGB array (256, 256, 3).
    Returns None on permanent failure.
    """
    key = (z, x, y)
    if key in _TILE_CACHE:
        return _TILE_CACHE[key]

    url = _TILE_URL.format(z=z, x=x, y=y)
    headers = {"User-Agent": "CRI-TerrainEngine/1.0 (climate-risk-engine)"}

    for attempt in range(_RETRY + 1):
        try:
            req  = Request(url, headers=headers)
            resp = urlopen(req, timeout=_TIMEOUT)
            data = resp.read()
            img  = PILImage.open(io.BytesIO(data)).convert("RGB")
            arr  = np.asarray(img, dtype=np.uint8)      # (256, 256, 3)
            _TILE_CACHE[key] = arr
            log.debug("Fetched tile z=%d x=%d y=%d", z, x, y)
            return arr
        except Exception as exc:
            if attempt < _RETRY:
                time.sleep(1.0)
                log.warning("Tile z=%d x=%d y=%d attempt %d failed: %s",
                            z, x, y, attempt + 1, exc)
            else:
                log.error("Tile z=%d x=%d y=%d permanently failed: %s", z, x, y, exc)
                return None


def _decode_terrarium(rgb: np.ndarray) -> np.ndarray:
    """
    Convert terrarium uint8 RGB tile → float64 elevation array (metres).
    Formula: elevation = R * 256 + G + B / 256 - 32768
    """
    R = rgb[:, :, 0].astype(np.float64)
    G = rgb[:, :, 1].astype(np.float64)
    B = rgb[:, :, 2].astype(np.float64)
    return R * 256.0 + G + B / 256.0 - 32_768.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_elevation_grid(
    lat:       float,
    lon:       float,
    radius_km: float = 2.5,
    zoom:      int   = _ZOOM,
) -> ElevationGrid:
    """
    Download AWS terrain tiles around (lat, lon) and return an ElevationGrid.

    Parameters
    ----------
    lat, lon    : asset coordinates (decimal degrees)
    radius_km   : half-width of the bounding box to fetch (km)
    zoom        : tile zoom level; 14 → ~9.5 m/px, 13 → ~19 m/px

    Returns
    -------
    ElevationGrid with `elevations` as a float64 numpy array.
    """
    # ------------------------------------------------------------------ #
    # 1. Compute bounding box in degrees
    # ------------------------------------------------------------------ #
    deg_per_km_lat = 1.0 / 111.32                  # ~0.008983°/km
    deg_per_km_lon = 1.0 / (111.32 * math.cos(math.radians(lat)))

    radius_deg_lat = radius_km * deg_per_km_lat
    radius_deg_lon = radius_km * deg_per_km_lon

    north = lat + radius_deg_lat
    south = lat - radius_deg_lat
    west  = lon - radius_deg_lon
    east  = lon + radius_deg_lon

    # ------------------------------------------------------------------ #
    # 2. Determine tile range
    # ------------------------------------------------------------------ #
    tx_min, ty_min = _lat_lon_to_tile(north, west, zoom)   # top-left tile
    tx_max, ty_max = _lat_lon_to_tile(south, east, zoom)   # bottom-right tile

    n_tx = tx_max - tx_min + 1
    n_ty = ty_max - ty_min + 1
    log.info("Fetching %d×%d tiles (zoom=%d) for (%g, %g) r=%.1f km",
             n_tx, n_ty, zoom, lat, lon, radius_km)

    # ------------------------------------------------------------------ #
    # 3. Download and stitch tiles into one big elevation mosaic
    # ------------------------------------------------------------------ #
    mosaic_h = n_ty * _TILE_SIZE
    mosaic_w = n_tx * _TILE_SIZE
    mosaic   = np.full((mosaic_h, mosaic_w), np.nan, dtype=np.float64)

    for j, ty in enumerate(range(ty_min, ty_max + 1)):
        for i, tx in enumerate(range(tx_min, tx_max + 1)):
            rgb = _fetch_tile(zoom, tx, ty)
            if rgb is None:
                continue
            elev = _decode_terrarium(rgb)
            row0 = j * _TILE_SIZE
            col0 = i * _TILE_SIZE
            mosaic[row0:row0 + _TILE_SIZE, col0:col0 + _TILE_SIZE] = elev

    # ------------------------------------------------------------------ #
    # 4. Compute exact pixel extents of the mosaic
    # ------------------------------------------------------------------ #
    mosaic_north, mosaic_west = _tile_top_left(tx_min, ty_min, zoom)
    mosaic_south, mosaic_east = _tile_top_left(tx_max + 1, ty_max + 1, zoom)

    lat_per_px = (mosaic_north - mosaic_south) / mosaic_h
    lon_per_px = (mosaic_east  - mosaic_west)  / mosaic_w

    # ------------------------------------------------------------------ #
    # 5. Crop to bounding box
    # ------------------------------------------------------------------ #
    row_north = int((mosaic_north - north) / lat_per_px)
    row_south = int((mosaic_north - south) / lat_per_px)
    col_west  = int((west - mosaic_west)   / lon_per_px)
    col_east  = int((east - mosaic_west)   / lon_per_px)

    row_north = max(0, min(mosaic_h - 1, row_north))
    row_south = max(0, min(mosaic_h - 1, row_south))
    col_west  = max(0, min(mosaic_w - 1, col_west))
    col_east  = max(0, min(mosaic_w - 1, col_east))

    if row_north >= row_south or col_west >= col_east:
        # Fallback: return full mosaic
        row_north, row_south = 0, mosaic_h
        col_west,  col_east  = 0, mosaic_w

    cropped = mosaic[row_north:row_south, col_west:col_east]
    rows, cols = cropped.shape

    # ------------------------------------------------------------------ #
    # 6. Locate asset pixel
    # ------------------------------------------------------------------ #
    # Top-left corner of cropped grid
    crop_north = mosaic_north - row_north * lat_per_px
    crop_west  = mosaic_west  + col_west  * lon_per_px

    asset_row = int((crop_north - lat) / lat_per_px)
    asset_col = int((lon - crop_west)  / lon_per_px)
    asset_row = max(0, min(rows - 1, asset_row))
    asset_col = max(0, min(cols - 1, asset_col))

    # ------------------------------------------------------------------ #
    # 7. Pixel size in metres (for slope / area calculations)
    # ------------------------------------------------------------------ #
    cell_size_lat_m, _ = _tile_pixel_size_m(lat, zoom)

    log.info("Grid %d×%d px, cell=%.1f m, asset=[%d,%d], elev=%.0f m",
             rows, cols, cell_size_lat_m,
             asset_row, asset_col, float(cropped[asset_row, asset_col]))

    return ElevationGrid(
        lat        = lat,
        lon        = lon,
        radius_km  = radius_km,
        cell_size_m= cell_size_lat_m,
        rows       = rows,
        cols       = cols,
        asset_row  = asset_row,
        asset_col  = asset_col,
        elevations = cropped,
    )
