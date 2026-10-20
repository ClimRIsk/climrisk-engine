"""Elevation tiles for the 3D viewer's terrain mesh: open LiDAR where it exists, the global tiles everywhere else.

The viewer builds its terrain from Terrarium-encoded 256 x 256 elevation tiles. For tiles inside LiDAR cover (England 1 m, Netherlands 0.5 m, US 3DEP where it is 3 m or finer),
`lidar_tile` returns a tile of the same encoding built from the LiDAR bare-earth grid, so the mesh shows embankments, dikes and floodplain steps the 10-30 m tiles lose. Pixels the LiDAR
does not cover are left to the caller to fill from the global tile (the returned mask says which are LiDAR).
"""
from __future__ import annotations

import io
import math
from typing import Optional, Tuple

import numpy as np


def tile_bounds(z: int, x: int, y: int) -> Tuple[float, float, float, float]:
    """(west, south, east, north) of an XYZ web-mercator tile in degrees."""
    n = 2.0 ** z
    west, east = x / n * 360.0 - 180.0, (x + 1) / n * 360.0 - 180.0
    north = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    south = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return west, south, east, north


def encode_terrarium(h: np.ndarray) -> np.ndarray:
    """Float heights (m) -> uint8 RGB in the Terrarium encoding (v = h + 32768; R = v // 256, G = v % 256, B = fractional * 256)."""
    v = np.clip(np.nan_to_num(h, nan=0.0) + 32768.0, 0, 65535.999)
    r = np.floor(v / 256.0)
    g = np.floor(v - r * 256.0)
    b = np.floor((v - np.floor(v)) * 256.0)
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def decode_terrarium(rgb: np.ndarray) -> np.ndarray:
    a = rgb.astype(float)
    return a[..., 0] * 256.0 + a[..., 1] + a[..., 2] / 256.0 - 32768.0


def lidar_tile(z: int, x: int, y: int, size: int = 256, fetch_grid=None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(heights m (size x size), is_lidar mask) for an XYZ tile, or None when no open LiDAR covers its centre."""
    from . import lidar_dem
    w, s, e, n = tile_bounds(z, x, y)
    lat_c, lon_c = (s + n) / 2.0, (w + e) / 2.0
    if lidar_dem._which(lat_c, lon_c) is None:
        return None
    half_km = max(abs(e - w) * 111.32 * math.cos(math.radians(lat_c)), abs(n - s) * 111.32) / 2.0 * 1.15
    g = (fetch_grid or lidar_dem.try_grid)(lat_c, lon_c, half_km)
    if g is None:
        return None
    px = (np.arange(size) + 0.5) / size
    lons = w + px * (e - w)
    lats = n - px * (n - s)
    LO, LA = np.meshgrid(lons, lats)
    h = lidar_dem.sample_lonlat(g, LO.ravel(), LA.ravel()).reshape(size, size)
    mask = np.isfinite(h)
    if not mask.any():
        return None
    return h, mask


def tile_png(h: np.ndarray) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(encode_terrarium(h), "RGB").save(buf, format="PNG")
    return buf.getvalue()
