"""
connectors/worldcover.py — ESA WorldCover 2021 (10 m) land cover from AWS Open Data.

Used for land-cover composition around an asset. It replaces NDBI for built-up
detection: NDBI cannot separate bare soil from concrete (arid Pilbara scored
85 % "built-up"), whereas WorldCover is a trained 11-class product with ~77 %
global overall accuracy (ESA WorldCover v200 validation report).

Source: s3://esa-worldcover/v200/2021/map/  (public, no key)
Tiles  : 3° × 3°, named by lower-left corner, e.g. N21E090
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

from .sentinel2_pixels import _read_band, _disk_mask, _downsample, _flat, asset_bbox, _grid_shape

log = logging.getLogger(__name__)

_URL = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
        "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")

CLASSES = {
    10: "tree_cover", 20: "shrubland", 30: "grassland", 40: "cropland",
    50: "built_up", 60: "bare_sparse", 70: "snow_ice", 80: "permanent_water",
    90: "herbaceous_wetland", 95: "mangroves", 100: "moss_lichen",
}


def _tile_name(lat: float, lon: float) -> str:
    la = int(math.floor(lat / 3.0) * 3)
    lo = int(math.floor(lon / 3.0) * 3)
    return f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}"


_GRID_CACHE: dict[tuple, tuple[np.ndarray, list]] = {}


def lc_grid(lat: float, lon: float, radius_km: float = 2.5) -> tuple[np.ndarray, list]:
    """Raw 10 m WorldCover class grid on the shared EPSG:4326 window (cached)."""
    key = (round(lat, 4), round(lon, 4), radius_km)
    if key in _GRID_CACHE:
        return _GRID_CACHE[key]
    bbox = asset_bbox(lat, lon, radius_km)
    shape = _grid_shape(lat, radius_km, 10.0)
    w, s, e, n = bbox
    tiles = {_tile_name(y, x) for y in (s, n) for x in (w, e)}
    lc = np.full(shape, np.nan, dtype=np.float32)
    errors = []
    for t in sorted(tiles):
        try:
            part = _read_band(_URL.format(tile=t), bbox, shape, "nearest")[0]
            lc = np.where(np.isnan(lc), part, lc)
        except Exception as exc:  # ocean tiles do not exist
            errors.append(f"{t}: {exc}")
    if len(_GRID_CACHE) > 64:
        _GRID_CACHE.clear()
    _GRID_CACHE[key] = (lc, errors)
    return lc, errors


def get_land_cover(lat: float, lon: float, radius_km: float = 2.5,
                   layer_px: int = 128, include_layer: bool = True) -> dict:
    """
    Land-cover fractions within the analysis window and within 250 m / 1 km of
    the asset, plus the class at the asset pixel.
    """
    shape = _grid_shape(lat, radius_km, 10.0)
    lc, errors = lc_grid(lat, lon, radius_km)

    valid = np.isfinite(lc)
    if not valid.any():
        return {"available": False, "errors": errors, "source": "ESA WorldCover 2021 v200"}

    ctr = (shape[0] // 2, shape[1] // 2)

    def fractions(mask):
        v = (valid & mask).sum()
        if not v:
            return {}
        vals, counts = np.unique(lc[valid & mask], return_counts=True)
        return {CLASSES.get(int(k), str(int(k))): round(float(c / v), 4)
                for k, c in sorted(zip(vals, counts), key=lambda kc: -kc[1])}

    at = lc[ctr]
    out = {
        "available": True,
        "source": "ESA WorldCover 2021 v200 · 10 m · AWS Open Data",
        "class_at_asset": CLASSES.get(int(at), None) if np.isfinite(at) else None,
        "area": fractions(np.ones(shape, bool)),
        "within_250m": fractions(_disk_mask(shape, ctr, 25)),
        "within_1km": fractions(_disk_mask(shape, ctr, 100)),
        "errors": errors,
    }
    out["built_up_within_250m"] = out["within_250m"].get("built_up", 0.0)
    out["tree_cover_within_1km"] = out["within_1km"].get("tree_cover", 0.0)
    out["flammable_veg_within_1km"] = round(sum(
        out["within_1km"].get(k, 0.0) for k in ("tree_cover", "shrubland", "grassland")), 4)
    out["wetland_within_1km"] = round(sum(
        out["within_1km"].get(k, 0.0) for k in ("herbaceous_wetland", "mangroves")), 4)

    if include_layer:
        # Nearest-neighbour downsample keeps class codes intact
        H, W = shape
        ri = np.linspace(0, H - 1, layer_px).round().astype(int)
        ci = np.linspace(0, W - 1, layer_px).round().astype(int)
        out["layer"] = _flat(lc[np.ix_(ri, ci)], 0)
        out["layer_shape"] = (layer_px, layer_px)
    return out
