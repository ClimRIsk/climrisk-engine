"""Floodplain position from open LiDAR bare earth: height above nearest drainage (HAND) at the pin.

Scope: England only (Environment Agency 1 m DTM). That is where the method was tested, on 12 flooded towns scored against the Environment Agency's
Recorded Flood Outlines (`scripts/lidar_flood_benchmark.py`): ROC area median 0.96 for HAND from LiDAR against 0.76 from the global terrain tiles; LiDAR
better in all 12 towns. In that test (sample balanced 50/50 inside and outside recorded outlines, so shares below are likelihood-style, not probabilities):
  HAND < 1 m: 86-95 % of points lay inside a recorded outline    1-5 m: 79-88 %    5-10 m: 42 %    10 m or more: 3 %
It is a SUSCEPTIBILITY (the site sits on ground the river can reach), not a flood depth or a return period, and it ignores defences built after the
LiDAR survey, culverts and drainage. The Netherlands LiDAR is used for elevation only: HAND means little in pumped, flat polders.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)
BANDS = ((1.0, "HIGH"), (5.0, "MODERATE"), (10.0, "LOW"))       # HAND below 1 m -> HIGH, below 5 m -> MODERATE, below 10 m -> LOW, otherwise no floor


def hand_and_rel(dem: np.ndarray, res: float, channel_km2: float = 1.0):
    """(height above the lowest ground within 1 km, HAND for channels draining at least `channel_km2`) for a DEM in metres."""
    from scipy.ndimage import minimum_filter
    from pysheds.view import Raster
    if not hasattr(np, "in1d"):
        np.in1d = np.isin                                          # type: ignore[attr-defined]
    from .pysheds_analyzer import _NODATA, _build_pysheds_grid
    valid = np.isfinite(dem)
    k = max(3, int(1000 / res) | 1)
    rel = dem - minimum_filter(np.where(valid, dem, np.inf), size=k, mode="nearest")
    grd, raster = _build_pysheds_grid(np.where(valid, dem, _NODATA), 0.0, 0.0, res, res, _NODATA)
    cond = grd.resolve_flats(grd.fill_depressions(grd.fill_pits(raster)))
    fdir = grd.flowdir(cond)
    acc = np.asarray(grd.accumulation(fdir), float)
    ch = acc * (res / 1000.0) ** 2 >= channel_km2
    hand = np.full(dem.shape, np.nan)
    if ch.any():
        h = np.asarray(grd.compute_hand(fdir, cond, Raster(ch, viewfinder=fdir.viewfinder)), float)
        h[(h < 0) | (h > 1e4) | ~valid] = np.nan
        hand = h
    return rel, hand


def band_for(hand_m: Optional[float]) -> Optional[str]:
    if hand_m is None or not np.isfinite(hand_m):
        return None
    for lim, name in BANDS:
        if hand_m < lim:
            return name
    return None


def flood_position(lat: float, lon: float, radius_km: float = 2.5) -> dict:
    """{available, source, hand_m, rel_m, floor} - available False when LiDAR does not cover the point or anything fails."""
    try:
        from ..connectors.terrain_tiles import get_elevation_grid
        g = get_elevation_grid(lat, lon, radius_km=radius_km)
        if "Environment Agency" not in g.source:
            return {"available": False, "reason": "no open LiDAR for this location (England only); global terrain is too coarse for this screen"}
        rel, hand = hand_and_rel(g.elevations, g.cell_size_m)
        r, c = g.asset_row, g.asset_col
        h = hand[r, c]
        if not np.isfinite(h):                                     # pin on a channel cell or masked ground: nearest valid cell within 25 m
            k = max(1, int(25 / g.cell_size_m))
            win = hand[max(0, r - k): r + k + 1, max(0, c - k): c + k + 1]
            h = np.nanmin(win) if np.isfinite(win).any() else np.nan
        rl = rel[r, c]
        h = None if not np.isfinite(h) else round(float(h), 2)
        return {"available": h is not None, "source": g.source, "cell_m": round(float(g.cell_size_m), 1), "hand_m": h,
                "rel_m": None if not np.isfinite(rl) else round(float(rl), 2), "floor": band_for(h),
                **({} if h is not None else {"reason": "no drainage network could be derived at the pin"})}
    except Exception as e:                                         # noqa: BLE001
        log.warning("LiDAR floodplain screen failed at (%g, %g): %s", lat, lon, str(e)[:120])
        return {"available": False, "reason": f"LiDAR floodplain screen could not run ({type(e).__name__})"}
