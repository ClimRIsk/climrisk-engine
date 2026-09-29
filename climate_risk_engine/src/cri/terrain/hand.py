"""
terrain/hand.py — Height Above Nearest Drainage (HAND) and relative elevation.

HAND = vertical distance from a cell down to the drainage cell it flows into
(D8). It is the standard terrain descriptor for flood susceptibility (Rennó et
al. 2008; Nobre et al. 2011; NOAA National Water Model inundation mapping) and,
unlike raw elevation, it is comparable across regions: 1 m above a channel
means the same thing in Houston and in Dhaka.

Why here: a backtest against 75k FEMA NFIP claims (Harris County, TX; see
scripts/backtest_flood_nfip.py) showed the original hand-weighted flood score
had no skill (AUC 0.44), while elevation-type signals did.

Data: AWS terrain tiles (~10 m) via connectors.terrain_tiles — global, no key.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from pysheds.view import Raster

from ..connectors.terrain_tiles import get_elevation_grid
from .pysheds_analyzer import (_NODATA, _build_pysheds_grid, _deg_per_m_lat, _deg_per_m_lon)

log = logging.getLogger(__name__)


def hand_metrics(lat: float, lon: float, radius_km: float = 2.5,
                 channel_km2: tuple[float, ...] = (0.1, 1.0),
                 zoom: int = 14) -> dict:
    """
    HAND at the asset for each channel-initiation threshold (km² of upstream
    area), the median HAND within 250 m, and the asset's elevation percentile
    within the window.
    """
    eg = get_elevation_grid(lat, lon, radius_km=radius_km, zoom=zoom)
    dem = eg.elevations.astype(np.float64)
    ar, ac, sp_m = eg.asset_row, eg.asset_col, eg.cell_size_m
    valid = ~np.isnan(dem)

    cell_lat, cell_lon = sp_m * _deg_per_m_lat(lat), sp_m * _deg_per_m_lon(lat)
    top_lat, left_lon = lat + ar * cell_lat, lon - ac * cell_lon
    grd, raster = _build_pysheds_grid(np.where(valid, dem, _NODATA), top_lat, left_lon,
                                      cell_lat, cell_lon, _NODATA)
    cond = grd.resolve_flats(grd.fill_depressions(grd.fill_pits(raster)))
    fdir = grd.flowdir(cond)
    acc = np.asarray(grd.accumulation(fdir), dtype=np.float64)
    cell_km2 = (sp_m / 1000.0) ** 2

    yy, xx = np.ogrid[:dem.shape[0], :dem.shape[1]]
    near = (yy - ar) ** 2 + (xx - ac) ** 2 <= (250.0 / sp_m) ** 2

    out: dict = {"cell_size_m": round(sp_m, 1), "elevation_m": round(float(dem[ar, ac]), 2)}
    for thr in channel_km2:
        channels = acc * cell_km2 >= thr
        if not channels.any():
            out[f"hand_m_at_{thr}km2"] = None
            continue
        mask = Raster(channels, viewfinder=fdir.viewfinder)
        hand = np.asarray(grd.compute_hand(fdir, cond, mask), dtype=np.float64)
        hand[(hand < 0) | (hand > 1e4) | ~valid] = np.nan
        v = hand[ar, ac]
        out[f"hand_m_at_{thr}km2"] = None if np.isnan(v) else round(float(v), 2)
        out[f"hand_median_250m_at_{thr}km2"] = round(float(np.nanmedian(hand[near])), 2) \
            if np.isfinite(hand[near]).any() else None

    e = dem[valid]
    out["elev_percentile_in_window"] = round(float((e < dem[ar, ac]).mean()), 3)
    out["elev_minus_window_p10_m"] = round(float(dem[ar, ac] - np.percentile(e, 10)), 2)
    return out
