"""
terrain/hand.py — Height Above Nearest Drainage (HAND) and area flood features.

HAND = vertical distance from a cell down to the drainage cell it flows into
(D8). It is the standard terrain descriptor for flood susceptibility (Rennó et
al. 2008; Nobre et al. 2011; NOAA National Water Model inundation mapping) and,
unlike raw elevation, it is comparable across regions: 1 m above a channel
means the same thing in Houston and in Dhaka.

Two uses
  hand_metrics(lat, lon)        point metrics at a site (kept for diagnostics)
  area_flood_features(...)      share of BUILT-UP land that sits within x m of
                                drainage inside a region (tract polygon, or a
                                buffer around an asset). A backtest against FEMA
                                NFIP claims showed point-at-centroid features
                                have no skill at tract level, so the area
                                version is what gets validated.

Data: AWS terrain tiles (~10 m, zoom 14; ~19 m at zoom 13 for large areas) and
ESA WorldCover 10 m — both global, no key.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

if not hasattr(np, "in1d"):          # NumPy >= 2.4 removed it; pysheds still calls it
    np.in1d = np.isin               # type: ignore[attr-defined]
from pysheds.view import Raster

from ..connectors.terrain_tiles import get_elevation_grid
from .pysheds_analyzer import (_NODATA, _build_pysheds_grid, _deg_per_m_lat, _deg_per_m_lon)

log = logging.getLogger(__name__)

CHANNEL_KM2 = (0.1, 1.0)          # channel-initiation thresholds (upstream area)
HAND_LEVELS_M = (0.5, 1.0, 2.0, 5.0)
_WC_BUILT = 50                     # ESA WorldCover class: built-up


@dataclass
class HandGrid:
    lat: float
    lon: float
    radius_km: float
    dem: np.ndarray
    valid: np.ndarray
    hand: dict                      # channel_km2 → HAND array (m, NaN where undefined)
    top_lat: float
    left_lon: float
    cell_lat_deg: float
    cell_lon_deg: float
    cell_m: float
    asset_rc: tuple[int, int]

    @property
    def transform(self):
        from affine import Affine
        return Affine(self.cell_lon_deg, 0.0, self.left_lon, 0.0, -self.cell_lat_deg, self.top_lat)


def hand_grid(lat: float, lon: float, radius_km: float = 2.5, zoom: Optional[int] = None,
              channel_km2: tuple[float, ...] = CHANNEL_KM2) -> HandGrid:
    """DEM → conditioned D8 flow → HAND for each channel threshold."""
    if zoom is None:
        zoom = 14 if radius_km <= 4.0 else 13      # keep grids ≲ 1000² cells
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

    hands = {}
    for thr in channel_km2:
        channels = acc * cell_km2 >= thr
        if not channels.any():
            continue
        h = np.asarray(grd.compute_hand(fdir, cond, Raster(channels, viewfinder=fdir.viewfinder)),
                       dtype=np.float64)
        h[(h < 0) | (h > 1e4) | ~valid] = np.nan
        hands[thr] = h
    return HandGrid(lat, lon, radius_km, dem, valid, hands, top_lat, left_lon,
                    cell_lat, cell_lon, sp_m, (ar, ac))


def builtup_on_grid(hg: HandGrid) -> Optional[np.ndarray]:
    """ESA WorldCover built-up mask resampled (nearest) onto the HAND grid."""
    try:
        from ..connectors.worldcover import lc_grid
        lc, _ = lc_grid(hg.lat, hg.lon, hg.radius_km)
    except Exception as exc:        # noqa: BLE001
        log.warning("WorldCover unavailable: %s", exc)
        return None
    H, W = hg.dem.shape
    ri = np.minimum((np.arange(H) * lc.shape[0] / H).astype(int), lc.shape[0] - 1)
    ci = np.minimum((np.arange(W) * lc.shape[1] / W).astype(int), lc.shape[1] - 1)
    return lc[np.ix_(ri, ci)] == _WC_BUILT


def area_flood_features(hg: HandGrid, region: np.ndarray, built: Optional[np.ndarray]) -> dict:
    """
    Features over `region` (bool mask on the HAND grid):
      builtup_km2, region_km2
      share_built_hand_lt_{h}m_at_{thr}km2   share of built-up cells with HAND < h
      median_built_hand_at_{thr}km2
      share_land_hand_lt_1m_at_{thr}km2      all land, for areas with no built-up
    """
    cell_km2 = (hg.cell_m / 1000.0) ** 2
    reg = region & hg.valid
    out: dict = {"region_km2": round(float(reg.sum() * cell_km2), 3),
                 "cell_size_m": round(hg.cell_m, 1)}
    b = (built & reg) if built is not None else None
    out["builtup_km2"] = round(float(b.sum() * cell_km2), 3) if b is not None else None
    for thr, h in hg.hand.items():
        land = h[reg]
        land = land[np.isfinite(land)]
        out[f"share_land_hand_lt_1m_at_{thr}km2"] = round(float((land < 1.0).mean()), 4) if land.size else None
        if b is None or not b.any():
            continue
        hb = h[b]
        hb = hb[np.isfinite(hb)]
        if not hb.size:
            continue
        out[f"median_built_hand_at_{thr}km2"] = round(float(np.median(hb)), 2)
        for lv in HAND_LEVELS_M:
            out[f"share_built_hand_lt_{lv}m_at_{thr}km2"] = round(float((hb < lv).mean()), 4)
    return out


def built_elevation_stats(lat: float, lon: float, radius_km: float = 2.5, within_m: float = 1000.0) -> dict:
    """
    Elevation of BUILT-UP land within `within_m` of a site (no flow routing —
    reuses the cached terrain tiles and WorldCover grid).

    Validated signal (scripts/backtest_flood_area.py, 180 census tracts in
    Harris TX, East Baton Rouge LA, Charleston SC vs 95k FEMA NFIP claims):
    tracts whose built-up land sits below 3 m had ~2.7× the median claim rate
    per built-up km² and 67 % fell in the top-claim fifth (vs 8 % in the bottom).
    Between 3 and 40 m elevation showed little separation. Scope: low-lying
    coastal-plain terrain; absolute elevation does not transfer to inland valleys.
    """
    from ..connectors.worldcover import lc_grid
    eg = get_elevation_grid(lat, lon, radius_km=radius_km, zoom=14)
    dem = eg.elevations.astype(np.float64)
    lc, _ = lc_grid(lat, lon, radius_km)
    H, W = dem.shape
    ri = np.minimum((np.arange(H) * lc.shape[0] / H).astype(int), lc.shape[0] - 1)
    ci = np.minimum((np.arange(W) * lc.shape[1] / W).astype(int), lc.shape[1] - 1)
    built = lc[np.ix_(ri, ci)] == _WC_BUILT
    yy, xx = np.ogrid[:H, :W]
    near = (yy - eg.asset_row) ** 2 + (xx - eg.asset_col) ** 2 <= (within_m / eg.cell_size_m) ** 2
    sel = built & near & np.isfinite(dem)
    if sel.sum() < 20:                                   # < ~0.002 km² built-up: use all land
        sel = near & np.isfinite(dem)
        basis = "all land (little built-up within radius)"
    else:
        basis = "built-up land (ESA WorldCover)"
    e = dem[sel]
    return {"median_elev_m": round(float(np.median(e)), 2),
            "share_below_3m": round(float((e < 3.0).mean()), 3),
            "within_m": within_m, "basis": basis}


def disk_mask(hg: HandGrid, radius_m: float) -> np.ndarray:
    H, W = hg.dem.shape
    ar, ac = hg.asset_rc
    yy, xx = np.ogrid[:H, :W]
    return (yy - ar) ** 2 + (xx - ac) ** 2 <= (radius_m / hg.cell_m) ** 2


def hand_metrics(lat: float, lon: float, radius_km: float = 2.5,
                 channel_km2: tuple[float, ...] = CHANNEL_KM2, zoom: int = 14) -> dict:
    """Point HAND at the site + median within 250 m + relative elevation (diagnostics)."""
    hg = hand_grid(lat, lon, radius_km, zoom, channel_km2)
    ar, ac = hg.asset_rc
    near = disk_mask(hg, 250.0)
    out: dict = {"cell_size_m": round(hg.cell_m, 1), "elevation_m": round(float(hg.dem[ar, ac]), 2)}
    for thr in channel_km2:
        h = hg.hand.get(thr)
        if h is None:
            out[f"hand_m_at_{thr}km2"] = None
            continue
        v = h[ar, ac]
        out[f"hand_m_at_{thr}km2"] = None if np.isnan(v) else round(float(v), 2)
        out[f"hand_median_250m_at_{thr}km2"] = round(float(np.nanmedian(h[near])), 2) \
            if np.isfinite(h[near]).any() else None
    e = hg.dem[hg.valid]
    out["elev_percentile_in_window"] = round(float((e < hg.dem[ar, ac]).mean()), 3)
    out["elev_minus_window_p10_m"] = round(float(hg.dem[ar, ac] - np.percentile(e, 10)), 2)
    return out
