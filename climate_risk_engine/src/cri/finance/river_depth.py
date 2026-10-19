"""River flood depth at a site for small and mid rivers: HAND + a synthetic rating curve (the approach behind national inundation-mapping systems).

  1. Terrain: best open DEM (1 m LiDAR where it exists), height above nearest drainage (HAND) for streams draining >= 1 km2 (`terrain.lidar_flood.hand_and_rel`).
  2. Flow: GloFAS river discharge (Open-Meteo flood API, 1991-2023, daily, 5 km) at the pin -> annual maxima -> Gumbel -> return-level flow for the requested
     return period and the 2-year flow as the bankfull flow.
  3. Stage: Manning normal depth in a wide rectangular channel, width from a regime relation W = 3.7 Q^0.5, channel slope fitted to the streambed near the pin,
     n = 0.04. Depth at the pin = stage above the nearest drainage minus HAND (zero when the pin is above the stage).

What it is: an estimate of how far up the valley side a 1-in-N flow reaches, scaled to the river's own size. What it is not: a hydraulic model of the reach.
No defences, culverts, bridges, floodplain storage or flow attenuation; GloFAS reanalysis under-reads big peaks and is 5 km, so the flow may belong to a bigger
or smaller river than the one at the pin (the distance to the GloFAS river cell is returned and the model declines beyond 6 km); n and the regime width are
generic. Benchmarked against the Environment Agency Recorded Flood Outlines in `scripts/river_depth_benchmark.py` / `docs/RIVER_DEPTH_BENCHMARK.md`.
"""
from __future__ import annotations

import logging
import math
from typing import Callable, Optional

import numpy as np

log = logging.getLogger(__name__)
MANNING_N = 0.04
WIDTH_COEF = 3.7                 # W = 3.7 Q^0.5 (m, m3/s): regime-theory order of magnitude for lowland UK/European alluvial channels


def manning_depth(q: float, slope: float, n: float = MANNING_N) -> float:
    """Normal depth (m) in a wide rectangular channel of regime width carrying q (m3/s) on bed slope `slope`."""
    if q <= 0 or slope <= 0:
        return 0.0
    w = WIDTH_COEF * math.sqrt(q)
    return float((q * n / (w * math.sqrt(slope))) ** 0.6)


def flow_quantiles(maxima, rps=(2, 100)) -> dict:
    from ..modellab import hydrology
    mu, beta = hydrology.gumbel_fit(maxima)
    return {rp: float(max(0.0, hydrology.gumbel_quantile(mu, beta, rp))) for rp in rps}


def channel_slope(z: np.ndarray, acc_km2: np.ndarray, r: int, c: int, res: float, channel_km2: float = 1.0, reach_m: float = 700.0) -> Optional[float]:
    """Bed slope of the stream nearest the pin: elevation vs along-valley position of channel cells within `reach_m` of the nearest channel cell."""
    ch = acc_km2 >= channel_km2
    if not ch.any():
        return None
    ys, xs = np.nonzero(ch)
    d = np.hypot(ys - r, xs - c)
    i = int(np.argmin(d))
    if d[i] * res > 1500:
        return None
    # follow the biggest stream: among channel cells near that one, keep the cells whose accumulation is within a factor of 3 of the local maximum
    near = np.hypot(ys - ys[i], xs - xs[i]) * res <= reach_m
    ys, xs = ys[near], xs[near]
    a = acc_km2[ys, xs]
    keep = a >= a.max() / 3.0
    ys, xs = ys[keep], xs[keep]
    if len(ys) < 5:
        return None
    pts = np.c_[xs * res, ys * res].astype(float)
    pts -= pts.mean(0)
    u, s, vt = np.linalg.svd(pts, full_matrices=False)
    t = pts @ vt[0]
    if np.ptp(t) < 100:
        return None
    zz = z[ys, xs]
    slope = abs(float(np.polyfit(t, zz, 1)[0]))
    return float(min(max(slope, 1e-4), 0.05))


def rating_for(depth_m: float) -> str:
    return "NEGLIGIBLE" if depth_m < 0.05 else "LOW" if depth_m < 0.15 else "MODERATE" if depth_m < 0.30 else "HIGH" if depth_m < 0.60 else "CRITICAL"


def site_river_depth(lat: float, lon: float, rp: int = 100, radius_km: float = 2.5, dem=None, maxima=None, snap_km: Optional[float] = None,
                     fetch: Optional[Callable] = None, keep_grids: bool = False) -> dict:
    """{available, depth_m, rating, hand_m, stage_m, q_rp_m3s, q_bankfull_m3s, slope, ...}; available False with a reason when anything is missing."""
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return {"available": False, "reason": "non-finite coordinates"}
    try:
        from ..terrain.lidar_flood import hand_and_rel
        if dem is None:
            from ..connectors.terrain_tiles import get_elevation_grid
            dem = get_elevation_grid(lat, lon, radius_km=radius_km)
        z = np.asarray(dem.elevations, float)
        res = float(dem.cell_size_m)
        r, c = dem.asset_row, dem.asset_col
        if maxima is None:
            from ..modellab import data as D
            j = (D.river_discharge(lat, lon, 1991, 2023, **({"fetch": fetch} if fetch else {})))
            t = j["daily"]["time"]
            q = j["daily"]["river_discharge"]
            by_year: dict = {}
            for ti, qi in zip(t, q):
                if qi is not None:
                    by_year[ti[:4]] = max(by_year.get(ti[:4], 0.0), float(qi))
            maxima = [v for v in by_year.values()]
            if len(maxima) < 15:
                return {"available": False, "reason": "river-flow record too short"}
            snap_km = math.hypot((j["latitude"] - lat) * 111.32, (j["longitude"] - lon) * 111.32 * math.cos(math.radians(lat)))
        if snap_km is not None and snap_km > 6.0:
            return {"available": False, "reason": f"nearest modelled river cell is {snap_km:.0f} km from the pin"}
        qs = flow_quantiles(maxima, (2, rp))
        rel, hand = hand_and_rel(z, res)
        acc_dummy = None
        # accumulation for slope: recompute channel mask cheaply from HAND==0 cells (channel cells have HAND 0)
        chan = np.isfinite(hand) & (hand <= 0.05)
        slope = channel_slope(z, np.where(chan, 1.0, 0.0), r, c, res, channel_km2=0.5)
        if slope is None:
            return {"available": False, "reason": "no stream channel within 1.5 km of the pin"}
        h = hand[r, c]
        if not np.isfinite(h):
            k = max(1, int(25 / res))
            win = hand[max(0, r - k): r + k + 1, max(0, c - k): c + k + 1]
            h = np.nanmin(win) if np.isfinite(win).any() else np.nan
        if not np.isfinite(h):
            return {"available": False, "reason": "no drainage network could be derived at the pin"}
        stage = manning_depth(qs[rp], slope)
        bank = manning_depth(qs[2], slope)
        depth = max(0.0, stage - float(h))
        out = {"available": True, "depth_m": round(depth, 2), "rating": rating_for(depth), "return_period": rp, "hand_m": round(float(h), 2), "stage_m": round(stage, 2),
               "bankfull_depth_m": round(bank, 2), "q_rp_m3s": round(qs[rp], 1), "q_bankfull_m3s": round(qs[2], 1), "slope": round(slope, 5),
               "glofas_snap_km": None if snap_km is None else round(snap_km, 1), "terrain_source": getattr(dem, "source", "unknown"), "cell_m": round(res, 1)}
        if keep_grids:
            out["_hand"], out["_stage"] = hand, stage
        return out
    except Exception as e:                                                  # noqa: BLE001
        log.warning("river depth model failed at (%g, %g): %s", lat, lon, str(e)[:160])
        return {"available": False, "reason": f"river depth model could not run ({type(e).__name__}: {str(e)[:100]})"}
