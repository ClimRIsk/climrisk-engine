"""Surface-water (rain-on-grid) flood depth at a site: local-inertial shallow-water solver on the best open terrain, driven by a design storm.

Pipeline: best terrain (open LiDAR where it exists, else global tiles) -> block-averaged to at most `cell_cap` cells across -> design daily rainfall for the
return period from ERA5 annual maxima (Gumbel) -> storm of `storm_hours` carrying a stated share of the daily total -> SCS curve-number runoff (built-up share
from ESA WorldCover raises the curve number) -> a drainage allowance removed from built-up ground -> the same solver the model lab uses (`modellab.flood.simulate`,
LISFLOOD-FP local-inertial scheme, mass-conserving, verified in `modellab.flood.run_benchmarks`) with free outflow at the edges.

What it is: a screening estimate of ponding and overland flow depth from rain alone. What it is not: a calibrated urban drainage model. Sewers, culverts,
pumps and buildings are not represented except through the drainage allowance; ERA5 daily rain smooths convective bursts, so depths are a floor in storm-prone
climates; the storm shape and duration share are assumptions. Compared with the Environment Agency's recorded surface-water flood outlines in
`scripts/pluvial_benchmark.py` / `docs/PLUVIAL_BENCHMARK.md`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Callable, Optional

import numpy as np

log = logging.getLogger(__name__)
CACHE = Path.home() / ".cache" / "climrisk" / "pluvial"
DRAINAGE_MM_H = 12.0          # rain removed per hour from fully built-up ground (the allowance used in national UK surface-water mapping for urban sewers)
DURATION_SHARE = {1.0: 0.35, 3.0: 0.50, 6.0: 0.65}      # share of the daily total that falls in the design storm (an assumption, stated in every result)


def _block_mean(a, f):
    if f <= 1:
        return a
    r, c = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    return np.nanmean(a[:r, :c].reshape(r // f, f, c // f, f), axis=(1, 3))


def _builtup_share(lat, lon, radius_km):
    try:
        from ..connectors.worldcover import lc_grid
        g, _ = lc_grid(lat, lon, radius_km)
        g = np.asarray(g)
        return float((g == 50).mean())
    except Exception:                                           # noqa: BLE001
        return None


def rating_for(depth_m: float) -> str:
    return "NEGLIGIBLE" if depth_m < 0.05 else "LOW" if depth_m < 0.15 else "MODERATE" if depth_m < 0.30 else "HIGH" if depth_m < 0.60 else "CRITICAL"


def site_pluvial(lat: float, lon: float, rp: int = 100, radius_km: float = 0.6, storm_hours: float = 1.0, cell_cap: int = 240, manning: float = 0.05,
                 dem=None, daily_rain_mm: Optional[float] = None, builtup: Optional[float] = None, keep_grids: bool = False,
                 rain_fetch: Optional[Callable] = None, drainage_mm_h: Optional[float] = None, open_boundary: bool = True,
                 storm_share: Optional[float] = None, sim_hours_after: float = 2.0) -> dict:
    """{available, depth_m (max within 15 m of the pin), rating, wet_hours, daily_rp_mm, storm_mm, runoff_mm, cell_m, terrain_source, ...}"""
    from ..modellab import flood, hydrology
    from scipy import ndimage as ndi
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return {"available": False, "reason": "non-finite coordinates"}
    try:
        if dem is None:
            from ..connectors.terrain_tiles import get_elevation_grid
            dem = get_elevation_grid(lat, lon, radius_km=radius_km)
        z0 = np.asarray(dem.elevations, float)
        if np.isnan(z0[max(0, dem.asset_row - 2): dem.asset_row + 3, max(0, dem.asset_col - 2): dem.asset_col + 3]).all():
            return {"available": False, "reason": "no terrain at the pin"}
        f = max(1, int(math.ceil(max(z0.shape) / cell_cap)))
        z = _block_mean(np.where(np.isnan(z0), np.nanmedian(z0), z0), f)
        dx = float(dem.cell_size_m) * f
        lidar = "LiDAR" in str(getattr(dem, "source", "")) or "3DEP" in str(getattr(dem, "source", ""))
        if not lidar:
            z = ndi.median_filter(z, size=3, mode="nearest")        # single-cell pits and spikes in the global product would become fake puddles
        ai, aj = min(z.shape[0] - 1, dem.asset_row // f), min(z.shape[1] - 1, dem.asset_col // f)
        if builtup is None:
            builtup = _builtup_share(lat, lon, radius_km)
        b = 0.0 if builtup is None else float(min(1.0, builtup))
        cn = 75.0 if builtup is None else 75.0                    # pervious ground: moderate soils, mixed cover (no soil product is read)
        if daily_rain_mm is None:
            from ..modellab import data as D
            key = CACHE / (hashlib.md5(f"{lat:.2f}{lon:.2f}".encode()).hexdigest() + ".json")
            am = None
            if rain_fetch is None and key.exists():
                am = json.loads(key.read_text())
            if am is None:
                am = (rain_fetch or D.annual_max_daily_rain)(lat, lon)
                if rain_fetch is None:
                    try:
                        CACHE.mkdir(parents=True, exist_ok=True)
                        key.write_text(json.dumps({"maxima_mm": am["maxima_mm"], "years": am["years"], "source": am["source"]}))
                    except Exception:                               # noqa: BLE001
                        pass
            mu, beta = hydrology.gumbel_fit(am["maxima_mm"])
            daily_rain_mm = float(hydrology.gumbel_quantile(mu, beta, rp))
            rain_src = am["source"]
        else:
            rain_src = "supplied"
        share = storm_share if storm_share is not None else DURATION_SHARE.get(float(storm_hours), 0.5)
        drain = DRAINAGE_MM_H if drainage_mm_h is None else float(drainage_mm_h)
        storm_mm = daily_rain_mm * share
        step_s = 300.0
        hy = hydrology.block_hyetograph(storm_mm, storm_hours, step_s)
        # Two surfaces mixed by the built-up share: pervious ground sheds SCS-CN excess; built-up ground is impervious and sheds all rain
        # beyond what the sewers take (DRAINAGE_MM_H). The mix is applied uniformly over the window (no spatial routing of sewer capacity).
        eff_pervious = hydrology.effective_rain_series(hy, cn)
        eff_built = np.maximum(hy - drain * (step_s / 3600.0), 0.0)
        eff = (1.0 - b) * eff_pervious + b * eff_built
        duration = (storm_hours + sim_hours_after) * 3600
        res = flood.simulate(z, dx, duration, rain_m_s=(np.arange(len(eff)) * step_s, eff / 1000.0 / step_s), manning_n=manning, open_boundary=open_boundary,
                             probe=(ai, aj), record_every_s=600.0)
        k = max(1, int(round(15.0 / dx)))
        win = res.h_max[max(0, ai - k): ai + k + 1, max(0, aj - k): aj + k + 1]
        depth = float(win.max())
        out = {"available": True, "depth_m": round(depth, 3), "rating": rating_for(depth), "wet_hours": round(float(res.t_wet[ai, aj] / 3600.0), 2),
               "return_period": rp, "daily_rp_mm": round(daily_rain_mm, 1), "storm_hours": storm_hours, "storm_share_of_daily": share, "storm_mm": round(storm_mm, 1),
               "runoff_mm": round(float(eff.sum()), 1), "curve_number_pervious": round(cn, 1), "builtup_share": None if builtup is None else round(builtup, 3),
               "drainage_allowance_mm_h": drain, "open_boundary": open_boundary, "cell_m": round(dx, 1), "terrain_source": getattr(dem, "source", "unknown"), "rain_source": rain_src,
               "mass_error": float(res.mass_error), "max_courant": float(res.max_courant)}
        if keep_grids:
            out["_h_max"], out["_z"], out["_asset"] = res.h_max, z, (ai, aj)
        return out
    except Exception as e:                                                          # noqa: BLE001
        log.warning("pluvial model failed at (%g, %g): %s", lat, lon, str(e)[:160])
        return {"available": False, "reason": f"surface-water model could not run ({type(e).__name__}: {str(e)[:100]})"}


# ── catchment-scale surface-water screen (11 Nov 2026) ────────────────────────────────────────────────────────────────────────────────────────────
# Why a second entry point: `site_pluvial` above runs on a 600 m window with a 1-3 h burst and a 12 mm/h sewer allowance. That cannot see a flat city whose water
# arrives from kilometres upslope and stays for a day (Chennai, 1-2 December 2015: 0.07 m modelled at Velachery, where residents stood in 1 to 2 m). This one runs
# the same solver on a ~12 km window, with a 24 h storm sized from the largest of ERA5, NASA POWER and the nearest rain gauge (`rain_design`), and reports a RANGE:
# the drains-working case (12 mm/h removed from built-up ground, the national UK mapping allowance) and the drains-overwhelmed case (none removed; design storms
# are by definition larger than the sewer standard, and in Chennai the outfalls were themselves under water). The rating uses the overwhelmed case.
SW_RADIUS_KM = 6.0
SW_CELL_CAP = 200
SW_STORM_H = 24.0


def _sw_key(lat: float, lon: float, rp: int) -> str:
    return hashlib.md5(f"sw1|{lat:.3f}|{lon:.3f}|{rp}".encode()).hexdigest()


def site_surface_water(lat: float, lon: float, rp: int = 100, radius_km: float = SW_RADIUS_KM, cell_cap: int = SW_CELL_CAP, storm_hours: float = SW_STORM_H,
                       bounds: bool = True, keep_grids: bool = False, rain: Optional[dict] = None, dem=None, use_cache: bool = True) -> dict:
    """{available, depth_m (drains overwhelmed, the rating basis), depth_working_m (drains working; None when bounds=False), rating, rain{...}, cell_m, ...}.

    `rain` may be a `rain_design.design()` result (tests, or a caller that already holds one). With keep_grids the 2-D arrays `_z`, `_h_max` (and `_asset`, `_dx`) are returned.
    The disk cache holds the summary and the arrays so a repeat call, or the 3D twin, does not rerun the solver."""
    kp = CACHE / (_sw_key(lat, lon, rp) + ".npz")
    if use_cache and rain is None and dem is None and kp.exists():
        try:
            z_ = np.load(kp, allow_pickle=False)
            meta = json.loads(str(z_["meta"]))
            if not (bounds and meta.get("depth_working_m") is None):
                if keep_grids:
                    meta["_z"], meta["_h_max"], meta["_asset"], meta["_dx"] = z_["z"], z_["h"], tuple(int(v) for v in z_["asset"]), float(meta["cell_m"])
                return meta
        except Exception:                                                   # noqa: BLE001
            pass
    if rain is None:
        try:
            from . import rain_design as RD
            rain = RD.design(lat, lon, rp=rp)
        except Exception as e:                                              # noqa: BLE001
            rain = {"available": False, "reason": f"{type(e).__name__}: {str(e)[:100]}"}
    if not rain.get("available"):
        return {"available": False, "reason": "design rainfall unavailable: " + str(rain.get("reason", "unknown"))[:200]}
    d1 = float(rain["d1_mm"])
    base = site_pluvial(lat, lon, rp=rp, radius_km=radius_km, cell_cap=cell_cap, storm_hours=storm_hours, storm_share=1.0, daily_rain_mm=d1, drainage_mm_h=0.0,
                        open_boundary=True, keep_grids=True, dem=dem, sim_hours_after=2.0)
    if not base.get("available"):
        return base
    low = None
    if bounds:
        low = site_pluvial(lat, lon, rp=rp, radius_km=radius_km, cell_cap=cell_cap, storm_hours=storm_hours, storm_share=1.0, daily_rain_mm=d1, drainage_mm_h=DRAINAGE_MM_H,
                           open_boundary=True, dem=dem, sim_hours_after=2.0)
    by = rain.get("by_source", {})
    g = by.get("gauge")
    grid_best = max([s["d1_mm"] for n, s in by.items() if n != "gauge" and "d1_mm" in s] or [0.0])
    rain_out = {"d1_mm": round(d1, 1), "d3_mm": rain.get("d3_mm"), "source_used": rain.get("source_used", {}).get("1d"), "gauge_km": None if not g else g.get("km"),
                "gauge_d1_mm": None if not g else g.get("d1_mm"), "grid_best_d1_mm": round(grid_best, 1) if grid_best else None,
                "largest_in_record": rain.get("largest_in_record", {}).get("1d"), "by_source": {n: {k: v for k, v in s.items() if k in ("source", "d1_mm", "max1_mm", "max1_date", "km")} for n, s in by.items()},
                "failed_sources": rain.get("failed_sources") or {}}
    out = {k: v for k, v in base.items() if not k.startswith("_")}
    out.update({"depth_m": base["depth_m"], "rating": base["rating"], "depth_working_m": None if low is None or not low.get("available") else low["depth_m"],
                "rating_working": None if low is None or not low.get("available") else low["rating"], "rain": rain_out, "radius_km": radius_km, "storm_hours": storm_hours,
                "basis": "drains overwhelmed (none removed); the drains-working case is depth_working_m"})
    if use_cache and dem is None:
        try:
            CACHE.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(kp, meta=json.dumps(out), z=base["_z"].astype(np.float32), h=base["_h_max"].astype(np.float32), asset=np.array(base["_asset"]))
        except Exception:                                                   # noqa: BLE001
            pass
    if keep_grids:
        out["_z"], out["_h_max"], out["_asset"], out["_dx"] = base["_z"], base["_h_max"], base["_asset"], float(base["cell_m"])
    return out
