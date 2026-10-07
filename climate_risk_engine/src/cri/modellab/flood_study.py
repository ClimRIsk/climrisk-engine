"""
Flood study: the engine plans, builds, verifies, stress-tests and (if observations exist) validates a 2-D inundation model.

The order mirrors a thesis chapter: problem → physics choice → data → build → verification → sensitivity → comparison with
observations → limits. Every decision is written to the lab notebook with its reason.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

from . import calib, flood, hydrology, metrics

NOW = time.time


def _block_mean(a: np.ndarray, f: int) -> np.ndarray:
    if f <= 1:
        return a
    r, c = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    return a[:r, :c].reshape(r // f, f, c // f, f).mean(axis=(1, 3))


def plan(radius_km: float, cell_m: float, max_cells: int = 90) -> dict:
    """Choose grid, solver and storm from the scale of the problem, and say why."""
    extent_m = 2 * radius_km * 1000
    f = max(1, int(np.ceil(extent_m / (cell_m * max_cells))))
    dx = cell_m * f
    return {"factor": f, "dx": dx, "n": int(extent_m // dx), "extent_m": extent_m,
            "why_grid": (f"The study area is {extent_m / 1000:.1f} km across and the terrain data resolve {cell_m:.0f} m. A flood model must be re-run many times "
                         f"(verification, sensitivity), so it is solved on a {dx:.0f} m grid ({int(extent_m // dx)} × {int(extent_m // dx)} cells)"
                         + (f", averaged from {f}×{f} source cells." if f > 1 else ", the native resolution."))}


MODEL_CHOICE = [
    ("Local-inertial shallow-water (chosen)", "Keeps the pressure-gradient and friction terms of the shallow-water equations and drops only convective acceleration. "
     "It is the standard for urban and floodplain inundation (LISFLOOD-FP) because it stays stable on coarse grids and conserves mass, and for gradually-varying flood waves it matches the full equations closely."),
    ("Full shallow-water equations (rejected)", "Needs sub-10 m terrain and a time step set by the fastest wave; the data here cannot justify that cost, and the extra physics only matters for dam-breaks and hydraulic jumps."),
    ("Height-above-nearest-drainage / bathtub (rejected as the main model)", "Static: it fills terrain to a level without time, friction or rainfall volume, so it overstates flooding in flat land that water cannot reach in a single storm. Kept as the engine's fast screen."),
]


def run(lat: float, lon: float, name: str = "Study site", radius_km: float = 1.5, return_period: int = 100, storm_hours: float = 6.0, cn: float = 80.0,
        manning: float = 0.05, dem=None, rain_fetch: Optional[Callable] = None, observed: Optional[dict] = None, reference_depth: Optional[Callable] = None,
        sensitivity: bool = True, progress: Optional[Callable] = None) -> dict:
    nb: list = []
    t_start = NOW()

    def note(step, title, detail, **kw):
        nb.append({"step": step, "title": title, "detail": detail, "t": round(NOW() - t_start, 1), **kw})
        if progress:
            try:
                progress(f"{step}: {title}")
            except Exception:                                         # noqa: BLE001
                pass

    assumptions = [f"Soil and land-use runoff is represented by one SCS curve number, CN = {cn:g}.",
                   f"Surface roughness is one Manning n = {manning:g} s m^(-1/3) over the whole area.",
                   "Drainage infrastructure (sewers, pumps, culverts) is not represented; all rain not lost to infiltration runs over the surface.",
                   "Rain falls uniformly over the area; the area is small enough for that to hold.",
                   "Edges of the study area drain freely (normal-depth outflow); no backwater from rivers or sea is imposed."]
    limits = ["Terrain is a global 30 m-class elevation product; buildings, embankments and culverts narrower than a cell are invisible, so local depths can be wrong by more than the headline number suggests.",
              "ERA5 reanalysis smooths rainfall extremes: the design storm is a floor, not a ceiling."]

    # 1 ── data
    if dem is None:
        from ..connectors.terrain_tiles import get_elevation_grid
        dem = get_elevation_grid(lat, lon, radius_km=radius_km)
    z0 = np.asarray(dem.elevations, float)
    P = plan(radius_km, dem.cell_size_m)
    z = _block_mean(z0, P["factor"])
    z_raw = z.copy()
    from scipy import ndimage as _ndi
    z = _ndi.median_filter(z, size=3, mode="nearest")           # global elevation products carry single-cell pits and spikes (vegetation, buildings, noise)
    cond_change = float(np.abs(z - z_raw).mean())
    ny, nx = z.shape
    ai, aj = min(ny - 1, dem.asset_row // P["factor"]), min(nx - 1, dem.asset_col // P["factor"])
    note("1", "Problem and scale", f"Where does water go at {name} ({lat:.4f}, {lon:.4f}) in a {return_period}-year rainstorm, how deep and for how long? " + P["why_grid"])
    note("1b", "Terrain conditioning", f"Global elevation products contain isolated single-cell pits and spikes (canopy, buildings, noise). Left alone, each pit becomes a fake puddle that collects rain and has no outlet. "
                                      f"A 3×3 median filter was applied (mean change {cond_change:.2f} m), and flooded area counts only connected patches of at least 4 cells.")
    note("2", "Choice of physics", " ".join(f"{a}: {b}" for a, b in MODEL_CHOICE))

    # 2 ── rainfall
    rain = {"ok": False}
    try:
        am = (rain_fetch or hydrology_fetch)(lat, lon)
        mu, beta = hydrology.gumbel_fit(am["maxima_mm"])
        d_T = hydrology.gumbel_quantile(mu, beta, return_period)
        rain = {"ok": True, "daily_T_mm": d_T, "mu": mu, "beta": beta, "n_years": len(am["years"]), "maxima": am["maxima_mm"], "years": am["years"], "source": am["source"]}
        note("3", "Design rainfall", f"{len(am['years'])} annual maxima of daily rain at the site ({am['source']}) fitted with a Gumbel distribution (μ = {mu:.1f} mm, β = {beta:.1f} mm): the {return_period}-year daily rain is {d_T:.0f} mm.")
    except Exception as e:                                           # noqa: BLE001
        d_T = 150.0
        limits.append(f"Rainfall statistics could not be fetched ({type(e).__name__}); a generic {d_T:.0f} mm/day design storm was used instead. Treat all depths as illustrative.")
        note("3", "Design rainfall", f"Rainfall record unavailable ({e}); falling back to a generic {d_T:.0f} mm/day design storm and flagging every result as illustrative.", warn=True)
    frac6 = 0.65
    storm_mm = d_T * frac6
    assumptions.append(f"The storm lasts {storm_hours:g} h and carries {frac6:.0%} of the {return_period}-year daily total (a standard tropical depth–duration ratio).")

    step_s = 300.0
    hy = hydrology.block_hyetograph(storm_mm, storm_hours, step_s)
    eff = hydrology.effective_rain_series(hy, cn)
    q_excess = hydrology.scs_cn_runoff(storm_mm, cn)
    note("4", "Runoff", f"The {storm_mm:.0f} mm storm produces {q_excess:.0f} mm of runoff after infiltration (SCS curve number {cn:g}); the rest soaks in.")
    duration = (storm_hours + 3.0) * 3600
    ts = np.arange(len(eff)) * step_s
    rates = eff / 1000.0 / step_s

    def run_once(n_manning=manning, rain_scale=1.0, cn_=cn, dx_z=z, keep=False):
        hy_ = hydrology.block_hyetograph(storm_mm * rain_scale, storm_hours, step_s)
        eff_ = hydrology.effective_rain_series(hy_, cn_)
        return flood.simulate(dx_z, P["dx"], duration, rain_m_s=(np.arange(len(eff_)) * step_s, eff_ / 1000.0 / step_s), manning_n=n_manning, open_boundary=True,
                              probe=(ai, aj), record_every_s=600.0)

    # 3 ── build + run
    t0 = NOW()
    res = run_once()
    note("5", "Model run", f"{res.steps:,} time steps (max Courant number {res.max_courant:.2f}) in {NOW() - t0:.0f} s; {res.vol_in / 1e6:.2f} million m³ of runoff entered, "
                           f"{res.vol_out / 1e6:.2f} million m³ left through the edges, {res.vol_stored / 1e6:.2f} million m³ is stored on the surface at the end.")

    # 4 ── verification
    t0 = NOW()
    bench = flood.run_benchmarks()
    run_checks = [{"name": "Mass balance of this run", "metric": "relative volume error", "value": res.mass_error, "limit": 1e-6, "ok": res.mass_error < 1e-6,
                   "note": "Rain in − outflow − water stored = 0."},
                  {"name": "Stability of this run", "metric": "max Courant number", "value": res.max_courant, "limit": 1.0, "ok": res.max_courant < 1.0,
                   "note": "The time step never outran the fastest wave."}]
    verification = bench + run_checks
    nfail = sum(1 for v in verification if not v["ok"])
    note("6", "Verification", f"{len(verification) - nfail} of {len(verification)} tests passed in {NOW() - t0:.0f} s: still water stays still, mass is conserved, the model reproduces the Manning normal depth, and the answer does not change with the time step, and halving the cell size changes it by under 5%." if not nfail else
         f"{nfail} verification test(s) FAILED — the results below must not be used.", warn=bool(nfail))

    site = res.h_max[max(0, ai - 1):ai + 2, max(0, aj - 1):aj + 2]
    site_depth = float(site.max())
    site_hours = float(res.t_wet[ai, aj] / 3600)
    wet_mask = _clusters(res.h_max > 0.3, 4)
    area_km2 = float(wet_mask.sum() * P["dx"] ** 2 / 1e6)
    out = {"site_depth_m": site_depth, "wet_hours": site_hours, "flooded_area_km2": area_km2, "domain_km2": float(z.size * P["dx"] ** 2 / 1e6), "max_depth_m": float(res.h_max.max()),
           "site_elev_m": float(z[ai, aj]), "domain_relief_m": float(z.max() - z.min())}

    # 5 ── sensitivity
    sens: list = []
    if sensitivity and nfail == 0:
        t0 = NOW()
        def f_(p):
            r = run_once(n_manning=p["manning"], rain_scale=p["rain"], cn_=p["cn"])
            return float(r.h_max[max(0, ai - 1):ai + 2, max(0, aj - 1):aj + 2].max())
        sens = calib.oat_sensitivity(f_, {"manning": manning, "rain": 1.0, "cn": cn}, {"manning": (manning * 0.6, manning * 1.6), "rain": (0.7, 1.3), "cn": (max(55, cn - 12), min(95, cn + 10))})
        names = {"manning": "Surface roughness (Manning n)", "rain": "Storm rainfall depth", "cn": "Infiltration (curve number)"}
        for r in sens:
            r["label"] = names[r["param"]]
        note("7", "Sensitivity", f"Each uncertain input was varied alone over a plausible range ({NOW() - t0:.0f} s). The depth at the site is most sensitive to {sens[0]['label'].lower()} "
                                 f"(swing {sens[0]['swing']:.2f} m, elasticity {sens[0]['elasticity']:+.2f}).")

    # 6 ── observations
    calibration = {"done": False}
    validation = {"done": False}
    ref = None
    if observed and observed.get("points"):
        pts = observed["points"]
        def obs_cell(p):
            if "row" in p:
                return int(p["row"]) // P["factor"], int(p["col"]) // P["factor"]
            return ai, aj
        half = max(1, len(pts) // 2) if len(pts) >= 4 else len(pts)
        train, test = pts[:half], pts[half:]
        def err_n(nv):
            r = run_once(n_manning=nv)
            return metrics.rmse([p["depth_m"] for p in train], [float(r.h_max[obs_cell(p)]) for p in train])
        g = calib.grid_search(err_n, np.linspace(manning * 0.5, manning * 2.0, 6))
        calibration = {"done": True, "param": "Manning n", "best": g["best"], "train_rmse_m": g["error"], "n_points": len(train), "trials": g["trials"]}
        r2 = run_once(n_manning=g["best"])
        if test:
            o = [p["depth_m"] for p in test]; s_ = [float(r2.h_max[obs_cell(p)]) for p in test]
            validation = {"done": True, "n_points": len(test), "rmse_m": metrics.rmse(o, s_), "bias_m": metrics.bias(o, s_), "r": metrics.pearson_r(o, s_), "obs": o, "sim": s_,
                          "source": observed.get("source", "supplied observations")}
            note("8", "Calibration and validation", f"Manning n tuned on {len(train)} observation(s) to {g['best']:.3f}; checked on {len(test)} held-out observation(s): RMSE {validation['rmse_m']:.2f} m, bias {validation['bias_m']:+.2f} m.")
        else:
            note("8", "Calibration", f"Manning n tuned to {g['best']:.3f} on {len(train)} observation(s). There were too few observations to hold any back, so this is calibrated, NOT validated.", warn=True)
        res = r2
    else:
        note("8", "Comparison with observations", "No observed flood depths or extents were supplied, so the model is verified but NOT calibrated and NOT validated. Supply flood marks, gauge depths or a satellite flood mask to change that.", warn=True)

    if reference_depth is None:
        try:
            from ..climate.flood_depth import get_flood_depth
            reference_depth = lambda la, lo, rp: get_flood_depth(la, lo, rp)      # noqa: E731
        except Exception:                                            # noqa: BLE001
            reference_depth = None
    if reference_depth is not None:
        try:
            d = float(reference_depth(lat, lon, return_period))
            if d >= 0:
                ref = {"depth_m": d, "source": "JRC Global River Flood Hazard Maps", "model_m": site_depth}
                note("9", "Intercomparison", f"The independent JRC global river-flood hazard map gives {d:.2f} m at the site for the {return_period}-year flood; this model's rain-driven depth is {site_depth:.2f} m. "
                                             "They answer different questions (river overflow vs local rainfall) so a gap is expected; it is context, not validation.")
        except Exception:                                            # noqa: BLE001
            pass

    status = ("verified, calibrated and validated" if validation["done"] else "verified and calibrated, not validated" if calibration["done"] else "verified only") if nfail == 0 else "FAILED verification"
    res_dict = {"domain": "flood", "name": name, "lat": lat, "lon": lon, "status": status, "ok": nfail == 0, "return_period": return_period, "storm_mm": storm_mm, "storm_hours": storm_hours,
            "daily_T_mm": d_T, "runoff_mm": q_excess, "rain": rain, "plan": P, "model_choice": MODEL_CHOICE, "out": out, "verification": verification, "sensitivity": sens,
            "calibration": calibration, "validation": validation, "reference": ref, "assumptions": assumptions, "limitations": limits, "notebook": nb,
            "grids": {"z": z, "h_max": res.h_max, "t_wet": res.t_wet, "dx": P["dx"], "asset": (ai, aj), "series": res.series, "hyetograph_mm": hy, "effective_mm": eff, "step_s": step_s},
            "next_steps": None, "params": {"manning": manning, "cn": cn, "radius_km": radius_km}, "elapsed_s": round(NOW() - t_start, 1)}
    res_dict["next_steps"] = next_steps(res_dict)
    return res_dict


def _clusters(mask: np.ndarray, min_cells: int) -> np.ndarray:
    """Keep only connected wet patches of at least min_cells cells (8-connectivity)."""
    from scipy import ndimage as ndi
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return mask
    sizes = ndi.sum(mask, lab, index=np.arange(1, n + 1))
    keep = np.isin(lab, 1 + np.flatnonzero(sizes >= min_cells))
    return keep


def next_steps(r: dict) -> list:
    out = []
    if not r["validation"]["done"]:
        out.append("Obtain observations to calibrate and validate: flood marks, gauge depths, or a Sentinel-1 flood mask of a past event. Until then treat the depths as scenario estimates.")
    if r["sensitivity"]:
        top = r["sensitivity"][0]
        out.append(f'The answer moves most with {top["label"].lower()} (swing {top["swing"]:.2f} m). Pinning that input down with local data is the cheapest way to tighten the result.')
    out.append("Replace the global 30 m terrain with a survey or LiDAR terrain model, and add culverts and embankments as features; both change local depths more than any physics refinement.")
    if r["rain"].get("ok"):
        out.append("ERA5 smooths extreme rain. Re-run with gauge-based intensity–duration–frequency curves if the municipality has them.")
    return out


def hydrology_fetch(lat, lon):
    from . import data
    return data.annual_max_daily_rain(lat, lon)
