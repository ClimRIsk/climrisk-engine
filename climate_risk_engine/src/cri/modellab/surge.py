"""Storm surge and extreme sea level.

Two independent lines of evidence, kept apart so each can check the other:

A. Observed extremes. Where a NOAA CO-OPS tide gauge lies within 100 km, the engine takes its record of annual-maximum water level above that year's mean sea level (decades long),
   fits an extreme-value distribution, hindcasts it (fit on the early years, predict the recent ones) and reports return levels of the storm tide.  Public-domain data; US coasts.

B. Physics. For tropical cyclones (IBTrACS best tracks, the same Holland vortex as the cyclone study) the engine computes the surge each storm raises at the site as the sum of
   * the inverse barometer, ΔP / (ρ_w g): about 1 cm per hPa of pressure deficit, with the Holland pressure profile p(r) = p_c + (p_n − p_c) exp(−(R_max/r)^B), and
   * the wind setup against the coast on a shelf of length L and mean depth h (open sea-floor depth from the terrain tiles): S = α ρ_a C_d U_on² L / (ρ_w g h)  (Dean & Dalrymple 1984),
   where U_on is the onshore component of the surface wind and C_d the wind-speed-dependent drag (Powell et al. 2003, capped).  α (order 1) absorbs what a one-dimensional shelf cannot
   represent (bay and funnel geometry, wave setup, finite response time). It is learned from storms at tide gauges, and every use is scored by leave-one-out against the surge those gauges recorded.
   Storm rate and the size distribution of surge peaks give return levels of surge (Poisson process + exponential peaks over threshold), checked by the same hindcast as the cyclone study.

Future extreme sea level = today's level + the relative sea-level rise of IPCC AR6 (slr.py). Surge is above the predicted tide; add the local high tide for a total water level.
"""
from __future__ import annotations

import datetime
import math

import numpy as np

from . import coastal as C, counts as CT, evt, plots, slr
from .dossier import NoModel
from .generic_report import Notebook
from ..climate import tropical_cyclone as TC

RHO_W, RHO_A, G = 1025.0, 1.15, 9.81
PN = 1010.0                                  # ambient pressure, hPa
INFLOW_DEG = 20.0


def inverse_barometer_m(dp_hpa):
    """Rise of the sea surface under a pressure deficit: Δη = Δp / (ρ g). 1 hPa -> 0.99 cm."""
    return np.asarray(dp_hpa, float) * 100.0 / (RHO_W * G)


def drag_coefficient(u):
    """Wind-speed dependent drag, Powell et al. (2003)-like: grows with wind, saturates near 2.5e-3."""
    return np.minimum((0.8 + 0.065 * np.asarray(u, float)) * 1e-3, 2.5e-3)


def wind_setup_m(u_on, L_m, h_m, cd=None):
    """Steady wind setup against a coast: S = ρ_a C_d U² L / (ρ_w g h) for the onshore wind U and a shelf of length L and mean depth h (positive onshore only)."""
    u = np.maximum(np.asarray(u_on, float), 0.0)
    cd = drag_coefficient(u) if cd is None else cd
    return RHO_A * cd * u ** 2 * L_m / (RHO_W * G * max(h_m, 1.0))


def pc_from_wind(vmax_kt):
    """Central pressure (hPa) from maximum wind (kt), Atkinson & Holliday (1977): V = 6.7 (1010 − pc)^0.644."""
    return PN - (np.asarray(vmax_kt, float) / 6.7) ** (1.0 / 0.644)


def holland_pressure(r_km, pc, pn, rm, b):
    x = (np.asarray(rm, float) / np.maximum(r_km, 0.5)) ** b
    return pc + (pn - pc) * np.exp(-x)


def shelf(lat: float, lon: float, tile_fetch=None) -> dict:
    """Open-sea direction, shelf length L (km, to 200 m depth, cap 300) and mean depth h (m) from the open terrain tiles.
    The seaward bearing is the one that reaches deep water (30 m) soonest, so a bay or harbour behind the site is not mistaken for the open sea."""
    best = None
    for b in range(0, 360, 30):
        p = C.profile(lat, lon, b, 300.0, 5.0, 8, tile_fetch)
        depth = np.where(np.isfinite(p), np.maximum(-p, 0.0), 0.0)
        sea = np.where(depth >= 1.0)[0]
        if not len(sea) or sea[0] > 12:                    # sea must start within 60 km
            continue
        deep = np.where(depth >= 30.0)[0]
        key = (int(deep[0]) if len(deep) else 1000 + int(-depth.mean()))
        if best is None or key < best[0]:
            best = (key, b, depth, int(sea[0]) * 5.0)
    if best is None:
        return {"ok": False}
    _, brg, depth, sea_km = best
    deep = np.where(depth >= 200.0)[0]
    n = max(int(deep[0]) if len(deep) else len(depth), 2)
    seg = depth[:n]
    h = float(max(np.mean(seg[seg > 0]) if (seg > 0).any() else 5.0, 3.0))
    return {"ok": True, "bearing_to_sea": brg, "L_km": float((n - 1) * 5.0 + 5.0), "h_m": h, "sea_km": sea_km, "capped": not len(deep)}


def storm_series(lat: float, lon: float, ix: np.ndarray, T: dict, sh: dict) -> dict:
    """Time series (30-minute steps) of the inverse barometer at the site and the (unit-α) wind setup from the wind stress averaged over the whole shelf, with the shelf's finite response time."""
    s = TC._interp_storm(ix, T)
    r = TC._haversine_km(s["lat"], s["lon"], lat, lon)
    vkt = s["vmax"]; v = vkt * TC.KT
    rm = np.where(np.isfinite(s["rmw"]) & (s["rmw"] > 0), s["rmw"] * TC.NMI, TC.rmax_km(v, s["lat"]))
    b = TC.holland_b(v, rm, s["lat"])
    order = np.argsort(T["t"][ix]); tt0 = T["t"][ix][order].astype(float); pp0 = T["pres"][ix][order].astype(float); okp = np.isfinite(pp0) & (pp0 > 850)
    pres_obs = np.interp(s["t"], tt0[okp], pp0[okp]) if okp.sum() >= 2 else np.full(len(s["t"]), np.nan)
    pc = np.where(np.isfinite(pres_obs), pres_obs, pc_from_wind(vkt))
    pc = np.minimum(pc, PN - 1.0)
    ib = inverse_barometer_m(PN - holland_pressure(r, pc, PN, rm, b))
    north = lat >= 0
    landward = (sh["bearing_to_sea"] + 180.0) % 360
    L_m, h = sh["L_km"] * 1000.0, sh["h_m"]
    pts = [(lat, lon)] + [C.destination(lat, lon, sh["bearing_to_sea"], d) for d in np.linspace(0.0, sh["L_km"], 6)[1:]]
    tau = np.zeros(len(s["t"]))
    for pla, plo in pts:
        rr = TC._haversine_km(s["lat"], s["lon"], pla, plo)
        w = TC.holland_wind(rr, v, rm, b)
        brg = TC._bearing_deg(s["lat"], s["lon"], pla, plo)
        toward = (brg - 90.0 - INFLOW_DEG) % 360 if north else (brg + 90.0 + INFLOW_DEG) % 360
        u_on = np.maximum(w * np.cos(np.radians(toward - landward)), 0.0)
        tau += RHO_A * drag_coefficient(u_on) * u_on ** 2
    tau = np.nan_to_num(tau / len(pts))
    s_eq = tau * L_m / (RHO_W * G * max(h, 1.0))
    tr = max(0.5, L_m / math.sqrt(G * max(h, 1.0)) / 3600.0 / 2.0)           # hours: half the shelf's long-wave travel time
    dt = float(np.median(np.diff(s["t"]))) if len(s["t"]) > 1 else 0.5
    a = 1.0 - math.exp(-dt / tr); sw = np.zeros(len(s["t"])); cur = 0.0
    for i, e in enumerate(s_eq):
        cur += a * (e - cur); sw[i] = cur
    ib = np.where(np.isfinite(ib), ib, 0.0)
    return {"t": s["t"], "ib": ib, "sw": sw, "r": r, "wind": TC.holland_wind(r, v, rm, b)}


def peak(series: dict, alpha: float) -> tuple:
    tot = series["ib"] + alpha * series["sw"]
    i = int(np.argmax(tot))
    return float(tot[i]), float(series["t"][i]), float(np.min(series["r"]))


def storm_time(T: dict, si: int, t_h: float) -> datetime.datetime:
    """Absolute UTC time of a track record: the bundled track times are hours since 1980-01-01 00:00 UTC."""
    return datetime.datetime(1980, 1, 1) + datetime.timedelta(hours=float(t_h))


def fit_alpha(ib, sw, obs) -> float:
    ib, sw, obs = map(lambda a: np.asarray(a, float), (ib, sw, obs))
    d = sw @ sw
    return float(max((sw @ (obs - ib)) / d, 0.0)) if d > 0 else 1.0


def loo_skill(ib, sw, obs) -> dict:
    """Leave-one-out prediction of observed peak surge with the one-parameter model; skill against predicting the mean of the others."""
    ib, sw, obs = map(lambda a: np.asarray(a, float), (ib, sw, obs))
    n = len(obs); pred = np.zeros(n); base = np.zeros(n)
    for k in range(n):
        m = np.arange(n) != k
        a = fit_alpha(ib[m], sw[m], obs[m])
        pred[k] = ib[k] + a * sw[k]; base[k] = obs[m].mean()
    mse = float(np.mean((pred - obs) ** 2)); mse0 = float(np.mean((base - obs) ** 2))
    r = float(np.corrcoef(pred, obs)[0, 1]) if n > 2 and np.std(pred) > 0 and np.std(obs) > 0 else float("nan")
    return {"n": n, "r": r, "rmse": math.sqrt(mse), "bias": float(np.mean(pred - obs)), "skill": float(1 - mse / mse0) if mse0 > 0 else float("nan"), "alpha": fit_alpha(ib, sw, obs), "pred": pred.tolist(), "obs": obs.tolist()}


def benchmarks() -> list:
    out = []
    out.append({"name": "Inverse barometer: 1 hPa raises the sea 0.99 cm", "ok": abs(float(inverse_barometer_m(1.0)) - 0.00995) < 5e-5, "detail": f"{float(inverse_barometer_m(1.0)) * 100:.3f} cm"})
    s = float(wind_setup_m(30.0, 100e3, 20.0, cd=2.6e-3))
    out.append({"name": "Wind setup: 30 m/s over a 100 km, 20 m deep shelf raises the coast 1.34 m (hand value: τ = 1.15 × 2.6e-3 × 30² = 2.69 Pa; S = τL/ρ_w g h)", "ok": abs(s - 1.338) < 0.01, "detail": f"{s:.3f} m"})
    out.append({"name": "Offshore wind raises no setup", "ok": float(wind_setup_m(-20.0, 100e3, 20.0)) == 0.0, "detail": "0 m"})
    out.append({"name": "Setup scales with the square of the wind (doubling the wind quadruples it)", "ok": abs(float(wind_setup_m(40.0, 1e5, 20.0, cd=2e-3)) / float(wind_setup_m(20.0, 1e5, 20.0, cd=2e-3)) - 4.0) < 1e-9, "detail": "×4"})
    p0 = float(holland_pressure(0.5, 940.0, 1010.0, 30.0, 1.5)); p1 = float(holland_pressure(2000.0, 940.0, 1010.0, 30.0, 1.5))
    out.append({"name": "Holland pressure: central value at the eye, ambient far away", "ok": abs(p0 - 940.0) < 0.5 and abs(p1 - 1010.0) < 1.0, "detail": f"{p0:.1f} hPa at 0.5 km, {p1:.1f} hPa at 2000 km"})
    pc = float(pc_from_wind(100.0))
    out.append({"name": "Atkinson–Holliday: 100 kt corresponds to about 943 hPa (hand value: (100/6.7)^(1/0.644) = 66.5 hPa deficit)", "ok": abs(pc - 943.5) < 1.0, "detail": f"{pc:.1f} hPa"})
    rng = np.random.default_rng(3)
    sw = rng.uniform(0.2, 2.5, 60); ib = rng.uniform(0.0, 0.8, 60); obs = ib + 1.3 * sw + rng.normal(0, 0.05, 60)
    sk = loo_skill(ib, sw, obs)
    out.append({"name": "Calibration recovers a known coefficient (true α = 1.3) and the leave-one-out skill is high", "ok": abs(sk["alpha"] - 1.3) < 0.05 and sk["skill"] > 0.95, "detail": f'α = {sk["alpha"]:.2f}, skill {sk["skill"]:.2f}'})
    return out


def _gauge_events(lat, lon, T, sh, sta, fetch, n_max=24, since=1996, search_km=500.0):
    """Storms near the gauge since `since`, with modelled components and the peak surge the gauge recorded around them."""
    d = TC._haversine_km(T["lat"], T["lon"], sta["lat"], sta["lon"])
    cand = []
    for si in np.unique(T["storm"][d <= search_km]):
        if int(T["season"][si]) < since:
            continue
        ix = np.where(T["storm"] == si)[0]
        ser = storm_series(sta["lat"], sta["lon"], ix, T, sh)
        pk, t_pk, rmin = peak(ser, 1.0)
        if pk >= 0.05:
            cand.append((pk, int(si), ser, t_pk, rmin))
    cand.sort(key=lambda c: -c[0])
    rows = []
    for pk, si, ser, t_pk, rmin in cand[:n_max]:
        t0 = storm_time(T, si, t_pk)
        a, b = (t0 - datetime.timedelta(days=3.5)), (t0 + datetime.timedelta(days=3.5))
        try:
            w = C.window(sta["id"], a.strftime("%Y%m%d"), b.strftime("%Y%m%d"), fetch)
        except Exception:                                              # noqa: BLE001
            continue
        if len(w["surge"]) < 100:
            continue
        i = int(np.argmax(ser["ib"] + ser["sw"]))
        rows.append({"storm": str(T["name"][si]).title(), "season": int(T["season"][si]), "si": si, "ib": float(ser["ib"][i]), "sw": float(ser["sw"][i]), "rmin_km": rmin, "obs": float(max(w["surge"]) - float(np.median(w["surge"]))), "tot": float(max(w["observed"])), "date": t0, "series": ser})
    return rows


def _annual_extremes(sta_id, fetch, with_trend=False):
    a = C.annual(sta_id, 1950, 2025, fetch)
    ys, x = [], []
    for y, m, h in zip(a["years"], a["msl_mm"], a["highest_m"]):
        if np.isfinite(h):
            ys.append(y); x.append(h - m / 1000.0)
    if not with_trend:
        return ys, x
    ay = np.array(a["years"], float); am = np.array(a["msl_mm"], float) / 1000.0
    c = np.polyfit(ay, am, 1) if len(ay) >= 5 else [0.0, 0.0]
    return ys, x, (lambda yr: float(c[0] * yr + c[1]))


def patch_gap_years(ys, x, rows, msl_at):
    """Years with too little monthly data (a gauge outage, e.g. Galveston, September 2008, Hurricane Ike) are filled from the highest water the hourly record shows around the storms of that year,
    measured above that year's trend mean sea level. This only ever adds years the monthly statistics dropped; it never changes a year that has complete data."""
    ys, x = list(ys), list(x); added = []
    best = {}
    for r in rows:
        d = r.get("date")
        if d is None or "tot" not in r:
            continue
        yr = d.year
        if yr in ys:
            continue
        v = r["tot"] - msl_at(yr)
        if yr not in best or v > best[yr][0]:
            best[yr] = (v, r["storm"])
    for yr, (v, nm) in sorted(best.items()):
        ys.append(yr); x.append(v); added.append((yr, nm, v))
    o = np.argsort(ys)
    return [ys[i] for i in o], [x[i] for i in o], added


def run(lat: float, lon: float, name: str = "Study site", progress=None, fetch=None, tile_fetch=None, T: dict = None, global_alpha: float = None, events_fetch=None, search_km: float = 600.0) -> dict:
    nb = Notebook(progress)
    nb("1", "Problem", f"How high does the sea stand above normal during a storm at {name} ({lat:.3f}, {lon:.3f}) once in 10, 50 and 100 years, can the engine reproduce the surges that tide gauges actually recorded, and how much higher will that be with the rising sea?")
    T = T or TC.tracks()
    sh = shelf(lat, lon, tile_fetch)
    if not sh["ok"]:
        raise NoModel("No sea within 60 km of this site: storm surge does not reach it.")
    el = C.elevation(lat, lon, 11, tile_fetch)
    nb("2", "Shelf", f'Seaward direction {sh["bearing_to_sea"]}° ({sh["sea_km"]:.0f} km away); shelf length {sh["L_km"]:.0f} km{" (capped)" if sh["capped"] else ""} to the 200 m isobath, mean depth {sh["h_m"]:.0f} m (open terrain tiles). Ground elevation {el:.0f} m.')
    bench = benchmarks(); nfail = sum(1 for b in bench if not b["ok"])
    nb("3", "Verification", f"{len(bench) - nfail} of {len(bench)} tests passed: inverse barometer, wind setup against its hand value, quadratic wind law, Holland pressure limits, calibration recovery." if not nfail else f"{nfail} FAILED.", warn=bool(nfail))
    # storms at the site
    d = TC._haversine_km(T["lat"], T["lon"], lat, lon)
    near = np.unique(T["storm"][d <= search_km])
    sta = None
    try:
        sta = C.nearest_station(lat, lon, 100.0, fetch)
    except Exception:                                                  # noqa: BLE001
        sta = None
    # --- A. observed extremes at the gauge
    A = None
    if sta:
        try:
            ys, xs, msl_at = _annual_extremes(sta["id"], fetch, True)
            if len(ys) >= 20:
                bm = evt.best_model(xs); hc = evt.hindcast(xs)
                A = {"station": sta, "msl_at": msl_at, "patched": [], "years": ys, "x": xs, "model": bm["params"], "kind": bm["kind"], "hc": hc, "levels": {T_: evt.level(bm["params"], T_) for T_ in (10, 50, 100)}, "ci": evt.bootstrap_levels(xs, n=150)}
                nb("4", "Observed extremes", f'{sta["name"]} gauge ({sta["km"]:.0f} km away), {len(ys)} years ({ys[0]}–{ys[-1]}): 50-year storm tide {A["levels"][50]:.2f} m above the year\'s mean sea level; hindcast ' + ("passed." if hc["ok"] and hc["passed"] else "FAILED." if hc["ok"] else "not possible."), warn=not (hc["ok"] and hc["passed"]))
        except Exception as e:                                         # noqa: BLE001
            nb("4", "Observed extremes", f"Gauge record unavailable ({str(e)[:80]}).", warn=True)
    # --- B. physics
    peaks_all = []
    alpha, skill, ev_rows = (global_alpha if global_alpha is not None else 1.0), None, []
    if sta and len(near):
        try:
            ev_rows = (events_fetch or _gauge_events)(lat, lon, T, shelf(sta["lat"], sta["lon"], tile_fetch), sta, fetch)
        except Exception as e:                                         # noqa: BLE001
            nb("5", "Gauge hindcast", f"Could not assemble gauge events ({str(e)[:80]}).", warn=True); ev_rows = []
        if A and ev_rows:
            ys2, xs2, added = patch_gap_years(A["years"], A["x"], ev_rows, A["msl_at"])
            if added:
                bm = evt.best_model(xs2); hc = evt.hindcast(xs2)
                A.update({"years": ys2, "x": xs2, "model": bm["params"], "kind": bm["kind"], "hc": hc, "levels": {T_: evt.level(bm["params"], T_) for T_ in (10, 50, 100)}, "ci": evt.bootstrap_levels(xs2, n=150), "patched": added})
                nb("4b", "Gap years", "Years the monthly gauge statistics dropped (outage) were filled from the hourly record around recorded storms: " + "; ".join(f"{y} ({n}: {v:.2f} m)" for y, n, v in added) + f". Revised 50-year storm tide {A['levels'][50]:.2f} m.")
        if len(ev_rows) >= 6:
            skill = loo_skill([r["ib"] for r in ev_rows], [r["sw"] for r in ev_rows], [r["obs"] for r in ev_rows])
            alpha = skill["alpha"]
            nb("5", "Gauge hindcast", f'{skill["n"]} storms since 1996 at {sta["name"]}: leave-one-out correlation {skill["r"]:.2f}, RMSE {skill["rmse"]:.2f} m, bias {skill["bias"]:+.2f} m, skill against the mean surge {skill["skill"]:.2f}; learned wind-setup coefficient α = {alpha:.2f}.', warn=skill["skill"] <= 0.2)
        else:
            nb("5", "Gauge hindcast", f"Only {len(ev_rows)} usable storm records at the gauge; not enough to score the model.", warn=True)
    rows = []
    surge_pk = []
    seasons = []
    for si in near:
        ix = np.where(T["storm"] == si)[0]
        ser = storm_series(lat, lon, ix, T, sh)
        pk, t_pk, rmin = peak(ser, alpha)
        rows.append((int(T["season"][si]), pk, rmin, str(T["name"][si]).title(), int(si)))
    y0, y1 = int(T["season"].min()), int(T["season"].max()); years = list(range(y0, y1 + 1)); n_y = len(years)
    pk = np.array([r[1] for r in rows], float); ss = np.array([r[0] for r in rows], int)
    thr = 0.5
    n_over = int((pk >= thr).sum())
    B = None
    if n_over >= 3:
        cnt = np.array([int(np.sum((ss == y) & (pk >= thr))) for y in years])
        hcB = CT.rate_hindcast(cnt)
        sc, nn = CT.pot_fit(pk[pk >= thr], thr)
        lv = {T_: float(CT.pot_level(thr, sc, n_over / n_y, T_)) for T_ in (10, 50, 100)}
        B = {"rate": n_over / n_y, "n": n_over, "scale": sc, "levels": lv, "hc": hcB, "thr": thr}
        nb("6", "Storm surge at the site", f'{n_over} storms in {n_y} seasons raised at least {thr:.1f} m (rate {B["rate"]:.2f}/yr); peak size above that follows an exponential with scale {sc:.2f} m. 10/50/100-year surge: {lv[10]:.1f} / {lv[50]:.1f} / {lv[100]:.1f} m.')
    else:
        nb("6", "Storm surge at the site", f"Only {n_over} tropical cyclones in {n_y} seasons would raise the sea by {thr:.1f} m or more here: no tropical-cyclone surge rate can be fitted.", warn=True)
    if B is None and A is None:
        raise NoModel("No tide gauge within 100 km and too few tropical cyclones to fit a surge rate: the engine does not guess a storm-tide level here.")
    # cross-check: modelled surge + high tide vs observed storm tide
    xc = None
    if A and B:
        ob = A["levels"][50]; md = B["levels"][50]
        xc = {"observed_50": ob, "model_surge_50": md, "agree": (0.5 <= md / ob <= 2.0) if ob > 0.3 else None, "note": f"Observed 50-year storm tide at the gauge {ob:.2f} m above mean sea level (includes tide); modelled cyclone surge alone {md:.2f} m."}
    # --- sea-level rise uplift
    up = None
    try:
        loc = slr.nearest(lat, lon)
        up = {s: {y: slr.uplift(loc["i"], s, y, 2) for y in (2050, 2100)} for s in ("ssp245", "ssp585")}
        up_hi = {s: {y: slr.uplift(loc["i"], s, y, 4) for y in (2050, 2100)} for s in ("ssp245", "ssp585")}
        base_year = 2025
        for s in up:
            for y in up[s]:
                up[s][y] -= slr.uplift(loc["i"], s, base_year, 2); up_hi[s][y] -= slr.uplift(loc["i"], s, base_year, 4)
    except Exception:                                                  # noqa: BLE001
        up = None
    # --- status
    if nfail:
        status = "FAILED verification"
    elif skill is not None and skill["skill"] > 0.2 and skill["r"] > 0.5:
        status = "verified and validated against tide-gauge surges"
    elif skill is not None:
        status = "verified; tide-gauge hindcast FAILED (the surge model has no skill against recorded surges)"
    elif A and A["hc"]["ok"] and A["hc"]["passed"]:
        status = "verified and hindcast-validated (observed gauge extremes)"
    elif A:
        status = "verified; gauge extremes hindcast FAILED" if A["hc"]["ok"] else "verified only (gauge record too short to hindcast)"
    else:
        status = "verified only (no tide gauge within 100 km to check against)"
    fmt = lambda v: f"{v:.2f}"
    head = []
    if B:
        head += [(f'{B["levels"][10]:.1f} m', "10-year cyclone surge"), (f'{B["levels"][50]:.1f} m', "50-year cyclone surge"), (f'{B["levels"][100]:.1f} m', "100-year cyclone surge")]
    if A:
        head.append((f'{A["levels"][50]:.2f} m', "50-year storm tide at the gauge (above mean sea level)"))
    if skill:
        head.append((f'r = {skill["r"]:.2f}', "gauge hindcast correlation"))
    blocks = [("h2", "question", "The question"), ("p", f"How high does the sea stand during a storm at {name}, how often, can the model reproduce what tide gauges recorded, and what will the rising sea add? A surge is the part of the water level the storm adds on top of the tide; it is what overtops defences, so its return levels and the sea-level rise under them are the two numbers a coastal asset needs."),
              ("h2", "physics", "The physics the engine chose"),
              ("p", "Low pressure lifts the sea surface by about a centimetre per hectopascal (the inverse barometer: **Δη = Δp / ρg**). Wind drags the surface water toward the coast, and where the shelf is shallow and long the water piles up against the shore: **S = α ρₐ C_d U² L / (ρ_w g h)**. Both are computed along each cyclone's track from the Holland vortex used in the cyclone study; α, the one empirical constant, is learned from storms that tide gauges recorded and scored by leave-one-out."),
              ("p", f"Here the shelf extends {sh['L_km']:.0f} km seaward ({sh['bearing_to_sea']}°) with a mean depth of {sh['h_m']:.0f} m, so a 30 m/s onshore wind would hold the sea up by about {float(wind_setup_m(30.0, sh['L_km'] * 1000, sh['h_m'])):.1f} m before the coast's geometry changes it."), ("basis",)]
    if A:
        blocks += [("h2", "observed", "What the tide gauge recorded"),
                   ("fig", plots.return_level_plot(A["x"], lambda T_: evt.level(A["model"], T_), {k: v for k, v in A["ci"].items()}, "m above the year's mean sea level", f'{A["station"]["name"]}: highest water level of each year'), "Black dots: the highest water level in each year above that year's mean sea level; band: 90 % bootstrap range."),
                   ("table", ["Return period", "Storm tide above mean sea level (m)", "90 % range (m)"], [[f"{T_} years", fmt(A["levels"][T_]), f'{A["ci"][T_][0]:.2f} – {A["ci"][T_][1]:.2f}' if T_ in A["ci"] else "n/a"] for T_ in (10, 50, 100)],
                    f'{A["station"]["name"]}, {len(A["years"])} years ({A["years"][0]}–{A["years"][-1]}); includes the astronomical tide. Hindcast: ' + ("passed." if A["hc"]["ok"] and A["hc"]["passed"] else "failed: " + (A["hc"].get("bias_note") or "the early record does not predict the recent one.") if A["hc"]["ok"] else "record too short."))]
    if skill:
        blocks += [("h2", "hindcast", "Does the model reproduce recorded surges?"),
                   ("fig", plots.scatter_fit(skill["pred"], skill["obs"], [0, max(skill["obs"] + skill["pred"])], [0, max(skill["obs"] + skill["pred"])], "modelled surge, leave-one-out (m)", "recorded surge (m)", f'{sta["name"]}: {skill["n"]} storms'), "Each dot is a cyclone since 1996; the line is perfect agreement. The recorded surge is the highest water level minus the predicted tide within three days of the storm."),
                   ("table", ["Storm", "Season", "Closest (km)", "Modelled pressure term (m)", "Modelled wind term (m, α = 1)", "Recorded surge (m)"], [[r["storm"], r["season"], f'{r["rmin_km"]:.0f}', f'{r["ib"]:.2f}', f'{r["sw"]:.2f}', f'{r["obs"]:.2f}'] for r in ev_rows[:15]], f'Leave-one-out: r = {skill["r"]:.2f}, RMSE {skill["rmse"]:.2f} m, bias {skill["bias"]:+.2f} m, skill {skill["skill"]:.2f} (0 = no better than the average surge). α = {alpha:.2f}.'),
                   ("callout", ("The surge model reproduces what the gauge recorded." if skill["skill"] > 0.2 and skill["r"] > 0.5 else "The surge model has little or no skill at this gauge: shelf geometry, bays and wave setup matter more than a one-dimensional shelf captures. Treat cyclone surge numbers as indicative."), "good" if skill["skill"] > 0.2 and skill["r"] > 0.5 else "bad")]
    if B:
        blocks += [("h2", "surge", "Cyclone surge at the site"),
                   ("table", ["Return period", "Surge above the tide (m)"], [[f"{T_} years", fmt(B["levels"][T_])] for T_ in (10, 50, 100)],
                    f'{B["n"]} cyclones in {n_y} seasons raised at least {thr:.1f} m. ' + (f"Wind-setup coefficient α = {alpha:.2f} " + ("learned from the gauge above." if skill else "is the uncalibrated physical value (1.0): no gauge here to learn it from, so treat as indicative.")))]
        hb = B["hc"]
        blocks += [("callout", f'Rate hindcast: {hb["rate_train"]:.2f}/yr in the first part of the record predicts {hb["expected"]:.1f} storms in the last {hb["n_test"]} seasons; observed {hb["observed"]} (p = {hb["p"]:.2f}).', "good" if hb["p"] > 0.05 else "warn")]
    if xc:
        blocks += [("callout", xc["note"] + (" They are of the same order." if xc["agree"] else " They differ by more than a factor of two." if xc["agree"] is False else ""), "good" if xc["agree"] else "warn")]
    if up:
        base = (A["levels"][100] if A else (B["levels"][100] if B else 0.0))
        rows_up = [[f"{s_.upper().replace('SSP', 'SSP')}", str(y), f'{up[s_][y]:+.2f}', f'{base + up[s_][y]:.2f}'] for s_ in up for y in (2050, 2100)]
        blocks += [("h2", "future", "With the rising sea"),
                   ("table", ["Pathway", "Year", "Rise since 2025 (m, median)", f'100-year level ({"storm tide" if A else "surge"}) then (m)'], rows_up, "The extreme level is shifted up by the median relative sea-level rise (IPCC AR6) from the same location: a storm of the same strength rides on a higher sea. Wave setup, changing storm intensity and tide-surge interaction are not included.")]
    blocks += [("h2", "result", "What to take from this"),
               ("p", ("Cyclone surge: " + f"{B['levels'][50]:.1f} m in 50 years. " if B else "") + ("The gauge's 50-year storm tide is " + f"{A['levels'][50]:.2f} m above mean sea level. " if A else "") + (f"Ground here is {el:.0f} m above the sea" + (": below the 100-year level, so a design storm can reach the site." if el < (A["levels"][100] if A else (B["levels"][100] if B else 99)) else ".") if el is not None else ""))]
    lim = ["The physics is a one-dimensional shelf model: bays, estuaries, river deltas and shelf geometry that funnels or spreads water are only absorbed in one coefficient, so site-specific hydrodynamic modelling is needed for design.",
           "Only tropical cyclones are modelled in B; extratropical storms (nor'easters, North Sea surges) appear only through the gauge record in A.",
           "Surge is above the predicted tide; total water level also needs the local high tide and wave run-up.",
           "The recorded surge is the highest hourly water level above the predicted tide within ±3.5 days of the modelled peak, minus the median residual of that window (a slowly varying sea level is not storm surge).",
           "Gauge data: NOAA CO-OPS (public domain). Tracks: IBTrACS (NOAA NCEI). Sea-level projection: " + slr.ATTRIBUTION]
    return {"domain": "surge", "label": "Storm surge", "kind": "surge", "name": name, "lat": lat, "lon": lon, "title": f"Storm surge at {name}: how high, how often, and does the model match the gauges?",
            "standfirst": "The engine modelled the surge raised by each tropical cyclone since 1980 and " + ("scored it against the surges the nearest tide gauge recorded" + (f" (correlation {skill['r']:.2f})" if skill else "") if sta else "found no tide gauge within 100 km to check against") + (f"; 50-year cyclone surge {B['levels'][50]:.1f} m" if B else "") + f". Status: {status}.",
            "status": status, "ok": nfail == 0, "headline": head or [("none", "no surge numbers")], "blocks": blocks, "verification": bench, "sensitivity": [],
            "assumptions": ["Surge peaks are independent between storms (Poisson process).", "The shelf geometry from open terrain tiles represents the sea-floor the wind acts on."], "limitations": lim,
            "next_steps": ["For design, run a hydrodynamic model (e.g. ADCIRC, Delft3D) for the site with this study as the screening level.", "Add the local high tide and wave run-up to the surge for a total water level.", "Where no gauge is near, obtain a local tide record to learn the wind-setup coefficient."],
            "notebook": nb.rows, "numbers": {"source": "IBTrACS + NOAA CO-OPS (public domain)", "levels": ({T_: B["levels"][T_] for T_ in (10, 50, 100)} if B else ({T_: A["levels"][T_] for T_ in (10, 50, 100)} if A else {})),
                                          "ci": ({str(k): v for k, v in A["ci"].items()} if A else None), "n_years": (len(A["years"]) if A else None), "gauge": sta, "observed": ({"levels": A["levels"], "hindcast": {k: v for k, v in A["hc"].items() if k != "pit"}} if A else None),
                                          "model": ({"levels": B["levels"], "rate": B["rate"]} if B else None), "alpha": alpha, "skill": ({k: v for k, v in skill.items() if k not in ("pred", "obs")} if skill else None), "crosscheck": xc, "shelf": sh, "uplift": up, "elevation_m": el},
            "elapsed_s": round(__import__("time").time() - nb.t0, 1)}
