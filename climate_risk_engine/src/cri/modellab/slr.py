"""Sea-level rise and shoreline retreat.

Projections: IPCC AR6 relative sea-level change (Fox-Kemper et al. 2021; Garner et al. 2021 / Kopp et al. 2023 FACTS framework), medium-confidence "total" projections including
vertical land motion, relative to 1995-2014, at 1,030 tide-gauge sites and 65,000 one-degree grid points, five SSP scenarios, five quantiles (5, 17, 50, 83, 95 %), decades 2030, 2050, 2070, 2100.
Licence: CC BY 4.0 (Zenodo 10.5281/zenodo.6382554), commercial use permitted with the required citation. The compact table is bundled (cri/data/slr_ar6.npz).

Shoreline retreat: the Bruun rule, R = S / tan(beta), with S the rise in sea level and tan(beta) the slope of the active beach profile. It applies to sandy, erodible coasts only; the study says so.
Backtest: where a NOAA CO-OPS tide gauge lies within 150 km, the observed 1995-2024 trend is compared with the rate AR6 implies for 2005-2030.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import coastal as C, plots
from .dossier import NoModel
from .generic_report import Notebook

ATTRIBUTION = ("IPCC AR6 sea-level projections: Fox-Kemper, B. et al. (2021) Ocean, Cryosphere and Sea Level Change, in Climate Change 2021: The Physical Science Basis (IPCC AR6 WG1, ch. 9); "
               "Garner, G. G. et al. (2021) IPCC AR6 Sea-Level Rise Projections, version 20210809, PO.DAAC; Kopp, R. E. et al. (2023) The Framework for Assessment of Changes To Sea-level (FACTS), Geosci. Model Dev. 16, 7461. Zenodo 10.5281/zenodo.6382554 (CC BY 4.0).")
SCEN_LABEL = {"ssp119": "SSP1-1.9", "ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5", "ssp370": "SSP3-7.0", "ssp585": "SSP5-8.5"}
BRUUN_SLOPES = (0.05, 0.02, 0.01)       # steep gravel/coarse sand ... typical sand ... gentle fine sand: tan(beta)


@lru_cache(maxsize=1)
def _load():
    p = Path(__file__).resolve().parent.parent / "data" / "slr_ar6.npz"
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in z.files}


def nearest(lat: float, lon: float, gauge_km: float = 60.0, grid_km: float = 300.0) -> dict:
    """AR6 location for a point: a tide gauge within gauge_km if there is one (it carries local land motion), else the nearest grid point within grid_km."""
    d = _load()
    dist = C.haversine_km(lat, lon, d["lat"].astype(float), np.where(d["lon"] > 180, d["lon"] - 360, d["lon"]).astype(float))
    names = d["names"]
    is_g = np.array([not str(n).startswith("grid_") for n in names])
    ig = np.where(is_g)[0]
    if len(ig):
        j = ig[int(np.argmin(dist[ig]))]
        if dist[j] <= gauge_km:
            return {"i": int(j), "name": str(names[j]), "km": float(dist[j]), "gauge": True}
    j = int(np.argmin(dist))
    if dist[j] > grid_km:
        raise NoModel(f"No sea-level projection location within {grid_km:.0f} km (nearest is {dist[j]:.0f} km away).")
    return {"i": j, "name": str(names[j]), "km": float(dist[j]), "gauge": not str(names[j]).startswith("grid_")}


def rsl(i: int) -> dict:
    """{scenario: {year: [q05, q17, q50, q83, q95] in metres}} at AR6 location index i (relative to 1995-2014)."""
    d = _load(); out = {}
    for si, s in enumerate(d["scenarios"]):
        out[str(s)] = {}
        for yi, y in enumerate(d["years"]):
            v = d["data"][si, yi, :, i].astype(float)
            out[str(s)][int(y)] = [float("nan") if x == -32768 else x / 1000.0 for x in v]
    return out


def uplift(i: int, scenario: str, year: int, q: int = 2) -> float:
    """Relative sea-level rise (m) at the AR6 location by a year, interpolating between the stored decades (0 at 2005, the middle of the baseline)."""
    r = rsl(i)[scenario]
    ys = [2005] + sorted(r)
    vs = [0.0] + [r[y][q] for y in sorted(r)]
    return float(np.interp(year, ys, vs))


def bruun_retreat(rise_m: float, tan_beta: float) -> float:
    """Shoreline retreat (m) for a rise in sea level, Bruun (1962): R = S / tan(beta)."""
    return float(rise_m / tan_beta)


def benchmarks() -> list:
    out = []
    out.append({"name": "Bruun rule: 0.5 m of rise on a 1:50 profile retreats the shore 25 m", "ok": abs(bruun_retreat(0.5, 0.02) - 25.0) < 1e-9, "detail": f"{bruun_retreat(0.5, 0.02):.1f} m"})
    d = _load()
    q = d["data"].astype(float)
    good = (q > -32768).all(axis=2).all(axis=1)[0]
    ok_order = bool(np.all(np.diff(q[:, :, :, :200], axis=2) >= -1)) if q.size else False
    out.append({"name": "Quantiles are ordered (5 % <= 17 % <= 50 % <= 83 % <= 95 %) at every checked location, year and scenario", "ok": ok_order, "detail": "first 200 locations"})
    i = int(np.where(~np.array([str(n).startswith("grid_") for n in d["names"]]))[0][0])
    med = [rsl(i)["ssp245"][y][2] for y in (2030, 2050, 2070, 2100)]
    out.append({"name": "Median rise increases with time", "ok": all(b > a for a, b in zip(med, med[1:])), "detail": " < ".join(f"{m:.2f}" for m in med)})
    r = rsl(i)
    out.append({"name": "Higher-emission scenarios give more rise by 2100", "ok": r["ssp119"][2100][2] < r["ssp245"][2100][2] < r["ssp585"][2100][2], "detail": f'{r["ssp119"][2100][2]:.2f} < {r["ssp245"][2100][2]:.2f} < {r["ssp585"][2100][2]:.2f} m'})
    la, lo = float(d["lat"][i]), float(d["lon"][i]); lo = lo - 360 if lo > 180 else lo
    nn = nearest(la, lo)
    out.append({"name": "A gauge's own coordinates find that gauge", "ok": nn["i"] == i and nn["km"] < 1.0, "detail": f'{nn["name"]} at {nn["km"]:.2f} km'})
    out.append({"name": "Great-circle distance: 1° of latitude is 111.2 km", "ok": abs(float(C.haversine_km(0, 0, 1, 0)) - 111.19) < 0.1, "detail": f'{float(C.haversine_km(0, 0, 1, 0)):.2f} km'})
    return out


def _trend(years, v):
    """OLS slope (mm/yr) with a 95 % interval widened for lag-1 autocorrelation of the residuals."""
    x = np.asarray(years, float); y = np.asarray(v, float)
    xm = x - x.mean(); b = float((xm * (y - y.mean())).sum() / (xm ** 2).sum()); res = y - y.mean() - b * xm
    r1 = float(np.corrcoef(res[:-1], res[1:])[0, 1]) if len(res) > 3 else 0.0
    r1 = max(0.0, min(0.9, r1)); neff = max(4.0, len(x) * (1 - r1) / (1 + r1))
    se = math.sqrt((res ** 2).sum() / max(len(x) - 2, 1) / (xm ** 2).sum()) * math.sqrt(len(x) / neff)
    return b, 1.96 * se


def backtest(lat: float, lon: float, i: int, fetch=None) -> dict:
    """Observed 1995-2024 sea-level trend at the nearest NOAA gauge against the AR6 rate for 2005-2030 (all scenarios, likely 17-83 % range)."""
    try:
        st = C.nearest_station(lat, lon, 150.0, fetch)
    except Exception as e:                                             # noqa: BLE001
        return {"ok": False, "reason": f"tide-gauge list unavailable ({str(e)[:80]})"}
    if not st:
        return {"ok": False, "reason": "no NOAA tide gauge within 150 km (the check is available on US coasts; other coasts need PSMSL/UHSLC data)"}
    try:
        a = C.annual(st["id"], 1995, 2024, fetch)
    except Exception as e:                                             # noqa: BLE001
        return {"ok": False, "reason": f"gauge {st['name']} returned no data ({str(e)[:80]})"}
    ys = [y for y in a["years"] if 1995 <= y <= 2024]
    if len(ys) < 20:
        return {"ok": False, "reason": f"gauge {st['name']} has only {len(ys)} complete years since 1995"}
    ms = [a["msl_mm"][a["years"].index(y)] for y in ys]
    slope, ci = _trend(ys, ms)
    r = rsl(i)
    rates = [r[s][2030][q] * 1000.0 / 25.0 for s in r for q in (1, 2, 3)]      # likely range (17-83 %) of every scenario
    lo, hi = min(rates), max(rates)
    overlap = (slope + ci) >= lo and (slope - ci) <= hi
    return {"ok": True, "station": st["name"], "km": st["km"], "obs_mm_yr": slope, "obs_ci": ci, "ar6_range_mm_yr": [lo, hi], "n_years": len(ys), "verdict": "pass" if overlap else "fail",
            "summary": f'Observed {slope:.1f} ± {ci:.1f} mm/yr at {st["name"]} ({len(ys)} years, 1995–2024) against {lo:.1f} to {hi:.1f} mm/yr implied by AR6 for 2005–2030 (likely range, all pathways): ' + ("consistent." if overlap else "NOT consistent (local land motion, a gauge move or a regional ocean anomaly can cause this).")}


def coast_info(lat: float, lon: float, tile_fetch=None) -> dict:
    """Distance to the sea and ground elevation from the open terrain tiles (8 bearings, 2 km steps to 50 km). None distance = inland."""
    el = C.elevation(lat, lon, 11, tile_fetch)
    best = None
    for b in range(0, 360, 45):
        p = C.profile(lat, lon, b, 50.0, 2.0, 10, tile_fetch)
        k = np.where(p <= -1.0)[0]
        if len(k) and (best is None or k[0] * 2.0 < best[0]):
            best = (k[0] * 2.0, b)
    return {"elevation_m": el, "sea_km": (best[0] if best else None), "sea_bearing": (best[1] if best else None)}


def run(lat: float, lon: float, name: str = "Study site", progress=None, fetch=None, tile_fetch=None, backtest_fetch=None) -> dict:
    nb = Notebook(progress)
    nb("1", "Problem", f"How much will the sea rise at {name} ({lat:.3f}, {lon:.3f}) by 2050 and 2100 under each emissions pathway, how sure is that, and what does it mean for the shoreline?")
    ci = coast_info(lat, lon, tile_fetch)
    if ci["sea_km"] is None and ci["elevation_m"] is not None and ci["elevation_m"] > 30:
        raise NoModel(f"{name} is inland (no sea within 50 km and ground at {ci['elevation_m']:.0f} m): sea-level rise does not reach it directly.")
    loc = nearest(lat, lon)
    nb("2", "Location", f'Nearest AR6 projection point: {loc["name"]} ({loc["km"]:.0f} km away, {"a tide gauge, so local land motion is included" if loc["gauge"] else "a one-degree grid point, so local land subsidence is only partly captured"}). Ground elevation {ci["elevation_m"]:.0f} m, sea {"about %.0f km away" % ci["sea_km"] if ci["sea_km"] is not None else "not found within 50 km"}.')
    proj = rsl(loc["i"])
    bench = benchmarks()
    nfail = sum(1 for b in bench if not b["ok"])
    nb("3", "Verification", f"{len(bench) - nfail} of {len(bench)} tests passed: the Bruun formula reproduces its hand value, quantiles are ordered, the rise grows with time and with emissions, and a gauge's own coordinates find that gauge." if not nfail else f"{nfail} test(s) FAILED.", warn=bool(nfail))
    bt = backtest(lat, lon, loc["i"], backtest_fetch)
    nb("4", "Backtest", bt["summary"] if bt["ok"] else "Not possible: " + bt["reason"], warn=not bt["ok"] or bt["verdict"] == "fail")
    status = ("FAILED verification" if nfail else "verified; backtest against the tide gauge FAILED" if bt["ok"] and bt["verdict"] == "fail" else
              "verified and backtested against a tide gauge" if bt["ok"] else "verified only (no tide gauge within 150 km to check against)")
    yrs = (2030, 2050, 2070, 2100)
    rows = []
    for s in ("ssp119", "ssp126", "ssp245", "ssp370", "ssp585"):
        for y in yrs:
            q = proj[s][y]
            rows.append([SCEN_LABEL[s], str(y), f"{q[2]:.2f}", f"{q[1]:.2f} to {q[3]:.2f}", f"{q[0]:.2f} to {q[4]:.2f}"])
    m245_50, m245_00, m585_00, hi585 = proj["ssp245"][2050][2], proj["ssp245"][2100][2], proj["ssp585"][2100][2], proj["ssp585"][2100][4]
    head = [(f"{m245_50:.2f} m", "2050, middle pathway (median)"), (f"{m245_00:.2f} m", "2100, middle pathway (median)"), (f"{m585_00:.2f} m", "2100, very high emissions (median)"), (f"{hi585:.2f} m", "2100, very high emissions, 95th percentile")]
    sandy_note = ("Bruun retreat applies only if this coast is sandy and free to move; rock, cliffs and engineered shores do not retreat this way.")
    near_coast = ci["sea_km"] is not None and ci["sea_km"] <= 2.0
    retreat_rows = []
    if near_coast:
        for s, y in (("ssp245", 2050), ("ssp245", 2100), ("ssp585", 2100)):
            S = proj[s][y][2]
            retreat_rows.append([f"{SCEN_LABEL[s]} {y}", f"{S:.2f}"] + [f"{bruun_retreat(S, t):.0f}" for t in BRUUN_SLOPES])
    years = [2005] + list(yrs)
    series = {SCEN_LABEL[s]: [0.0] + [proj[s][y][2] for y in yrs] for s in ("ssp126", "ssp245", "ssp585")}
    from .. import story as S_
    plt = S_._mpl(); fig, ax = plt.subplots(figsize=(7.4, 3.8))
    cols = {"SSP1-2.6": "#16a34a", "SSP2-4.5": "#0ea5e9", "SSP5-8.5": "#dc2626"}
    for lab, v in series.items():
        ax.plot(years, v, color=cols[lab], lw=2, label=lab)
    s5 = "ssp585"; ax.fill_between(years, [0] + [proj[s5][y][0] for y in yrs], [0] + [proj[s5][y][4] for y in yrs], color="#dc2626", alpha=0.12, lw=0, label="SSP5-8.5, 5–95 %")
    ax.set_ylabel("relative sea level (m, vs 1995–2014)"); ax.set_xlabel("year"); ax.legend(frameon=False, fontsize=9)
    ax.set_title(f"Sea level at {loc['name'].title()}: median by pathway", loc="left", fontsize=11, fontweight="bold", color=S_.INK); fig.tight_layout()
    fig_png = S_._png(fig)
    blocks = [("h2", "question", "The question"), ("p", f"How high will the sea stand at {name} in the decades an asset will live through, and what does that do to the shoreline? Sea level is one of the few climate hazards with a long, physically grounded projection, so the engine uses the IPCC's own numbers and then asks whether they match what the nearest tide gauge has recorded."),
              ("h2", "physics", "The physics and the data"),
              ("p", "Relative sea-level change is the sum of ocean thermal expansion and circulation, melting of mountain glaciers and of the Greenland and Antarctic ice sheets, changes in land water storage, and vertical land motion (sinking or rising ground). The IPCC AR6 combines process models for each term into probability distributions; the engine reads the medium-confidence projections for the nearest point."),
              ("p", "Shoreline retreat from the rise follows the Bruun rule, **R = S / tan β**: a rise S pushes the beach profile up and landward until the same profile shape sits on the higher sea, so retreat is the rise divided by the slope of the active profile."),
              ("h2", "data", "Projected sea-level rise"), ("fig", fig_png, "Median relative sea level by emissions pathway; the band is the 5th to 95th percentile for the highest pathway."),
              ("table", ["Pathway", "Year", "Median (m)", "Likely range, 17–83 % (m)", "Very likely range, 5–95 % (m)"], rows, "Metres of relative sea-level rise since 1995–2014 (centred on 2005), including local land motion where the projection point is a tide gauge. Rise beyond 2100 under high emissions is deeply uncertain because of ice-sheet instability.")]
    if bt["ok"]:
        blocks += [("h2", "backtest", "Does it match the tide gauge?"), ("p", bt["summary"]), ("callout", "The projection is consistent with the observed rate." if bt["verdict"] == "pass" else "The observed rate is outside the projected range: the local record does not match the projection.", "good" if bt["verdict"] == "pass" else "bad")]
    else:
        blocks += [("h2", "backtest", "Does it match the tide gauge?"), ("callout", "Not checked: " + bt["reason"] + ".", "warn")]
    if retreat_rows:
        blocks += [("h2", "retreat", "What it does to the shoreline"), ("table", ["Case", "Rise (m)", "Retreat if steep (1:20), m", "Typical (1:50), m", "Gentle (1:100), m"], retreat_rows, sandy_note)]
    else:
        blocks += [("h2", "retreat", "What it does to the shoreline"), ("p", f"The site is {'not on the open shore' if ci['sea_km'] is None else 'about %.0f km from the sea' % ci['sea_km']}, so the shoreline-retreat rule is not applied to it. " + sandy_note)]
    blocks += [("h2", "result", "What to take from this"), ("p", f"By 2050 the median rise is {m245_50:.2f} m on the middle pathway and by 2100 it is {m245_00:.2f} m, with a plausible high end near {hi585:.1f} m. Permanent flooding of low ground comes slowly; the immediate effect is that every storm surge and high tide starts from a higher base (see the storm surge study).")]
    lim = ["Projections are scenarios, not forecasts; the pathway society follows is not predicted.", "The IPCC medium-confidence distribution excludes low-likelihood ice-sheet collapse outcomes that could add more than a metre by 2100 (its low-confidence storylines are not used here).",
           "A one-degree grid point is not a tide gauge: local subsidence (deltas, groundwater pumping) can add millimetres a year that a grid point misses.", "Ground elevation comes from open terrain tiles (tens of metres horizontal, metre-level vertical): not a survey.", "Attribution: " + ATTRIBUTION]
    return {"domain": "sealevel", "label": "Sea-level rise", "kind": "slr", "name": name, "lat": lat, "lon": lon, "title": f"Sea-level rise at {name}: how much, how sure, and what it does to the shore",
            "standfirst": f"The engine read the IPCC AR6 projections for {name}: {m245_50:.2f} m by 2050 and {m245_00:.2f} m by 2100 on the middle pathway (up to {hi585:.1f} m at the 95th percentile of the highest). " + (bt["summary"] if bt["ok"] else "No tide gauge was close enough to check them.") + f" Status: {status}.",
            "status": status, "ok": nfail == 0, "headline": head, "blocks": blocks, "verification": bench, "sensitivity": [],
            "assumptions": ["Relative sea level at the nearest projection point applies to the site.", "Mean sea level only; surge and tide are in the storm surge study."], "limitations": lim,
            "next_steps": ["Combine with the storm surge study: future extreme water level = today's extreme + the rise.", "Survey the site's elevation against a local vertical datum before any design decision.", "Add local land-motion data (GNSS, InSAR) where subsidence is suspected."],
            "notebook": nb.rows, "numbers": {"source": "IPCC AR6 relative sea level (CC BY 4.0)", "location": loc, "coast": ci, "rsl": {s: {str(y): v for y, v in d.items()} for s, d in proj.items()}, "backtest": bt, "retreat": retreat_rows}, "elapsed_s": round(__import__("time").time() - nb.t0, 1)}
