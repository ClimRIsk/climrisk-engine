"""Hazards under future climate by the delta method: the site's observed daily weather is kept and only the CMIP6 ensemble's change in annual-mean temperature and precipitation (World Bank CCKP, CC0) is applied,
then the engine's own physics (soil water balance, Canadian Fire Weather Index) is run again on the altered weather. Tropical-cyclone intensity uses the published expert-assessment scaling (Knutson et al. 2020).

What this does and does not capture: warming and the mean rain change, yes. Changes in how rain is distributed through the year, in wind, in dry-spell length, and in humidity beyond the two bounding assumptions, no. Each limitation is stated in the report.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from . import scenarios as SC

SSP_SHOW = ("ssp245", "ssp585")
PER_SHOW = ("2040-2059", "2080-2099")
ET_PER_K = (0.0, 0.03, 0.06)                      # fractional rise of reference evapotranspiration per kelvin of warming: none / central / high (assumption, stated in the report)
# Global-mean warming of 2081-2100 relative to 1995-2014 (IPCC AR6 WG1 Table SPM.1, 1850-1900 values 1.8/2.7/3.6/4.4 minus 0.85); used only for the cyclone scaling.
GMST_2081_2100 = {"ssp126": 0.95, "ssp245": 1.85, "ssp370": 2.75, "ssp585": 3.55}
KNUTSON = {"wind": (0.05, 0.01, 0.10), "rain_rate": 0.14, "cat45_share": 0.13, "per": 2.0}      # per 2 degC of warming: median (range) increase of lifetime-peak wind; rain rate; share of category 4-5 (Knutson et al. 2020, BAMS 101, E303)
KNUTSON_CITE = "Knutson, T. et al. (2020), Tropical Cyclones and Climate Change Assessment: Part II. Projected Response to Anthropogenic Warming, Bull. Amer. Meteor. Soc. 101, E303–E322."
ATTRIBUTION = SC.ATTRIBUTION


def drivers(lat: float, lon: float, fetch=None) -> dict:
    """{ssp: {period: {"dT": median shift, "P": median precipitation ratio, "dT_rng": (p10,p90), "P_rng": (p10,p90)}}} from the CMIP6 ensemble at the grid cell."""
    t = SC.change_factors("tas", lat, lon, fetch=fetch, ssps=SSP_SHOW)
    p = SC.change_factors("pr", lat, lon, fetch=fetch, ssps=SSP_SHOW)
    out = {"cell": t["cell"], "ssp": {}}
    for s in SSP_SHOW:
        out["ssp"][s] = {}
        for per in PER_SHOW:
            a, b = t["ssp"].get(s, {}).get(per), p["ssp"].get(s, {}).get(per)
            if a and b:
                out["ssp"][s][per] = {"dT": a["median"], "P": b["median"], "dT_rng": (a["p10"], a["p90"]), "P_rng": (b["p10"], b["p90"])}
    return out


def _rows(dr, fn, fmtfn):
    rows, nums = [], []
    for s in SSP_SHOW:
        for per in PER_SHOW:
            d = dr["ssp"].get(s, {}).get(per)
            if not d:
                continue
            r = fn(d)
            rows.append([SC.SSP_LABEL[s], per, f'{d["dT"]:+.1f} °C', f'{(d["P"] - 1) * 100:+.0f} %'] + fmtfn(r))
            nums.append({"ssp": s, "period": per, "dT": d["dT"], "P": d["P"], **r})
    return rows, nums


# ── drought ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def drought_projection(sim_fn, P, E0, doy, burn: int, dr: dict) -> dict:
    """sim_fn(P, E0, doy) -> soil moisture series (the fitted model). Baseline drought = days below the 10th percentile of the simulated baseline. Returns rows for the report."""
    base = sim_fn(P, E0, doy)[burn:]
    thr = float(np.percentile(base, 10)); m0 = float(base.mean())

    def fn(d):
        out = {}
        for k in ET_PER_K:
            sim = sim_fn(P * d["P"], E0 * (1.0 + k * d["dT"]), doy)[burn:]
            out[k] = (float(sim.mean() - m0), float((sim < thr).mean()))
        return {"by_et": {str(k): v for k, v in out.items()}}

    def fm(r):
        c = r["by_et"][str(ET_PER_K[1])]; lo = r["by_et"][str(ET_PER_K[0])]; hi = r["by_et"][str(ET_PER_K[2])]
        return [f'{c[0]:+.3f}', f'{100 * c[1]:.0f} % ({100 * lo[1]:.0f}–{100 * hi[1]:.0f} %)']
    rows, nums = _rows(dr, fn, fm)
    return {"rows": rows, "numbers": nums, "baseline_mean": m0, "threshold": thr}


# ── fire weather ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def fire_projection(temp, rh, wind_kmh, rain, months, years, dr: dict, tropical: bool) -> dict:
    """Re-run the Canadian FWI on altered weather. Two bounds for humidity: relative humidity unchanged, and specific humidity unchanged (relative humidity falls as the air warms)."""
    from . import fire, heat
    temp = np.asarray(temp, float); rh = np.asarray(rh, float); rain = np.asarray(rain, float); yrs = np.asarray(years)

    def summary(t, h, r):
        f = np.array(fire.run_series(t.tolist(), np.clip(h, 1, 100).tolist(), wind_kmh, r.tolist(), months, tropical=tropical)["fwi"])
        keep = yrs > yrs.min()
        ann = [float(f[yrs == y].max()) for y in sorted(set(yrs[keep].tolist()))]
        d30 = [int((f[yrs == y] >= 30).sum()) for y in sorted(set(yrs[keep].tolist()))]
        return float(np.mean(ann)), float(np.mean(d30))
    b = summary(temp, rh, rain)

    def fn(d):
        t2 = temp + d["dT"]; r2 = rain * d["P"]
        rh_q = rh * heat.sat_vapour_pressure_hpa(temp) / heat.sat_vapour_pressure_hpa(t2)
        a = summary(t2, rh, r2); c = summary(t2, rh_q, r2)
        return {"ann_const_rh": a[0], "ann_const_q": c[0], "d30_const_rh": a[1], "d30_const_q": c[1]}

    def fm(r):
        lo, hi = sorted((r["ann_const_rh"], r["ann_const_q"])); l2, h2 = sorted((r["d30_const_rh"], r["d30_const_q"]))
        return [f"{lo:.0f}–{hi:.0f}", f"{l2:.0f}–{h2:.0f}"]
    rows, nums = _rows(dr, fn, fm)
    return {"rows": rows, "numbers": nums, "baseline_annual_max": b[0], "baseline_days30": b[1]}


# ── tropical cyclone ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def cyclone_scaling(levels: dict) -> dict:
    """Design winds scaled by the Knutson et al. (2020) median (and range) change of lifetime-peak wind per 2 degC of warming, using AR6 global warming of each pathway at 2081-2100 (linear)."""
    rows, nums = [], []
    for s, w in GMST_2081_2100.items():
        f, lo, hi = (KNUTSON["wind"][i] * w / KNUTSON["per"] for i in range(3))
        row = [SC.SSP_LABEL[s], f"{w:+.1f} °C", f"{f * 100:+.1f} % ({lo * 100:+.1f} to {hi * 100:+.1f} %)"]
        for T in (50, 100):
            v = levels[T]
            row.append(f"{v * (1 + f):.0f} ({v * (1 + lo):.0f}–{v * (1 + hi):.0f}) m/s")
        rows.append(row); nums.append({"ssp": s, "warming": w, "wind_change": f, "range": (lo, hi)})
    return {"rows": rows, "numbers": nums, "rain_rate": KNUTSON["rain_rate"] * 1.0, "cite": KNUTSON_CITE}
