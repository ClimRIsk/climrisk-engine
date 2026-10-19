"""Climate scenario composer: one pathway + one year (+ a return period) -> the physical inputs the 3D twin, the agent and the reports use.

Everything comes from the engine's own, separately verified results, and each component carries a stated confidence and its limits:
  rain      CMIP6 ensemble change in the annual maximum one-day rainfall at the grid cell (World Bank CCKP, CC0), p10 / median / p90 of the models  [modellab.scenarios]
  river     flood depth scales with flow roughly as Q^0.6 in a wide channel (Manning); flow is taken to scale with the Rx1day ratio, a screening assumption
  sea       IPCC AR6 relative sea-level rise (includes local land motion at tide-gauge sites), from the 2020 reference the mapped coastal floods use  [modellab.slr]
  heat      CMIP6 change in the annual maximum daily temperature
  wind      tropical-cyclone peak wind scaled by the Knutson et al. (2020) expert assessment with the pathway's global warming (low confidence; only where cyclones reach the site)
Nothing here forecasts a particular storm: it moves the level of a mapped, return-period event to what that event would look like in the chosen climate.
"""
from __future__ import annotations

import math
from typing import Optional

from .modellab import futureclimate as FC, scenarios as SC

SSPS = ("ssp126", "ssp245", "ssp370", "ssp585")
YEARS = (2030, 2050, 2070, 2090)
MANNING_EXP = 0.6                     # h ~ Q^(3/5), wide rectangular channel
REF_YEAR = 2020                       # the 'today' reference of the mapped coastal scenarios


def period_of(year: int) -> str:
    for p in SC.PERIODS:
        a, b = (int(x) for x in p.split("-"))
        if a <= year <= b:
            return p
    return SC.PERIODS[0] if year < 2020 else SC.PERIODS[-1]


def warming_at(ssp: str, year: int) -> float:
    """Global-mean warming relative to 1995-2014 (K): linear from 0 in 2015 to the AR6 2081-2100 value, held beyond."""
    w = FC.GMST_2081_2100[ssp]
    return max(0.0, w * min(1.0, (year - 2015) / (2090 - 2015)))


def _rain(lat, lon, ssp, per, fetch):
    fac = SC.change_factors("rain", lat, lon, fetch=fetch, ssps=(ssp,))
    e = (fac["ssp"].get(ssp) or {}).get(per)
    if not e:
        raise RuntimeError("no rainfall projection at this grid cell")
    return {"p10": e["p10"], "median": e["median"], "p90": e["p90"], "baseline_mm_day": fac["baseline_1995_2014"], "cell": fac["cell"]}


def _heat(lat, lon, ssp, per, fetch):
    fac = SC.change_factors("heat", lat, lon, fetch=fetch, ssps=(ssp,))
    e = (fac["ssp"].get(ssp) or {}).get(per)
    return {"p10": e["p10"], "median": e["median"], "p90": e["p90"], "baseline_c": fac["baseline_1995_2014"]} if e else None


def _sea(lat, lon, ssp, year, tile_fetch):
    from .modellab import slr
    try:
        ci = slr.coast_info(lat, lon, tile_fetch)
        if ci["sea_km"] is None and (ci["elevation_m"] or 0) > 30:
            return {"available": False, "why": "inland: no sea within 50 km and ground above 30 m"}
        loc = slr.nearest(lat, lon)
    except Exception as e:                                         # noqa: BLE001
        return {"available": False, "why": str(e)[:160]}
    out = {"available": True, "location": loc["name"], "km": loc["km"], "gauge": loc["gauge"], "sea_km": ci["sea_km"], "elevation_m": ci["elevation_m"], "add_m": {}}
    for lab, qi in (("p17", 1), ("median", 2), ("p83", 3), ("p95", 4)):
        out["add_m"][lab] = slr.uplift(loc["i"], ssp, year, qi) - slr.uplift(loc["i"], ssp, REF_YEAR, qi)
    return out


def _wind(lat, lon, ssp, year):
    try:
        from .climate import tropical_cyclone as TC
        h = TC.site_hazard(lat, lon)
        exposed = (h.annual_rate_ge_ts or 0.0) >= 0.05
        failed = None
    except Exception as exc:                                       # noqa: BLE001
        exposed, failed = False, type(exc).__name__
    w = warming_at(ssp, year)
    med, lo, hi = (FC.KNUTSON["wind"][i] * w / FC.KNUTSON["per"] for i in range(3))
    unknown_note = (f"Cyclone exposure could not be determined here ({failed}), so wind is left unchanged. That is NOT evidence that the "
                    "site is outside cyclone areas.") if failed else None
    return {"tc_exposed": exposed, "tc_status": "unavailable" if failed else "ok", "warming_k": w, "scale": {"median": 1 + med, "low": 1 + lo, "high": 1 + hi} if exposed else {"median": 1.0, "low": 1.0, "high": 1.0},
            "note": unknown_note or "Tropical cyclones reach this site; peak wind scaled by the published assessment (+5 % per 2 °C, range 1-10 %)." if exposed else unknown_note or "Tropical-cyclone winds do not reach this site often enough to scale; other wind extremes are not projected, so wind is unchanged."}


def compose(lat: float, lon: float, ssp: str = "ssp245", year: int = 2070, fetch=None, tile_fetch=None, with_sea: bool = True) -> dict:
    if ssp not in SSPS:
        raise ValueError(f"unknown pathway {ssp!r}; choose one of {SSPS}")
    year = int(max(2025, min(2099, year)))
    per = period_of(year)
    rain = _rain(lat, lon, ssp, per, fetch)
    heat = None
    try:
        heat = _heat(lat, lon, ssp, per, fetch)
    except Exception:                                              # noqa: BLE001
        heat = None
    sea = _sea(lat, lon, ssp, year, tile_fetch) if with_sea else {"available": False, "why": "not requested"}
    wind = _wind(lat, lon, ssp, year)
    fac = {k: rain[k] ** MANNING_EXP for k in ("p10", "median", "p90")}
    conf = [
        {"factor": "Heavier extreme rain", "level": "medium", "why": f"CMIP6 ensemble median {100 * (rain['median'] - 1):+.0f} % (models range {100 * (rain['p10'] - 1):+.0f} to {100 * (rain['p90'] - 1):+.0f} %); delta method on the annual maximum; check the site's backtest in the rain study."},
        {"factor": "River flood depth", "level": "low", "why": "depth change from the rainfall ratio through h ∝ Q^0.6 is a screening rule; catchment response (soil, drainage, snow) is not modelled."},
        {"factor": "Sea-level rise", "level": ("medium-high" if sea.get("available") and sea.get("gauge") else "medium") if sea.get("available") else "n/a",
         "why": ("IPCC AR6 at a tide gauge, so local land motion is included; the engine backtests it against the gauge in the sea-level study." if sea.get("gauge") else "IPCC AR6 at a one-degree grid point: local subsidence is only partly captured.") if sea.get("available") else sea.get("why", "")},
        {"factor": "Cyclone wind", "level": "low" if wind["tc_exposed"] else "n/a", "why": wind["note"]},
        {"factor": "Extreme heat", "level": "medium-high" if heat else "n/a", "why": "CMIP6 change in the annual maximum daily temperature; the site's backtest is in the heat study." if heat else "not available at this cell"},
    ]
    return {"ssp": ssp, "ssp_label": SC.SSP_LABEL[ssp], "year": year, "period": per, "warming_k": warming_at(ssp, year), "rain": rain, "river_depth_factor": fac, "sea": sea, "heat": heat, "wind": wind, "confidence": conf,
            "limits": ["Scenarios are not forecasts: they depend on the emissions pathway, and the p10-p90 range covers model differences only.",
                       "The mapped flood event keeps its return period; the climate moves its level. Changes in how often the event occurs are not applied.",
                       "Wave setup, tide-surge interaction, drainage, defences and changes in catchment land use are not included."],
            "attribution": [SC.ATTRIBUTION, "IPCC AR6 sea-level projections (Fox-Kemper et al. 2021; Garner et al. 2021), CC BY 4.0.", FC.KNUTSON_CITE]}


# ── applying a composed scenario to a 3D twin ────────────────────────────────────────────────────────────────────────────────
def pick_base(twin: dict, kind: Optional[str], rp: int, sea_available: bool) -> Optional[dict]:
    fs = twin["scenarios"]["flood"]
    want = []
    if kind in (None, "auto"):
        want = (["coastal", "river"] if sea_available else ["river", "coastal"])
    else:
        want = [kind]
    for k in want:
        cands = [s for s in fs if s["kind"] == k and (k != "coastal" or (s.get("year") or 0) <= 2020)]
        if cands:
            return min(cands, key=lambda s: abs((s["rp"] or 0) - rp))
    return None


def twin_forcing(twin: dict, comp: dict, rp: int = 100, kind: Optional[str] = None, band: str = "median") -> dict:
    """Which mapped scenario to replay and how many metres to add to its water level, for the composed climate. band: median | high (p83 sea, p90 rain) | low."""
    sea_ok = bool(comp["sea"].get("available"))
    base = pick_base(twin, kind, rp, sea_ok)
    out = {"scenario_id": base["id"] if base else None, "base_label": base["label"] if base else None, "add_m": 0.0, "wind_scale": comp["wind"]["scale"]["high" if band == "high" else "low" if band == "low" else "median"], "derivation": []}
    if not base:
        out["derivation"].append("The twin has no mapped flood scenario to replay; only wind and heat change.")
        return out
    if base["kind"] == "coastal":
        add = comp["sea"]["add_m"]["p83" if band == "high" else "p17" if band == "low" else "median"] if sea_ok else 0.0
        out["add_m"] = round(add, 2)
        out["derivation"].append(f"{base['label']} is replayed with {add:+.2f} m of relative sea-level rise since {REF_YEAR} ({comp['ssp_label']}, {comp['year']}; AR6 {'tide gauge ' + comp['sea']['location'] if comp['sea'].get('gauge') else 'grid point'}).")
    else:
        f = comp["river_depth_factor"]["p90" if band == "high" else "p10" if band == "low" else "median"]
        h = max(base.get("depth_p90_m") or 0.0, base.get("depth_m") or 0.0)
        dry = h < 1.0
        h = max(h, 1.0)                    # where the site itself is dry, the regional water the solver routes in is still about a metre deep for a 1-in-100 river flood: an assumption, said in the derivation
        add = h * (f - 1.0)
        out["add_m"] = round(add, 2)
        out["derivation"].append(f"{base['label']}: rainfall ×{comp['rain']['p90' if band == 'high' else 'p10' if band == 'low' else 'median']:.2f} gives flow ×~{comp['rain']['median'] ** 1.0:.2f} and depth ×{f:.2f} (Q^0.6) on a {h:.1f} m reference depth{' (assumed: the mapped depth at the site is under 1 m, so the regional flood level is used)' if dry else ''}: {add:+.2f} m on the water level.")
    if comp["wind"]["tc_exposed"]:
        out["derivation"].append(f"Cyclone wind ×{out['wind_scale']:.3f} (Knutson et al. 2020, low confidence).")
    return out


def apply_to_twin(twin: dict, comp: dict, rp: int = 100, kind: Optional[str] = None, band: str = "median", run_solver: bool = True) -> dict:
    """Compose -> forcing -> re-run the 2D solver on the twin's terrain (not a depth shift) -> evaluate every structure. Returns what the viewer and the agent need."""
    from . import asset_twin as AT
    fo = twin_forcing(twin, comp, rp, kind, band)
    sim = None
    if fo["scenario_id"] and run_solver and twin["site"].get("hydraulics", True):
        sims = twin.setdefault("sims", {})
        if fo["scenario_id"] not in sims:                          # 'now' must come from the same solver, or the comparison mixes methods
            b = AT.simulate_flood(twin, fo["scenario_id"]); b.pop("block", None); sims[fo["scenario_id"]] = b
        sim = AT.simulate_flood(twin, fo["scenario_id"], add_m=fo["add_m"])
        sim.pop("block", None)
    comps = [c for c in twin["components"] if not c.get("removed") and c.get("on_site", True)]
    if any(c["kind"] != "building" for c in comps):
        comps = [c for c in comps if c["kind"] != "building"]

    def tally(hz):
        return AT.evaluate(comps, twin["site"], hz, None, twin.get("linear"))["tally"]
    base_sc = next((s for s in twin["scenarios"]["flood"] if s["id"] == fo["scenario_id"]), None)
    res = {"forcing": fo, "composed": comp, "sim": sim, "now": None, "future": None, "n": len(comps)}
    if base_sc:
        base_sim = (twin.get("sims") or {}).get(base_sc["id"])
        hz0 = {"flood": ({"comp": base_sim["comp"], "q": base_sc["q"], "wet_share": base_sc["wet_share"]} if base_sim and base_sim.get("comp") and not base_sim.get("failed") else {"q": base_sc["q"], "wet_share": base_sc["wet_share"]})}
        if sim and sim.get("comp") and not sim.get("failed"):
            hz1 = {"flood": {"comp": sim["comp"], "q": base_sc["q"], "wet_share": base_sc["wet_share"]}}
        else:
            hz1 = {"flood": {"q": base_sc["q"], "wet_share": base_sc["wet_share"]}, "what_if": {"flood_add_m": fo["add_m"]}}
        t0, t1 = tally(hz0)["flood"], tally(hz1)["flood"]
        res["now"] = {"equipment_inundated": t0["equipment_inundated"] + t0["submerged"], "wet": t0["wet_floor"]}
        res["future"] = {"equipment_inundated": t1["equipment_inundated"] + t1["submerged"], "wet": t1["wet_floor"], "solver": bool(sim and sim.get("comp"))}
    return res
