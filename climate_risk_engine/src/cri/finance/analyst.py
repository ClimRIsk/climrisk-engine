"""
finance/analyst.py — the Risk Analyst agent: one asset (or a company's assets) through
four stages, each grounded in data the engine holds locally or reads from open sources.

  1. Physical risk   cyclone (IBTrACS + Holland), river flood (JRC 90 m, RP 10–500),
                     coastal flood (Deltares 90 m, 2018 & 2050 sea level), flood
                     defences (FLOPROS) + the satellite site analysis when available
  2. Policy          carbon-pricing instruments in force / in the pipeline for the
                     asset's state/province and country and its sector (World Bank)
  3. Transition      NGFS Phase 5 carbon price and sector-activity paths, 7 scenarios,
                     3 models, combined with today's policy design
  4. Financial       Monte Carlo loss distribution to 2050 (physical damage, carbon
                     cost, stranded capital) — mean, median, 95th/99th percentiles,
                     expected shortfall, as money and as % of asset value

It then writes findings and actions ("this asset has this problem; it can be resolved
through this"). Every number comes from the engine; the optional LLM (Groq) only turns
the structured result into prose and never invents figures.
"""
from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Optional

import numpy as np

from . import abatement, asset_value, earnings, geo, policy, transition, vulnerability
from .distribution import PASS_THROUGH, STRANDING_KEYS, simulate
from .flood_hazard import site_flood

MAIN_SCENARIOS = ["Net Zero 2050", "Delayed transition", "Current Policies"]


def _fmt_usd(x: Optional[float]) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    a = abs(x)
    if a >= 1e9:
        return f"${x/1e9:,.2f}bn"
    if a >= 1e6:
        return f"${x/1e6:,.1f}m"
    if a >= 1e3:
        return f"${x/1e3:,.0f}k"
    return f"${x:,.0f}"


# ── register lookups ─────────────────────────────────────────────────────────

def _register():
    from ..data.asset_register import _load
    return _load()


def nearest_register_asset(lat: float, lon: float, max_km: float = 1.0) -> Optional[int]:
    R = _register()
    d = np.hypot((R["lat"] - lat) * 111.0, (R["lon"] - lon) * 111.0 * math.cos(math.radians(lat)))
    i = int(np.argmin(d))
    return i if d[i] <= max_km else None


def register_asset(i: int) -> dict:
    R = _register()
    f = lambda a: None if not np.isfinite(a) else float(a)      # noqa: E731
    return {"register_index": int(i), "name": str(R["name"][i]), "lat": float(R["lat"][i]), "lon": float(R["lon"][i]),
            "iso3": str(R["countries"][R["country"][i]]), "sector": str(R["sectors"][R["sector"][i]]),
            "atype": str(R["atype"][i]), "capacity": f(R["capacity"][i]), "emissions_t": f(R["emissions_t"][i]),
            "source": str(R["source"][i]), "elev_m": f(R["elev_m"][i]),
            "tc_rate_ts": float(R["tc_rate_ts"][i]), "tc_rp100_ms": float(R["tc_rp100_ms"][i]),
            "owners": _owners_of(int(i))}


@lru_cache(maxsize=1)
def _owner_index() -> dict:
    R = _register()
    idx: dict = {}
    for k, c in R["_companies"].items():
        for j in c["assets"]:
            idx.setdefault(j, []).append(c["name"])
    return idx


def _owners_of(i: int) -> list:
    return _owner_index().get(i, [])[:5]


# ── stage 1: physical ────────────────────────────────────────────────────────

def _tc_params(lat: float, lon: float, pre: Optional[dict] = None) -> dict:
    """λ (storms/yr ≥ 17.5 m/s) and σ of the exponential wind tail, from the site model."""
    if pre and pre.get("tc_rate_ts") is not None:
        lam, v100, worst, top = pre["tc_rate_ts"], pre["tc_rp100_ms"], None, []
    else:
        from ..climate.tropical_cyclone import site_hazard
        hz = site_hazard(lat, lon)
        lam, v100 = hz.annual_rate_ge_ts, float(hz.rp_wind_ms.get(100, 0.0))
        worst, top = hz.worst, hz.top_events[:3]
    sigma = (v100 - 17.5) / math.log(lam * 100) if lam * 100 > 1.0 and v100 > 17.5 else None
    return {"lam": float(lam), "sigma": sigma, "rp100_ms": v100, "worst": worst, "top_events": top,
            "category_100yr": _saffir(v100)}


def _saffir(v: float) -> str:
    lab = "below tropical-storm force"
    for thr, name in [(17.5, "tropical storm"), (33, "Cat 1"), (43, "Cat 2"), (50, "Cat 3"), (58, "Cat 4"), (70, "Cat 5")]:
        if v >= thr:
            lab = name
    return lab


def _flood_curves(fl: dict, cont: str) -> tuple[dict, dict]:
    riv, cst = fl.get("river") or {}, fl.get("coastal") or {}
    river = {"available": bool(riv.get("available")), "source": riv.get("source"),
             "note": riv.get("note") or riv.get("reason")}
    if riv.get("available") and not riv.get("footprint_on_water"):
        rps = riv["rps"]
        dep = [riv["depths"][str(rp)] for rp in rps]
        river.update(rps=rps, mean_depth_m=[round(float(np.mean(d)), 2) if d else 0.0 for d in dep],
                     wet_share=[round(float(np.mean(np.array(d) > 0.05)), 2) if d else 0.0 for d in dep],
                     damage=[round(float(np.mean(vulnerability.flood_damage(d, cont))), 4) if d else 0.0 for d in dep])
    coast = {"available": bool(cst.get("available")), "source": cst.get("source"), "note": cst.get("reason")}
    if cst.get("available"):
        rps = cst["rps"]
        for y, tag in ((2018, ""), (2050, "_2050")):
            dep = [cst["depths"].get(f"{y}_{rp}") or [] for rp in rps]
            coast["mean_depth_m" + tag] = [round(float(np.mean(d)), 2) if d else 0.0 for d in dep]
            coast["damage" + tag] = [round(float(np.mean(vulnerability.flood_damage(d, cont))), 4) if d else 0.0 for d in dep]
        coast["rps"] = rps
    return river, coast


def _severity(aal_pct: float, rp100_ratio: float) -> str:
    if aal_pct >= 1.0 or rp100_ratio >= 0.4:
        return "HIGH"
    if aal_pct >= 0.2 or rp100_ratio >= 0.15:
        return "MODERATE"
    if aal_pct >= 0.02 or rp100_ratio >= 0.03:
        return "LOW"
    return "NEGLIGIBLE"


# ── actions ──────────────────────────────────────────────────────────────────

_ABATEMENT = {
    "power:coal": "Set a dated retirement or conversion plan (biomass/ammonia co-firing only as a bridge); "
                  "contract replacement renewables + storage; evaluate CCS only where storage is nearby.",
    "power:gas": "Improve heat rate, prepare for hydrogen blending, and weigh CCS or conversion to peaking duty; "
                 "hedge carbon exposure through power contracts with carbon pass-through clauses.",
    "power:oil": "Switch to gas or renewables-plus-storage; oil-fired generation carries the highest cost per MWh.",
    "cement": "Cut clinker ratio (calcined clay LC3, slag, fly ash), raise alternative-fuel share, and prepare "
              "CCS readiness for process emissions; set up product-level embedded-emissions MRV for CBAM.",
    "steel:BF/BOF": "Plan transition to scrap-EAF or DRI (gas → hydrogen) at the next relining; secure scrap and "
                    "renewable power; build product-level MRV for CBAM.",
    "steel:EAF": "Contract renewable power and raise scrap quality — EAF emissions are mostly from electricity.",
    "steel:DRI": "Prepare for hydrogen-based DRI; source low-carbon power.",
    "refining": "Electrify process heat, add CCS on hydrogen/SMR units, and plan throughput reduction in line "
                "with falling oil demand.",
    "chemicals:ammonia": "Capture the concentrated CO2 stream from ammonia production (lowest-cost CCS) and "
                         "prepare for electrolytic hydrogen feed; CBAM applies to fertilisers.",
    "aluminium:smelting": "Secure low-carbon power (PPAs) — power is the dominant emissions source; CBAM applies.",
}


def _asset_life(commissioned: Optional[int], retire: Optional[int], act_key: Optional[str]):
    """Validated plant-life input for the simulation and a note on what it means."""
    from .distribution import DEFAULT_DESIGN_LIFE, DESIGN_LIFE
    if not commissioned and not retire:
        return None, None
    now = int(transition.YEARS[0])
    if commissioned is not None and not (1850 <= int(commissioned) <= now):
        raise ValueError(f"commissioned_year must be between 1850 and {now}")
    if retire is not None and not (now <= int(retire) <= 2100):
        raise ValueError(f"retirement_year must be between {now} and 2100")
    if commissioned and retire and int(retire) <= int(commissioned):
        raise ValueError("retirement_year must be after commissioned_year")
    d = DESIGN_LIFE.get(act_key or "", DEFAULT_DESIGN_LIFE)
    parts = []
    if commissioned:
        parts.append(f"Commissioned {int(commissioned)}: design life sampled {d[0]}–{d[2]} years (mode {d[1]}, analyst judgement for "
                     "the sector), which sets the remaining life and undepreciated value used for stranding; the plant keeps operating "
                     "unless a retirement year is given (refurbishment past design life is common).")
    if retire:
        parts.append(f"Retirement fixed at {int(retire)} (legal phase-out or planned closure): no emissions or earnings after it.")
    return {"commissioned": int(commissioned) if commissioned else None, "retire": int(retire) if retire else None,
            "design": d}, " ".join(parts)


def _cbam(share: Optional[float], pol: dict, sector: str, atype: str, ckey: Optional[str]):
    """EU CBAM exposure on exports: only for in-scope products made outside the EU ETS area."""
    if share is None:
        return None, ("Product is in EU CBAM scope but the share exported to the EU is not given, so the border charge is not "
                      "modelled — give export_share_eu (0–1)." if pol.get("cbam_exporter") else None)
    if not (0 <= float(share) <= 1):
        raise ValueError("export_share_eu must be between 0 and 1")
    if not pol.get("cbam_exporter"):
        return None, "export_share_eu ignored: the asset is inside the EU ETS area or its product is outside CBAM scope."
    if not share:
        return None, None
    eu = next((i for i in policy._wb()["instruments"] if i["id"] == "ETS_EU"), None)
    p_now = (policy._price_latest(eu)[0] if eu else None) or 0.0
    drv = transition.drivers("DEU", sector, atype, ckey)
    return ({"share": float(share), "price": drv["price"], "p_now": float(p_now)},
            f"CBAM: {100 * float(share):.0f}% of emissions are embedded in EU exports and pay the EU carbon price less the price paid at "
            f"home, on the share not covered by free allocation (2.5% in 2026 rising to 100% in 2034); EU price = max of today's ETS "
            f"price (${p_now:.0f}/t) and the NGFS EU path.")


def _actions(res: dict, cost_key: Optional[str], act_key: Optional[str]) -> list:
    acts = []
    ph = res["physical"]
    fin = res["financial"]
    tc = ph["cyclone"]
    riv, cst = ph["river_flood"], ph["coastal_flood"]
    hz = fin["physical"]["by_hazard_aal_pct"] if fin else {}
    if tc["severity"] in ("MODERATE", "HIGH"):
        acts.append({"hazard": "tropical_cyclone", "problem": f"1-in-100-year wind {tc['rp100_ms']:.0f} m/s ({tc['category_100yr']}) "
                     f"— modelled average annual damage {hz.get('cyclone', 0):.2f}% of value.",
                     "action": "Verify roof, cladding and door ratings against the 1-in-100 wind; anchor rooftop and "
                               "yard equipment; storm-mode shutdown procedure; review windstorm insurance limits and deductibles."})
    if riv.get("severity") in ("MODERATE", "HIGH"):
        i100 = riv["rps"].index(100) if 100 in riv.get("rps", []) else -1
        d100 = riv["mean_depth_m"][i100] if i100 >= 0 else None
        acts.append({"hazard": "river_flood", "problem": f"Undefended 1-in-100 river flood ≈ {d100} m over the site; local "
                     f"defence standard ≈ 1-in-{riv['protection_rp']:.0f} years (FLOPROS).",
                     "action": f"Raise or wet-proof critical equipment (switchgear, MCCs, control rooms) above "
                               f"{(d100 or 0) + 0.5:.1f} m; demountable barriers at openings; flood emergency plan; confirm "
                               "the actual defence standard with the local water authority."})
    if cst.get("severity") in ("MODERATE", "HIGH"):
        acts.append({"hazard": "coastal_flood", "problem": "Storm-surge flooding modelled at the site; sea-level rise to 2050 "
                     "raises depths and frequency.", "action": "Plan for the 2050 1-in-100 surge level in any "
                     "capex on low-lying plant; check the design level of seawalls/levees; relocate critical stock."})
    sc = fin["scenarios"].get("Delayed transition") or next(iter(fin["scenarios"].values()), None) if fin else None
    if sc and res["transition"]["emissions_t"]:
        tn = sc["transition_npv"]["mean"]
        v = res["asset"]["value"]["mode"]
        ab = sc.get("abatement") or {}
        if v and tn > 0.02 * v or tn > 5e6:
            problem = (f"Carbon cost to 2050 (delayed transition, present value) {_fmt_usd(tn)} if nothing is done"
                       + (f" ({100 * tn / v:.0f}% of asset value)" if v else ""))
            if ab.get("enabled"):
                rc = ab["response_cost_npv"]["mean"]
                problem += f"; with the sector's abatement levers deployed as prices rise {_fmt_usd(rc)} ({_fmt_usd(tn - rc)} avoided)"
            acts.append({"hazard": "carbon_cost", "problem": problem + ".",
                         "action": _ABATEMENT.get(cost_key or "", "Build a site decarbonisation plan (efficiency, "
                                                  "electrification, fuel switch) and price carbon internally at the NGFS path.")})
        if ab.get("enabled"):
            ls = [l for l in ab["levers"] if l.get("median_start_year")]
            if ls:
                acts.append({"hazard": "abatement", "problem": "Modelled response: " + "; ".join(
                    f"{l['name']} starts about {l['median_start_year']} ({100 * l['prob_started_by_2040']:.0f}% of runs by 2040)"
                    for l in ls[:3]) + ".",
                    "action": "Get site-specific quotes for these levers (cost per tonne, lead time) and rerun with them in "
                              "place of the library ranges; start permitting and procurement now for the long-lead options."})
        e = sc.get("earnings")
        if e and e.get("covenants") and e["covenants"]["by_horizon"].get("2040", 0) > 0.1:
            acts.append({"hazard": "covenants", "problem": f"Covenant breach probability by 2040 "
                         f"{100 * e['covenants']['by_horizon']['2040']:.0f}% (delayed transition).",
                         "action": "Renegotiate leverage and coverage covenants with climate-adjusted headroom, or phase "
                                   "abatement capex ahead of the price path."})
        if sc.get("stranded") and sc["stranded"]["mean"] > 0:
            acts.append({"hazard": "stranding", "problem": f"Sector output falls under transition scenarios — expected "
                         f"stranded capital {_fmt_usd(sc['stranded']['mean'])} (delayed transition).",
                         "action": "Shorten depreciation lives, avoid life-extension capex, and test repurposing "
                                   "(storage, grid services, conversion) before the next major outage."})
    if res["policy"].get("cbam_exporter"):
        acts.append({"hazard": "cbam", "problem": "Exports to the EU face the CBAM carbon charge from 2026 (payable 2027).",
                     "action": "Measure verified embedded emissions per product now; defaults are punitive. Domestic "
                               "carbon prices paid are deductible."})
    return acts


# ── orchestrator ─────────────────────────────────────────────────────────────

def _water(lat: float, lon: float, sector: Optional[str]) -> Optional[dict]:
    """WRI Aqueduct 4.0 context (not monetised in the loss distribution)."""
    try:
        from ..connectors.wri_aqueduct import point, stress_path
        r = point(lat, lon, sector)
        if not r.get("found"):
            return None
        b = r["baseline"]
        return {"water_stress": b["bws"], "water_depletion": b["bwd"], "drought": b["drr"],
                "groundwater_decline": b["gtd"], "overall": r["overall_water_risk"],
                "water_stress_path_bau": stress_path(r, "bau"), "admin1": r["admin1"], "source": r["source"],
                "note": "Context only — water stress is not monetised in the loss distribution."}
    except Exception:
        return None


def analyse_asset(lat: float, lon: float, name: Optional[str] = None, sector: Optional[str] = None,
                  atype: Optional[str] = None, capacity: Optional[float] = None, emissions_t: Optional[float] = None,
                  value_usd: Optional[float] = None, iso3: Optional[str] = None, register_index: Optional[int] = None,
                  site: Optional[dict] = None, radius_m: float = 250.0, discount: float = 0.07, n: int = 4000,
                  narrative_llm: bool = False, _flood: Optional[dict] = None, _common: Optional[dict] = None,
                  financials: Optional[dict] = None, abatement_on: bool = True, levers: Optional[list] = None,
                  commissioned_year: Optional[int] = None, retirement_year: Optional[int] = None,
                  export_share_eu: Optional[float] = None) -> dict:
    t0 = time.time()
    steps = []
    reg = None
    if register_index is None and not sector:
        register_index = nearest_register_asset(lat, lon)
    if register_index is not None:
        reg = register_asset(register_index)
        reg["match_m"] = round(1000.0 * math.hypot((reg["lat"] - lat) * 111.0, (reg["lon"] - lon) * 111.0 * math.cos(math.radians(lat))))
        name = name or reg["name"]
        sector = sector or reg["sector"]
        atype = atype if atype is not None else reg["atype"]
        capacity = capacity if capacity is not None else reg["capacity"]
        emissions_t = emissions_t if emissions_t is not None else reg["emissions_t"]
    adm = geo.admin_at(lat, lon)
    iso3 = iso3 or (reg or {}).get("iso3") or adm.get("iso3")
    cont, tcr = geo.continent(iso3), geo.tc_region(iso3)
    ckey = asset_value.cost_key(sector or "", atype or "")
    steps.append({"stage": "identify", "text": f"{name or 'Site'} — {sector or 'sector unknown'} {atype or ''} in "
                  f"{adm.get('name') or '?'}, {iso3 or '?'}" + (f"; matched register asset ({reg['source']})" if reg else "")})

    # 1 physical
    tc = _tc_params(lat, lon, reg)
    fl = _flood if _flood is not None else site_flood(lat, lon, radius_m, elev_m=(reg or {}).get("elev_m"))
    river, coast = _flood_curves(fl, cont)
    river["protection_rp"] = float(adm.get("prot_river_rp") or 0)
    coast["protection_rp"] = float(adm.get("prot_coast_rp") or 0)
    steps.append({"stage": "physical", "text": "Cyclone event set, JRC river and Deltares coastal flood depths, "
                  "FLOPROS defences" + (", satellite site analysis" if site else "")})

    # 2 policy
    pol = policy.applicable(iso3, adm.get("name"), sector or "", atype or "", ckey)
    steps.append({"stage": "policy", "text": f"{len(pol['in_force'])} instrument(s) in force, {len(pol['pipeline'])} in "
                  f"the pipeline; effective price ${pol['effective_price_usd_t']}/t"})

    # 3 transition
    drv = transition.drivers(iso3, sector or "", atype or "", ckey)
    cs_path = policy.charged_share_path(pol["charged_share"], transition.YEARS)
    act_key = transition.activity_key(sector or "", atype or "", ckey)
    pt_key = ("el" if pol["policy_sector"] == "el" else "refining" if ckey == "refining" else "cement" if ckey == "cement"
              else "steel" if (ckey or "").startswith("steel") else "chemicals" if (ckey or "").startswith(("chemicals", "petro"))
              else pol["policy_sector"])
    central = transition.central_paths(drv, pol["effective_price_usd_t"])
    steps.append({"stage": "transition", "text": f"NGFS Phase 5: {len(drv['price'])} scenarios × {len(drv['models'])} "
                  f"models for {iso3 or 'World'}" + (f"; activity = {drv['activity_variable']}" if drv['activity_variable'] else "")})

    # 4 financial
    val = asset_value.estimate(sector or "", atype or "", capacity, emissions_t, iso3, value_usd)
    fin_in = earnings.normalise(financials, act_key) if financials else None        # raises FinancialsError (a ValueError)
    ab_lib = abatement.library_for(act_key, ckey)
    ab_levers = abatement.user_levers(levers) if levers else ab_lib["levers"]
    ab_cfg = {"enabled": bool(abatement_on and emissions_t and ab_levers), "levers": ab_levers}
    life_in, life_note = _asset_life(commissioned_year, retirement_year, act_key)
    cbam_in, cbam_note = _cbam(export_share_eu, pol, sector or "", atype or "", ckey)
    case = {"value": val if val.get("mode") else None, "financials": fin_in, "asset_life": life_in,
            "tc": {"lam": tc["lam"], "sigma": tc["sigma"], "region": tcr},
            "river": {"rps": river.get("rps"), "damage": river.get("damage"), "protection": river["protection_rp"]}
            if river.get("damage") else None,
            "coast": {"rps": coast.get("rps"), "damage": coast.get("damage"), "damage_2050": coast.get("damage_2050"),
                      "protection": coast["protection_rp"]} if coast.get("damage") else None,
            "transition": {"emissions_t": emissions_t or 0.0, "explicit_price": pol["effective_price_usd_t"],
                           "charged_share_path": cs_path, "drivers": drv, "pass_through_key": pt_key,
                           "activity_key": act_key, "abatement": ab_cfg, "cbam": cbam_in}}
    fin = simulate(case, n=n, discount=discount, common=_common)
    steps.append({"stage": "financial", "text": f"Monte Carlo {n:,} draws × 25 years, discount {discount:.0%} real"})

    # severities from the simulation
    ph = fin["physical"]
    by = ph["by_hazard_aal_pct"]

    def rp100_ratio(key):
        h = case.get(key) or {}
        if not h.get("rps"):
            return 0.0
        d = np.interp(np.log(100), np.log(h["rps"]), h["damage"])
        return float(d) if 100 > (h.get("protection") or 0) else 0.0
    vh = vulnerability.V_HALF[tcr][0]
    tc["severity"] = _severity(by.get("cyclone", 0.0), float(vulnerability.tc_damage(tc["rp100_ms"], vh)))
    # "assessed" is False when no flood map was read for this site (not fetched, unavailable, footprint on water, or above the
    # coastal cut-off). Severity stays NEGLIGIBLE for compatibility, but it means "no loss modelled", not "low risk".
    river["assessed"] = bool(river.get("damage"))
    coast["assessed"] = bool(coast.get("damage"))
    river["severity"] = _severity(by.get("river", 0.0), rp100_ratio("river")) if river["assessed"] else "NEGLIGIBLE"
    coast["severity"] = _severity(by.get("coast", 0.0), rp100_ratio("coast")) if coast["assessed"] else "NEGLIGIBLE"

    res = {
        "asset": {"name": name, "lat": lat, "lon": lon, "iso3": iso3, "admin1": adm.get("name"), "sector": sector,
                  "type": atype, "capacity": capacity, "emissions_t": emissions_t, "owners": (reg or {}).get("owners"),
                  "register_source": (reg or {}).get("source"), "register_match": None if not reg else {"name": reg["name"], "distance_m": reg.get("match_m")}, "value": val},
        "physical": {"cyclone": tc, "river_flood": river, "coastal_flood": coast,
                     "defences": {"river_rp": river["protection_rp"], "coastal_rp": coast["protection_rp"],
                                  "source": adm.get("source")},
                     "site_analysis": None if not site else {"headline": site.get("headline"),
                                                              "risk_signal": site.get("risk_signal"),
                                                              "findings": [{"hazard": f["hazard"], "severity": f["severity"],
                                                                            "summary": f.get("summary")} for f in site.get("findings", [])]},
                     "vulnerability": vulnerability.describe(cont, tcr),
                     "water": _water(lat, lon, sector)},
        "policy": pol,
        "transition": {"emissions_t": emissions_t, "scenarios": central, "models": drv["models"],
                       "regions": drv["regions"], "activity_variable": drv["activity_variable"],
                       "abatement": {"enabled": ab_cfg["enabled"], "library": ab_lib["key"], "note": ab_lib["note"],
                                     "user_levers": bool(levers),
                                     "levers": [{k: l[k] for k in ("name", "share", "mac", "lead", "avail", "basis", "source")}
                                                for l in ab_levers]},
                       "plant_life": {"commissioned": commissioned_year, "retirement": retirement_year,
                                      "design_life": (life_in or {}).get("design"), "note": life_note},
                       "cbam": {"export_share_eu": export_share_eu if cbam_in else None, "note": cbam_note},
                       "assumptions": {"emissions": ("abated when the carbon price the plant faces reaches a lever's levelised cost "
                                                     "(sector levers, lead times, irreversible); the unabated cost is shown as cost of inaction"
                                                     if ab_cfg["enabled"] else
                                                     "held at today's level unless sector output falls (no abatement response modelled)"),
                                       "future_charged_share": transition.FUTURE_CS,
                                       "pass_through": PASS_THROUGH.get(pt_key, PASS_THROUGH["ind"]),
                                       "stranding": act_key in STRANDING_KEYS}},
        "financial": fin,
        "steps": steps,
    }
    res["actions"] = _actions(res, ckey, act_key)
    res["headline"] = _headline(res)
    res["narrative"] = _narrative(res)
    if narrative_llm:
        res["narrative_llm"] = _llm_narrative(res)
    res["sources"] = ["NOAA IBTrACS v04r01", "JRC CEMS-GloFAS river flood hazard maps v2.1",
                      "Deltares Global Flood Maps (Planetary Computer)", "FLOPROS v1 (Scussolini et al. 2016)",
                      "Eberenz et al. 2021 / Emanuel 2011 wind damage", "JRC depth-damage (Huizinga et al. 2017)",
                      "World Bank Carbon Pricing Dashboard (May 2026)", "NGFS Phase 5 via IIASA",
                      "Climate TRACE v6 / EPA GHGRP 2023 (asset, owners, emissions)"]
    res["caveats"] = [
        "Screening-grade model: validate asset values, defence standards and emissions with the owner before decisions.",
        "Direct physical damage only — business interruption and supply-chain losses are not included.",
        "Cyclone and river-flood frequencies are historical (no climate-change uplift); coastal depths include "
        "sea-level rise to 2050.",
        "NGFS carbon prices are model shadow prices; how much reaches a plant depends on policy design (sampled).",
    ]
    if ab_cfg["enabled"]:
        res["caveats"].append("Abatement uses sector cost ranges from published studies (some analyst judgement, flagged per lever), "
                              "held flat in real terms; the operator reacts to the price it sees and cannot foresee the scenario. "
                              "Replace with site quotes where you have them.")
    elif emissions_t and not ab_cfg["enabled"]:
        res["caveats"].append("No abatement response included (" + (ab_lib["note"] or "switched off") + ") — results are the cost of inaction.")
    for note in (life_note, cbam_note):
        if note:
            res["caveats"].append(note)
    if not life_in and act_key in STRANDING_KEYS:
        res["caveats"].append("Plant age unknown: remaining life is sampled (10–30 years) and the undepreciated share of "
                              "value 15–60%. Give the commissioning year (and any legal retirement year) to replace both.")
    if fin_in:
        res["caveats"].append("Earnings layer: financials must cover the same perimeter as the emissions; ratings and default "
                              "probabilities are a coverage-based proxy, not an internal credit rating; debt refinancing is not modelled.")
        res["earnings_input"] = {k: v for k, v in fin_in.items()}
    else:
        res["caveats"].append("No financials given: EBITDA, leverage, covenant and credit effects are not computed.")
    if not emissions_t:
        res["caveats"].insert(0, "Transition (carbon-cost) loss not assessed: no emissions known for this asset — "
                                 "give emissions_t, or capacity for a register-type estimate.")
        res["headline"] = res["headline"].rstrip(".") + ". Physical risk only — no emissions given, so carbon cost is not included."
    if not (res["asset"].get("value") or {}).get("mode"):
        res["caveats"].insert(0, "No asset value: money figures are zero — give value_usd (or capacity for an estimate).")
    res["elapsed_s"] = round(time.time() - t0, 1)
    return res


def _headline(r: dict) -> str:
    a, fin = r["asset"], r["financial"]
    sc = fin["scenarios"].get("Delayed transition") or next(iter(fin["scenarios"].values()), {})
    v = a["value"].get("mode")
    tot = sc.get("total_npv", {})
    parts = [f"{a['name'] or 'This asset'}"]
    ph = fin["physical"]
    if not v and not a.get("emissions_t"):
        parts.append(f"average annual physical damage {ph['aal_pct']:.2f}% of asset value, 1-in-100-year annual damage "
                     f"{100 * ph['annual_damage_ratio_rp']['100']:.1f}% — no emissions or value on record here: enter "
                     "the asset value and annual emissions for money figures and the transition analysis")
    elif v:
        parts.append(f"expected climate loss to 2050 {_fmt_usd(tot.get('mean'))} ({tot.get('pct_of_value', {}).get('mean', 0):.1f}% of "
                     f"{_fmt_usd(v)} value) under a delayed transition; 1-in-20 outcome {_fmt_usd(tot.get('p95'))}")
    else:
        parts.append(f"carbon cost to 2050 {_fmt_usd(sc.get('transition_npv', {}).get('mean'))} (present value, delayed transition)")
    top = max([("cyclone", r["physical"]["cyclone"]["severity"]), ("river flood", r["physical"]["river_flood"]["severity"]),
               ("coastal flood", r["physical"]["coastal_flood"]["severity"])],
              key=lambda x: ["NEGLIGIBLE", "LOW", "MODERATE", "HIGH"].index(x[1]))
    if top[1] != "NEGLIGIBLE":
        parts.append(f"highest of the three hazards priced in this loss (cyclone, river flood, coastal flood): {top[0]} ({top[1]})")
    # hazards the site analysis rates but this loss does not price must not vanish behind a 0.00% headline
    sa = (r["physical"].get("site_analysis") or {}).get("findings") or []
    priced = {"flood", "tropical_cyclone"}
    unpriced = [f"{str(f['hazard']).replace('_', ' ')} ({f['severity']})" for f in sa
                if f.get("hazard") not in priced and f.get("severity") in ("HIGH", "MODERATE")]
    if unpriced:
        parts.append("not priced in this loss: " + ", ".join(unpriced) + " — 0.00% physical damage here means the priced hazards "
                     "(cyclone, river and coastal flood) are low, not that the site is safe")
    if v and (a["value"].get("quality") in ("proxy", "benchmark")):
        parts.append("the asset value is itself an estimate (" + ("from emissions" if a["value"].get("quality") == "proxy" else "from capacity") +
                     "), so the percentage of value is indicative only; give the asset value for a firm figure")
    return "; ".join(parts) + "."


def _narrative(r: dict) -> list:
    a, ph, pol, fin = r["asset"], r["physical"], r["policy"], r["financial"]
    out = []
    tc = ph["cyclone"]
    out.append(f"Physical: cyclone {tc['severity'].lower()} — {tc['lam']:.2f} tropical-storm-force events a year on "
               f"average, 1-in-100-year wind {tc['rp100_ms']:.0f} m/s ({tc['category_100yr']}). River flood "
               f"{ph['river_flood'].get('severity', 'n/a').lower()}, coastal flood {ph['coastal_flood'].get('severity', 'n/a').lower()}; "
               f"defences ≈ 1-in-{ph['defences']['river_rp']:.0f} (river) and 1-in-{ph['defences']['coastal_rp']:.0f} "
               f"(coast). Modelled average annual damage {fin['physical']['aal_pct']:.2f}% of value.")
    if pol["in_force"]:
        out.append(f"Policy: {pol['price_setting_instrument']} sets an effective ${pol['effective_price_usd_t']}/t today "
                   f"({(pol['charged_share'] or {}).get('rule', '')}). " + " ".join(pol["notes"][:2]))
    else:
        out.append("Policy: no explicit carbon price covers this asset today. " + " ".join(pol["notes"][1:3]))
    rows = []
    for s in MAIN_SCENARIOS:
        sc = fin["scenarios"].get(s)
        if sc:
            ab = sc.get("abatement") or {}
            rows.append(f"{sc['label']}: {_fmt_usd(sc['transition_npv']['mean'])} carbon cost PV if unabated"
                        + (f", {_fmt_usd(ab['response_cost_npv']['mean'])} with abatement" if ab.get("enabled") else "")
                        + (f", stranded {_fmt_usd(sc['stranded']['mean'])}" if sc.get("stranded") else ""))
    if rows and a.get("emissions_t"):
        out.append(f"Transition ({a['emissions_t']:,.0f} tCO2e/yr): " + "; ".join(rows) + ".")
    if a["value"].get("mode"):
        sc = fin["scenarios"].get("Delayed transition") or {}
        t = sc.get("total_npv") or {}
        out.append(f"Financial: total loss to 2050 (PV) mean {_fmt_usd(t.get('mean'))}, median {_fmt_usd(t.get('p50'))}, "
                   f"95th percentile {_fmt_usd(t.get('p95'))}, expected shortfall (worst 5%) {_fmt_usd(t.get('es95'))}. "
                   f"Asset value basis: {a['value']['basis']}.")
        if (sc.get("abatement") or {}).get("enabled") and sc.get("total_npv_unabated"):
            out.append(f"Cost of inaction: without any abatement the mean loss would be "
                       f"{_fmt_usd(sc['total_npv_unabated']['mean'])}.")
    e = (fin["scenarios"].get("Delayed transition") or {}).get("earnings")
    if e:
        h = (e.get("horizons") or {}).get("2040") or {}
        bits = []
        if h.get("ebitda_change_pct_median") is not None:
            bits.append(f"median EBITDA {h['ebitda_change_pct_median']:+.0f}% vs today by 2040")
        if h.get("leverage_median") is not None:
            bits.append(f"net debt/EBITDA {h['leverage_median']:.1f}×")
        if h.get("rating_median"):
            bits.append(f"synthetic rating {h['rating_median']} (today {e['credit']['baseline']['rating']})")
        if (e.get("covenants") or {}).get("by_horizon", {}).get("2040") is not None:
            bits.append(f"covenant breach probability {100 * e['covenants']['by_horizon']['2040']:.0f}%")
        if bits:
            out.append("Earnings (delayed transition): " + ", ".join(bits) + ".")
    return out


def _llm_narrative(r: dict) -> Optional[str]:
    try:
        from ..ai.narrator import _call_llm
    except Exception:                                      # noqa: BLE001
        return None
    facts = "\n".join(r["narrative"]) + "\nActions:\n" + "\n".join(f"- {x['problem']} → {x['action']}" for x in r["actions"])
    return _call_llm("You are a climate risk analyst writing for a credit committee. Use ONLY the numbers given. "
                     "Four short paragraphs: physical risk, policy, transition, financial impact; then 3 prioritised "
                     "actions. No new figures.", facts, max_tokens=700)


# ── company / portfolio ──────────────────────────────────────────────────────

def analyse_company(keys: list, top_n: int = 30, n: int = 2000, discount: float = 0.07,
                    values: Optional[dict] = None) -> dict:
    """Run every asset of a company (from the register) and aggregate the loss distribution."""
    from ..data.asset_register import company_exposure
    exp = company_exposure(keys)
    if not exp.get("found"):
        return {"found": False, "keys": keys}
    assets = exp["assets"]
    for a in assets:
        a["_kw"] = {"name": a["name"], "register_index": a["i"], "value_usd": (values or {}).get(str(a["i"]))}
    return _portfolio(assets, exp["companies"], top_n, n, discount,
                      f"flood depths fetched live for the top {top_n} assets by emissions")


_FIN_COLS = ("revenue_usd", "ebitda_usd", "da_usd", "interest_usd", "net_debt_usd", "book_value_usd", "tax_rate",
             "electricity_mwh", "fuel", "fuel_gj", "commodity", "price_realisation", "operating_leverage",
             "sustaining_capex_usd", "covenant_max_leverage", "covenant_min_icr", "lgd")


def _int_or_none(v):
    try:
        return None if v in (None, "") or v != v else int(float(v))
    except (TypeError, ValueError):
        raise ValueError(f"not a year: {v!r}")


def _num_or_none(v):
    try:
        return None if v in (None, "") or v != v else float(v)
    except (TypeError, ValueError):
        raise ValueError(f"not a number: {v!r}")


def _site_financials(s: dict, k: int) -> Optional[dict]:
    """Per-site financial columns → earnings-layer input (needs revenue_usd and ebitda_usd; a lone one is an error)."""
    f = {c: s[c] for c in _FIN_COLS if s.get(c) not in (None, "")}
    if not f:
        return None
    if "revenue_usd" not in f or "ebitda_usd" not in f:
        raise ValueError(f"site {s.get('name') or k + 1}: financial columns need both revenue_usd and ebitda_usd")
    return f


def analyse_sites(sites: list, label: str = "Portfolio", top_n: int = 30, n: int = 2000, discount: float = 0.07) -> dict:
    """The same four-stage analysis and aggregation for a client's own site list.
    sites: [{name, lat, lon, sector?, type?, capacity?, emissions_t?, value_usd?}] — no register snapping:
    a site without a sector is analysed as sector "other" rather than borrowing a nearby register plant."""
    if not sites:
        return {"found": False, "note": "no sites"}
    assets = []
    for k, s in enumerate(sites):
        lat, lon = float(s["lat"]), float(s["lon"])
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError(f"site {s.get('name') or k + 1}: invalid coordinates")
        assets.append({"i": k, "name": s.get("name") or f"Site {k + 1}", "lat": lat, "lon": lon,
                       "country": s.get("country") or "", "sector": s.get("sector") or "other",
                       "emissions_t": s.get("emissions_t"), "elev_m": None,
                       "_kw": {"name": s.get("name") or f"Site {k + 1}", "sector": s.get("sector") or "other",
                               "atype": s.get("type"), "capacity": s.get("capacity"),
                               "emissions_t": s.get("emissions_t"), "value_usd": s.get("value_usd"),
                               "financials": _site_financials(s, k),
                               "commissioned_year": _int_or_none(s.get("commissioned_year")),
                               "retirement_year": _int_or_none(s.get("retirement_year")),
                               "export_share_eu": _num_or_none(s.get("export_share_eu"))}})
    return _portfolio(assets, [label], top_n, n, discount,
                      f"flood depths fetched live for the top {top_n} sites by emissions (then by list order)")


def _earn_row(sc: Optional[dict]) -> Optional[dict]:
    e = (sc or {}).get("earnings")
    if not e:
        return None
    h = (e.get("horizons") or {}).get("2040") or {}
    return {"ebitda_change_pct": h.get("ebitda_change_pct_median"), "leverage": h.get("leverage_median"),
            "rating": h.get("rating_median"), "pd": h.get("pd_mean"),
            "covenant_breach_prob": (e.get("covenants") or {}).get("by_horizon", {}).get("2040"),
            "climate_impairment_mean": ((e.get("impairment") or {}).get("climate_attributable") or {}).get("mean")}


def _portfolio(assets: list, companies: list, top_n: int, n: int, discount: float, flood_note: str) -> dict:
    t0 = time.time()
    order = sorted(range(len(assets)), key=lambda k: -(assets[k]["emissions_t"] or 0))
    live = set(order[:top_n])
    floods = {}
    with ThreadPoolExecutor(8) as pool:
        futs = {k: pool.submit(site_flood, assets[k]["lat"], assets[k]["lon"], 250.0, None, assets[k]["elev_m"]) for k in live}
        for k, f in futs.items():
            try:
                floods[k] = f.result(timeout=90)
            except Exception:                              # noqa: BLE001
                floods[k] = {"river": {"available": False}, "coastal": {"available": False}}
    common = {"u_model": np.random.default_rng(11).uniform(0, 1, n)}
    per, tot, vals = [], {}, []
    for k in order:
        a = assets[k]
        r = analyse_asset(a["lat"], a["lon"], n=n, discount=discount,
                          _flood=floods.get(k, {"river": {"available": False, "reason": "not fetched (outside top assets)"},
                                                "coastal": {"available": False, "reason": "not fetched"}}),
                          _common=dict(common, seed=1000 + k), **a["_kw"])
        for s, sc in r["financial"]["scenarios"].items():
            tot.setdefault(s, []).append(sc.pop("_total_draws"))
        vd = r["financial"].pop("_value_draws", None)
        if vd is not None:
            vals.append(vd)
        per.append({"i": a["i"], "name": a["name"], "country": a["country"] or r["asset"].get("iso3"),
                    "sector": a["sector"] or r["asset"].get("sector"), "lat": a["lat"], "lon": a["lon"],
                    "emissions_t": r["asset"].get("emissions_t"), "value_mode": r["asset"]["value"].get("mode"),
                    "value_quality": r["asset"]["value"].get("quality"),
                    "flood_assessed": k in live, "physical_aal_pct": r["financial"]["physical"]["aal_pct"],
                    "cyclone": r["physical"]["cyclone"]["severity"],
                    "river": r["physical"]["river_flood"].get("severity") if r["physical"]["river_flood"].get("assessed") else "NOT ASSESSED",
                    "coast": r["physical"]["coastal_flood"].get("severity") if r["physical"]["coastal_flood"].get("assessed") else "NOT ASSESSED",
                    "river_assessed": bool(r["physical"]["river_flood"].get("assessed")),
                    "coast_assessed": bool(r["physical"]["coastal_flood"].get("assessed")),
                    "carbon_price_now": r["policy"]["effective_price_usd_t"],
                    "instrument": r["policy"]["price_setting_instrument"],
                    "npv_delayed_mean": (r["financial"]["scenarios"].get("Delayed transition") or {}).get("total_npv", {}).get("mean"),
                    "npv_nz_mean": (r["financial"]["scenarios"].get("Net Zero 2050") or {}).get("total_npv", {}).get("mean"),
                    "npv_cp_mean": (r["financial"]["scenarios"].get("Current Policies") or {}).get("total_npv", {}).get("mean"),
                    "earnings_delayed_2040": _earn_row(r["financial"]["scenarios"].get("Delayed transition")),
                    "top_action": (r["actions"][0]["action"] if r["actions"] else None)})
    V = sum((p["value_mode"] or 0) for p in per)
    from .distribution import _stats
    port = {}
    for s, arrs in tot.items():
        x = np.sum(np.stack(arrs), axis=0)
        hist_hi = float(np.percentile(x, 99.5)) or 1.0
        counts, edges = np.histogram(np.clip(x, 0, hist_hi), bins=40, range=(0, hist_hi))
        port[s] = {"label": transition.label(s), "total_npv": _stats(x, np.sum(np.stack(vals), axis=0) if vals else None),
                   "histogram": {"edges": np.round(edges, 0).tolist(), "counts": counts.tolist()}}
    per.sort(key=lambda p: -(p["npv_delayed_mean"] or 0))
    return {"found": True, "companies": companies, "n_assets": len(per), "portfolio_value_mode": V,
            "flood_assessed_assets": len(live),
            "physical_coverage": {"assets": len(per), "river_flood_assessed": sum(1 for p in per if p["river_assessed"]),
                                  "coastal_flood_assessed": sum(1 for p in per if p["coast_assessed"]),
                                  "physical_loss_is_lower_bound": any(not p["river_assessed"] for p in per),
                                  "note": "Physical loss in the scenarios counts flood only for assets whose flood maps were read; "
                                          "every other asset contributes cyclone loss only, so the portfolio physical loss is a lower bound "
                                          "unless every asset was assessed."},
            "scenarios": port, "assets": per,
            "method": "Each asset run through the four-stage analyst; " + flood_note + " (others: cyclone + policy + "
                      "transition only). Transition model draws are shared across assets (common scenario "
                      "uncertainty); physical events independent. % of value is taken per draw against that draw's "
                      "portfolio value, so it cannot exceed 100%.",
            "elapsed_s": round(time.time() - t0, 1)}
