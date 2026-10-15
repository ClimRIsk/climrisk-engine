"""
cri/stresstest.py — one linked stress test across hazard, operations, supply chain, carbon and cash.

The engine already has the pieces (site hazard profile, DESNZ freight factors, carbon cost). Run alone they cannot answer
"the typhoon cuts the grid, a Tier-2 supplier is hit, the rush shipment goes by air, and the air shipment triggers a border
tariff". This module joins them in one timeline and prices the decision with every cost counted together.

Rules, the same as the rest of the engine:
  * Engine-backed numbers (site hazard, return period of the storm, freight emissions) say which engine run they came from.
  * Scenario facts the user supplies (fuel hours, delay weeks, tariff rate) are echoed back as inputs, never presented as results.
  * A missing input stays null and the answer says it is not modelled. Nothing is silently assumed to be zero.
  * Costs are added together; the "in isolation" figures are shown beside the joint figure so the interaction is visible.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Callable, Optional

# Saffir–Simpson (NHC), 1-minute sustained wind, m/s (kt × 0.5144)
CATEGORY_WIND_MS = {1: (33, 42), 2: (43, 49), 3: (50, 58), 4: (58, 70), 5: (70, 90)}
HAZ_KEYS = ["cyclone", "river_flood", "coastal_flood", "thermal_stress"]


class StressError(ValueError):
    pass


def _f(x) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise StressError(f"{x!r} is not a number")
    if not math.isfinite(v):
        raise StressError("numbers must be finite")
    return v


def _pos(x, name) -> Optional[float]:
    v = _f(x)
    if v is not None and v < 0:
        raise StressError(f"{name} cannot be negative")
    return v


# ── engine-backed pieces ─────────────────────────────────────────────────────

def _default_hazards(lat: float, lon: float, name: str) -> dict:
    from . import siteprofile as SP
    return SP.profile(lat, lon, name, only=HAZ_KEYS, budget_s=150.0)


def return_period_of_wind(wind_ms: float, rp_wind: dict) -> dict:
    """Where a given peak wind sits on the site's own return-period curve (log-linear between the engine's return levels)."""
    pts = sorted((float(k), float(v)) for k, v in (rp_wind or {}).items() if v is not None)
    if len(pts) < 2:
        return {"status": "null", "reason": "no return-period winds for this site"}
    if wind_ms <= pts[0][1]:
        return {"status": "modelled", "rp_years": None, "bound": f"<{pts[0][0]:g}", "note": f"below the 1-in-{pts[0][0]:g}-year wind ({pts[0][1]:g} m/s)"}
    if wind_ms >= pts[-1][1]:
        return {"status": "modelled", "rp_years": None, "bound": f">{pts[-1][0]:g}",
                "note": f"above the engine's 1-in-{pts[-1][0]:g}-year wind ({pts[-1][1]:g} m/s): outside the modelled range"}
    for (r0, w0), (r1, w1) in zip(pts, pts[1:]):
        if w0 <= wind_ms <= w1:
            t = (wind_ms - w0) / (w1 - w0) if w1 > w0 else 0.0
            rp = math.exp(math.log(r0) + t * (math.log(r1) - math.log(r0)))
            return {"status": "modelled", "rp_years": round(rp), "bound": None, "note": f"about a 1-in-{round(rp)}-year wind at this site"}
    return {"status": "null", "reason": "could not place the wind on the curve"}


def _site_hazard(site: dict, scen_wind: Optional[float], hazard_fn: Callable) -> dict:
    """Engine hazard profile for one site, cut down to what the stress test uses."""
    out = {"id": site.get("id"), "name": site.get("name"), "role": site.get("role"), "lat": site.get("lat"), "lon": site.get("lon")}
    lat, lon = _f(site.get("lat")), _f(site.get("lon"))
    if lat is None or lon is None:
        out.update(status="null", reason="no coordinates given, so the engine could not run its hazards for this site",
                   hazards={})
        return out
    try:
        p = hazard_fn(lat, lon, site.get("name") or "Site")
    except Exception as e:                                                   # noqa: BLE001
        out.update(status="null", reason=f"hazard run failed ({type(e).__name__}: {str(e)[:120]})", hazards={})
        return out
    hz = {}
    for k in HAZ_KEYS:
        e = (p.get("hazards") or {}).get(k)
        if not e:
            continue
        hz[k] = {"status": e.get("status"), "label": e.get("label"), "value": e.get("value"), "unit": e.get("unit"),
                 "severity": e.get("severity"), "reason": e.get("reason"), "caveat": e.get("caveat")}
    out.update(status="modelled", hazards=hz, from_cache=bool(p.get("from_cache")), source="siteprofile.profile (engine)")
    cy = (p.get("hazards") or {}).get("cyclone") or {}
    rpw = ((cy.get("metrics") or {}).get("rp_wind_ms"))
    if scen_wind is not None:
        out["storm_placement"] = return_period_of_wind(scen_wind, rpw) if rpw else {
            "status": "null", "reason": cy.get("reason") or "the engine has no cyclone return levels at this point"}
        out["scenario_wind_ms"] = scen_wind
        if rpw:
            out["site_rp_wind_ms"] = rpw
    return out


# ── operations timeline ──────────────────────────────────────────────────────

def operations(ev: dict, horizon_days: int = 60) -> dict:
    """Daily capacity of the facility after the storm: grid, generator fuel, road access and workers."""
    grid_out = _pos(ev.get("grid_outage_days"), "grid_outage_days")
    fuel_h = _pos(ev.get("generator_fuel_hours"), "generator_fuel_hours")
    access = _pos(ev.get("access_cut_days"), "access_cut_days")
    absent = _pos(ev.get("workers_absent_pct"), "workers_absent_pct")
    refuel = _pos(ev.get("refuel_after_days"), "refuel_after_days")
    miss = [k for k, v in (("grid_outage_days", grid_out), ("generator_fuel_hours", fuel_h), ("access_cut_days", access), ("workers_absent_pct", absent)) if v is None]
    if grid_out is None and access is None:
        return {"status": "null", "reason": "no outage or access-cut duration given", "inputs_missing": miss}
    fuel_days = (fuel_h or 0.0) / 24.0
    cover_until = fuel_days if refuel is None else max(fuel_days, refuel + fuel_days)   # refuel arrives on day `refuel`, gives another fuel_days
    go = grid_out or 0.0
    ac = access or 0.0
    ab = (absent if absent is not None else 100.0 if ac > 0 else 0.0) / 100.0         # unknown absence during a cut is treated as total, and said so
    cap = []
    for d in range(horizon_days):
        grid_up = d >= go
        powered = grid_up or d < cover_until
        workers = 1.0 - ab if d < ac else 1.0
        cap.append(round((workers if powered else 0.0), 4))
    lost = sum(1.0 - c for c in cap)
    first_full = next((d for d, c in enumerate(cap) if c >= 1.0 and all(x >= 1.0 for x in cap[d:])), None)
    out = {"status": "modelled", "capacity_by_day": cap, "production_days_lost": round(lost, 2), "full_capacity_from_day": first_full,
           "generator_cover_days": round(cover_until, 2), "fuel_gap_days": round(max(0.0, go - cover_until), 2),
           "inputs": {k: v for k, v in (("grid_outage_days", grid_out), ("generator_fuel_hours", fuel_h), ("access_cut_days", access),
                                        ("workers_absent_pct", absent), ("refuel_after_days", refuel)) if v is not None},
           "inputs_missing": miss}
    if absent is None and ac > 0:
        out["assumption"] = "workers_absent_pct not given: during the access cut the plant is treated as having no workforce (worst case)."
    return out


# ── supply side ──────────────────────────────────────────────────────────────

def supply(sup: dict, inv: dict, supplier_hazard: Optional[dict] = None, restart_day: float = 0.0) -> dict:
    """Component buffer vs supplier delay vs alternates. The buffer is only drawn down once the plant is producing again (restart_day)."""
    if not sup:
        return {"status": "null", "reason": "no supplier disruption described"}
    dmin, dmax = _pos(sup.get("delay_weeks_min"), "delay_weeks_min"), _pos(sup.get("delay_weeks_max"), "delay_weeks_max")
    if dmin is None and dmax is None:
        return {"status": "null", "reason": "no supplier delay given"}
    dmin = dmin if dmin is not None else dmax
    dmax = dmax if dmax is not None else dmin
    if dmax < dmin:
        dmin, dmax = dmax, dmin
    buf = _pos((inv or {}).get("component_buffer_days"), "component_buffer_days")
    alts = []
    for a in sup.get("alternates") or []:
        backlog = _pos(a.get("backlog_weeks"), "backlog_weeks")
        qual = _pos(a.get("qualification_days"), "qualification_days")
        pre = a.get("preapproved")
        if pre is True and qual is None:
            qual = 0.0
        avail = None if backlog is None or qual is None else round(qual + backlog * 7.0, 1)
        alts.append({"name": a.get("name"), "preapproved": pre, "backlog_weeks": backlog, "qualification_days": qual,
                     "available_from_day": avail, "price_premium_pct": _pos(a.get("price_premium_pct"), "price_premium_pct"),
                     "note": None if avail is not None else "qualification time or backlog not given, so its start date is unknown"})
    known = [a["available_from_day"] for a in alts if a["available_from_day"] is not None]
    first_alt = min(known) if known else None
    d_lo, d_hi = dmin * 7.0, dmax * 7.0

    def resume_day(delay_d: float) -> float:
        return delay_d if first_alt is None else min(delay_d, first_alt)

    def gap(delay_d: float) -> Optional[float]:
        if buf is None:
            return None
        return round(max(0.0, resume_day(delay_d) - (restart_day + buf)), 1)

    res = {"status": "modelled" if buf is not None else "partial",
           "delay_days": {"min": d_lo, "max": d_hi}, "component_buffer_days": buf,
           "shortfall_days": {"if_delay_min": gap(d_lo), "if_delay_max": gap(d_hi)},
           "alternates": alts, "preapproved_alternates": sum(1 for a in alts if a["preapproved"] is True),
           "earliest_alternate_day": first_alt,
           "alternate_beats_supplier": None if first_alt is None else bool(first_alt < d_hi),
           "inputs_missing": [] if buf is not None else ["component_buffer_days"]}
    if buf is not None:
        res["restart_day_used"] = restart_day
        res["resume_day"] = {"if_delay_min": round(max(restart_day, resume_day(d_lo)), 1), "if_delay_max": round(max(restart_day, resume_day(d_hi)), 1)}
        res["buffer_needed_days"] = {"if_delay_min": round(max(0.0, resume_day(d_lo) - restart_day), 1),
                                     "if_delay_max": round(max(0.0, resume_day(d_hi) - restart_day), 1)}
        res["buffer_gap_days"] = {"if_delay_min": round(max(0.0, res["buffer_needed_days"]["if_delay_min"] - buf), 1),
                                  "if_delay_max": round(max(0.0, res["buffer_needed_days"]["if_delay_max"] - buf), 1)}
    return res


# ── freight, tariff and cash ─────────────────────────────────────────────────

def _freight_factors(batch: dict) -> dict:
    from .lca import cfp
    from .lca.engine import factors
    ship = factors()["freight"]["ship"]
    sea = _pos(batch.get("sea_kgco2e_per_tkm"), "sea_kgco2e_per_tkm")
    sea_note = "user-supplied factor"
    if sea is None:
        sea = ship["kgco2e_per_tkm"] + (ship["wtt_kgco2e_per_tkm"] or 0.0)
        sea_note = (f"{ship['label']} (DESNZ 2025, incl. WTT). This is a bulk-carrier average; a container service has a higher "
                    "per-tonne-km factor, so enter your carrier's factor if you have one.")
    air = _pos(batch.get("air_kgco2e_per_tkm"), "air_kgco2e_per_tkm")
    air_note = "user-supplied factor"
    if air is None:
        air = cfp._desnz(cfp.AIR_IDS["kg"]) + cfp._desnz(cfp.AIR_IDS["wtt"])
        air_note = "DESNZ 2026 freight flights, international, excl. radiative forcing, incl. WTT"
    return {"sea": sea, "air": air, "sea_basis": sea_note, "air_basis": air_note}


def _option(label: str, air_share: float, B: dict, T: dict, P: dict, ff: dict, base: dict) -> dict:
    """Cost and emissions of shipping `air_share` of the batch by air and the rest by sea. Every figure is a sum of its parts."""
    units, mass, value = B["units"], B["mass_t"], B["value_usd"]
    air_t, sea_t = mass * air_share, mass * (1 - air_share)
    air_km, sea_km = B.get("air_km"), B.get("sea_km")
    air_cost = None if B.get("air_usd_per_t") is None else air_t * B["air_usd_per_t"]
    sea_cost = None if B.get("sea_usd_per_t") is None else sea_t * B["sea_usd_per_t"]
    transport_kg = None
    if air_km is not None and sea_km is not None:
        transport_kg = air_t * air_km * ff["air"] + sea_t * sea_km * ff["sea"]
    embodied = B.get("embodied_kgco2e_per_unit")
    per_unit_total = None if transport_kg is None or embodied is None else embodied + transport_kg / units
    per_unit_transport = None if transport_kg is None else transport_kg / units
    air_units = units * air_share

    # tariff
    tariff, tariff_note, triggered = None, None, None
    rate = T.get("rate_pct")
    trig = T.get("trigger") or {}
    if rate is None:
        tariff_note = "tariff rate not given"
    else:
        if trig.get("type") == "intensity_threshold":
            thr = trig.get("kgco2e_per_unit")
            scope = T.get("scope", "batch")
            if thr is None:
                tariff_note = "tariff threshold not given"
            elif scope == "consignment":
                # each consignment is tested on its own: the air consignment carries embodied + its air transport
                if air_units > 0 and embodied is not None and air_km is not None:
                    air_pu = embodied + (air_t * air_km * ff["air"]) / air_units
                    triggered_air = air_pu > thr
                else:
                    triggered_air = None
                sea_units = units - air_units
                if sea_units > 0 and embodied is not None and sea_km is not None:
                    sea_pu = embodied + (sea_t * sea_km * ff["sea"]) / sea_units
                    triggered_sea = sea_pu > thr
                else:
                    triggered_sea = None
                if None in ([triggered_air] if air_units > 0 else []) + ([triggered_sea] if sea_units > 0 else []):
                    tariff_note = "embodied footprint or distances missing, so the tariff trigger cannot be tested"
                else:
                    taxed = (air_units * B["value_usd"] / units if triggered_air else 0.0) + (sea_units * B["value_usd"] / units if triggered_sea else 0.0)
                    tariff, triggered = taxed * rate / 100.0, bool(triggered_air or triggered_sea)
            else:
                if per_unit_total is None:
                    tariff_note = "embodied footprint or distances missing, so the tariff trigger cannot be tested"
                else:
                    triggered = per_unit_total > thr
                    tariff = value * rate / 100.0 if triggered else 0.0
        elif trig.get("type") == "any_air":
            triggered = air_share > 0
            tariff = value * rate / 100.0 if triggered else 0.0
        elif trig.get("type") == "always":
            triggered, tariff = True, value * rate / 100.0
        else:
            tariff_note = "tariff trigger not given (use intensity_threshold, any_air or always)"

    # delivery and penalty: the batch is on time only for the part that flies
    sea_days, air_days, window = B.get("sea_days"), B.get("air_days"), B.get("delivery_window_days")
    start = P.get("start_day", 0.0) or 0.0                  # day the batch can leave (production restart / component arrival)
    late_days_sea = None if (sea_days is None or window is None) else max(0.0, start + sea_days - window)
    late_days_air = None if (air_days is None or window is None) else max(0.0, start + air_days - window)
    penalty = None
    pct_wk, cap, grace = P.get("late_penalty_pct_per_week"), P.get("penalty_cap_pct"), P.get("grace_days") or 0.0
    if pct_wk is not None and late_days_sea is not None and late_days_air is not None:
        def pen(ld, share):
            weeks = max(0.0, ld - grace) / 7.0
            pct = weeks * pct_wk
            if cap is not None:
                pct = min(pct, cap)
            return value * share * pct / 100.0
        penalty = pen(late_days_sea, 1 - air_share) + pen(late_days_air, air_share)
    cogs = B.get("cogs_usd")
    parts = {"air_freight_usd": air_cost, "sea_freight_usd": sea_cost, "tariff_usd": tariff, "late_penalty_usd": penalty}
    missing = [k for k, v in parts.items() if v is None]
    freight = None if air_cost is None or sea_cost is None else air_cost + sea_cost
    cost_known = sum(v for v in parts.values() if v is not None)
    profit = None if (cogs is None or missing) else value - cogs - cost_known
    return {"label": label, "air_share": round(air_share, 3), "emissions_t": None if transport_kg is None else round(transport_kg / 1000.0, 2),
            "transport_kgco2e_per_unit": None if per_unit_transport is None else round(per_unit_transport, 3),
            "footprint_kgco2e_per_unit": None if per_unit_total is None else round(per_unit_total, 3),
            "tariff_triggered": triggered, "tariff_note": tariff_note,
            "late_days": {"sea": late_days_sea, "air": late_days_air},
            "costs_usd": {k: (None if v is None else round(v, 2)) for k, v in parts.items()},
            "freight_usd": None if freight is None else round(freight, 2),
            "total_extra_cost_usd": None if missing else round(cost_known, 2),
            "profit_usd": None if profit is None else round(profit, 2),
            "margin_pct": None if profit is None else round(100.0 * profit / value, 2),
            "components_missing": missing}


def freight_decision(batch: dict, tariff: dict, penalty: dict, restart_day: float = 0.0) -> dict:
    """Price sea, air and split options with freight, tariff and late penalty added together; show the isolated figures beside the joint one."""
    if not batch:
        return {"status": "null", "reason": "no product batch described"}
    B = dict(batch)
    for k in ("units", "mass_t", "value_usd"):
        B[k] = _pos(B.get(k), k)
    if not B["units"] or not B["mass_t"] or not B["value_usd"]:
        return {"status": "null", "reason": "batch needs units, mass_t and value_usd"}
    for k in ("production_days", "sea_km", "air_km", "sea_days", "air_days", "delivery_window_days", "sea_usd_per_t", "air_usd_per_t", "cogs_usd", "embodied_kgco2e_per_unit"):
        B[k] = _pos(B.get(k), k)
    T = dict(tariff or {})
    T["rate_pct"] = _pos(T.get("rate_pct"), "rate_pct")
    P = dict(penalty or {})
    P["late_penalty_pct_per_week"] = _pos(P.get("late_penalty_pct_per_week"), "late_penalty_pct_per_week")
    P["penalty_cap_pct"] = _pos(P.get("penalty_cap_pct"), "penalty_cap_pct")
    P["grace_days"] = _pos(P.get("grace_days"), "grace_days")
    P["start_day"] = restart_day
    ff = _freight_factors(B)
    base = {}
    opts = [_option("Stay on sea freight", 0.0, B, T, P, ff, base), _option("All by air", 1.0, B, T, P, ff, base)]
    shares = [i / 20.0 for i in range(1, 20)]
    grid = [_option(f"{int(s * 100)}% by air", s, B, T, P, ff, base) for s in shares]
    comparable = [o for o in opts + grid if o["total_extra_cost_usd"] is not None]
    best = min(comparable, key=lambda o: o["total_extra_cost_usd"]) if comparable else None
    sea_o, air_o = opts
    out = {"status": "modelled" if comparable else "partial", "options": opts, "split_grid": grid,
           "best": None if best is None else {"label": best["label"], "air_share": best["air_share"], "total_extra_cost_usd": best["total_extra_cost_usd"],
                                              "margin_pct": best["margin_pct"]},
           "emission_factors": {"sea_kgco2e_per_tkm": round(ff["sea"], 5), "air_kgco2e_per_tkm": round(ff["air"], 5), "sea_basis": ff["sea_basis"], "air_basis": ff["air_basis"]}}
    if sea_o["footprint_kgco2e_per_unit"] and air_o["footprint_kgco2e_per_unit"]:
        out["air_footprint_multiple"] = round(air_o["footprint_kgco2e_per_unit"] / sea_o["footprint_kgco2e_per_unit"], 2)
    if sea_o["transport_kgco2e_per_unit"] and air_o["transport_kgco2e_per_unit"]:
        out["air_transport_multiple"] = round(air_o["transport_kgco2e_per_unit"] / sea_o["transport_kgco2e_per_unit"], 1)
    # interaction: what the air route costs on its own terms vs together with the tariff it triggers
    ac = air_o["costs_usd"]
    if None not in (ac["air_freight_usd"], ac["tariff_usd"], ac["late_penalty_usd"], sea_o["costs_usd"]["late_penalty_usd"], sea_o["costs_usd"]["sea_freight_usd"]):
        sea_total = sea_o["total_extra_cost_usd"]
        isolated_freight = ac["air_freight_usd"] - sea_o["costs_usd"]["sea_freight_usd"]
        out["interaction"] = {
            "freight_premium_of_air_usd": round(isolated_freight, 2),
            "tariff_triggered_by_air_usd": round(ac["tariff_usd"] - (sea_o["costs_usd"]["tariff_usd"] or 0.0), 2),
            "late_penalty_avoided_by_air_usd": round(sea_o["costs_usd"]["late_penalty_usd"] - ac["late_penalty_usd"], 2),
            "air_minus_sea_total_usd": round(air_o["total_extra_cost_usd"] - sea_total, 2),
            "note": "Looked at one at a time, air looks like a freight premium that buys back a penalty. Counted together it also triggers the tariff; the last figure is the net.",
        }
        # break-even tariff rate at which the two routes cost the same (linear in the rate), if the tariff only applies to air
        if (sea_o["costs_usd"]["tariff_usd"] or 0.0) == 0.0 and air_o["tariff_triggered"] and T.get("rate_pct"):
            per_pct = ac["tariff_usd"] / T["rate_pct"]
            other = air_o["total_extra_cost_usd"] - ac["tariff_usd"]
            be = (sea_total - other) / per_pct if per_pct else None
            out["breakeven"] = {"tariff_rate_pct_where_air_equals_sea": None if be is None else round(be, 2),
                                "reading": ("Air is cheaper than sea at any tariff rate below this." if be is not None and be > 0
                                            else "Air costs more than sea even with no tariff, on these inputs.")}
        if sea_o["late_days"]["sea"] is not None and P.get("late_penalty_pct_per_week"):
            out["lateness"] = {"sea_late_days": sea_o["late_days"]["sea"], "air_late_days": air_o["late_days"]["air"]}
    return out


# ── the four questions ───────────────────────────────────────────────────────

def _pct(a, b):
    return None if not b else round(100.0 * a / b, 1)


def answers(spec: dict, hz: list, ops: dict, sup: dict, fr: dict) -> list:
    out = []
    # 1 visibility
    sup_site = next((h for h in hz if h.get("role") == "supplier"), None)
    lag = _pos((spec.get("supplier") or {}).get("detection_lag_days"), "detection_lag_days")
    pre = (sup or {}).get("preapproved_alternates")
    parts, miss, ev = [], [], []
    if sup_site and sup_site.get("status") == "modelled":
        cy = (sup_site["hazards"].get("cyclone") or {})
        parts.append(f"Before the storm: the engine already rates the supplier site's cyclone hazard {cy.get('severity') or 'not assessed'}"
                     + (f" ({cy['value']:.1f} m/s 1-in-100 wind)" if isinstance(cy.get("value"), (int, float)) else "")
                     + ", so its exposure is knowable in advance rather than discovered on Day 4.")
        ev.append("siteprofile cyclone, river_flood, coastal_flood at the supplier coordinates")
    else:
        miss.append("supplier coordinates (needed to know in advance that the supplier is in the same storm path)")
    if lag is not None:
        parts.append(f"Actual detection: {lag:g} days after the event (your input).")
    else:
        parts.append("How long your team would take to notice is not known: no detection time or supplier-monitoring input was given.")
        miss.append("detection_lag_days")
    if sup and sup.get("alternates") is not None:
        n = len(sup["alternates"])
        parts.append(f"Alternate components: {n} listed, {pre} pre-approved." if n else "No alternate source is listed.")
        if n and not pre:
            parts.append("None is pre-approved, so the engine counts qualification time before any alternate can ship.")
    else:
        miss.append("alternate suppliers")
    out.append({"id": "visibility", "question": "Visibility: how long to realise the Tier-2 supplier is offline, and are alternates pre-approved?",
                "status": "partial" if miss else "modelled", "answer": " ".join(parts), "evidence": ev, "inputs_missing": miss})
    # 2 joint financial buffers
    inter = fr.get("interaction")
    if inter:
        txt = (f"Counted together: air freight costs ${inter['freight_premium_of_air_usd']:,.0f} more than sea, triggers ${inter['tariff_triggered_by_air_usd']:,.0f} of tariff, "
               f"and avoids ${inter['late_penalty_avoided_by_air_usd']:,.0f} of late penalties. Net, air is "
               f"${abs(inter['air_minus_sea_total_usd']):,.0f} {'more expensive' if inter['air_minus_sea_total_usd'] > 0 else 'cheaper'} than staying on sea.")
        if fr.get("best"):
            txt += f" Lowest total: {fr['best']['label']} (extra cost ${fr['best']['total_extra_cost_usd']:,.0f}, margin {fr['best']['margin_pct']}%)." if fr["best"].get("margin_pct") is not None else \
                f" Lowest total: {fr['best']['label']} (extra cost ${fr['best']['total_extra_cost_usd']:,.0f})."
        cash = _pos((spec.get("finance") or {}).get("cash_buffer_usd"), "cash_buffer_usd")
        if cash is not None and fr.get("best"):
            txt += f" Against a ${cash:,.0f} cash buffer that is {_pct(fr['best']['total_extra_cost_usd'], cash)}% of the buffer."
        out.append({"id": "financial_buffers", "question": "Financial buffers: does the model price freight cost and tariff together?", "status": "modelled", "answer": txt,
                    "evidence": ["freight_decision: DESNZ freight factors, tariff trigger, late penalty summed in one calculation"],
                    "inputs_missing": [] if cash is not None else ["finance.cash_buffer_usd (to compare against your liquidity)"]})
    else:
        miss = sorted({m for o in (fr.get("options") or []) for m in o["components_missing"]})
        out.append({"id": "financial_buffers", "question": "Financial buffers: does the model price freight cost and tariff together?", "status": "null",
                    "answer": "Not modelled yet: some of the cost components have no input, and a missing cost is never counted as zero.",
                    "evidence": [], "inputs_missing": miss or ["batch, tariff and penalty inputs"]})
    # 3 inventory buffer
    if sup and sup.get("component_buffer_days") is not None:
        s = sup["shortfall_days"]
        txt = (f"Buffer held: {sup['component_buffer_days']:g} days. Supplier delay {sup['delay_days']['min']:.0f}–{sup['delay_days']['max']:.0f} days. "
               f"Production stops {s['if_delay_min']:g}–{s['if_delay_max']:g} days for lack of the component"
               + ("" if sup.get("earliest_alternate_day") is None else f" (best known alternate starts day {sup['earliest_alternate_day']:g})") + ". "
               f"To ride through you would need {sup['buffer_needed_days']['if_delay_min']:g}–{sup['buffer_needed_days']['if_delay_max']:g} days of buffer, "
               f"i.e. {sup['buffer_gap_days']['if_delay_min']:g}–{sup['buffer_gap_days']['if_delay_max']:g} days more than you hold.")
        if ops.get("status") == "modelled":
            txt += f" Separately the storm costs {ops['production_days_lost']:g} production-day equivalents (grid, fuel, access)."
        out.append({"id": "inventory_buffer", "question": "Operational redundancy: days of buffer for the critical component?", "status": "modelled", "answer": txt,
                    "evidence": ["supply(): buffer vs delay range vs alternates"], "inputs_missing": []})
    else:
        out.append({"id": "inventory_buffer", "question": "Operational redundancy: days of buffer for the critical component?", "status": "null",
                    "answer": "Not known: the days of component buffer you hold were not given, so the shortfall is not computed. The delay itself is "
                              + ("not given either." if not sup or sup.get("status") == "null" else f"{sup['delay_days']['min']:.0f}–{sup['delay_days']['max']:.0f} days."),
                    "evidence": [], "inputs_missing": ["inventory.component_buffer_days"]})
    # 4 governance
    g = spec.get("governance") or {}
    owner, limit, esc = g.get("decision_owner"), _pos(g.get("authority_limit_usd"), "authority_limit_usd"), _pos(g.get("escalation_hours"), "escalation_hours")
    deadline = None
    miss4 = []
    if fr.get("best") and fr.get("options"):
        sea, air = fr["options"][0], fr["options"][1]
        delta = None if sea["total_extra_cost_usd"] is None or air["total_extra_cost_usd"] is None else abs(air["total_extra_cost_usd"] - sea["total_extra_cost_usd"])
        rule = f"Decision rule from the numbers: choose '{fr['best']['label']}' unless the supplier delay or tariff changes"
        if fr.get("breakeven", {}).get("tariff_rate_pct_where_air_equals_sea") is not None:
            rule += f"; shipping everything by air only beats staying on sea if the tariff is below {fr['breakeven']['tariff_rate_pct_where_air_equals_sea']:.1f}%"
        rule += "."
        txt = rule
        if delta is not None and limit is not None:
            txt += f" The gap between the two routes is ${delta:,.0f}, " + ("within" if delta <= limit else "above") + f" the ${limit:,.0f} authority limit" + ("." if delta <= limit else ", so it needs escalation.")
        elif limit is None:
            miss4.append("governance.authority_limit_usd")
        out_status = "partial" if (not owner or limit is None) else "modelled"
    else:
        txt = "The trade-off is not priced yet, so no decision rule can be derived."
        out_status = "null"
        miss4.append("batch, tariff and penalty inputs")
    if not owner:
        txt += " No decision owner is recorded: that is a governance gap in itself, since a short clock leaves no time to find out who decides."
        miss4.append("governance.decision_owner")
    else:
        txt += f" Decision owner on record: {owner}" + (f", escalation in {esc:g} h." if esc is not None else ".")
    txt += " Reputation cost of a missed delivery is not quantified by the engine; only the contractual penalty is."
    out.append({"id": "governance", "question": "Adaptive governance: who decides between the deadline and the tariff?", "status": out_status, "answer": txt,
                "evidence": ["freight_decision best option and break-even"], "inputs_missing": miss4})
    return out


def hazards_only(spec: dict, hazard_fn: Optional[Callable] = None) -> dict:
    """Standalone: the engine's site hazards for each site in the spec, with the scenario storm placed on each site's own return-period curve."""
    hazard_fn = hazard_fn or _default_hazards
    h = spec.get("hazard") or {}
    wind = _pos(h.get("wind_ms"), "wind_ms")
    if wind is None and h.get("category") is not None:
        c = int(h["category"])
        if c not in CATEGORY_WIND_MS:
            raise StressError("typhoon category must be 1–5")
        wind = sum(CATEGORY_WIND_MS[c]) / 2.0
    return {"scenario_wind_ms": wind, "sites": [_site_hazard(dict(x), wind, hazard_fn) for x in (spec.get("sites") or [])]}


# ── orchestration ────────────────────────────────────────────────────────────

def run(spec: dict, hazard_fn: Optional[Callable] = None) -> dict:
    """Run the linked stress test. `spec` is described in the module docstring and in WORKED_EXAMPLE."""
    if not isinstance(spec, dict):
        raise StressError("spec must be an object")
    hazard_fn = hazard_fn or _default_hazards
    hz_in = spec.get("hazard") or {}
    cat = hz_in.get("category")
    wind = _pos(hz_in.get("wind_ms"), "wind_ms")
    wind_note = None
    if wind is None and cat is not None:
        c = int(cat)
        if c not in CATEGORY_WIND_MS:
            raise StressError("typhoon category must be 1–5")
        lo, hi = CATEGORY_WIND_MS[c]
        wind = (lo + hi) / 2.0
        wind_note = f"Category {c} (Saffir–Simpson, 1-min sustained {lo}–{hi} m/s): midpoint {wind:g} m/s used."
    sites = [dict(s) for s in (spec.get("sites") or [])]
    hz = [_site_hazard(s, wind, hazard_fn) for s in sites]
    ops = operations(spec.get("event") or {})
    sup_site = next((h for h in hz if h.get("role") == "supplier"), None)
    restart = float(ops["full_capacity_from_day"]) if ops.get("status") == "modelled" and ops.get("full_capacity_from_day") is not None else 0.0
    sup = supply(spec.get("supplier") or {}, spec.get("inventory") or {}, sup_site, restart)
    prod_days = _pos((spec.get("batch") or {}).get("production_days"), "production_days")
    assumptions = []
    if prod_days is None:
        assumptions.append("batch.production_days not given: no production time after restart is added, so lateness is a lower bound.")
    if sup.get("status") in ("modelled", "partial") and sup.get("resume_day"):
        restart_lo = sup["resume_day"]["if_delay_min"] + (prod_days or 0.0)
        restart_hi = sup["resume_day"]["if_delay_max"] + (prod_days or 0.0)
    else:
        restart_lo = restart_hi = restart + (prod_days or 0.0)
    # delivery lateness is priced at the planning case (upper end of the supplier delay) and at the lower end
    fr_hi = freight_decision(spec.get("batch") or {}, spec.get("tariff") or {}, spec.get("penalty") or {}, restart_hi)
    fr_lo = freight_decision(spec.get("batch") or {}, spec.get("tariff") or {}, spec.get("penalty") or {}, restart_lo) if restart_lo != restart_hi else None
    ans = answers(spec, hz, ops, sup, fr_hi)
    timeline = [{"phase": "Day 0 — physical shock", "what": [x for x in [wind_note, f"Scenario peak wind {wind:g} m/s" if wind else None] if x]
                 + [h["storm_placement"]["note"] for h in hz if h.get("storm_placement", {}).get("note")][:1],
                 "engine_numbers": [{"site": h["name"], "storm_placement": h.get("storm_placement")} for h in hz if h.get("storm_placement")]},
                {"phase": "Days 1–4 — operations", "what": [], "engine_numbers": ops if ops.get("status") == "modelled" else [ops.get("reason")]},
                {"phase": "Day 4 — supply chain", "what": [], "engine_numbers": sup},
                {"phase": "Day 10 — freight, tariff and cash", "what": [], "engine_numbers": fr_hi}]
    nulls = sorted({m for a in ans for m in a["inputs_missing"]})
    return {"name": spec.get("name") or "Stress test", "as_of": spec.get("as_of"), "sites": hz, "scenario_wind_ms": wind, "wind_note": wind_note,
            "operations": ops, "supply": sup, "freight": fr_hi, "freight_if_supplier_delay_is_shortest": fr_lo, "ship_ready_day": {"low": restart_lo, "high": restart_hi},
            "answers": ans, "assumptions": assumptions, "timeline": timeline, "inputs_missing": nulls,
            "inputs_echo": {k: spec.get(k) for k in ("event", "supplier", "inventory", "batch", "tariff", "penalty", "governance", "finance") if spec.get(k)},
            "rules": ["Hazard numbers, return-period placement and freight emissions are engine results; everything under inputs_echo is what you supplied.",
                      "A missing input is null and the answer says so. A missing cost is never counted as zero.",
                      "Costs of freight, tariff and late penalty are added in one calculation; the isolated figures are shown beside the joint one.",
                      "Screening-grade: it prices the decision on your inputs; it does not forecast the storm, the supplier's recovery or the tariff outcome."]}


# ── worked example (the user's "Squeeze on District 4" scenario, with illustrative inputs) ───────────

WORKED_EXAMPLE = {
    "name": "The Squeeze on District 4 (illustrative inputs — replace with yours)",
    "as_of": "2028-10",
    "hazard": {"kind": "tropical_cyclone", "category": 4},
    "sites": [{"id": "hub", "name": "Contract manufacturing hub, Zone 4", "role": "manufacturing", "lat": 20.80, "lon": 106.78},
              {"id": "tier2", "name": "Tier-2 component supplier", "role": "supplier", "lat": 20.86, "lon": 106.68}],
    "event": {"grid_outage_days": 9, "generator_fuel_hours": 48, "access_cut_days": 5, "workers_absent_pct": 80},
    "supplier": {"name": "Tier-2 component supplier", "component": "specialised component", "delay_weeks_min": 4, "delay_weeks_max": 6,
                 "detection_lag_days": 4,
                 "alternates": [{"name": "Alternate A", "preapproved": False, "backlog_weeks": 8, "qualification_days": 14},
                                {"name": "Alternate B", "preapproved": False, "backlog_weeks": 10, "qualification_days": 14}]},
    "inventory": {"component_buffer_days": 14},
    "batch": {"units": 20000, "mass_t": 40, "value_usd": 4000000, "cogs_usd": 3500000, "embodied_kgco2e_per_unit": 20,
              "production_days": 7, "sea_km": 18000, "air_km": 9500, "sea_days": 35, "air_days": 4, "delivery_window_days": 45,
              "sea_usd_per_t": 150, "air_usd_per_t": 4500},
    "tariff": {"rate_pct": 15, "trigger": {"type": "intensity_threshold", "kgco2e_per_unit": 30}, "scope": "batch"},
    "penalty": {"late_penalty_pct_per_week": 2.0, "penalty_cap_pct": 10, "grace_days": 3},
    "governance": {"decision_owner": None, "authority_limit_usd": None, "escalation_hours": None},
    "finance": {"cash_buffer_usd": 1500000},
}


# ── chat: pull what a narrative states, save it as a draft ────────────────────

_DRAFT = Path.home() / ".climrisk" / "stress_draft.json"


def looks_like_stress_scenario(text: str) -> bool:
    low = (text or "").lower()
    groups = [r"typhoon|hurricane|cyclone|flood|heat ?wave", r"supplier|tier[\s-]?[123]|supply chain", r"tariff|cbam|border|carbon (?:tax|price)|ets",
              r"air freight|sea freight|shipping|logistics", r"stress[\s-]?test|scenario|phase \d"]
    hits = sum(1 for g in groups if re.search(g, low))
    return hits >= 4 and len(low) > 250 or (hits >= 3 and re.search(r"stress[\s-]?test|squeeze|phase \d", low) is not None and len(low) > 250)


def extract_spec(text: str) -> dict:
    """Pick the numbers a scenario narrative states. Anything it does not state stays absent (null), never guessed."""
    low = text.lower()
    spec: dict = {"name": (re.search(r"[\"“]([^\"”]{4,80})[\"”]", text) or [None, None])[1] or "Stress test"}
    m = re.search(r"categor(?:y|ies)\s*([1-5])|cat\s*([1-5])\b", low)
    if m:
        spec["hazard"] = {"kind": "tropical_cyclone", "category": int(m.group(1) or m.group(2))}
    y = re.search(r"\b(20[2-9]\d)\b", text)
    if y:
        spec["as_of"] = y.group(1)
    ev, sup, bat, tar = {}, {}, {}, {}
    m = re.search(r"(\d+(?:\.\d+)?)\s*hours?\s+of\s+(?:fuel|diesel)|(\d+(?:\.\d+)?)\s*hours?\s+(?:of\s+)?fuel", low)
    if m:
        ev["generator_fuel_hours"] = float(m.group(1) or m.group(2))
    m = re.search(r"(\d+(?:\.\d+)?)(?:\s*[-–]\s*|[\s\-–]*to[\s\-–]*)(\d+(?:\.\d+)?)\s*weeks?", low)
    if m:
        sup["delay_weeks_min"], sup["delay_weeks_max"] = float(m.group(1)), float(m.group(2))
    else:
        m = re.search(r"(\d+(?:\.\d+)?)\s*weeks?\s+(?:delay|late)", low)
        if m:
            sup["delay_weeks_min"] = sup["delay_weeks_max"] = float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:green\s+)?(?:border\s+)?tariff|tariff[^.]{0,40}?(\d+(?:\.\d+)?)\s*%", low)
    if m:
        tar["rate_pct"] = float(m.group(1) or m.group(2))
        tar["trigger"] = {"type": "any_air"} if re.search(r"air freight[^.]{0,120}tariff|tariff[^.]{0,120}air freight", low) else {"type": "always"}
    m = re.search(r"(?:day|within)\s*(\d+)\s*days?[^.]{0,60}(?:delivery|window)|delivery window[^.]{0,30}?(\d+)\s*days", low)
    if m:
        bat["delivery_window_days"] = float(m.group(1) or m.group(2))
    if ev:
        spec["event"] = ev
    if sup:
        sup["name"] = "Tier-2 supplier"
        spec["supplier"] = sup
    if bat:
        spec["batch"] = bat
    if tar:
        spec["tariff"] = tar
    return spec


def save_draft(spec: dict) -> None:
    try:
        _DRAFT.parent.mkdir(parents=True, exist_ok=True)
        _DRAFT.write_text(json.dumps(spec))
    except OSError:
        pass


def load_draft() -> Optional[dict]:
    try:
        return json.loads(_DRAFT.read_text())
    except (OSError, ValueError):
        return None
