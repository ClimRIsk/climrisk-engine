"""
One ship, one voyage: fuel, emissions, EU ETS and FuelEU exposure, costs, disruption what-ifs and mitigation.

Derived, not assumed
  fuel         = fuel_per_nm(reference) x (speed / reference speed)^2 x distance      (Admiralty/cube law: power ~ v^3, time ~ 1/v, so fuel per nm ~ v^2; exponent is an input)
                 the reference rate comes from the ship's own record (MRV file or the user), never from a typical-ship table
  emissions    = fuel mass x FuelEU Annex II defaults (lca/marine_fuels.py): tank-to-wake and well-to-wake
  ETS scope    = 100% if both ports are in the EEA, 50% if one is, 0% otherwise; phase-in 40% (2024), 70% (2025), 100% (2026+)
  FuelEU scope = 100% if both ports are in EU Member States, 50% if one is
  costs        = only the components supplied (bunker price, charter/opex per day, canal and port costs, EUA price). Missing ones are listed, not defaulted.
  diversion    = extra distance as a percentage: given by the user, or the +30% East Asia-Europe Cape average the Commission cites from UNCTAD 2025 (flagged as an average)
Not modelled: weather speed loss, hull fouling, load condition, currents, port fuel, shore power, pooling/banking outcomes, penalty (unless a rate is given).
This is a planning screen with stated assumptions, not a noon-report or a verified MRV number.
"""
from __future__ import annotations

import math
from datetime import date
from typing import Optional

from ..lca import marine_fuels as MF

EU27 = {"AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE"}
EEA = EU27 | {"IS", "NO", "LI"}
ETS_PHASE = {2024: 0.4, 2025: 0.7}            # 2026 onwards: 1.0
DIVERSION_DEFAULT_PCT = 30.0
REF_FUELS = ("hfo", "lfo", "mdo_mgo")


class VoyageInputError(ValueError):
    def __init__(self, msg, missing=None):
        super().__init__(msg)
        self.missing = missing or []


def gc_nm(a: dict, b: dict) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (a["lat"], a["lon"], b["lat"], b["lon"]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 3440.065 * math.asin(math.sqrt(h))


def _gc_points(a: dict, b: dict, n: int) -> list:
    la1, lo1, la2, lo2 = map(math.radians, (a["lat"], a["lon"], b["lat"], b["lon"]))
    d = gc_nm(a, b) / 3440.065
    if d == 0:
        return [(a["lat"], a["lon"])]
    out = []
    for i in range(n):
        f = i / (n - 1)
        A, B = math.sin((1 - f) * d) / math.sin(d), math.sin(f * d) / math.sin(d)
        x = A * math.cos(la1) * math.cos(lo1) + B * math.cos(la2) * math.cos(lo2)
        y = A * math.cos(la1) * math.sin(lo1) + B * math.cos(la2) * math.sin(lo2)
        z = A * math.sin(la1) + B * math.sin(la2)
        out.append((math.degrees(math.atan2(z, math.hypot(x, y))), math.degrees(math.atan2(y, x))))
    return out


def scope_share(o_cc: Optional[str], d_cc: Optional[str], zone: set) -> Optional[float]:
    if not o_cc or not d_cc:
        return None
    oi, di = o_cc.upper() in zone, d_cc.upper() in zone
    return 1.0 if (oi and di) else 0.5 if (oi or di) else 0.0


def suez_relevant(o: dict, d: dict) -> bool:
    """Heuristic only: one port on the Atlantic/Mediterranean side of Suez (lon -30..40, lat>5) and the other east of it (lon 40..150, lat>-12)."""
    def west(p): return -30 <= p["lon"] <= 40 and p["lat"] > 5
    def east(p): return 40 < p["lon"] <= 150 and p["lat"] > -12
    return (west(o) and east(d)) or (west(d) and east(o))


def _burn(v: dict, dist: float, speed: float, exp: float) -> dict:
    vref, k = v.get("reference_speed_kn"), v.get("fuel_kg_per_nm_ref")
    miss = [n for n, x in (("reference speed (kn)", vref), ("fuel per nm at that speed, or daily consumption at a service speed", k)) if not x]
    if miss:
        raise VoyageInputError("the ship's fuel record is incomplete", miss)
    fuel_t = k * (speed / vref) ** exp * dist / 1000.0
    return {"fuel_t": fuel_t, "hours": dist / speed, "days": dist / speed / 24.0, "kg_per_nm": k * (speed / vref) ** exp}


def _emis(fuel_key: str, tonnes: float, year: int) -> dict:
    r = MF.wtw(fuel_key)
    e_mj = tonnes * 1e6 * r["lcv_mj_per_g"]
    si = MF.ship_intensity([{"fuel": fuel_key, "tonnes": tonnes}], year)
    return {"fuel": r["label"], "fuel_key": fuel_key, "tonnes": tonnes, "energy_mj": e_mj, "ttw_co2_t": e_mj * r["ttw_co2"] / 1e6, "ttw_co2e_t": si["ttw_tco2e"], "wtw_co2e_t": si["wtw_tco2e"],
            "intensity": si["ghg_intensity_gco2e_per_mj"], "limit": si["fueleu_limit_gco2e_per_mj"], "notes": si["notes"]}


def _price_eua(market: dict):
    if market.get("eua_price") is not None:
        return float(market["eua_price"]), market.get("currency", "EUR"), "given"
    try:
        from ..carbon_markets import _eua_price_default
        p = _eua_price_default()
        if p is not None:
            return float(p), "USD", "engine default: latest World Bank Carbon Pricing Dashboard EU ETS price (USD/t); give a current EUA price to override"
    except Exception:                                                  # noqa: BLE001
        pass
    return None, None, "no price available"


def _costs(fuel_t, days, ets_t, ets_price, ets_ccy, market: dict, extra_days: float = 0.0, extra_fuel_t: float = 0.0) -> dict:
    """Only supplied components. Returns lines and a total over those lines."""
    lines, missing = [], []
    bp = market.get("bunker_usd_per_t")
    if bp is not None:
        lines.append({"item": "Bunker fuel", "usd": fuel_t * bp, "basis": f"{fuel_t:,.0f} t x {bp:,.0f} USD/t (price you gave)"})
    else:
        missing.append("bunker price (USD/t)")
    cd = market.get("charter_usd_per_day")
    if cd is not None:
        lines.append({"item": "Time (charter hire or daily opex)", "usd": days * cd, "basis": f"{days:.1f} days x {cd:,.0f} USD/day (you gave)"})
    else:
        missing.append("daily charter hire or opex (USD/day)")
    if ets_t and ets_price is not None:
        lines.append({"item": f"EU ETS allowances ({ets_ccy})", "usd": None, ets_ccy.lower(): ets_t * ets_price, "basis": f"{ets_t:,.0f} t x {ets_price:,.2f} {ets_ccy}/t"})
    for k, lab in (("canal_usd", "Canal tolls"), ("port_usd", "Port and agency costs"), ("other_usd", "Other costs you gave")):
        if market.get(k) is not None:
            lines.append({"item": lab, "usd": float(market[k]), "basis": "as given"})
    usd = sum(l["usd"] for l in lines if l.get("usd") is not None)
    ets_line = next((l for l in lines if l["item"].startswith("EU ETS")), None)
    return {"lines": lines, "total_usd_supplied_components": usd, "ets_in_own_currency": ({ets_ccy: ets_line[ets_ccy.lower()]} if ets_line else None), "missing_inputs": missing,
            "note": "Total = the USD components above. ETS is shown in its own currency and not added: no exchange rate was supplied."}


def _cyclones(o: dict, d: dict, n: int = 9) -> dict:
    try:
        from ..finance.analyst import _tc_params
        pts = []
        for lat, lon in _gc_points(o, d, n):
            p = _tc_params(lat, lon)
            pts.append({"lat": round(lat, 1), "lon": round(((lon + 180) % 360) - 180, 1), "storms_per_year": p["lam"], "rp100_wind_ms": p["rp100_ms"], "category_100yr": p["category_100yr"]})
        top = sorted(pts, key=lambda x: -x["storms_per_year"])[:3]
        return {"samples": len(pts), "most_exposed": top, "max_storms_per_year": top[0]["storms_per_year"] if top else None,
                "note": "Sampled along the great circle, which is not the sailed track (it crosses land on many routes). Rates are annual climatology from the engine's tropical-cyclone model, not a seasonal forecast; the season and the actual track matter."}
    except Exception as e:                                           # noqa: BLE001
        return {"error": f"{type(e).__name__}: {str(e)[:120]}", "note": "Cyclone exposure not assessed (null, not zero)."}


def assess_voyage(vessel: dict, distance_nm: Optional[float], speed_kn: Optional[float] = None, fuel: Optional[str] = None,
                  origin: Optional[dict] = None, dest: Optional[dict] = None, year: Optional[int] = None, market: Optional[dict] = None,
                  speed_options_kn: Optional[list] = None, delay_days: Optional[list] = None, diversion_pct: Optional[float] = None,
                  speed_exponent: float = 2.0, fueleu_penalty_eur_per_t_vlsfoe: Optional[float] = None) -> dict:
    """vessel: {name, imo, gt, reference_speed_kn, fuel_kg_per_nm_ref, design_speed_kn?}; origin/dest: {name, lat, lon, country_code}. Raises VoyageInputError if something essential is missing."""
    market = dict(market or {})
    year = year or date.today().year
    origin, dest = origin or {}, dest or {}
    limits, flags = [], []
    gc = None
    if origin.get("lat") is not None and dest.get("lat") is not None:
        gc = gc_nm(origin, dest)
    if not distance_nm:
        if gc is None:
            raise VoyageInputError("no distance and no port positions", ["distance (nm) or both port positions"])
        distance_nm = gc
        flags.append(f"Distance is the great-circle {gc:,.0f} nm between the ports, a lower bound: the sailed route is longer. Give the route distance for a firm number.")
    elif gc is not None and distance_nm < gc * 0.98:
        flags.append(f"The distance given ({distance_nm:,.0f} nm) is shorter than the great-circle distance ({gc:,.0f} nm): check it.")
    speed = speed_kn or vessel.get("reference_speed_kn")
    if not speed:
        raise VoyageInputError("no speed", ["speed (kn)"])
    if not speed_kn:
        flags.append(f"Speed not given: the ship's reference speed ({speed:.1f} kn) is used.")
    vref = vessel.get("reference_speed_kn")
    if vref and (speed < 0.5 * vref or (vessel.get("design_speed_kn") and speed > vessel["design_speed_kn"])):
        flags.append("The speed is far from the ship's reference speed or above its design speed: the v^2 scaling is an extrapolation there.")
    if vessel.get("class_proxy"):
        flags += list(vessel.get("notes") or [])
    b = _burn(vessel, distance_nm, speed, speed_exponent)
    fuel_given = fuel is not None
    fuel_key = fuel or "hfo"
    if not fuel_given:
        flags.append("Fuel type not given: HFO is the reference fuel for the headline, and the table shows LFO and MDO/MGO as well.")
    E = _emis(fuel_key, b["fuel_t"], year)
    alt_fuels = [] if fuel_given else [_emis(k, b["fuel_t"], year) for k in REF_FUELS if k != fuel_key]
    # scope
    s_ets, s_fe = scope_share(origin.get("country_code"), dest.get("country_code"), EEA), scope_share(origin.get("country_code"), dest.get("country_code"), EU27)
    phase = ETS_PHASE.get(year, 1.0 if year >= 2026 else None)
    ets = {"scope_share": s_ets, "phase_in": phase}
    px, ccy, pbasis = _price_eua(market)
    if s_ets is None:
        ets["status"] = "unknown: the country of one port was not resolved. Give it (for example 'Rotterdam, NL')."
        ets_t = None
    else:
        basis_t = E["ttw_co2_t"] if year <= 2025 else E["ttw_co2e_t"]
        ets_t = basis_t * s_ets * (phase or 0)
        ets.update({"basis": "CO2 only (2024-25)" if year <= 2025 else "CO2 + CH4 + N2O from 2026 (as I understand the Directive; not confirmed in the retrieved text)",
                    "chargeable_t": ets_t, "price": px, "price_currency": ccy, "price_basis": pbasis, "cost": (ets_t * px if px is not None else None),
                    "status": "applies" if ets_t else "no ETS surrender for this voyage",
                    "note": "Applies to ships of 5,000 GT and above" + ("" if vessel.get("gt") is None else (f"; this ship: {vessel['gt']:,.0f} GT" + ("" if vessel['gt'] >= 5000 else " (below the threshold)")))
                            + ". At-berth and in-port emissions at EEA ports are also in scope and are not in this voyage figure."})
        if vessel.get("gt") is not None and vessel["gt"] < 5000:
            ets["chargeable_t"], ets["cost"], ets["status"] = 0.0, 0.0, "below 5,000 GT: out of scope"
            ets_t = 0.0
    # FuelEU
    fe = {"scope_share": s_fe}
    if s_fe is None:
        fe["status"] = "unknown: the country of one port was not resolved."
    else:
        e_scope = E["energy_mj"] * s_fe
        bal = (E["limit"] - E["intensity"]) * e_scope / 1e6
        fe.update({"intensity_gco2e_per_mj": E["intensity"], "limit_gco2e_per_mj": E["limit"], "energy_in_scope_mj": e_scope, "compliance_balance_tco2e": bal,
                   "status": ("surplus" if bal >= 0 else "deficit") if s_fe else "no FuelEU energy in scope",
                   "note": "Voyage-level contribution to the ship's annual balance, not the annual result: other voyages, at-berth energy, pooling and banking decide the year."})
        if fueleu_penalty_eur_per_t_vlsfoe is not None and bal < 0:
            lcv_vlsfo = 41000.0
            fe["penalty_eur"] = abs(bal) * 1e6 / (E["intensity"] * lcv_vlsfo) * fueleu_penalty_eur_per_t_vlsfoe
            fe["penalty_basis"] = "|balance| / (actual intensity x 41,000 MJ/t) x the rate you gave: my reading of FuelEU Annex IV, not in the retrieved text; confirm before relying on it."
        else:
            fe["penalty_eur"] = None
            fe["penalty_basis"] = "not computed: the penalty rate was not in the retrieved text and none was given."
    cost = _costs(b["fuel_t"], b["days"], ets_t, px, ccy, market)
    # scenarios
    scen = []
    for sp in (speed_options_kn or [round(speed * 0.9, 1), round(speed * 0.8, 1)]):
        if sp <= 0 or abs(sp - speed) < 1e-9:
            continue
        bb = _burn(vessel, distance_nm, sp, speed_exponent)
        ee = _emis(fuel_key, bb["fuel_t"], year)
        ets_s = None if s_ets is None else (ee["ttw_co2_t"] if year <= 2025 else ee["ttw_co2e_t"]) * s_ets * (phase or 0)
        cs = _costs(bb["fuel_t"], bb["days"], ets_s, px, ccy, market)
        scen.append({"name": f"Speed {sp:g} kn", "speed_kn": sp, "days": bb["days"], "fuel_t": bb["fuel_t"], "ttw_co2e_t": ee["ttw_co2e_t"], "wtw_co2e_t": ee["wtw_co2e_t"],
                     "d_days": bb["days"] - b["days"], "d_fuel_t": bb["fuel_t"] - b["fuel_t"], "d_ttw_co2e_t": ee["ttw_co2e_t"] - E["ttw_co2e_t"], "d_cost_usd": (cs["total_usd_supplied_components"] - cost["total_usd_supplied_components"]) if cost["lines"] else None,
                     "complete": not cs["missing_inputs"]})
    div = None
    pct = diversion_pct if diversion_pct is not None else (DIVERSION_DEFAULT_PCT if (origin.get("lat") is not None and dest.get("lat") is not None and suez_relevant(origin, dest)) else None)
    if pct is not None:
        dd = distance_nm * (1 + pct / 100.0)
        bd = _burn(vessel, dd, speed, speed_exponent)
        ed = _emis(fuel_key, bd["fuel_t"], year)
        ets_d = None if s_ets is None else (ed["ttw_co2_t"] if year <= 2025 else ed["ttw_co2e_t"]) * s_ets * (phase or 0)
        cd = _costs(bd["fuel_t"], bd["days"], ets_d, px, ccy, market)
        div = {"extra_distance_pct": pct, "distance_nm": dd, "days": bd["days"], "fuel_t": bd["fuel_t"], "ttw_co2e_t": ed["ttw_co2e_t"], "d_days": bd["days"] - b["days"], "d_fuel_t": bd["fuel_t"] - b["fuel_t"],
               "d_ttw_co2e_t": ed["ttw_co2e_t"] - E["ttw_co2e_t"], "d_cost_usd": (cd["total_usd_supplied_components"] - cost["total_usd_supplied_components"]) if cost["lines"] else None,
               "basis": ("your figure" if diversion_pct is not None else f"+{DIVERSION_DEFAULT_PCT:g}%: the East Asia-Europe Cape-of-Good-Hope average the Commission cites from UNCTAD 2025; this route may differ. The Suez/Red Sea route is judged from port positions only (heuristic)."),
               "note": "The extra distance is applied to the distance you gave: if that figure already is the Cape route, ignore this block. Suez canal tolls avoided are not netted off (they stay in the base case if you gave canal_usd)."}
    delays = []
    for dl in (delay_days if delay_days is not None else [1, 3, 7]):
        rem_h = (b["days"] - dl) * 24.0
        entry = {"delay_days": dl, "label": "sensitivity grid, not a forecast" if delay_days is None else "your figure"}
        if rem_h > 0 and dl > 0:
            # a ship that is dl days late over the full voyage must sail faster to arrive on the original schedule
            v_req = distance_nm / rem_h
            bb = _burn(vessel, distance_nm, v_req, speed_exponent)
            entry.update({"speed_to_recover_kn": v_req, "extra_fuel_t": bb["fuel_t"] - b["fuel_t"], "extra_ttw_co2e_t": _emis(fuel_key, bb["fuel_t"], year)["ttw_co2e_t"] - E["ttw_co2e_t"],
                          "extra_bunker_usd": ((bb["fuel_t"] - b["fuel_t"]) * market["bunker_usd_per_t"] if market.get("bunker_usd_per_t") is not None else None),
                          "above_design_speed": bool(vessel.get("design_speed_kn") and v_req > vessel["design_speed_kn"])})
        if market.get("charter_usd_per_day") is not None:
            entry["time_cost_of_delay_usd"] = dl * market["charter_usd_per_day"]
        delays.append(entry)
    cyc = _cyclones(origin, dest) if gc is not None else {"note": "Cyclone exposure not assessed: port positions not resolved."}
    # mitigation
    alts = []
    for k in ("lng_otto_medium", "lng_lbsi", "methanol_ng", "lpg_propane", "nh3_ng_fuel_cell", "h2_ng_fuel_cell"):
        try:
            r = MF.wtw(k)
        except Exception:                                              # noqa: BLE001
            continue
        alts.append({"fuel": r["label"], "wtw_gco2e_per_mj": r["wtw"], "ttw_gco2e_per_mj": r["ttw"], "wtw_vs_current_pct": 100.0 * (r["wtw"] / E["intensity"] - 1.0),
                     "wtw_tco2e_same_energy": E["energy_mj"] * r["wtw"] / 1e6, "meets_fueleu_limit": r["wtw"] <= E["limit"], "cost": "needs a price per tonne from you; fuel mass for the same energy = energy / LCV"})
    mit = []
    if scen:
        best = min(scen, key=lambda x: x["fuel_t"])
        mit.append({"measure": "Slow steaming", "effect": f"{best['name']}: fuel {best['d_fuel_t']:+,.0f} t, TtW CO2e {best['d_ttw_co2e_t']:+,.0f} t, time {best['d_days']:+.1f} days" + (f", cost {best['d_cost_usd']:+,.0f} USD (supplied components)" if best['d_cost_usd'] is not None else ""),
                    "caveat": "Time has a cost (charter, cargo, schedule). The v^2 scaling is a first-order approximation."})
    if fe.get("compliance_balance_tco2e") is not None and fe["compliance_balance_tco2e"] < 0:
        mit.append({"measure": "FuelEU: pooling or a lower-WtW fuel", "effect": f"voyage deficit {abs(fe['compliance_balance_tco2e']):,.0f} tCO2e. Options in the Regulation as I understand it: pool with a surplus ship, bank/borrow within limits, or burn a lower-intensity fuel (table).", "caveat": "Regulation text not retrieved here; confirm Articles 20-21 before acting."})
    if ets_t:
        mit.append({"measure": "EU ETS: reduce chargeable emissions", "effect": f"every 1% of fuel saved on the in-scope voyage saves ~{ets_t*0.01:,.0f} t of allowances" + (f" (~{ets_t*0.01*px:,.0f} {ccy})" if px is not None else ""), "caveat": "Do not restructure port calls to dodge scope: the Regulation has anti-circumvention provisions."})
    if delays:
        mit.append({"measure": "Recover delay by speeding up", "effect": "see the delay table: fuel rises roughly with the square of speed, so catching up costs far more than the delay's time cost unless the cargo or berth window is very valuable.", "caveat": "Check against design speed and the charter's speed warranty."})
    mit.append({"measure": "Weather and Red Sea risk", "effect": "weather routing and early diversion decisions cannot be quantified from the data here; the cyclone table shows where on the great-circle line the storm climatology is highest.", "caveat": "null, not zero"})
    limits = ["Fuel burn scales from the ship's own reference record by v^2; weather, fouling, load and currents are not modelled.",
              "Emission factors are FuelEU Annex II defaults (fossil pathways); biofuels and e-fuels need certified values.",
              "Distance is as given, or the great-circle lower bound.",
              "ETS and FuelEU scope uses the country of each port as resolved; transhipment, outermost-region and last-port-of-call rules are not modelled.",
              "Costs include only the components supplied; there is no default price anywhere in this calculation.",
              "No claim of an MRV-verified figure: this is a planning screen."]
    return {"vessel": {k: vessel.get(k) for k in ("name", "imo", "ship_type", "gt", "dwt", "class_proxy", "reference_speed_kn", "fuel_kg_per_nm_ref", "design_speed_kn", "notes")},
            "voyage": {"origin": origin.get("name"), "destination": dest.get("name"), "distance_nm": distance_nm, "great_circle_nm": gc, "speed_kn": speed, "days": b["days"], "year": year},
            "fuel": {"t": b["fuel_t"], "kg_per_nm": b["kg_per_nm"], "type": E["fuel"]}, "emissions": {"ttw_co2e_t": E["ttw_co2e_t"], "wtw_co2e_t": E["wtw_co2e_t"], "ttw_co2_t": E["ttw_co2_t"], "intensity_gco2e_per_mj": E["intensity"], "alt_reference_fuels": alt_fuels, "notes": E["notes"]},
            "ets": ets, "fueleu": fe, "cost": cost, "scenarios": {"speed": scen, "diversion": div, "delay": delays}, "cyclones": cyc, "mitigation": mit, "alternative_fuels": alts, "flags": flags, "limits": limits,
            "sources": ["FuelEU Annex II defaults via the Commission guidance (lca/marine_fuels.py)", "EU ETS / FuelEU maritime scope: European Commission pages as retrieved 28 Oct 2026", "Diversion: UNCTAD 2025 via the Commission (average, not this route)", "Cyclones: engine tropical-cyclone model"]}
