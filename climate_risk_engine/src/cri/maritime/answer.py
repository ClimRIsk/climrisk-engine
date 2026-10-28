"""
Plain-language front end to the voyage model: reads a sentence like
"MV Aurora is docked at the Port of Singapore and is moving to Rotterdam, 16000 NM, 14 knots",
finds the ship, resolves the ports, and returns a reply plus tables. Used by the resident agent and the API.

What it will not do: invent the ship's fuel burn. If no record is found it still gives everything that does not depend on it
(timing, scope under EU ETS/FuelEU, disruption exposure, emissions per tonne of fuel) and names exactly what to supply.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

from . import classes as CL
from . import vessel as VS
from . import voyage as V
from ..lca import marine_fuels as MF

_SHIP = re.compile(r"\b(ship|vessel|tanker|bulker|bulk carrier|container ?ship|boxship|ro-?ro|carrier|MV|M/V|MT|M/T)\b", re.I)
_VOY = re.compile(r"\b(\d[\d,\.]*\s*(nm|nautical)|docked|berthed|moored|sailing|voyage|moving to|heading to|bound for|steam(?:ing)? to|port of)\b", re.I)
_NAME_STOP = {"is", "was", "has", "will", "which", "that", "moves", "moving", "sails", "sailing", "docked", "berthed", "moored", "currently", "now", "at", "in", "and", "from", "to", "with", "of", "the", "a", "an", "carrying", "loaded", "ballast", "heading", "bound", "going", "if", "how", "what"}
FUEL_WORDS = [(r"\b(vlsfo|hsfo|ifo|hfo|heavy fuel|fuel oil|residual)\b", "hfo"), (r"\b(lfo|light fuel)\b", "lfo"), (r"\b(mgo|mdo|marine gas ?oil|marine diesel|diesel)\b", "mdo_mgo"),
              (r"\b(lng)\b", "lng_otto_medium"), (r"\bmethanol\b", "methanol_ng"), (r"\bammonia\b", "nh3_ng_fuel_cell"), (r"\blpg\b", "lpg_propane")]


def matches(text: str) -> bool:
    t = text or ""
    return bool(_SHIP.search(t) and _VOY.search(t) and re.search(r"\bto\b|bound for|towards?", t, re.I)) or bool(re.search(r"\bIMO\s*\d{7}\b", t, re.I) and _VOY.search(t))


def _num(s):
    return float(s.replace(",", "")) if s else None


def parse(text: str) -> dict:
    t = text
    out = {"flags": []}
    m = re.search(r"\bIMO\s*(\d{7})\b", t, re.I)
    if m:
        out["imo"] = m.group(1)
    nm = re.search(r"\b(?i:ship|vessel|tanker|bulker|carrier|container ?ship)\s+(?i:called|named)?\s*[\"“']?([A-Z0-9][\w\-']*(?:\s+[A-Z0-9][\w\-']*){0,3})", t) or re.search(r"\b(?:MV|M/V|MT|M/T)\s+([A-Z0-9][\w\-']*(?:\s+[A-Z0-9][\w\-']*){0,3})", t)
    if nm:
        words = []
        for w in nm.group(1).split():
            if w.lower() in _NAME_STOP:
                break
            words.append(w)
        while words and words[0] in ("MV", "M/V", "MT", "M/T"):
            words = words[1:]
        if words and not re.fullmatch(r"(?i)imo", words[0]):
            out["name"] = " ".join(words)
    m = re.search(r"\b(?:docked|berthed|moored|alongside|lying|located|sitting|currently)\b[^.]*?\b(?:in|at)\s+(?:the\s+)?(?:[Pp]ort\s+of\s+)?([A-Z][\w\-']+(?:\s+[A-Z][\w\-']+){0,2})", t) or re.search(r"\bfrom\s+(?:the\s+)?(?:[Pp]ort\s+of\s+)?([A-Z][\w\-']+(?:\s+[A-Z][\w\-']+){0,2})", t)
    if m:
        out["origin"] = re.sub(r"\s+Port$", "", m.group(1)).strip()
    m = re.search(r"\b(?:moving|moves|move|sail(?:s|ing)?|head(?:s|ing)?|going|go|bound|steam(?:s|ing)?|voyage|travel(?:s|ling)?|transit(?:s|ing)?|proceed(?:s|ing)?)\b[^.]*?\b(?:to|for|towards?)\s+(?:the\s+)?(?:[Pp]ort\s+of\s+)?([A-Z][\w\-']+(?:\s+[A-Z][\w\-']+){0,2})", t) or re.search(r"\bto\s+(?:the\s+)?(?:[Pp]ort\s+of\s+)?([A-Z][\w\-']+(?:\s+[A-Z][\w\-']+){0,2})", t)
    if m:
        out["dest"] = re.sub(r"\s+Port$", "", m.group(1)).strip()
    m = re.search(r"(\d[\d,\.]*)\s*(?:nm|n\.?m\.?|nautical\s+miles?)\b", t, re.I)
    if m:
        out["distance_nm"] = _num(m.group(1))
    m = re.search(r"(?:at|speed(?: of)?|steaming at|\bwith)\s*(\d+(?:\.\d+)?)\s*(?:kn|kts?|knots?)\b", t, re.I)
    if m:
        out["speed_kn"] = _num(m.group(1))
    for rx, k in FUEL_WORDS:
        if re.search(rx, t, re.I):
            out["fuel"] = k
            if k == "lng_otto_medium":
                out["flags"].append("LNG engine type not given: a medium-speed Otto engine (methane slip default) is used. Give the engine type for a firm number.")
            if re.search(r"vlsfo|hsfo|ifo", t, re.I):
                out["flags"].append("VLSFO/HSFO/IFO grades are treated as the FuelEU 'HFO' class (RME-RMK); confirm for your grade.")
            break
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:t|tonnes?|tons?|mt)\s*(?:/|per|a)\s*day(?:[^.]*?\bat\s+(\d+(?:\.\d+)?)\s*(?:kn|kts?|knots?))?", t, re.I)
    if m:
        out["daily_t"] = _num(m.group(1))
        if m.group(2):
            out["daily_speed_kn"] = _num(m.group(2))
    for key, rx in (("dwt", r"(\d[\d,\.]*)\s*(k?)\s*(?:dwt|deadweight)"), ("teu", r"(\d[\d,\.]*)\s*(k?)\s*teu"), ("gt", r"(\d[\d,\.]*)\s*(k?)\s*(?:gt|gross\s+ton)"), ("cbm", r"(\d[\d,\.]*)\s*(k?)\s*(?:cbm|m3|m³)")):
        m = re.search(rx, t, re.I)
        if m:
            out[key] = _num(m.group(1)) * (1000.0 if m.group(2) else 1.0)
    st = CL.classify(t)
    if st:
        out["ship_type"] = st
    mk = {}
    m = re.search(r"(?:bunker|fuel)[^.\d$€]{0,25}(?:price|cost|at)?[^.\d$€]{0,8}[$€]?\s*(\d[\d,\.]*)\s*(?:usd)?\s*(?:/|per)\s*(?:t|tonne|ton|mt)", t, re.I) or re.search(r"[$]\s*(\d[\d,\.]*)\s*(?:/|per)\s*(?:t|tonne|ton|mt)\b", t, re.I)
    if m:
        mk["bunker_usd_per_t"] = _num(m.group(1))
    m = re.search(r"(?:charter|hire|opex|day rate)[^.\d$]{0,25}[$]?\s*(\d[\d,\.]*)\s*(?:usd)?\s*(?:/|per)\s*day", t, re.I) or re.search(r"[$]\s*(\d[\d,\.]*)\s*(?:/|per)\s*day\b", t, re.I)
    if m:
        mk["charter_usd_per_day"] = _num(m.group(1))
    m = re.search(r"(?:eua|ets|allowance|carbon)[^.\d$€]{0,20}[€$]?\s*(\d+(?:\.\d+)?)\s*(?:eur|€|usd|\$)?\s*(?:/|per)\s*t", t, re.I)
    if m:
        mk["eua_price"] = _num(m.group(1)); mk["currency"] = "EUR" if re.search(r"€|eur", t, re.I) else "USD"
    out["market"] = mk
    m = re.search(r"\b(20[2-4]\d)\b", t)
    if m:
        out["year"] = int(m.group(1))
    return out


def _place(name: str, geocode: Callable) -> Optional[dict]:
    if not name:
        return None
    for q in (f"port of {name}", name):
        g = geocode(q)
        if g:
            return {"name": name, "lat": g["lat"], "lon": g["lon"], "country_code": (g.get("country_code") or "").upper() or None, "source": "OpenStreetMap geocoder"}
    return None


def resolve_vessel(p: dict, data_dir=None, fleets=None) -> dict:
    """Vessel record from user-supplied consumption, else the fleet files. Returns {vessel|None, how, missing, candidates, files}."""
    key = p.get("imo") or p.get("name")
    v = None
    how = None
    cand, files, why = [], [], None
    if key:
        f = VS.find(key, data_dir, fleets)
        v, cand, files, why = f["vessel"], f["candidates"], f["files"], f["why"]
        how = f"fleet file ({', '.join(files)})" if v else None
    if p.get("daily_t"):
        spd = p.get("daily_speed_kn") or p.get("speed_kn")
        if spd:
            rec = dict(v or {"name": p.get("name"), "imo": p.get("imo")})
            rec.update({"reference_speed_kn": spd, "fuel_kg_per_nm_ref": p["daily_t"] * 1000.0 / (24.0 * spd)})
            rec["notes"] = list(rec.get("notes") or []) + ["fuel rate from the daily consumption you gave"]
            return {"vessel": rec, "how": "daily consumption you gave", "missing": [], "candidates": [], "files": files, "why": None}
    if v and not v.get("missing"):
        return {"vessel": v, "how": how, "missing": [], "candidates": [], "files": files, "why": None}
    if p.get("ship_type") and p.get("speed_kn"):
        unit = CL.UNIT_OF[p["ship_type"]]
        try:
            cv = CL.vessel_from_class(p["ship_type"], p["speed_kn"], size=p.get(unit), size_unit=unit, dwt=p.get("dwt"), name=p.get("name"))
            if v:
                cv.update({k: v[k] for k in ("name", "imo", "gt") if v.get(k)})
            return {"vessel": cv, "how": "IMO DCS 2022 class median (no record for this ship)", "missing": [], "candidates": [], "files": files, "why": None}
        except ValueError as e:
            return {"vessel": v, "how": how, "missing": [str(e)], "candidates": cand, "files": files, "why": why}
    miss = list(v["missing"]) if v else ["the ship's daily fuel consumption and the speed it applies to (for example '45 t/day at 14 knots'), or its type and size (for example 'container ship 8000 TEU', 'bulk carrier 75,000 dwt') plus a speed, for an IMO class-median estimate"]
    return {"vessel": v, "how": how, "missing": miss, "candidates": cand, "files": files, "why": why}


def _fmt(x, d=0):
    return "n/a" if x is None else f"{x:,.{d}f}"


def answer(text: str, geocode: Callable, params: Optional[dict] = None, data_dir=None, fleets=None) -> dict:
    p = parse(text)
    if params:
        p.update({k: v for k, v in params.items() if v is not None})
    blocks, flags = [], list(p.get("flags", []))
    o, d = _place(p.get("origin"), geocode), _place(p.get("dest"), geocode)
    if not p.get("dest") or not p.get("origin"):
        return {"reply": "I need both ports: where the ship is now and where it is going (for example 'docked at Singapore, moving to Rotterdam, 16000 nm').", "blocks": [], "actions": [], "result": None}
    if not o or not d:
        bad = [n for n, x in ((p["origin"], o), (p["dest"], d)) if not x]
        return {"reply": f"I could not place {', '.join(bad)} on the map. Give the port with its country (for example 'Rotterdam, Netherlands') or coordinates.", "blocks": [], "actions": [], "result": None}
    rv = resolve_vessel(p, data_dir, fleets)
    market = p.get("market") or {}
    if rv["vessel"] and not rv["missing"]:
        try:
            r = V.assess_voyage(rv["vessel"], p.get("distance_nm"), p.get("speed_kn"), p.get("fuel"), o, d, p.get("year"), market, diversion_pct=p.get("diversion_pct"),
                                fueleu_penalty_eur_per_t_vlsfoe=p.get("fueleu_penalty"))
        except V.VoyageInputError as e:
            return {"reply": f"The voyage cannot be computed yet: {e}. Missing: {'; '.join(e.missing)}.", "blocks": [], "actions": [], "result": None}
        r["flags"] = flags + r["flags"]
        return _render_full(r, rv, o, d)
    return _render_partial(p, o, d, rv, flags, market)


def _render_full(r, rv, o, d) -> dict:
    vy, E, ets, fe, c = r["voyage"], r["emissions"], r["ets"], r["fueleu"], r["cost"]
    nm = r["vessel"].get("name") or r["vessel"].get("imo") or "The ship"
    rows = [["Distance", f"{_fmt(vy['distance_nm'])} nm", "as given" if vy["distance_nm"] != vy["great_circle_nm"] else "great-circle lower bound"],
            ["Speed / time", f"{vy['speed_kn']:.1f} kn / {vy['days']:.1f} days", "time = distance / speed"],
            ["Fuel burned", f"{_fmt(r['fuel']['t'])} t {r['fuel']['type']}", f"{r['fuel']['kg_per_nm']:.0f} kg/nm; ship record scaled by v^2 ({rv['how']})"],
            ["CO2e, tank-to-wake", f"{_fmt(E['ttw_co2e_t'])} t", "what leaves the funnel (CO2 + CH4 + N2O, AR4 GWP)"],
            ["CO2e, well-to-wake", f"{_fmt(E['wtw_co2e_t'])} t", "includes fuel production and supply"]]
    blocks = [{"type": "table", "head": ["Item", "Value", "Basis"], "rows": rows, "caption": f"{nm}: {vy['origin']} to {vy['destination']}, {vy['year']}"}]
    ets_txt = (f"{ets['status']}: {ets['scope_share']*100:.0f}% of the voyage in scope, phase-in {ets['phase_in']*100:.0f}%, {_fmt(ets.get('chargeable_t'))} t chargeable" + (f", about {_fmt(ets['cost'])} {ets['price_currency']}" if ets.get("cost") is not None else "")) if ets.get("scope_share") is not None else ets.get("status")
    fe_txt = (f"{fe['status']}: {fe['scope_share']*100:.0f}% of energy in scope, intensity {fe['intensity_gco2e_per_mj']:.2f} vs limit {fe['limit_gco2e_per_mj']:.2f} gCO2e/MJ, voyage balance {_fmt(fe['compliance_balance_tco2e'])} tCO2e") if fe.get("scope_share") is not None else fe.get("status")
    blocks.append({"type": "table", "head": ["Regime", "Result"], "rows": [["EU ETS", ets_txt], ["FuelEU Maritime", fe_txt]], "caption": (ets.get("price_basis") or "")})
    if c["lines"]:
        blocks.append({"type": "table", "head": ["Cost line", "USD", "Basis"], "rows": [[l["item"], _fmt(l["usd"]) if l.get("usd") is not None else "see note", l["basis"]] for l in c["lines"]] + [["Total of USD lines", _fmt(c["total_usd_supplied_components"]), c["note"]]]})
    if c["missing_inputs"]:
        blocks.append({"type": "note", "text": "Not costed, because no figure was given: " + "; ".join(c["missing_inputs"]) + ". Nothing was assumed."})
    sc = r["scenarios"]
    rows = []
    for s in sc["speed"]:
        rows.append([s["name"], f"{s['d_days']:+.1f} d", f"{s['d_fuel_t']:+,.0f} t", f"{s['d_ttw_co2e_t']:+,.0f} t", _fmt(s["d_cost_usd"]) if s["d_cost_usd"] is not None else "n/a"])
    if sc["diversion"]:
        x = sc["diversion"]
        rows.append([f"Diversion +{x['extra_distance_pct']:.0f}% distance", f"{x['d_days']:+.1f} d", f"{x['d_fuel_t']:+,.0f} t", f"{x['d_ttw_co2e_t']:+,.0f} t", _fmt(x["d_cost_usd"]) if x["d_cost_usd"] is not None else "n/a"])
    if rows:
        blocks.append({"type": "table", "head": ["Disruption / option", "Time", "Fuel", "TtW CO2e", "Cost (USD lines)"], "rows": rows,
                       "caption": (sc["diversion"]["basis"] + " " + sc["diversion"]["note"]) if sc["diversion"] else "Speed options scaled by v^2."})
    if sc["delay"]:
        blocks.append({"type": "table", "head": ["Delay", "Speed to recover it", "Extra fuel", "Extra TtW CO2e", "Extra bunker cost"],
                       "rows": [[f"{x['delay_days']} d", f"{x.get('speed_to_recover_kn', float('nan')):.1f} kn" + (" (above design speed)" if x.get("above_design_speed") else ""), f"{x.get('extra_fuel_t', float('nan')):,.0f} t", f"{x.get('extra_ttw_co2e_t', float('nan')):,.0f} t",
                                 _fmt(x.get("extra_bunker_usd")) if x.get("extra_bunker_usd") is not None else "n/a"] for x in sc["delay"]], "caption": sc["delay"][0]["label"]})
    cy = r["cyclones"]
    if cy.get("most_exposed"):
        blocks.append({"type": "table", "head": ["Most storm-exposed points on the line", "Tropical storms / yr", "100-yr wind", "Category"], "rows": [[f"{x['lat']}, {x['lon']}", f"{x['storms_per_year']:.2f}", f"{x['rp100_wind_ms']:.0f} m/s", x["category_100yr"]] for x in cy["most_exposed"]], "caption": cy["note"]})
    blocks.append({"type": "list", "items": [f"{m['measure']}: {m['effect']} ({m['caveat']})" for m in r["mitigation"]]})
    if r["alternative_fuels"]:
        blocks.append({"type": "table", "head": ["Alternative fuel (same energy)", "WtW gCO2e/MJ", "vs now", "Meets FuelEU limit"], "rows": [[a["fuel"], f"{a['wtw_gco2e_per_mj']:.1f}", f"{a['wtw_vs_current_pct']:+.0f}%", "yes" if a["meets_fueleu_limit"] else "no"] for a in r["alternative_fuels"]],
                       "caption": "Fossil-pathway defaults only; bio and e-fuels need certified values. Same-energy comparison ignores engine efficiency and price."})
    blocks.append({"type": "note", "text": " ".join(r["flags"] + r["limits"])})
    best = sc["speed"][-1] if sc["speed"] else None
    reply = ((f"Class estimate: I have no record for {nm}, so this uses the IMO's 2022 median for its type and size, not this ship's own fuel burn. " if r["vessel"].get("class_proxy") else "") +
             f"{nm}, {vy['origin']} to {vy['destination']}: {_fmt(vy['distance_nm'])} nm at {vy['speed_kn']:.1f} kn is {vy['days']:.1f} days and about {_fmt(r['fuel']['t'])} t of fuel, "
             f"which is {_fmt(E['ttw_co2e_t'])} t CO2e at the funnel ({_fmt(E['wtw_co2e_t'])} t well-to-wake). EU ETS: {ets.get('status')}. FuelEU: {fe.get('status')}."
             + (f" Total of the costs you gave: {_fmt(c['total_usd_supplied_components'])} USD." if c["lines"] else " Give a bunker price and a daily hire rate and I will cost it.")
             + (f" Slowing to {best['speed_kn']:g} kn saves {abs(best['d_fuel_t']):,.0f} t of fuel for {best['d_days']:.1f} more days." if best and best["d_fuel_t"] < 0 else "")
             + (f" A diversion would add {sc['diversion']['d_days']:.1f} days and {sc['diversion']['d_fuel_t']:,.0f} t of fuel." if sc["diversion"] else ""))
    return {"reply": reply, "blocks": blocks, "actions": [], "result": r}


def _render_partial(p, o, d, rv, flags, market) -> dict:
    dist = p.get("distance_nm")
    gc = V.gc_nm(o, d)
    dist_used = dist or gc
    sp = p.get("speed_kn")
    s_ets, s_fe = V.scope_share(o.get("country_code"), d.get("country_code"), V.EEA), V.scope_share(o.get("country_code"), d.get("country_code"), V.EU27)
    rows = [["Route", f"{o['name']} ({o.get('country_code') or '?'}) to {d['name']} ({d.get('country_code') or '?'})", o["source"]],
            ["Distance", f"{dist_used:,.0f} nm" + ("" if dist else " (great-circle lower bound)"), "as given" if dist else "port positions"]]
    if sp:
        rows.append(["Time at that speed", f"{dist_used / sp / 24:.1f} days", f"{dist_used:,.0f} nm / {sp:g} kn"])
    rows.append(["EU ETS scope", {1.0: "100% of the voyage", 0.5: "50% of the voyage (one end in the EEA)", 0.0: "outside scope", None: "unknown country"}[s_ets], "EEA ports; ships of 5,000 GT and above"])
    rows.append(["FuelEU scope", {1.0: "100% of the energy", 0.5: "50% of the energy (one end in an EU Member State)", 0.0: "outside scope", None: "unknown country"}[s_fe], "EU Member State ports"])
    blocks = [{"type": "table", "head": ["Item", "Result", "Basis"], "rows": rows}]
    fk = p.get("fuel") or "hfo"
    r = MF.wtw(fk)
    kgco2_per_t = r["lcv_mj_per_g"] * 1e6 * r["ttw"] / 1e6
    blocks.append({"type": "note", "text": f"Emissions per tonne of {r['label']} burned: {kgco2_per_t:,.0f} kg CO2e tank-to-wake, {r['lcv_mj_per_g'] * r['wtw']:,.0f} kg CO2e well-to-wake (FuelEU Annex II defaults). Multiply by the fuel the voyage burns."})
    if V.suez_relevant(o, d):
        pct = V.DIVERSION_DEFAULT_PCT
        extra = dist_used * pct / 100.0
        blocks.append({"type": "note", "text": f"If the Red Sea is avoided, the East Asia-Europe Cape average is about +{pct:.0f}% distance: +{extra:,.0f} nm" + (f", +{extra / sp / 24:.1f} days" if sp else "") + " (average the Commission cites from UNCTAD 2025; this route may differ)."})
    cy = V._cyclones(o, d)
    if cy.get("most_exposed"):
        blocks.append({"type": "table", "head": ["Most storm-exposed points on the great-circle line", "Tropical storms / yr", "100-yr wind"], "rows": [[f"{x['lat']}, {x['lon']}", f"{x['storms_per_year']:.2f}", f"{x['rp100_wind_ms']:.0f} m/s"] for x in cy["most_exposed"]], "caption": cy["note"]})
    ship = p.get("name") or (f"IMO {p['imo']}" if p.get("imo") else "the ship")
    need = rv["missing"] or ["the ship's daily fuel consumption and the speed it applies to (for example '45 t/day at 14 knots'), or its type and size (for example 'container ship 8000 TEU') plus a speed, for an IMO class-median estimate"]
    why = rv.get("why")
    if rv["candidates"]:
        why = "several ships match: " + ", ".join(f"{c['name']} (IMO {c['imo']})" for c in rv["candidates"][:6]) + ". Give the IMO number."
    blocks.append({"type": "note", "text": "I cannot give fuel, CO2 or cost for " + ship + " yet. Missing: " + "; ".join(need) + (f". {why}" if why else "") +
                   ". Open sources that hold this: the EU MRV (THETIS-MRV) annual report lists every ship over 5,000 GT calling at EU ports with its fuel, distance and hours at sea. Place it in data/vessels/ and the engine reads it. I have not guessed a consumption."})
    blocks.append({"type": "list", "items": ["Give the daily consumption at a speed, a bunker price (USD/t) and a daily hire or opex rate, and I will return fuel, CO2e (tank-to-wake and well-to-wake), ETS and FuelEU exposure, a cost build-up, slow-steaming, delay and diversion cases, and mitigation options."]})
    if flags:
        blocks.append({"type": "note", "text": " ".join(flags)})
    reply = (f"Route understood: {o['name']} to {d['name']}, {dist_used:,.0f} nm" + (f", {dist_used / sp / 24:.1f} days at {sp:g} kn" if sp else "") + f". ETS scope: {'none' if s_ets == 0 else 'unknown' if s_ets is None else f'{s_ets*100:.0f}%'}; FuelEU scope: {'none' if s_fe == 0 else 'unknown' if s_fe is None else f'{s_fe*100:.0f}%'}. "
             f"To give emissions and costs for {ship} I need its fuel burn: {'; '.join(need)}.")
    return {"reply": reply, "blocks": blocks, "actions": [], "result": None}
