"""
cri/resident/scenario.py — the scenario lab: build scenarios for a twin and rank protection solutions against them.

Scenario matrix  every mapped flood scenario × water-level shifts (+0, +0.3, +0.6, +1.0 m: stronger event or higher sea level) and
                 every design / all-cause wind × speed scales (×1.0, ×1.1, ×1.2), counted component by component.
Solutions        the measures the Asset Twin agent proposed (raise equipment, barrier, wind retrofit) plus their combinations.
Ranking          1) resolves the base scenarios (zero equipment inundated / beyond design), 2) robustness margin: how much extra
                 water level or wind speed the solution still withstands, 3) fewest measures. No cost data is invented — the lab
                 ranks by physical performance and says so.
Limit            water-level shifts re-use the solver's depth field and shift it (the viewer re-runs the solver on release); the
                 lab's margins are therefore screening-grade for flood and exact for wind.
"""
from __future__ import annotations

import itertools
from typing import Optional

from .. import asset_twin as AT

ADDS = [0.0, 0.3, 0.6, 1.0]
SCALES = [1.0, 1.1, 1.2]


def _onsite(twin: dict) -> list:
    comps = [c for c in twin["components"] if not c.get("removed")]
    on = [c for c in comps if c.get("on_site", True)]
    if any(c["kind"] != "building" for c in on):
        on = [c for c in on if c["kind"] != "building"]
    return on


def _flood_hz(twin: dict, sc: dict) -> dict:
    sim = (twin.get("sims") or {}).get(sc["id"])
    if sim and sim.get("comp") and not sim.get("failed"):
        return {"flood": {"comp": sim["comp"], "q": sc["q"], "wet_share": sc["wet_share"]}}
    return {"flood": {"q": sc["q"], "wet_share": sc["wet_share"]}}


def _count(tally: dict, bad: tuple) -> int:
    return sum(tally.get(k, 0) for k in bad)


def matrix(twin: dict) -> list:
    on = _onsite(twin); site = twin["site"]; rows = []
    for sc in AT._key_flood(twin, solver=True):
        hz = _flood_hz(twin, sc)
        for a in ADDS:
            h = dict(hz, what_if={"flood_add_m": a})
            t = AT.evaluate(on, site, h, None, twin.get("linear"))["tally"]["flood"]
            rows.append({"hazard": "flood", "scenario": sc["id"], "label": sc["label"], "shift": f"+{a:g} m", "add": a,
                         "wet": t["wet_floor"], "equipment_inundated": t["equipment_inundated"] + t["submerged"], "n": len(on)})
    for sc in twin["scenarios"]["wind"]:
        for s in SCALES:
            t = AT.evaluate(on, site, {"wind": {"v10_ms": sc["v10_ms"]}, "what_if": {"wind_scale": s}}, None)["tally"]["wind"]
            rows.append({"hazard": "wind", "scenario": sc["id"], "label": sc["label"], "shift": f"×{s:g}", "scale": s,
                         "beyond_design": t["severe"] + t["failure"], "n": len(on)})
    return rows


def _candidates(analysis: dict) -> list:
    fl, wd = [], []
    seen = set()
    for f in analysis.get("findings", []):
        for m in f.get("protection", []):
            k = (m["type"], m.get("delta_m"), m.get("height_m"), m.get("design_wind_ms"))
            if k in seen:
                continue
            seen.add(k)
            (wd if m["type"] == "wind_retrofit" else fl).append({kk: vv for kk, vv in m.items() if kk in ("type", "delta_m", "height_m", "design_wind_ms", "components", "why")})
    # keep the largest of each type so one measure covers every scenario
    def top(ms):
        by = {}
        for m in ms:
            cur = by.get(m["type"])
            mag = m.get("delta_m") or m.get("height_m") or m.get("design_wind_ms") or 0
            if cur is None or mag > (cur.get("delta_m") or cur.get("height_m") or cur.get("design_wind_ms") or 0):
                by[m["type"]] = m
        return list(by.values())
    return top(fl) + top(wd)


def _perf(twin: dict, measures: list) -> dict:
    on = _onsite(twin); site = twin["site"]
    flood_scs = AT._key_flood(twin, solver=True); wind_scs = [s for s in twin["scenarios"]["wind"] if s["kind"] in ("design", "allcause") and s.get("rp") in (50, 100, 250)]
    base_bad, margin_f = 0, None
    for sc in flood_scs:
        hz = _flood_hz(twin, sc)
        r = AT.evaluate(on, site, dict(hz, what_if={"flood_add_m": 0}), measures, twin.get("linear"))["tally"]["flood"]
        base_bad += r["equipment_inundated"] + r["submerged"]
    if flood_scs:
        margin_f = 0.0
        for a in [x / 10 for x in range(0, 21)]:
            ok = all((lambda t: t["equipment_inundated"] + t["submerged"] == 0)(
                AT.evaluate(on, site, dict(_flood_hz(twin, sc), what_if={"flood_add_m": a}), measures, twin.get("linear"))["tally"]["flood"]) for sc in flood_scs)
            if ok:
                margin_f = a
            else:
                break
    wind_bad, margin_w = 0, None
    for sc in wind_scs:
        t = AT.evaluate(on, site, {"wind": {"v10_ms": sc["v10_ms"]}}, measures)["tally"]["wind"]
        wind_bad += t["severe"] + t["failure"]
    if wind_scs:
        margin_w = 0.0
        for s in [1.0 + i * 0.05 for i in range(0, 13)]:
            ok = all((lambda t: t["severe"] + t["failure"] == 0)(AT.evaluate(on, site, {"wind": {"v10_ms": sc["v10_ms"]}, "what_if": {"wind_scale": s}}, measures)["tally"]["wind"]) for sc in wind_scs)
            if ok:
                margin_w = round(s - 1.0, 2)
            else:
                break
    return {"flood_equipment_inundated": base_bad, "wind_beyond_design": wind_bad,
            "flood_margin_m": margin_f, "wind_margin_pct": None if margin_w is None else round(margin_w * 100)}


def lab(twin: dict, analysis: Optional[dict] = None) -> dict:
    """Scenario matrix + ranked protection solutions for a built twin."""
    if analysis is None:
        analysis = AT.analyse(twin)
    AT.ensure_sims(twin)
    mx = matrix(twin)
    cands = _candidates(analysis)
    sets = [[]]
    for r in range(1, len(cands) + 1):
        sets += [list(c) for c in itertools.combinations(cands, r)]
    sols = []
    for ms in sets:
        p = _perf(twin, ms)
        name = " + ".join(m["type"].replace("_", " ") + f" {m.get('delta_m') or m.get('height_m') or m.get('design_wind_ms')}" for m in ms) or "do nothing"
        resolves = p["flood_equipment_inundated"] == 0 and p["wind_beyond_design"] == 0
        sols.append({"name": name, "measures": ms, "n_measures": len(ms), "resolves_base": resolves, **p})
    sols.sort(key=lambda s: (not s["resolves_base"], s["flood_equipment_inundated"] + s["wind_beyond_design"], -(s["flood_margin_m"] or 0) - (s["wind_margin_pct"] or 0) / 50, s["n_measures"]))
    best = next((s for s in sols if s["resolves_base"] and s["n_measures"] > 0), None)
    do_nothing = next(s for s in sols if s["n_measures"] == 0)
    if best:
        rec = (f"Recommended: {best['name']}. With it no equipment is inundated and nothing is beyond design wind in the base scenarios; it still holds up to "
               f"+{best['flood_margin_m']} m of extra water level and +{best['wind_margin_pct']}% wind speed."
               if best["flood_margin_m"] is not None and best["wind_margin_pct"] is not None else f"Recommended: {best['name']}.")
    elif do_nothing["flood_equipment_inundated"] == 0 and do_nothing["wind_beyond_design"] == 0:
        rec = "In the scenarios modelled, nothing at this site is inundated or beyond design wind — no protection is needed on this evidence. Margins below show how far it is from that edge."
    else:
        rec = "No single measure set resolves every scenario; the best partial option is first in the table."
    return {"matrix": mx, "solutions": sols[:8], "do_nothing": do_nothing, "recommendation": rec,
            "method": "Scenario matrix = mapped flood scenarios × water-level shifts and winds × speed scales, evaluated per component. Solutions ranked by physical performance; no costs are invented.",
            "limits": ["Flood margins shift the solver's depth field (screening-grade); open the 3D twin and drag the water-level slider to re-run the solver.",
                       "Event duration, drainage, waves and debris are not modelled; wind margins use the power-law profile and assumed design winds."]}
