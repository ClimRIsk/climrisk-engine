"""
cri/lca/cfp.py — ISO 14067:2018 carbon footprint of a product (CFP) study calculator.

Takes the A1–A3 inventory of any product template (cement, steel, aluminium, ammonia, custom) and runs the
calculation steps the standard asks for, so the engine produces the numbers and the study report itself:

  boundary        cradle-to-gate (a *partial* CFP in the standard's terms), cradle-to-customer (A1–A4),
                  cradle-to-grave (A1–C: the full CFP)
  A4              distribution: truck / rail / ship (DESNZ) and air (DESNZ freight flights; aircraft
                  emissions reported separately, radiative-forcing uplift shown separately)
  B               use stage over the service life; electricity follows a fixed factor or an NGFS grid
                  pathway year by year
  C               end of life: DESNZ waste-disposal factors by treatment (landfill, combustion, recycling…);
                  recycling credit NOT taken (cut-off / recycled-content approach)
  allocation      multi-output processes: mass, economic or energy allocation, with all three shown
  GHG types       fossil, biogenic emissions, biogenic removals, direct land-use change, aircraft — separately
  cut-off         mass and emission share of excluded flows against a stated criterion
  uncertainty     Monte Carlo over every stage → P5 / P50 / P95 of the boundary total
  sensitivity     ±10 % per flow, hotspots to 80 % of the total, effect of each methodological choice
  data quality    the ten ISO 14067 §6.3.5 criteria, scored from the inputs actually used
  study report    ISO 14067 clause 9 content as JSON, Markdown and Excel, with a communication statement

Verification: a third-party critical review is an act by people, not a calculation, so every report states
"not reviewed" until one is recorded.
"""
from __future__ import annotations

import dataclasses
import hashlib
import io
import json
from datetime import date
from typing import Optional

import numpy as np

from .engine import (CV_BY_DQ, LOW_CARBON_FLOOR_G_KWH, ElectricityUse, FuelUse, Inventory, Material, ProcessEmission,
                     TransportLeg, _fuel_terms, _ln, calculate, data_quality_summary, factors, grid_factor,
                     grid_index_paths, uncertainty_draws)

BOUNDARIES = {
    "cradle_to_gate": {"stages": ["A1–A3"], "label": "Partial CFP, cradle-to-gate (A1–A3)"},
    "cradle_to_customer": {"stages": ["A1–A3", "A4"], "label": "Partial CFP, cradle-to-customer (A1–A4)"},
    "cradle_to_grave": {"stages": ["A1–A3", "A4", "B", "C"], "label": "CFP, cradle-to-grave (A1–C)"},
}
AIR_IDS = {
    "kg": "desnz26:freighting-goods-freight-flights-international-to-from-non-uk-without-rf-tonne-km",
    "kg_rf": "desnz26:freighting-goods-freight-flights-international-to-from-non-uk-with-rf-tonne-km",
    "wtt": "desnz26:wtt-delivery-vehs-freight-wtt-freight-flights-international-to-from-non-uk-without-rf-tonne-km",
}
EOL_FACTOR_YEAR = 2026
TREATMENTS = ("landfill", "combustion", "closed_loop", "open_loop", "composting", "anaerobic_digestion")


class StudyError(ValueError):
    pass


# ── factor look-ups ───────────────────────────────────────────────────────────

def _desnz(fid: str) -> float:
    from ..carbon import factors as CF
    return float(CF.desnz(fid, "AR6")["kgco2e_per_unit"])


def eol_options() -> dict:
    """End-of-life materials and the treatments with a published DESNZ factor (kg CO2e per tonne)."""
    from ..carbon import factors as CF
    out: dict = {}
    for f in CF.lib()["desnz"]["factors"]:
        if f["path"][0] != "Waste disposal" or f["unit"] != "tonnes":
            continue
        mat = " > ".join(f["path"][1:])
        tr = (f["variant"] or "").lower().replace("-", "_").replace(" ", "_")
        out.setdefault(mat, {})[tr] = {"id": f["id"], "kgco2e_per_t": round(float(f["kgco2e"]), 4)}
    return {"source": CF.lib()["desnz"]["source"], "treatments": list(TREATMENTS), "materials": out,
            "note": "Recycling (closed_loop / open_loop) carries only the collection and processing burden; no credit "
                    "for avoided virgin production is taken (cut-off approach)."}


def _eol_lookup() -> dict:
    return eol_options()["materials"]


def _g_at(country: Optional[str], g_override: Optional[float], mult: float) -> tuple[float, str]:
    if g_override is not None:
        return float(g_override), "supplier-specific factor"
    g, src = grid_factor(country)
    if g > LOW_CARBON_FLOOR_G_KWH:
        g = LOW_CARBON_FLOOR_G_KWH + (g - LOW_CARBON_FLOOR_G_KWH) * mult
    return g, src


def _grid_mult(country: Optional[str], basis: str, year: int) -> float:
    if basis in (None, "", "static"):
        return 1.0
    paths = grid_index_paths(country)
    if basis not in paths["scenarios"]:
        raise StudyError(f"grid_basis must be 'static' or one of {sorted(paths['scenarios'])}")
    return float(np.interp(year, paths["years"], paths["scenarios"][basis]))


# ── allocation ────────────────────────────────────────────────────────────────

def allocation_shares(a: dict) -> dict:
    """Share of the multi-output process burden carried by the studied product under each method."""
    prod = a.get("product") or {}
    cops = a.get("coproducts") or []
    out: dict = {}
    for key, label, getter in (
            ("mass", "mass", lambda x: float(x.get("mass_t") or 0)),
            ("energy", "energy content", lambda x: float(x.get("energy_gj") or 0)),
            ("economic", "economic value", lambda x: float(x.get("mass_t") or 0) * float(x.get("price_per_t") or 0))):
        p = getter(prod)
        tot = p + sum(getter(c) for c in cops)
        if p > 0 and tot > 0 and all(getter(c) > 0 for c in cops):
            out[key] = {"share": p / tot, "basis": label}
    return out


def _scale(inv: Inventory, f: float, apply_to: str) -> Inventory:
    """Copy of the inventory with the A3 (and optionally A1/A2) flows scaled by the allocation factor."""
    s = dataclasses.replace(
        inv, fuels=[dataclasses.replace(x, gj=x.gj * f) for x in inv.fuels],
        electricity=[dataclasses.replace(x, kwh=x.kwh * f) for x in inv.electricity],
        process=[dataclasses.replace(x, kg_co2e=x.kg_co2e * f) for x in inv.process],
        materials=list(inv.materials), transport=list(inv.transport), exclusions=list(inv.exclusions),
        notes=list(inv.notes))
    if apply_to == "A1-A3":
        s.materials = [dataclasses.replace(x, t=x.t * f) for x in inv.materials]
        s.transport = [dataclasses.replace(x, t=x.t * f) for x in inv.transport]
    return s


# ── stages A4, B, C ───────────────────────────────────────────────────────────

def _stage_a4(legs: list[dict]) -> dict:
    F = factors()["freight"]
    rows, total, air, air_rf = [], 0.0, 0.0, 0.0
    for lg in legs or []:
        mode, t, km = lg["mode"], float(lg["t"]), float(lg["km"])
        label = lg.get("label") or mode
        if mode == "air":
            k = t * km * (_desnz(AIR_IDS["kg"]) + _desnz(AIR_IDS["wtt"]))
            rf = t * km * (_desnz(AIR_IDS["kg_rf"]) - _desnz(AIR_IDS["kg"]))
            air += k
            air_rf += rf
            basis = f"{t:g} t × {km:g} km × DESNZ 2026 freight flights, international (excl. RF, incl. WTT)"
        elif mode in F:
            fr = F[mode]
            k = t * km * (fr["kgco2e_per_tkm"] + (fr["wtt_kgco2e_per_tkm"] or 0.0))
            basis = f"{t:g} t × {km:g} km × {fr['label']} (DESNZ 2025, incl. WTT)"
        else:
            raise StudyError("distribution mode must be truck, rail, ship or air")
        total += k
        rows.append({"stage": "A4", "item": label, "kg_co2e": k, "basis": basis, "dq": int(lg.get("dq", 3)), "mode": mode})
    return {"total": total, "rows": rows, "aircraft_kg": air, "aircraft_rf_extra_kg": air_rf}


def _stage_b(use: Optional[dict], ref_year: int) -> dict:
    if not use:
        return {"total": 0.0, "rows": [], "by_year": [], "biogenic_co2": 0.0, "service_life_years": 0}
    life = int(use.get("service_life_years") or 0)
    if life <= 0 or life > 100:
        raise StudyError("use.service_life_years must be between 1 and 100")
    basis = use.get("grid_basis", "static")
    rows: list[dict] = []
    by_year = [0.0] * life
    biogenic = 0.0
    for e in use.get("electricity") or []:
        kwh, ctry, g_ov = float(e["kwh_per_year"]), e.get("country"), e.get("g_per_kwh")
        label = e.get("label", "Use-stage electricity")
        tot = 0.0
        for y in range(life):
            g, src = _g_at(ctry, g_ov, _grid_mult(ctry, basis if g_ov is None else "static", ref_year + y))
            v = kwh * g / 1000.0
            by_year[y] += v
            tot += v
        rows.append({"stage": "B", "item": label, "kg_co2e": tot, "dq": int(e.get("dq", 3)),
                     "basis": f"{kwh:,.4g} kWh/yr × {life} yr, grid basis '{basis if g_ov is None else 'supplier-specific'}' ({src})"})
    for fu in use.get("fuels") or []:
        t = _fuel_terms(FuelUse(fu["fuel"], float(fu["gj_per_year"]), int(fu.get("dq", 3))))
        yr = t["fossil_co2"] + t["ch4_n2o"] + t["wtt"]
        for y in range(life):
            by_year[y] += yr
        biogenic += t["biogenic_co2"] * life
        rows.append({"stage": "B", "item": f"{factors()['fuels'][fu['fuel']]['label']} in use", "kg_co2e": yr * life,
                     "dq": int(fu.get("dq", 3)), "basis": f"{fu['gj_per_year']:g} GJ/yr × {life} yr × IPCC 2006 + DESNZ WTT"})
    for o in use.get("other") or []:
        v = float(o["kg_co2e_per_year"])
        for y in range(life):
            by_year[y] += v
        rows.append({"stage": "B", "item": o.get("label", "Other use-stage emissions"), "kg_co2e": v * life,
                     "dq": int(o.get("dq", 3)), "basis": f"{v:g} kg CO2e/yr × {life} yr (user-supplied)"})
    return {"total": float(sum(by_year)), "rows": rows, "by_year": [round(x, 3) for x in by_year],
            "biogenic_co2": biogenic, "service_life_years": life, "grid_basis": basis}


def _stage_c(items: list[dict], legs: list[dict]) -> dict:
    lookup = _eol_lookup()
    rows, total = [], 0.0
    keymap = {k.lower(): k for k in lookup}
    for it in items or []:
        mat = keymap.get(str(it["material"]).lower())
        if not mat:
            raise StudyError(f"unknown end-of-life material '{it['material']}' — see GET /lca/eol-options")
        shares = {k: float(v) for k, v in (it.get("shares") or {}).items() if float(v) > 0}
        if abs(sum(shares.values()) - 1.0) > 0.01:
            raise StudyError(f"end-of-life shares for {mat} must sum to 1 (got {sum(shares.values()):.3f})")
        t = float(it["t"])
        for tr, sh in shares.items():
            if tr not in lookup[mat]:
                raise StudyError(f"no published factor for {mat} / {tr}; available: {sorted(lookup[mat])}")
            k = t * sh * lookup[mat][tr]["kgco2e_per_t"]
            total += k
            rows.append({"stage": "C", "item": f"{mat}: {tr.replace('_', ' ')}", "kg_co2e": k, "dq": 3,
                         "basis": f"{t * sh:g} t × {lookup[mat][tr]['kgco2e_per_t']:g} kg/t (DESNZ 2026 waste disposal)"})
    for lg in legs or []:
        a4 = _stage_a4([lg])
        total += a4["total"]
        rows.append({**a4["rows"][0], "stage": "C", "item": "Transport to treatment: " + a4["rows"][0]["item"]})
    return {"total": total, "rows": rows}


# ── cut-off ───────────────────────────────────────────────────────────────────

def _cutoff(spec: Optional[dict], inv: Inventory, gross_kg: float) -> dict:
    crit = float((spec or {}).get("criterion_pct", 1.0))
    excl = (spec or {}).get("excluded") or []
    declared = list(inv.exclusions)
    if not spec or not excl:
        return {"status": "not assessed", "criterion_pct": crit, "declared_exclusions": declared,
                "note": "No excluded flows listed, so the mass and emission shares of the cut-off flows cannot "
                        "be computed. Supply cutoff.excluded and cutoff.total_input_mass_t."}
    total_mass = float(spec.get("total_input_mass_t") or 0) or (sum(m.t for m in inv.materials)
                                                                 + sum(float(e.get("mass_t") or 0) for e in excl))
    rows, mass_ex, kg_ex = [], 0.0, 0.0
    for e in excl:
        m = float(e.get("mass_t") or 0)
        k = m * float(e["kgco2e_per_t"]) if e.get("kgco2e_per_t") is not None else None
        mass_ex += m
        kg_ex += k or 0.0
        rows.append({"name": e["name"], "mass_t": m, "kg_co2e": k,
                     "mass_share_pct": round(100 * m / total_mass, 3) if total_mass else None})
    mass_pct = 100 * mass_ex / total_mass if total_mass else None
    em_known = all(r["kg_co2e"] is not None for r in rows) and bool(rows)
    em_pct = 100 * kg_ex / (gross_kg + kg_ex) if em_known and (gross_kg + kg_ex) else None
    ok = (mass_pct is not None and mass_pct <= crit) and (em_pct is None or em_pct <= crit)
    return {"status": "within criterion" if ok and em_known else "mass within criterion; emissions unquantified"
            if ok else "criterion exceeded", "criterion_pct": crit, "excluded": rows,
            "excluded_mass_pct": None if mass_pct is None else round(mass_pct, 3),
            "excluded_emissions_pct": None if em_pct is None else round(em_pct, 3), "declared_exclusions": declared}


# ── data quality (ISO 14067 §6.3.5) ───────────────────────────────────────────

def _rate(score: float) -> str:
    return "good" if score <= 2 else "fair" if score <= 3 else "poor"


def _data_quality(inv: Inventory, ref_year: int, unc: dict, cutoff: dict, uses_air: bool, spec_inputs: dict) -> dict:
    G = factors()["grid"]
    countries = {e.country.upper() for e in inv.electricity if e.country and e.g_per_kwh is None}
    sup = [e for e in inv.electricity if e.g_per_kwh is not None]
    world = [e for e in inv.electricity if e.g_per_kwh is None and not e.country]
    years = [G["countries"][c]["year"] for c in countries if c in G["countries"]]
    if inv.transport or uses_air:
        years.append(2025)
    age = max([ref_year - y for y in years], default=0)
    t_score = 1 if age <= 2 else 2 if age <= 4 else 3 if age <= 6 else 4 if age <= 10 else 5
    g_score = 4 if world else 1
    dq = data_quality_summary(inv)
    ps = dq.get("primary_data_share") or 0.0
    te_score = 1 if ps >= 0.8 else 2 if ps >= 0.5 else 3 if ps >= 0.3 else 4 if ps >= 0.1 else 5
    half = (unc["p95"] - unc["p5"]) / 2 / unc["p50"] if unc["p50"] else 0
    pr_score = 1 if half <= 0.05 else 2 if half <= 0.10 else 3 if half <= 0.20 else 4 if half <= 0.35 else 5
    n_ex = len(inv.exclusions)
    co_score = min(5, 1 + n_ex + (1 if cutoff["status"] == "not assessed" else 0))
    rep = round((g_score + te_score) / 2)
    rows = [
        ("Time-related coverage", t_score, f"oldest time-sensitive factor in use is {age} yr before the reference year "
                                           f"{ref_year}" if years else "no time-sensitive factors in use"),
        ("Geographical coverage", g_score, "plant-country grid factor used" if not world else
         "world-average grid factor used because no plant country was given"),
        ("Technological coverage", te_score, f"{round(100 * ps)} % of emissions come from primary data (data-quality score ≤ 2)"),
        ("Precision", pr_score, f"P5–P95 half-range is {100 * half:.1f} % of the median"),
        ("Completeness", co_score, f"{n_ex} declared exclusion(s); cut-off {cutoff['status']}"),
        ("Representativeness", rep, "mean of geographical and technological coverage"),
        ("Consistency", 1, "one factor library, IPCC AR6 GWP100 and one boundary rule applied to every flow"),
        ("Reproducibility", 1, "every input and factor is returned and hashed (fingerprint)"),
        ("Sources of data", 1 if not world else 2, "all factors cite a published table (see source list)"),
        ("Uncertainty of the information", 1, f"Monte Carlo over every flow, {unc['n_draws']} draws"),
    ]
    sc = [r[1] for r in rows]
    return {"criteria": [{"criterion": a, "score": b, "rating": _rate(b), "basis": c} for a, b, c in rows],
            "overall_score": round(float(np.mean(sc)), 2), "overall_rating": _rate(float(np.mean(sc))),
            "scale": "1 = best … 5 = worst", "primary_data_share": ps,
            "emission_weighted_flow_score": dq.get("weighted_score")}


# ── the study ─────────────────────────────────────────────────────────────────

def run_study(inv: Inventory, spec: Optional[dict] = None, product: str = "product", country: Optional[str] = None,
              n: int = 5000, seed: int = 11) -> dict:
    spec = dict(spec or {})
    bkey = spec.get("boundary", "cradle_to_gate")
    if bkey not in BOUNDARIES:
        raise StudyError(f"boundary must be one of {sorted(BOUNDARIES)}")
    ref_year = int(spec.get("reference_year", 2025))
    stages_in = BOUNDARIES[bkey]["stages"]
    fingerprint = hashlib.sha256(json.dumps({"inv": dataclasses.asdict(inv), "spec": spec, "n": n, "seed": seed},
                                            sort_keys=True, default=str).encode()).hexdigest()[:16]

    # allocation
    alloc_spec = spec.get("allocation")
    alloc = None
    run_inv = inv
    if alloc_spec and alloc_spec.get("coproducts"):
        shares = allocation_shares(alloc_spec)
        method = alloc_spec.get("method", "economic")
        if method not in shares:
            raise StudyError(f"allocation method '{method}' needs the matching quantities for the product and every "
                             f"co-product (mass_t / energy_gj / mass_t × price_per_t)")
        apply_to = alloc_spec.get("apply_to", "A3")
        f = shares[method]["share"]
        run_inv = _scale(inv, f, apply_to)
        alloc = {"method": method, "applied_to": apply_to, "product_share": round(f, 4),
                 "all_methods": {k: round(v["share"], 4) for k, v in shares.items()},
                 "coproducts": [c["name"] for c in alloc_spec["coproducts"]],
                 "hierarchy": "ISO 14067 §6.4.6: avoid allocation first (subdivision / system expansion); where "
                              "unavoidable, use physical relationships, then other relationships such as economic value. "
                              "The share under every method is shown so the effect of the choice is visible."}

    base = calculate(run_inv)
    gate = base["total_kg_co2e"]
    gate_gross = base["gross_kg_co2e"]
    a4 = _stage_a4(spec.get("distribution") or [])
    b = _stage_b(spec.get("use"), ref_year) if "B" in stages_in else {"total": 0.0, "rows": [], "by_year": [],
                                                                      "biogenic_co2": 0.0, "service_life_years": 0}
    c = _stage_c(spec.get("end_of_life") or [], spec.get("eol_transport") or []) if "C" in stages_in else {"total": 0.0, "rows": []}
    a4_in = a4["total"] if "A4" in stages_in else 0.0
    air_in = a4["aircraft_kg"] if "A4" in stages_in else 0.0
    air_rf_in = a4["aircraft_rf_extra_kg"] if "A4" in stages_in else 0.0

    # GHG-type split
    luc = float(spec.get("luc_kgco2e") or 0.0)
    bio_em = base["biogenic_co2_kg"] + (b["biogenic_co2"] if "B" in stages_in else 0.0)
    uptake_share = float(spec.get("biogenic_uptake_share", 1.0))
    bio_rem = float(spec["biogenic_removal_kg_co2"]) if spec.get("biogenic_removal_kg_co2") is not None else bio_em * uptake_share
    fossil_all = gate + a4_in + b["total"] + c["total"]
    fossil_excl_air = fossil_all - air_in
    by_type = {"fossil_ghg": fossil_excl_air, "aircraft_ghg": air_in, "biogenic_co2_emissions": bio_em,
               "biogenic_co2_removals": (-bio_rem if bio_rem else 0.0), "direct_land_use_change": luc}
    total = fossil_excl_air + air_in + luc + bio_em - bio_rem

    by_stage = {"A1–A3 product stage": gate}
    if "A4" in stages_in:
        by_stage["A4 distribution"] = a4["total"]
    if "B" in stages_in:
        by_stage["B use stage"] = b["total"]
    if "C" in stages_in:
        by_stage["C end of life"] = c["total"]
    by_boundary = {k: None for k in BOUNDARIES}
    a4_all, b_all, c_all = a4["total"], None, None
    # totals under each boundary (use/EoL only when supplied)
    bu = _stage_b(spec.get("use"), ref_year) if spec.get("use") else {"total": 0.0}
    cc = _stage_c(spec.get("end_of_life") or [], spec.get("eol_transport") or []) if spec.get("end_of_life") else {"total": 0.0}
    by_boundary["cradle_to_gate"] = gate
    by_boundary["cradle_to_customer"] = gate + a4_all
    by_boundary["cradle_to_grave"] = gate + a4_all + bu["total"] + cc["total"]

    # uncertainty
    rng = np.random.default_rng(seed + 101)
    draws = uncertainty_draws(run_inv, n, seed)
    cv = lambda dq: CV_BY_DQ.get(int(dq), 0.2)                         # noqa: E731
    td = draws.copy()
    gate_unc = {"p5": float(np.percentile(draws, 5)), "p50": float(np.percentile(draws, 50)),
                "p95": float(np.percentile(draws, 95)), "n_draws": n}
    if "A4" in stages_in:
        for r in a4["rows"]:
            td += _ln(rng, r["kg_co2e"], np.hypot(cv(r["dq"]), 0.15), n)
    if "B" in stages_in:
        for r in b["rows"]:
            td += _ln(rng, r["kg_co2e"], np.hypot(cv(r["dq"]), 0.15), n)
        if b["total"]:
            td *= 1.0                                               # service life is a stated scenario, not a draw
    if "C" in stages_in:
        for r in c["rows"]:
            td += _ln(rng, r["kg_co2e"], 0.25, n)
    if luc:
        td += _ln(rng, luc, 0.5, n)
    q = np.percentile(td, [5, 50, 95])
    unc = {"p5": float(q[0]), "p50": float(q[1]), "p95": float(q[2]), "mean": float(td.mean()), "n_draws": n,
           "deterministic": total,
           "method": "lognormal per flow; CV from data-quality score combined with factor uncertainty (see /lca/factors); "
                     "end-of-life factors ±25 %; direct land-use change ±50 %; the service life and end-of-life shares "
                     "are stated scenario inputs, not sampled"}
    # biogenic net is added deterministically (default 0)
    for k in ("p5", "p50", "p95", "mean"):
        unc[k] += bio_em - bio_rem

    # contributions / hotspots / sensitivity
    items = [dict(d, stage=d["stage"]) for d in base["detail"]]
    if "A4" in stages_in:
        items += a4["rows"]
    if "B" in stages_in:
        items += b["rows"]
    if "C" in stages_in:
        items += c["rows"]
    if luc:
        items.append({"stage": "A1–A3", "item": "Direct land-use change (user-supplied)", "kg_co2e": luc,
                      "basis": "user-supplied dLUC emissions"})
    if base["captured_kg_co2"]:
        items.append({"stage": "A3", "item": "CO2 captured and stored (deducted)", "kg_co2e": -base["captured_kg_co2"],
                      "basis": f"capture rate {run_inv.capture_rate:.0%} of capturable CO2"})
    pos = sum(max(i["kg_co2e"], 0) for i in items) or 1.0
    ranked = sorted(items, key=lambda i: -abs(i["kg_co2e"]))
    cum, hot = 0.0, []
    for i in ranked:
        if i["kg_co2e"] <= 0:
            continue
        cum += i["kg_co2e"]
        hot.append({**{k: i[k] for k in ("stage", "item")}, "kg_co2e": round(i["kg_co2e"], 2),
                    "share_pct": round(100 * i["kg_co2e"] / pos, 1), "cumulative_pct": round(100 * cum / pos, 1)})
        if cum / pos >= 0.8:
            break
    tot_ref = total if total else 1.0
    sens = [{"item": i["item"], "stage": i["stage"], "kg_co2e": round(i["kg_co2e"], 2),
             "effect_of_plus_10pct_on_total_pct": round(100 * 0.10 * i["kg_co2e"] / tot_ref, 2)} for i in ranked[:12]]
    choices = []
    if alloc:
        for m, sh in alloc["all_methods"].items():
            alt = _scale(inv, sh, alloc["applied_to"])
            choices.append({"choice": f"allocation by {m}", "gate_kg_co2e": round(calculate(alt)["total_kg_co2e"], 1),
                            "change_vs_selected_pct": round(100 * (calculate(alt)["total_kg_co2e"] / gate - 1), 1) + 0.0 if gate else None})
    if inv.capture_rate:
        nc = dataclasses.replace(run_inv, capture_rate=0.0)
        choices.append({"choice": "no CO2 capture", "gate_kg_co2e": round(calculate(nc)["total_kg_co2e"], 1),
                        "change_vs_selected_pct": round(100 * (calculate(nc)["total_kg_co2e"] / gate - 1), 1) + 0.0 if gate else None})
    if bio_em:
        choices.append({"choice": "no biogenic uptake credit (biogenic CO2 counted as emitted)",
                        "total_kg_co2e": round(total + bio_rem, 1),
                        "change_vs_selected_pct": round(100 * bio_rem / tot_ref, 1)})
    if air_in:
        choices.append({"choice": "include aircraft radiative-forcing uplift", "total_kg_co2e": round(total + air_rf_in, 1),
                        "change_vs_selected_pct": round(100 * air_rf_in / tot_ref, 1)})
    for e in inv.electricity:
        if e.g_per_kwh is None:
            for scen in ("Net Zero 2050", "Current Policies"):
                paths = grid_index_paths(e.country)["scenarios"]
                if scen in paths:
                    m2 = float(np.interp(2035, grid_index_paths(e.country)["years"], paths[scen]))
                    alt = calculate(run_inv, grid_multiplier=m2)["total_kg_co2e"]
                    choices.append({"choice": f"gate emissions with the 2035 grid under NGFS {scen}",
                                    "gate_kg_co2e": round(alt, 1),
                                    "change_vs_selected_pct": round(100 * (alt / gate - 1), 1) if gate else None})
            break

    cut = _cutoff(spec.get("cutoff"), inv, gate_gross)
    sources = sorted({"IPCC 2006 Guidelines Vol.2 (stationary combustion) and Vol.3 (process emissions)",
                      "IPCC AR6 WG1 Table 7.15 (GWP100)", "Ember Yearly Electricity Data (CC BY 4.0) — grid factors",
                      "UK DESNZ GHG Conversion Factors 2025 (OGL v3.0) — freight and well-to-tank",
                      "NGFS Phase 5 via IIASA — grid pathways"}
                     | ({"UK DESNZ GHG Conversion Factors 2026 (OGL v3.0) — freight flights, waste disposal"}
                        if (air_in or c["rows"]) else set())
                     | {m.source for m in inv.materials if m.source})
    dqa = _data_quality(run_inv, ref_year, {**unc, "p50": unc["p50"] if unc["p50"] else 1.0}, cut,
                        bool(air_in), spec)

    label = BOUNDARIES[bkey]["label"]
    result = {
        "product": product, "country": country, "declared_unit": inv.declared_unit,
        "boundary": {"key": bkey, "label": label, "stages_included": stages_in,
                     "is_full_cfp": bkey == "cradle_to_grave",
                     "excluded_stages": [s for s in ("A4", "B", "C") if s not in " ".join(stages_in)]},
        "reference_year": ref_year, "gwp": "IPCC AR6 WG1 GWP100",
        "cfp_kg_co2e": round(total, 2),
        "uncertainty": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in unc.items()},
        "by_stage": {k: round(v, 2) for k, v in by_stage.items()},
        "a1_a3_by_component": {k: round(v, 2) for k, v in base["by_stage"].items()},
        "by_ghg_type": {k: round(v, 2) for k, v in by_type.items()},
        "total_by_boundary_if_all_supplied": {k: round(v, 2) for k, v in by_boundary.items()},
        "aircraft_rf_uplift_kg_co2e": round(air_rf_in, 2),
        "captured_co2_kg": round(base["captured_kg_co2"], 2),
        "use_stage": {"service_life_years": b["service_life_years"], "by_year_kg_co2e": b["by_year"],
                      "grid_basis": b.get("grid_basis")} if "B" in stages_in else None,
        "detail": [dict(i, kg_co2e=round(i["kg_co2e"], 3)) for i in items],
        "hotspots_to_80pct": hot, "sensitivity": sens, "methodological_choices": choices,
        "allocation": alloc, "cutoff": cut, "data_quality": dqa, "exclusions": base["exclusions"],
        "notes": run_inv.notes + [
            "End-of-life recycling takes no credit for avoided virgin production (cut-off approach)."]
        if "C" in stages_in else run_inv.notes,
        "factor_sources": sources, "fingerprint": fingerprint,
        "verification": {"status": "not reviewed", "critical_review": None,
                         "statement": "No third-party critical review has been recorded for this study."},
        "gate_uncertainty": {k: round(v, 2) if isinstance(v, float) else v for k, v in gate_unc.items()},
    }
    result["communication_statement"] = communication_statement(result)
    return result


def communication_statement(r: dict) -> str:
    u = r["uncertainty"]
    return (f"{r['boundary']['label']} of {r['product']} per {r['declared_unit']}: {r['cfp_kg_co2e']:,.0f} kg CO2e "
            f"(P5–P95 {u['p5']:,.0f}–{u['p95']:,.0f}), IPCC AR6 GWP100, reference year {r['reference_year']}. "
            f"Calculated with the ClimRisk LCA engine following ISO 14067:2018 methodology. "
            f"{r['verification']['statement']}"
            + ("" if r["boundary"]["is_full_cfp"] else " Stages after the gate/customer boundary are not included."))


# ── study report (ISO 14067 clause 9 content) ────────────────────────────────

def report(r: dict, spec: Optional[dict] = None) -> dict:
    spec = spec or {}
    gen = {"commissioner": spec.get("commissioner") or "not stated",
           "practitioner": spec.get("practitioner") or "not stated",
           "report_date": spec.get("report_date") or date.today().isoformat(),
           "report_version": spec.get("report_version", "1.0"), "study_fingerprint": r["fingerprint"]}
    sections = {
        "1_general": gen,
        "2_goal": {"intended_application": spec.get("intended_application") or "not stated",
                   "reasons_for_study": spec.get("reasons") or "not stated",
                   "intended_audience": spec.get("audience") or "not stated",
                   "public_comparative_assertion": bool(spec.get("public_comparative_assertion", False)),
                   "note": "A comparative assertion disclosed to the public requires a third-party critical review."},
        "3_scope": {"product": r["product"], "declared_unit": r["declared_unit"], "boundary": r["boundary"],
                    "time_period": f"reference year {r['reference_year']}",
                    "geography": r["country"] or "not given (world-average electricity)",
                    "gwp": r["gwp"], "allocation": r["allocation"] or "none applied (single-output process)",
                    "cutoff": r["cutoff"], "data_quality_requirements": "see section 6",
                    "recycling approach": "cut-off (recycled content); no credit for avoided burdens",
                    "biogenic_carbon": "reported separately; uptake share assumed "
                                       f"{spec.get('biogenic_uptake_share', 1.0)}",
                    "land_use_change": "direct LUC included only where supplied; indirect LUC excluded",
                    "aircraft": "reported separately; radiative-forcing uplift shown separately"},
        "4_inventory_and_data": {"flows": r["detail"], "exclusions": r["exclusions"], "sources": r["factor_sources"],
                                 "notes": r["notes"]},
        "5_results": {"cfp_kg_co2e": r["cfp_kg_co2e"], "uncertainty": r["uncertainty"], "by_stage": r["by_stage"],
                      "by_ghg_type": r["by_ghg_type"], "aircraft_rf_uplift_kg_co2e": r["aircraft_rf_uplift_kg_co2e"],
                      "captured_co2_kg": r["captured_co2_kg"]},
        "6_data_quality": r["data_quality"],
        "7_interpretation": {"hotspots_to_80pct": r["hotspots_to_80pct"], "sensitivity": r["sensitivity"],
                             "methodological_choices": r["methodological_choices"],
                             "limitations": _limitations(r)},
        "8_review_and_communication": {"verification": r["verification"],
                                       "communication_statement": r["communication_statement"]},
    }
    return {"title": f"Carbon footprint of {r['product']} ({r['declared_unit']}) — ISO 14067 study report",
            "sections": sections, "markdown": _markdown(r, sections)}


def _limitations(r: dict) -> list[str]:
    out = ["Factors are published defaults or user-supplied values; results are only as good as the inputs (see data quality)."]
    if not r["boundary"]["is_full_cfp"]:
        out.append(f"{r['boundary']['label']}: stages outside the boundary are not included, so this is a partial CFP.")
    if r["exclusions"]:
        out.append("Declared exclusions: " + "; ".join(r["exclusions"]))
    if r["cutoff"]["status"] == "not assessed":
        out.append("Cut-off was not assessed (no excluded-flow masses supplied).")
    if r["data_quality"]["primary_data_share"] < 0.5:
        out.append("Less than half of the emissions come from primary data; replace default factors with plant data.")
    return out


def _fmt(x) -> str:
    return f"{x:,.1f}" if isinstance(x, (int, float)) else str(x)


def _markdown(r: dict, s: dict) -> str:
    u = r["uncertainty"]
    L = [f"# {r['product'].title()} carbon footprint — ISO 14067 study report", "",
         f"**Result:** {r['cfp_kg_co2e']:,.1f} kg CO2e per {r['declared_unit']} "
         f"(P5–P95 {u['p5']:,.1f}–{u['p95']:,.1f}) · {r['boundary']['label']} · {r['gwp']}", "",
         "## 1. General", *[f"- {k.replace('_', ' ')}: {v}" for k, v in s["1_general"].items()], "",
         "## 2. Goal", *[f"- {k.replace('_', ' ')}: {v}" for k, v in s["2_goal"].items()], "",
         "## 3. Scope", f"- Declared unit: {r['declared_unit']}", f"- Boundary: {r['boundary']['label']}",
         f"- Reference year: {r['reference_year']}",
         f"- Allocation: {s['3_scope']['allocation'] if isinstance(s['3_scope']['allocation'], str) else s['3_scope']['allocation']['method'] + ' (product share ' + str(s['3_scope']['allocation']['product_share']) + ')'}",
         f"- Cut-off: {r['cutoff']['status']} (criterion {r['cutoff']['criterion_pct']} %)", "",
         "## 4. Inventory and data", "| Stage | Flow | kg CO2e | Basis |", "|---|---|---:|---|",
         *[f"| {d['stage']} | {d['item']} | {_fmt(d['kg_co2e'])} | {d.get('basis', '')} |" for d in r["detail"]], "",
         "Sources: " + "; ".join(r["factor_sources"]), "",
         "## 5. Results", "| Stage | kg CO2e |", "|---|---:|", *[f"| {k} | {_fmt(v)} |" for k, v in r["by_stage"].items()], "",
         "| GHG type (ISO 14067 §6.4.9.4) | kg CO2e |", "|---|---:|", *[f"| {k.replace('_', ' ')} | {_fmt(v)} |" for k, v in r["by_ghg_type"].items()], "",
         "## 6. Data quality (ISO 14067 §6.3.5)", "| Criterion | Score | Basis |", "|---|---:|---|",
         *[f"| {c['criterion']} | {c['score']} ({c['rating']}) | {c['basis']} |" for c in r["data_quality"]["criteria"]], "",
         f"Overall {r['data_quality']['overall_score']} ({r['data_quality']['overall_rating']}); 1 = best, 5 = worst.", "",
         "## 7. Interpretation", "Hotspots (to 80 % of the total):",
         *[f"- {h['item']} ({h['stage']}): {h['share_pct']} % (cumulative {h['cumulative_pct']} %)" for h in r["hotspots_to_80pct"]], "",
         "Methodological choices:", *[f"- {c['choice']}: {c.get('change_vs_selected_pct')} %" for c in r["methodological_choices"]], "",
         "Limitations:", *[f"- {x}" for x in s["7_interpretation"]["limitations"]], "",
         "## 8. Review and communication", f"- Verification: {r['verification']['status']}", f"- {r['communication_statement']}"]
    return "\n".join(L)


def report_xlsx(r: dict, rep: dict) -> bytes:
    from openpyxl import Workbook
    from openpyxl.styles import Font
    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    u = r["uncertainty"]
    rows = [("Product", r["product"]), ("Declared unit", r["declared_unit"]), ("Boundary", r["boundary"]["label"]),
            ("CFP (kg CO2e)", r["cfp_kg_co2e"]), ("P5", u["p5"]), ("P50", u["p50"]), ("P95", u["p95"]),
            ("GWP", r["gwp"]), ("Reference year", r["reference_year"]), ("Fingerprint", r["fingerprint"]),
            ("Verification", r["verification"]["status"]), ("Statement", r["communication_statement"])]
    for a, b in rows:
        ws.append([a, b])
    for c in ws["A"]:
        c.font = Font(bold=True)
    ws.column_dimensions["A"].width = 22
    ws.column_dimensions["B"].width = 100

    def sheet(name, header, data):
        w = wb.create_sheet(name)
        w.append(header)
        for c in w[1]:
            c.font = Font(bold=True)
        for d in data:
            w.append(d)
        for col in w.columns:
            w.column_dimensions[col[0].column_letter].width = 38
    sheet("Inventory", ["Stage", "Flow", "kg CO2e", "Basis"], [[d["stage"], d["item"], d["kg_co2e"], d.get("basis", "")] for d in r["detail"]])
    sheet("By stage", ["Stage", "kg CO2e"], [[k, v] for k, v in r["by_stage"].items()])
    sheet("By GHG type", ["Type", "kg CO2e"], [[k, v] for k, v in r["by_ghg_type"].items()])
    sheet("Data quality", ["Criterion", "Score", "Rating", "Basis"], [[c["criterion"], c["score"], c["rating"], c["basis"]] for c in r["data_quality"]["criteria"]])
    sheet("Hotspots", ["Flow", "Stage", "kg CO2e", "Share %", "Cumulative %"], [[h["item"], h["stage"], h["kg_co2e"], h["share_pct"], h["cumulative_pct"]] for h in r["hotspots_to_80pct"]])
    sheet("Sensitivity", ["Flow", "Stage", "kg CO2e", "+10% effect on total (%)"], [[s["item"], s["stage"], s["kg_co2e"], s["effect_of_plus_10pct_on_total_pct"]] for s in r["sensitivity"]])
    sheet("Methodological choices", ["Choice", "Result kg CO2e", "Change %"], [[c["choice"], c.get("gate_kg_co2e", c.get("total_kg_co2e")), c.get("change_vs_selected_pct")] for c in r["methodological_choices"]])
    sheet("Sources", ["Source"], [[s] for s in r["factor_sources"]])
    sheet("Report (markdown)", ["Line"], [[ln] for ln in rep["markdown"].split("\n")])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
