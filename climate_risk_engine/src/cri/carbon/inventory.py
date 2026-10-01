"""
Corporate GHG inventory (GHG Protocol Corporate Standard + Scope 2 Guidance + Scope 3 Standard; ISO 14064-1).

Inputs are the client's own activity data. Each record is calculated with a named factor, consolidated by the
chosen organisational boundary, and kept with its evidence reference and data-quality score so the final
numbers can be traced line by line (see assurance.py).

Scope 2 is reported both ways:
  location-based — grid-average factor for where the electricity is used;
  market-based   — contractual instruments first (EACs/PPAs/green tariffs), then a supplier-specific factor,
                   then the residual mix; where no residual mix exists the location-based factor is used and
                   this is disclosed (GHG Protocol Scope 2 Guidance, Table 6.3).
Scope 3 category 3 (fuel- and energy-related activities) is derived automatically from Scope 1 fuels
(well-to-tank) and Scope 2 electricity (upstream of generation, and T&D losses).
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from . import factors as F

S3_CATEGORIES = {
    1: "Purchased goods and services", 2: "Capital goods", 3: "Fuel- and energy-related activities",
    4: "Upstream transportation and distribution", 5: "Waste generated in operations", 6: "Business travel",
    7: "Employee commuting", 8: "Upstream leased assets", 9: "Downstream transportation and distribution",
    10: "Processing of sold products", 11: "Use of sold products", 12: "End-of-life treatment of sold products",
    13: "Downstream leased assets", 14: "Franchises", 15: "Investments",
}
CATEGORIES = {
    "s1_stationary": (1, "Stationary combustion"), "s1_mobile": (1, "Mobile combustion"),
    "s1_fugitive": (1, "Fugitive emissions (refrigerants, SF6)"), "s1_process": (1, "Process emissions"),
    "s2_electricity": (2, "Purchased electricity"), "s2_heat_steam": (2, "Purchased heat, steam and cooling"),
    **{f"s3_c{k}": (3, f"Cat {k} — {v}") for k, v in S3_CATEGORIES.items()},
}
METHODS = ("factor", "custom_factor", "spend", "direct", "electricity", "heat")
DEFAULT_DQ = {"factor": 2, "custom_factor": 2, "spend": 4, "direct": 2, "electricity": 2, "heat": 2}


@dataclass
class Entity:
    id: str
    name: str
    country: str
    equity_pct: float = 100.0
    operational_control: bool = True
    financial_control: bool = True


@dataclass
class Inventory:
    organisation: str
    reporting_year: int
    consolidation: str = "operational_control"      # operational_control | financial_control | equity_share
    gwp: str = "AR6"
    entities: list[Entity] = field(default_factory=list)
    records: list[dict] = field(default_factory=list)
    category_status: dict = field(default_factory=dict)   # "s3_c10": {"status": "not_relevant", "reason": "..."}
    base_year: Optional[int] = None
    revenue_musd: Optional[float] = None
    employees: Optional[float] = None
    intake_notes: list = field(default_factory=list)       # warnings raised while reading an uploaded workbook


class InventoryError(ValueError):
    pass


def _share(ent: Entity, approach: str) -> float:
    if approach == "equity_share":
        return max(0.0, min(100.0, ent.equity_pct)) / 100.0
    if approach == "financial_control":
        return 1.0 if ent.financial_control else 0.0
    if approach == "operational_control":
        return 1.0 if ent.operational_control else 0.0
    raise InventoryError("consolidation must be operational_control, financial_control or equity_share")


def _unit_ok(a: str, b: str) -> bool:
    n = lambda s: (s or "").lower().replace(" ", "").replace("(", "").replace(")", "")
    return n(a) == n(b)


def _calc(rec: dict, ent: Entity, gwp: str) -> dict:
    """Unconsolidated emissions of one record (kg CO2e) with full working."""
    m = rec.get("method")
    if m not in METHODS:
        raise InventoryError(f"record {rec.get('id')}: method must be one of {METHODS}")
    cat = rec.get("category")
    if cat not in CATEGORIES:
        raise InventoryError(f"record {rec.get('id')}: unknown category '{cat}'")
    scope = CATEGORIES[cat][0]
    out: dict = {"scope": scope, "category": cat, "warnings": [], "derived": []}

    def qty():
        q = rec.get("quantity")
        if q is None or float(q) < 0:
            raise InventoryError(f"record {rec.get('id')}: quantity must be ≥ 0")
        return float(q)

    if m == "factor":
        f = F.desnz(rec["factor_id"], gwp)
        if rec.get("unit") and not _unit_ok(rec["unit"], f["unit"]):
            raise InventoryError(f"record {rec.get('id')}: unit '{rec['unit']}' does not match factor unit '{f['unit']}'")
        q = qty()
        out.update(kgco2e=q * f["kgco2e_per_unit"], gases={g: q * v for g, v in f["gases"].items()},
                   working=f"{q:g} {f['unit']} × {f['kgco2e_per_unit']:.6g} kg CO2e/{f['unit']}",
                   factor={k: f[k] for k in ("id", "label", "source", "gwp", "unit")} | {"value": f["kgco2e_per_unit"]})
        if f.get("gwp_flag"):
            out["warnings"].append("factor has no AR6 restatement — AR5 value used")
        want = {1: "Scope 1", 2: "Scope 2", 3: "Scope 3"}[scope]
        if f["desnz_scope"] != want and not (scope == 3 and f["desnz_scope"] in ("Scope 1", "Scope 2")):
            out["warnings"].append(f"factor is a DESNZ {f['desnz_scope']} factor used in {want}")
        if scope == 1:
            w = F.wtt_for(rec["factor_id"])
            if w:
                out["derived"].append({"category": "s3_c3", "kgco2e": q * w["kgco2e_per_unit"],
                                       "working": f"WTT {q:g} {f['unit']} × {w['kgco2e_per_unit']:.6g}",
                                       "factor": {"id": w["id"], "source": w["source"], "gwp": w["gwp"]}})
    elif m == "custom_factor":
        q = qty()
        v = rec.get("kgco2e_per_unit")
        if v is None or not rec.get("factor_source"):
            raise InventoryError(f"record {rec.get('id')}: custom factors need kgco2e_per_unit and factor_source")
        out.update(kgco2e=q * float(v), gases={"CO2e": q * float(v)},
                   working=f"{q:g} {rec.get('unit', 'unit')} × {float(v):.6g} kg CO2e (supplier/custom)",
                   factor={"id": "custom", "label": rec.get("factor_label", "custom factor"),
                           "source": rec["factor_source"], "value": float(v), "unit": rec.get("unit")})
    elif m == "spend":
        amt = rec.get("amount")
        if amt is None or float(amt) < 0:
            raise InventoryError(f"record {rec.get('id')}: amount must be ≥ 0")
        fx = float(rec.get("fx_to_usd", 1.0 if (rec.get("currency", "USD").upper() == "USD") else 0) or 0)
        if fx <= 0:
            raise InventoryError(f"record {rec.get('id')}: give fx_to_usd for spend in {rec.get('currency')}")
        sf = F.spend_factor(rec["naics"], int(rec.get("spend_year", rec.get("year"))))
        usd = float(amt) * fx
        kg = usd * sf["kgco2e_per_usd_nominal"]
        out.update(kgco2e=kg, gases={"CO2e": kg},
                   working=f"{float(amt):,.0f} {rec.get('currency', 'USD')} × {fx:g} → ${usd:,.0f} × deflator "
                           f"{sf['deflator_to_2022']:.3f} × {sf['kgco2e_per_usd_2022']:.3f} kg/2022$ (NAICS {rec['naics']})",
                   factor={"id": f"useeio13:{rec['naics']}", "label": sf["title"], "source": sf["source"],
                           "gwp": sf["gwp"], "value": sf["kgco2e_per_usd_2022"], "unit": "kg CO2e / 2022 USD"})
        if ent.country and F._iso3(ent.country) != "USA":
            out["warnings"].append("spend-based factor reflects the US economy (USEEIO); treat as an estimate for non-US spend")
    elif m == "direct":
        v = rec.get("kgco2e")
        if v is None or float(v) < 0 or not rec.get("factor_source"):
            raise InventoryError(f"record {rec.get('id')}: direct records need kgco2e ≥ 0 and factor_source "
                                 "(e.g. supplier-reported, CEMS, PCAF run)")
        out.update(kgco2e=float(v), gases={"CO2e": float(v)}, working=f"{float(v):,.1f} kg CO2e reported",
                   factor={"id": "direct", "label": "reported/measured", "source": rec["factor_source"]})
    elif m == "electricity":
        if cat != "s2_electricity" and not cat.startswith("s3_"):
            raise InventoryError(f"record {rec.get('id')}: electricity method belongs in s2_electricity or a Scope 3 category")
        kwh = rec.get("kwh")
        if kwh is None or float(kwh) < 0:
            raise InventoryError(f"record {rec.get('id')}: kwh must be ≥ 0")
        kwh = float(kwh)
        country = rec.get("country") or ent.country
        if rec.get("location_g_per_kwh") is not None:
            loc = {"g_per_kwh": float(rec["location_g_per_kwh"]), "source": rec.get("location_source", "user-supplied"),
                   "dq": 2}
        else:
            loc = F.electricity_location(country, rec.get("us_state"), rec.get("egrid_subregion"))
        sub = rec.get("egrid_subregion") or loc.get("subregion")
        loc_kg = kwh * loc["g_per_kwh"] / 1000
        # market-based hierarchy
        covered = 0.0
        mkt_kg = 0.0
        mb_lines = []
        for ins in rec.get("instruments") or []:
            mwh = float(ins.get("mwh", 0))
            if mwh < 0:
                raise InventoryError(f"record {rec.get('id')}: instrument MWh must be ≥ 0")
            take = min(mwh * 1000, kwh - covered)
            g = float(ins.get("g_per_kwh", 0.0))
            if not ins.get("certificate"):
                out["warnings"].append("contractual instrument without certificate/contract reference")
            mkt_kg += take * g / 1000
            covered += take
            mb_lines.append(f"{take:,.0f} kWh {ins.get('type', 'EAC')} @ {g:g} g")
            if mwh * 1000 > take + 1e-6:
                out["warnings"].append("instruments exceed consumption — excess ignored")
        rest = kwh - covered
        if rest > 0:
            if rec.get("supplier_g_per_kwh") is not None:
                g, src = float(rec["supplier_g_per_kwh"]), rec.get("supplier_source", "supplier-specific")
            else:
                rm = F.electricity_residual(country, sub)
                if rm:
                    g, src = rm["g_per_kwh"], rm["source"]
                else:
                    g, src = loc["g_per_kwh"], "location-based factor (no residual mix available — disclosed)"
                    out["warnings"].append("market-based: no residual mix for this grid, location-based factor used")
            mkt_kg += rest * g / 1000
            mb_lines.append(f"{rest:,.0f} kWh @ {g:.1f} g ({src})")
        out.update(kgco2e=loc_kg, kgco2e_market=mkt_kg, gases={"CO2e": loc_kg},
                   working=f"{kwh:,.0f} kWh × {loc['g_per_kwh']:.1f} g/kWh (location); market: " + "; ".join(mb_lines),
                   factor={"id": "grid", "label": loc.get("basis", ""), "source": loc["source"], "value": loc["g_per_kwh"],
                           "unit": "g CO2e/kWh"})
        if loc.get("note"):
            out["warnings"].append(loc["note"])
        if cat == "s2_electricity":
            iso = F._iso3(country)
            if iso == "GBR":
                E = F.lib()["electricity"]["GBR"]
                td = next(x for x in E["desnz_td"] if "UK" in " ".join(x["path"]))["kgco2e"]
                wg = sum(x["kgco2e"] for x in E["desnz_wtt"])
                out["derived"].append({"category": "s3_c3", "kgco2e": kwh * (td + wg),
                                       "working": f"{kwh:,.0f} kWh × (T&D {td} + WTT {wg:.5f}) kg/kWh",
                                       "factor": {"id": "desnz26:uk-electricity-td+wtt", "source": "UK DESNZ 2026",
                                                  "gwp": "AR5 (DESNZ)"}})
            else:
                loss = F.td_loss_share(country, sub)
                up = F.upstream_generation_share(country)
                if loss:
                    l = loss["share"]
                    lost = kwh * l / (1 - l)
                    kg = lost * loc["g_per_kwh"] / 1000 * (1 + (up or 0))
                    out["derived"].append({"category": "s3_c3", "kgco2e": kg,
                                           "working": f"T&D losses {lost:,.0f} kWh ({l:.1%}, {loss['source']}) × {loc['g_per_kwh']:.1f} g",
                                           "factor": {"id": "td_losses", "source": loss["source"]}})
                if up:
                    kg = kwh * loc["g_per_kwh"] / 1000 * up
                    out["derived"].append({"category": "s3_c3", "kgco2e": kg,
                                           "working": f"upstream of generation {kwh:,.0f} kWh × {loc['g_per_kwh']:.1f} g × {up:.3f}",
                                           "factor": {"id": "ember_upstream", "source": "Ember lifecycle − direct"}})
                if not loss and not up:
                    out["warnings"].append("no T&D-loss or upstream data for this country — category 3 electricity part excluded")
    elif m == "heat":
        q = qty()
        if rec.get("kgco2e_per_kwh") is not None:
            v, src = float(rec["kgco2e_per_kwh"]), rec.get("factor_source", "supplier-specific")
        elif F._iso3(rec.get("country") or ent.country) == "GBR":
            f = F.desnz("desnz26:heat-and-steam-heat-and-steam-district-heat-and-steam-kwh-kwh", gwp)
            v, src = f["kgco2e_per_unit"], f["source"]
        else:
            raise InventoryError(f"record {rec.get('id')}: purchased heat outside the UK needs the supplier's kgco2e_per_kwh")
        out.update(kgco2e=q * v, kgco2e_market=q * v, gases={"CO2e": q * v}, working=f"{q:g} kWh × {v:.5g} kg/kWh",
                   factor={"id": "heat", "source": src, "value": v, "unit": "kg CO2e/kWh"})
    if scope == 2 and "kgco2e_market" not in out:
        out["kgco2e_market"] = out["kgco2e"]
    return out


def calculate(inv: Inventory) -> dict:
    ents = {e.id: e for e in inv.entities}
    if not ents:
        raise InventoryError("add at least one entity (the organisational boundary)")
    lines, errors, other_year = [], [], []
    ids = [str(r.get("id")) for r in inv.records if r.get("id") is not None]
    dup = sorted({x for x in ids if ids.count(x) > 1})
    if dup:
        errors.append(f"duplicate record id(s) {', '.join(dup)} — each record needs its own id for the audit trail")
    for i, rec in enumerate(inv.records):
        rec = dict(rec)
        rec.setdefault("id", f"r{i + 1}")
        ent = ents.get(rec.get("entity"))
        if ent is None:
            errors.append(f"record {rec['id']}: unknown entity '{rec.get('entity')}'")
            continue
        if int(rec.get("year", inv.reporting_year)) != inv.reporting_year:
            other_year.append(f"{rec['id']} ({rec.get('year')})")
            continue
        try:
            r = _calc(rec, ent, inv.gwp)
        except (InventoryError, KeyError) as exc:
            errors.append(str(exc).strip("'\""))
            continue
        share = _share(ent, inv.consolidation)
        dq = int(rec.get("dq", DEFAULT_DQ[rec["method"]]))
        base = {"record": rec["id"], "entity": ent.id, "entity_name": ent.name, "share": share, "dq": dq,
                "evidence": rec.get("evidence"), "notes": rec.get("notes"), "method": rec["method"],
                "description": rec.get("description", "")}
        lines.append(base | {"scope": r["scope"], "category": r["category"], "kgco2e_gross": r["kgco2e"],
                             "t_co2e": r["kgco2e"] * share / 1000,
                             "t_co2e_market": (r["kgco2e_market"] * share / 1000) if "kgco2e_market" in r else None,
                             "gases_t": {g: v * share / 1000 for g, v in r["gases"].items()},
                             "working": r["working"], "factor": r["factor"], "warnings": r["warnings"]})
        for d in r["derived"]:
            lines.append(base | {"scope": 3, "category": d["category"], "kgco2e_gross": d["kgco2e"],
                                 "t_co2e": d["kgco2e"] * share / 1000, "t_co2e_market": None,
                                 "gases_t": {"CO2e": d["kgco2e"] * share / 1000}, "working": d["working"],
                                 "factor": d["factor"], "warnings": [], "derived_from": rec["id"], "dq": max(dq, 3)})
    if errors:
        raise InventoryError("; ".join(errors))

    s1 = sum(l["t_co2e"] for l in lines if l["scope"] == 1)
    s2l = sum(l["t_co2e"] for l in lines if l["scope"] == 2)
    s2m = sum(l["t_co2e_market"] for l in lines if l["scope"] == 2)
    by_cat = defaultdict(float)
    for l in lines:
        by_cat[l["category"]] += l["t_co2e"]
    s3 = sum(v for k, v in by_cat.items() if k.startswith("s3_"))
    by_entity = defaultdict(lambda: defaultdict(float))
    for l in lines:
        by_entity[l["entity"]][f"scope{l['scope']}"] += l["t_co2e"]
    by_gas = defaultdict(float)
    for l in lines:
        if l["scope"] == 1:
            for g, v in l["gases_t"].items():
                by_gas[g] += v
    tot = s1 + s2l + s3
    dq_w = (sum(l["dq"] * l["t_co2e"] for l in lines) / tot) if tot else None

    completeness = []
    for k, (sc, name) in CATEGORIES.items():
        present = by_cat.get(k, 0) > 0 or any(l["category"] == k for l in lines)
        st = inv.category_status.get(k, {})
        completeness.append({"category": k, "scope": sc, "name": name, "reported": present,
                             "t_co2e": round(by_cat.get(k, 0.0), 3),
                             "status": "reported" if present else st.get("status", "missing" if sc == 3 else "none"),
                             "reason": st.get("reason")})
    checks = [{"level": "warning", "check": "intake", "message": n} for n in inv.intake_notes]
    if other_year:
        checks.append({"level": "warning", "check": "period",
                       "message": f"{len(other_year)} record(s) dated outside {inv.reporting_year} were not counted: "
                                  + ", ".join(other_year[:10]) + (" …" if len(other_year) > 10 else "")})
    for c in completeness:
        if c["status"] in ("excluded", "not_relevant") and not (c.get("reason") or "").strip():
            checks.append({"level": "error", "check": "completeness",
                           "message": f"{c['name']}: marked {c['status'].replace('_', ' ')} without a reason — "
                                      "the GHG Protocol requires exclusions to be disclosed and justified"})
        if c["status"] == "missing" and c["scope"] == 3:
            checks.append({"level": "error", "check": "completeness",
                           "message": f"{c['name']}: no data and no exclusion reason — report it or state why it is excluded"})
    for l in lines:
        if not l.get("evidence") and not l.get("derived_from"):
            checks.append({"level": "warning", "check": "evidence", "message": f"record {l['record']}: no evidence reference"})
        for w in l["warnings"]:
            checks.append({"level": "warning", "check": "method", "message": f"record {l['record']}: {w}"})
    if tot:
        spend_share = sum(l["t_co2e"] for l in lines if l["method"] == "spend") / tot
        if spend_share > 0.5:
            checks.append({"level": "warning", "check": "method",
                           "message": f"{spend_share:.0%} of the inventory is spend-based — replace large lines with activity or supplier data"})

    return {
        "organisation": inv.organisation, "reporting_year": inv.reporting_year, "consolidation": inv.consolidation,
        "gwp": inv.gwp,
        "totals_t": {"scope1": round(s1, 3), "scope2_location": round(s2l, 3), "scope2_market": round(s2m, 3),
                     "scope3": round(s3, 3), "total_location": round(tot, 3), "total_market": round(round(s1, 3) + round(s2m, 3) + round(s3, 3), 3)},
        "scope3_by_category_t": {S3_CATEGORIES[int(k[4:])]: round(v, 3) for k, v in sorted(by_cat.items(), key=lambda x: (len(x[0]), x[0])) if k.startswith("s3_")},
        "scope1_by_gas_t": {g: round(v, 3) for g, v in by_gas.items()},
        "by_category_t": {CATEGORIES[k][1]: round(v, 3) for k, v in by_cat.items()},
        "by_entity_t": {ents[e].name: {k: round(v, 3) for k, v in d.items()} for e, d in by_entity.items()},
        "intensity": {"t_per_musd_revenue": round((s1 + s2l) / inv.revenue_musd, 3) if inv.revenue_musd else None,
                      "t_per_employee": round((s1 + s2l) / inv.employees, 3) if inv.employees else None,
                      "basis": "Scope 1 + Scope 2 (location-based)"},
        "data_quality": {"weighted_score": round(dq_w, 2) if dq_w else None,
                         "scale": "1 = metered/invoiced primary data … 5 = estimate/proxy",
                         "share_primary": round(sum(l["t_co2e"] for l in lines if l["dq"] <= 2) / tot, 3) if tot else None},
        "completeness": completeness, "checks": checks, "lines": lines,
        "fingerprint": hashlib.sha256(json.dumps({"r": inv.records, "e": [e.__dict__ for e in inv.entities],
                                                  "c": inv.consolidation, "g": inv.gwp}, sort_keys=True,
                                                 default=str).encode()).hexdigest()[:16],
        "factor_library": F.lib()["version"],
        "standards": ["GHG Protocol Corporate Standard (rev.)", "GHG Protocol Scope 2 Guidance (2015)",
                      "GHG Protocol Corporate Value Chain (Scope 3) Standard", "ISO 14064-1:2018"],
    }


def base_year_recalculation(base_year_total_t: float, changes: list[dict], threshold: float = 0.05) -> dict:
    """GHG Protocol ch.5: recalculate the base year when structural changes, methodology changes or errors
    shift base-year emissions by more than the significance threshold (default 5 %)."""
    if base_year_total_t <= 0:
        raise InventoryError("base_year_total_t must be > 0")
    rows, cum = [], 0.0
    for c in changes:
        eff = float(c.get("base_year_effect_t", 0.0))
        cum += eff
        rows.append({"type": c.get("type"), "description": c.get("description"), "base_year_effect_t": eff,
                     "share": round(eff / base_year_total_t, 4)})
    need = abs(cum) / base_year_total_t > threshold
    return {"changes": rows, "cumulative_effect_t": round(cum, 3), "cumulative_share": round(cum / base_year_total_t, 4),
            "threshold": threshold, "recalculation_required": need,
            "recalculated_base_year_t": round(base_year_total_t + cum, 3) if need else base_year_total_t,
            "note": "Organic growth or decline does not trigger recalculation; acquisitions, divestments, "
                    "outsourcing/insourcing, methodology changes and discovered errors do."}


def from_dict(d: dict) -> Inventory:
    ents = [Entity(**e) for e in d.get("entities", [])]
    return Inventory(organisation=d["organisation"], reporting_year=int(d["reporting_year"]),
                     consolidation=d.get("consolidation", "operational_control"), gwp=d.get("gwp", "AR6"),
                     entities=ents, records=d.get("records", []), category_status=d.get("category_status", {}),
                     base_year=d.get("base_year"), revenue_musd=d.get("revenue_musd"), employees=d.get("employees"),
                     intake_notes=list(d.get("intake_notes") or []))
