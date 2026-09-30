"""
Real-estate decarbonisation and stranding (operational carbon, CRREM-style).

Per building: energy-use intensity (kWh/m²·yr) and operational carbon intensity (kg CO2e/m²·yr) from metered
energy by carrier, using the carbon-accounting factor library (grid factor for the building's location; fuels
from UK DESNZ 2026 restated to AR6). The intensity is projected to 2050 as the grid decarbonises (NGFS Phase 5
scenario, median of models) and with any planned retrofits, then compared year by year with a decarbonisation
pathway to find the **stranding year** — the first year the building is above the pathway — and the excess
emissions and their cost after that.

Pathways: the CRREM Global Pathways are the market reference. CRREM data is free for internal analysis with
attribution but embedding it in paid software needs a CRREM License Partner agreement, so the pathway is an
input here (paste or upload the CRREM values for the country and property type you are licensed to use, or
your own science-based pathway). Nothing is embedded or approximated.

Regulatory checks (where the country/property type matches):
  Netherlands — offices must hold at least EPC label C since 1 Jan 2023 (Bouwbesluit art. 5.11).
  England & Wales — non-domestic MEES: no continued letting below EPC E since 1 Apr 2023.
  France — Décret tertiaire: final energy −40 % by 2030, −50 % 2040, −60 % 2050 vs a reference year ≥ 2010
           (tertiary buildings ≥ 1,000 m²).
  EU — EPBD recast (Directive (EU) 2024/1275): minimum energy performance standards for the worst-performing
       non-residential stock (16 % by 2030, 26 % by 2033); national thresholds still being transposed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .carbon import factors as CF

EPC_ORDER = ["A+++", "A++", "A+", "A", "B", "C", "D", "E", "F", "G"]
FUEL_FACTORS = {   # carrier → DESNZ 2026 factor id (kWh gross CV basis where applicable)
    "natural_gas_kwh": "desnz26:fuels-gaseous-fuels-natural-gas-kwh-gross-cv",
    "fuel_oil_kwh": "desnz26:fuels-liquid-fuels-burning-oil-kwh-gross-cv",
    "lpg_kwh": "desnz26:fuels-gaseous-fuels-lpg-kwh-gross-cv",
    "diesel_kwh": "desnz26:fuels-liquid-fuels-diesel-average-biofuel-blend-kwh-gross-cv",
}


@dataclass
class Building:
    id: str
    name: str
    country: str
    property_type: str
    floor_area_m2: float
    electricity_kwh: float = 0.0
    natural_gas_kwh: float = 0.0
    fuel_oil_kwh: float = 0.0
    lpg_kwh: float = 0.0
    diesel_kwh: float = 0.0
    district_heat_kwh: float = 0.0
    district_heat_kgco2e_per_kwh: Optional[float] = None
    onsite_renewable_kwh: float = 0.0       # generated and consumed on site (reduces grid electricity)
    us_state: Optional[str] = None
    egrid_subregion: Optional[str] = None
    grid_g_per_kwh: Optional[float] = None   # override (e.g. national/utility factor)
    data_year: int = 2025
    epc_rating: Optional[str] = None
    is_let: bool = True
    value_usd: Optional[float] = None
    retrofits: list = field(default_factory=list)   # [{"year": 2028, "energy_reduction_pct": 25, "electrify_heat": true, "capex_usd": ..}]
    reference_year_final_energy_kwh: Optional[float] = None   # France Décret tertiaire reference


class REError(ValueError):
    pass


def _fuel_kg_per_kwh(fid: str) -> float:
    return CF.desnz(fid, "AR6")["kgco2e_per_unit"]


def _grid(b: Building) -> dict:
    if b.grid_g_per_kwh is not None:
        return {"g_per_kwh": float(b.grid_g_per_kwh), "source": "user-supplied"}
    return CF.electricity_location(b.country, b.us_state, b.egrid_subregion)


def _grid_multipliers(country: str, scenario: str, years: list) -> list:
    from .lca.engine import grid_index_paths
    p = grid_index_paths(CF._iso3(country))
    idx = p["scenarios"].get(scenario)
    if idx is None:
        raise REError(f"unknown scenario '{scenario}' — one of {sorted(p['scenarios'])}")
    base = float(np.interp(years[0], p["years"], idx)) or 1.0
    return [float(np.interp(y, p["years"], idx)) / base for y in years]


def _year_state(b: Building, y: int) -> dict:
    """Energy by carrier in year y after retrofits (electrify_heat moves fossil heat to electricity at a
    coefficient of performance of 3)."""
    e = {"electricity": max(0.0, b.electricity_kwh - b.onsite_renewable_kwh), "district_heat": b.district_heat_kwh,
         **{k: getattr(b, k) for k in FUEL_FACTORS}}
    for r in sorted(b.retrofits, key=lambda r: r["year"]):
        if y < int(r["year"]):
            continue
        f = 1 - float(r.get("energy_reduction_pct", 0)) / 100
        e = {k: v * f for k, v in e.items()}
        if r.get("electrify_heat"):
            heat = sum(e[k] for k in FUEL_FACTORS)
            e["electricity"] += heat / 3.0
            for k in FUEL_FACTORS:
                e[k] = 0.0
    return e


def assess_building(b: Building, pathway_kgco2e_m2: Optional[dict] = None, pathway_source: str = "",
                    scenario: str = "Current Policies", carbon_price_usd_t: float = 0.0,
                    annual_improvement_pct: float = 0.0, end_year: int = 2050) -> dict:
    if b.floor_area_m2 <= 0:
        raise REError(f"{b.id}: floor_area_m2 must be > 0")
    if b.district_heat_kwh > 0 and b.district_heat_kgco2e_per_kwh is None and CF._iso3(b.country) != "GBR":
        raise REError(f"{b.id}: district heat outside the UK needs the network's kgco2e_per_kwh")
    grid = _grid(b)
    years = list(range(b.data_year, end_year + 1))
    mult = _grid_multipliers(b.country, scenario, years)
    fuel_f = {k: _fuel_kg_per_kwh(fid) for k, fid in FUEL_FACTORS.items()}
    dh_f = (b.district_heat_kgco2e_per_kwh if b.district_heat_kgco2e_per_kwh is not None
            else CF.desnz("desnz26:heat-and-steam-heat-and-steam-district-heat-and-steam-kwh-kwh", "AR6")["kgco2e_per_unit"])
    ci, eui, tco2 = [], [], []
    for i, y in enumerate(years):
        e = _year_state(b, y)
        imp = (1 - annual_improvement_pct / 100) ** (y - b.data_year)
        e = {k: v * imp for k, v in e.items()}
        g = 25.0 + (grid["g_per_kwh"] - 25.0) * mult[i] if grid["g_per_kwh"] > 25 and b.grid_g_per_kwh is None else grid["g_per_kwh"]
        kg = e["electricity"] * g / 1000 + e["district_heat"] * dh_f + sum(e[k] * fuel_f[k] for k in FUEL_FACTORS)
        tco2.append(kg / 1000)
        ci.append(kg / b.floor_area_m2)
        eui.append((sum(e.values()) + (b.onsite_renewable_kwh if y == b.data_year else 0)) / b.floor_area_m2)
    out = {"id": b.id, "name": b.name, "country": b.country, "property_type": b.property_type,
           "floor_area_m2": b.floor_area_m2, "years": years,
           "eui_kwh_m2": round(eui[0], 1), "carbon_intensity_kgco2e_m2": round(ci[0], 2),
           "operational_tco2e": round(tco2[0], 2), "trajectory_kgco2e_m2": [round(v, 2) for v in ci],
           "grid": {"g_per_kwh": grid["g_per_kwh"], "source": grid["source"], "scenario": scenario,
                    "basis": "location-based grid factor, decarbonised along the NGFS Phase 5 scenario (median of models)"},
           "stranding": None, "regulatory": _regulatory(b)}
    if pathway_kgco2e_m2:
        pts = sorted((int(k), float(v)) for k, v in pathway_kgco2e_m2.items())
        xs, ys = zip(*pts)
        pw = [float(np.interp(y, xs, ys)) for y in years]
        above = [ci[i] > pw[i] + 1e-9 for i in range(len(years))]
        strand = next((years[i] for i, a in enumerate(above) if a), None)
        excess_t = [max(0.0, ci[i] - pw[i]) * b.floor_area_m2 / 1000 for i in range(len(years))]
        out["stranding"] = {"pathway_source": pathway_source or "user-supplied pathway",
                            "pathway_kgco2e_m2": [round(v, 2) for v in pw], "stranding_year": strand,
                            "aligned_today": not above[0],
                            "cumulative_excess_tco2e": round(sum(excess_t), 1),
                            "excess_cost_usd_at_price": round(sum(excess_t) * carbon_price_usd_t) if carbon_price_usd_t else None,
                            "carbon_price_usd_t": carbon_price_usd_t or None}
    capex = sum(float(r.get("capex_usd", 0) or 0) for r in b.retrofits)
    if capex:
        out["retrofit_capex_usd"] = capex
    return out


def _regulatory(b: Building) -> list:
    iso = CF._iso3(b.country)
    pt = (b.property_type or "").lower()
    rating = (b.epc_rating or "").upper().replace(" ", "")
    res = []

    def worse(r, limit):
        return r in EPC_ORDER and EPC_ORDER.index(r) > EPC_ORDER.index(limit)
    if iso == "NLD" and "office" in pt:
        if not rating:
            res.append({"rule": "NL offices ≥ label C (since 2023)", "status": "unknown", "note": "give the EPC label"})
        else:
            res.append({"rule": "NL offices ≥ label C (since 2023)", "status": "breach" if worse(rating, "C") else "ok"})
    if iso == "GBR" and b.is_let and "residential" not in pt:
        if not rating:
            res.append({"rule": "MEES non-domestic ≥ E (since Apr 2023)", "status": "unknown", "note": "give the EPC rating"})
        else:
            res.append({"rule": "MEES non-domestic ≥ E (since Apr 2023)", "status": "breach" if worse(rating, "E") else "ok"})
    if iso == "FRA" and b.floor_area_m2 >= 1000 and "residential" not in pt:
        if b.reference_year_final_energy_kwh:
            now = (b.electricity_kwh + b.natural_gas_kwh + b.fuel_oil_kwh + b.lpg_kwh + b.diesel_kwh + b.district_heat_kwh)
            cut = 1 - now / b.reference_year_final_energy_kwh
            res.append({"rule": "Décret tertiaire −40 % by 2030 vs reference", "status": "ok" if cut >= 0.40 else "gap",
                        "current_reduction_pct": round(cut * 100, 1)})
        else:
            res.append({"rule": "Décret tertiaire −40 % by 2030 vs reference", "status": "unknown",
                        "note": "give reference_year_final_energy_kwh"})
    eu = {"AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA", "DEU", "GRC", "HUN", "IRL", "ITA", "LVA",
          "LTU", "LUX", "MLT", "NLD", "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE"}
    if iso in eu and "residential" not in pt:
        res.append({"rule": "EU EPBD recast: MEPS for worst-performing 16 % (2030) / 26 % (2033) of non-residential stock",
                    "status": "monitor", "note": "national thresholds being transposed"})
    return res


def assess_portfolio(buildings: list, pathways: dict, scenario: str = "Current Policies",
                     carbon_price_usd_t: float = 0.0, annual_improvement_pct: float = 0.0) -> dict:
    """pathways: {"<ISO3>|<property type>": {"source": "...", "kgco2e_m2": {year: value}}} (or key "*" for all)."""
    rows, errors = [], []
    for b in buildings:
        key = f"{CF._iso3(b.country)}|{b.property_type}".lower()
        pw = next((v for k, v in pathways.items() if k.lower() == key), None) or pathways.get("*")
        try:
            rows.append(assess_building(b, (pw or {}).get("kgco2e_m2"), (pw or {}).get("source", ""), scenario,
                                        carbon_price_usd_t, annual_improvement_pct))
        except (REError, KeyError) as exc:
            errors.append(str(exc).strip("'\""))
    if errors:
        raise REError("; ".join(errors))
    area = sum(r["floor_area_m2"] for r in rows) or 1
    stranded = [r for r in rows if r["stranding"] and r["stranding"]["stranding_year"] is not None]
    by_year = {}
    for r in stranded:
        by_year.setdefault(r["stranding"]["stranding_year"], 0.0)
        by_year[r["stranding"]["stranding_year"]] += r["floor_area_m2"]
    cum, sa = 0.0, {}
    for y in sorted(by_year):
        cum += by_year[y]
        sa[y] = round(cum / area * 100, 1)
    return {"n_buildings": len(rows), "floor_area_m2": area,
            "operational_tco2e": round(sum(r["operational_tco2e"] for r in rows), 1),
            "weighted_carbon_intensity_kgco2e_m2": round(sum(r["carbon_intensity_kgco2e_m2"] * r["floor_area_m2"] for r in rows) / area, 2),
            "weighted_eui_kwh_m2": round(sum(r["eui_kwh_m2"] * r["floor_area_m2"] for r in rows) / area, 1),
            "share_of_area_stranded_by_year_pct": sa,
            "buildings_without_pathway": [r["id"] for r in rows if r["stranding"] is None],
            "regulatory_breaches": [{"id": r["id"], **x} for r in rows for x in r["regulatory"] if x["status"] in ("breach", "gap")],
            "buildings": rows, "scenario": scenario,
            "note": "Pathways are supplied by the user (e.g. CRREM under its data terms); none are embedded."}


def from_dict(d: dict) -> Building:
    return Building(**d)
