"""
cri/lca/products.py — screening templates for other CBAM goods, and a fully custom inventory.

  steel      route BF-BOF / EAF / DRI-EAF: IPCC 2006 Vol.3 Table 4.1 Tier 1 process CO2
             (BOF 1.46, EAF 0.08, DRI 0.70 t CO2/t) + plant electricity × grid
  aluminium  primary, prebake: IPCC Tier 1 anode CO2 1.6 t/t + smelter electricity × grid
             (+ alumina and PFCs only if the user supplies them — otherwise listed as exclusions)
  ammonia    natural-gas route: IPCC Tier 1 fuel requirement 30.2 GJ/t (combustion + feedstock)
             − CO2 recovered for urea, + electricity
  custom     any product: the user supplies fuels, electricity, process emissions, purchased
             materials (with factors) and transport; the engine applies the same factors,
             uncertainty and dynamic grid paths

Tier 1 templates are screening estimates. Electricity use is always a required input: it
differs by an order of magnitude between plants and no default is imposed.
"""
from __future__ import annotations

from typing import Optional

from .engine import (ElectricityUse, FuelUse, Inventory, Material, ProcessEmission, TransportLeg,
                     calculate, data_quality_summary, factors, grid_factor, grid_index_paths, uncertainty)

import numpy as np


def _dynamic(inv: Inventory, country: Optional[str]) -> dict:
    paths = grid_index_paths(country)
    years = list(range(2025, 2051))
    out = {}
    for s, idx in paths["scenarios"].items():
        iy = np.interp(years, paths["years"], idx)
        out[s] = [round(calculate(inv, grid_multiplier=float(g))["total_kg_co2e"], 1) for g in iy]
    return {"years": years, "scenarios": out, "grid_basis": paths["region_basis"], "plan_applied": False}


def _result(product: str, inv: Inventory, country: Optional[str], tier_note: str, n: int = 5000) -> dict:
    res = calculate(inv)
    return {
        "product": product, "country": country, "declared_unit": inv.declared_unit,
        "footprint_kg_co2e_per_t": round(res["total_kg_co2e"], 1),
        "uncertainty": {k: (round(v, 1) if isinstance(v, float) else v) for k, v in uncertainty(inv, n=n).items()},
        "by_stage": {k: round(v, 1) for k, v in res["by_stage"].items()},
        "detail": [dict(d, kg_co2e=round(d["kg_co2e"], 2)) for d in res["detail"]],
        "biogenic_co2_kg_per_t": round(res["biogenic_co2_kg"], 1),
        "captured_co2_kg_per_t": round(res["captured_kg_co2"], 1),
        "data_quality": data_quality_summary(inv),
        "exclusions": res["exclusions"], "notes": inv.notes + [tier_note],
        "trajectory": _dynamic(inv, country),
        "standard": "ISO 14067:2018-aligned cradle-to-gate screening (A1–A3); GWP100 IPCC AR6; biogenic CO2 reported "
                    "separately. Not a verified CFP study.",
    }


STEEL_BENCHMARK_NOTE = {
    "bf_bof": "Benchmark check: the worldsteel 2023 global average for BF-BOF is 2.32 t CO2 per t crude steel "
              "(Sustainability Indicators report, partial scope 3). The IPCC Tier 1 process factor plus electricity "
              "typically gives a LOWER figure for integrated mills because it omits externally sourced coke, "
              "fuel-gas and ore-preparation emissions. Treat this result as a floor; use plant data (Tier 2/3) "
              "or add fuels/materials before quoting it.",
    "eaf": "Benchmark check: the worldsteel 2023 global average for scrap-EAF is 0.70 t CO2 per t crude steel, "
           "dominated by grid electricity, alloys and electrodes. Results far below that need plant-specific evidence.",
    "dri_eaf": "Benchmark check: the worldsteel 2023 global average for DRI-EAF is 1.43 t CO2 per t crude steel. "
               "The Tier 1 route here omits ore pelletising and natural-gas supply chain emissions.",
}


def steel(route: str, kwh_per_t: float, country: Optional[str] = None, g_per_kwh: Optional[float] = None,
          extra_fuels: Optional[dict] = None, materials: Optional[list] = None, transport: Optional[list] = None,
          n: int = 5000) -> dict:
    P = factors()["process"]
    key = {"bf_bof": "steel_bof_tco2_per_t", "eaf": "steel_eaf_tco2_per_t", "dri_eaf": "steel_dri_tco2_per_t"}.get(route)
    if not key:
        raise ValueError("route must be bf_bof, eaf or dri_eaf")
    inv = Inventory("1 t crude steel")
    inv.process.append(ProcessEmission(f"Process CO2, {route.upper()} (IPCC Tier 1)", P[key]["value"] * 1000, 4))
    if route == "dri_eaf":
        inv.process.append(ProcessEmission("EAF step (IPCC Tier 1)", P["steel_eaf_tco2_per_t"]["value"] * 1000, 4))
    inv.electricity.append(ElectricityUse(kwh_per_t, country, g_per_kwh, 2, "Plant electricity"))
    for f, gj in (extra_fuels or {}).items():
        inv.fuels.append(FuelUse(f, gj, 2))
    inv.materials += [Material(**m) for m in (materials or [])]
    inv.transport += [TransportLeg(**t) for t in (transport or [])]
    if route in ("eaf", "dri_eaf") and not materials:
        inv.exclusions.append("Scrap / DRI feedstock upstream (supply factors to include)")
    if route == "bf_bof":
        inv.notes.append("IPCC Tier 1 BOF factor covers the integrated route (coke, sinter, iron- and steelmaking).")
    inv.notes.append(STEEL_BENCHMARK_NOTE[route])
    return _result("steel", inv, country, "Tier 1 screening (IPCC 2006 Vol.3 Table 4.1); use plant data for Tier 2/3.", n)


def aluminium(kwh_per_t: float, country: Optional[str] = None, g_per_kwh: Optional[float] = None,
              alumina_t: Optional[float] = None, alumina_kgco2e_per_t: Optional[float] = None,
              pfc_kgco2e_per_t: Optional[float] = None, n: int = 5000) -> dict:
    P = factors()["process"]
    inv = Inventory("1 t primary aluminium")
    inv.process.append(ProcessEmission("Anode CO2, prebake (IPCC Tier 1)", P["aluminium_prebake_tco2_per_t"]["value"] * 1000, 4))
    inv.electricity.append(ElectricityUse(kwh_per_t, country, g_per_kwh, 2, "Smelter electricity"))
    if alumina_t and alumina_kgco2e_per_t is not None:
        inv.materials.append(Material("Alumina", alumina_t, alumina_kgco2e_per_t, "user-supplied", 3))
    else:
        inv.exclusions.append("Alumina refining upstream (supply t alumina/t Al and its factor to include)")
    if pfc_kgco2e_per_t is not None:
        inv.process.append(ProcessEmission("PFC (anode effects)", pfc_kgco2e_per_t, 3, capturable=False))
    else:
        inv.exclusions.append("PFC emissions from anode effects (supply measured kg CO2e/t to include)")
    return _result("aluminium", inv, country, "Tier 1 screening (IPCC 2006 Vol.3 Table 4.10); electricity dominates.", n)


def ammonia(kwh_per_t: float, country: Optional[str] = None, g_per_kwh: Optional[float] = None,
            gas_gj_per_t: Optional[float] = None, co2_recovered_for_urea_kg_per_t: float = 0.0, n: int = 5000) -> dict:
    P = factors()["process"]
    gj = gas_gj_per_t if gas_gj_per_t is not None else P["ammonia_gas_gj_per_t"]["value"]
    inv = Inventory("1 t ammonia")
    inv.fuels.append(FuelUse("natural_gas", gj, 2 if gas_gj_per_t is not None else 4))
    if co2_recovered_for_urea_kg_per_t:
        inv.process.append(ProcessEmission("CO2 recovered for urea (deducted)", -co2_recovered_for_urea_kg_per_t, 2,
                                           capturable=False))
        inv.notes.append("CO2 recovered for urea is deducted here and must be counted where the urea is used.")
    inv.electricity.append(ElectricityUse(kwh_per_t, country, g_per_kwh, 2, "Plant electricity"))
    return _result("ammonia", inv, country, "Natural gas route; IPCC 2006 Vol.3 Table 3.1 Tier 1 fuel requirement "
                   "when plant gas use is not given (fuel and feedstock CO2 combined).", n)


def custom(declared_unit: str, fuels: list, electricity: list, process: list, materials: list, transport: list,
           country: Optional[str] = None, capture_rate: float = 0.0, exclusions: Optional[list] = None, n: int = 5000) -> dict:
    inv = Inventory(declared_unit, capture_rate=capture_rate, exclusions=list(exclusions or []))
    for f in fuels:
        if f["fuel"] not in factors()["fuels"]:
            raise ValueError(f"unknown fuel '{f['fuel']}' — see GET /lca/factors")
        inv.fuels.append(FuelUse(f["fuel"], float(f["gj"]), int(f.get("dq", 2))))
    for e in electricity:
        inv.electricity.append(ElectricityUse(float(e["kwh"]), e.get("country", country), e.get("g_per_kwh"),
                                              int(e.get("dq", 2)), e.get("label", "Electricity")))
    for p in process:
        inv.process.append(ProcessEmission(p["name"], float(p["kg_co2e"]), int(p.get("dq", 2)), bool(p.get("capturable", True))))
    for m in materials:
        inv.materials.append(Material(m["name"], float(m["t"]), float(m["kgco2e_per_t"]), m.get("source", "user-supplied"),
                                      int(m.get("dq", 3))))
    for t in transport:
        if t["mode"] not in factors()["freight"]:
            raise ValueError("transport mode must be truck, rail or ship")
        inv.transport.append(TransportLeg(t["mode"], float(t["t"]), float(t["km"]), int(t.get("dq", 3)), t.get("label", "")))
    return _result("custom", inv, country, "Custom inventory supplied by the user.", n)
