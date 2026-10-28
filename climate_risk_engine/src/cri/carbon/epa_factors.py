"""
US EPA GHG Emission Factors Hub (15 Jan 2025 edition): fuel combustion, electricity (eGRID subregions), purchased steam, scope 3 transport and travel.

Every result carries the table it came from. Factors are combustion only; upstream (well-to-tank) emissions are not in this source and are not added. GWPs are the AR5 values the Hub prints (CH4 28, N2O 265).
Units are the Hub's own (short ton, scf, gallon, mmBtu, lb/MWh); convert before calling, or use `amount_unit` where a fuel has a heat content.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

_PATH = Path(__file__).resolve().parents[1] / "data" / "epa_ghg_hub_2025.json"
LB_PER_KG = 2.2046226218
MWH_PER_KWH = 1e-3


class EpaFactorError(ValueError):
    pass


@lru_cache(maxsize=1)
def lib() -> dict:
    return json.loads(_PATH.read_text(encoding="utf-8"))


def _co2e(co2_kg: float, ch4_g: float, n2o_g: float) -> float:
    g = lib()["gwp100_ar5"]
    return co2_kg + ch4_g / 1000.0 * g["CH4"] + n2o_g / 1000.0 * g["N2O"]


def _src(table: str) -> dict:
    return {"source": lib()["source"], "table": table, "retrieved": lib()["retrieved"], "combustion_only": True}


def stationary(fuel: str, amount: float, unit: Optional[str] = None, hhv_mmbtu_per_unit: Optional[float] = None) -> dict:
    """Stationary combustion (Table 1). amount in the fuel's Hub unit (short ton, scf or gallon) or in mmBtu with unit='mmBtu'. A supplier heat content overrides the default."""
    T = lib()["table1_stationary"]
    if fuel not in T:
        raise EpaFactorError(f"unknown fuel {fuel!r}; choose from: {', '.join(sorted(T))}")
    if amount < 0:
        raise EpaFactorError("amount must be >= 0")
    f = T[fuel]
    hhv = hhv_mmbtu_per_unit if hhv_mmbtu_per_unit is not None else f["hhv_mmbtu_per_unit"]
    if unit in (None, f["unit"]):
        mmbtu, used = amount * hhv, f["unit"]
    elif unit.lower() == "mmbtu":
        mmbtu, used = amount, "mmBtu"
    else:
        raise EpaFactorError(f"{fuel}: unit must be {f['unit']!r} or 'mmBtu'")
    co2, ch4, n2o = mmbtu * f["kgco2_per_mmbtu"], mmbtu * f["gch4_per_mmbtu"], mmbtu * f["gn2o_per_mmbtu"]
    biogenic = f["category"].startswith("biomass")
    return {"fuel": fuel, "amount": amount, "unit": used, "heat_mmbtu": mmbtu, "heat_content_source": "supplier value given" if hhv_mmbtu_per_unit is not None else "Hub default HHV",
            "kg_co2": co2, "g_ch4": ch4, "g_n2o": n2o, "kg_co2e": _co2e(co2, ch4, n2o),
            "biogenic": biogenic, "note": ("biogenic fuel: report biogenic CO2 separately from the inventory total per GHG Protocol; kg_co2e includes it here" if biogenic else None),
            **_src("Table 1 Stationary Combustion")}


def mobile_co2(fuel: str, amount: float) -> dict:
    """Mobile combustion CO2 only (Table 2). CH4 and N2O depend on vehicle type and year (Hub Tables 3-5, not loaded): they are NOT included, and the result says so."""
    T = lib()["table2_mobile_co2"]
    if fuel not in T:
        raise EpaFactorError(f"unknown fuel {fuel!r}; choose from: {', '.join(sorted(T))}")
    f = T[fuel]
    return {"fuel": fuel, "amount": amount, "unit": f["unit"], "kg_co2": amount * f["kgco2_per_unit"], "kg_co2e": amount * f["kgco2_per_unit"],
            "excluded": "CH4 and N2O (vehicle-type and model-year tables not loaded), so kg_co2e is CO2 only and slightly low", **_src("Table 2 Mobile Combustion CO2")}


def electricity(subregion: str, mwh: float) -> dict:
    """Location-based electricity (Table 6, total output rates, eGRID2023)."""
    T = lib()["table6_egrid_total_output"]
    k = subregion.upper() if subregion.upper() in T else subregion
    if k not in T:
        raise EpaFactorError(f"unknown eGRID subregion {subregion!r}; choose from: {', '.join(sorted(T))}")
    f = T[k]
    co2, ch4, n2o = (mwh * f[x] / LB_PER_KG for x in ("lb_co2_per_mwh", "lb_ch4_per_mwh", "lb_n2o_per_mwh"))
    return {"subregion": k, "name": f["name"], "mwh": mwh, "kg_co2": co2, "kg_ch4": ch4, "kg_n2o": n2o, "kg_co2e": co2 + ch4 * lib()["gwp100_ar5"]["CH4"] + n2o * lib()["gwp100_ar5"]["N2O"],
            "grid_gross_loss_pct": f["grid_gross_loss_pct"],
            "note": "Grid gross loss is for scope 3 category 3 only; do not apply it to scope 2. Use total output rates for inventories, not non-baseload.", **_src("Table 6 Electricity (eGRID2023)")}


def steam_heat(mmbtu: float) -> dict:
    f = lib()["table7_steam_heat"]
    co2, ch4, n2o = mmbtu * f["kgco2_per_mmbtu"], mmbtu * f["gch4_per_mmbtu"], mmbtu * f["gn2o_per_mmbtu"]
    return {"mmbtu": mmbtu, "kg_co2": co2, "kg_co2e": _co2e(co2, ch4, n2o), "note": f["note"], **_src("Table 7 Steam and Heat")}


def travel_or_transport(kind: str, quantity: float, table: str = "table10_scope3_travel") -> dict:
    """Scope 3 distance-based factors (Table 8 freight/vehicles, Table 10 travel). quantity in the factor's unit."""
    if table not in ("table8_scope3_transport", "table10_scope3_travel"):
        raise EpaFactorError("table must be table8_scope3_transport or table10_scope3_travel")
    T = lib()[table]
    if kind not in T:
        raise EpaFactorError(f"unknown entry {kind!r}; choose from: {', '.join(sorted(T))}")
    f = T[kind]
    co2, ch4, n2o = quantity * f["kgco2_per_unit"], quantity * f["gch4_per_unit"], quantity * f["gn2o_per_unit"]
    return {"entry": kind, "quantity": quantity, "unit": f["unit"], "kg_co2": co2, "kg_co2e": _co2e(co2, ch4, n2o), **_src(table)}
