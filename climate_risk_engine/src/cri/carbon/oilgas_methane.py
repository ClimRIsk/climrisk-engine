"""
Oil and gas methane from equipment counts: US EPA 40 CFR 98 Subpart W Table W-1 default population emission factors.

  whole gas (scf)  = factor (scf/hour/unit) x units x hours
  CH4 (kg)         = whole gas x CH4 mole fraction x molar volume factor: ideal gas at 60 degF, 14.7 psia, MW 16.043 -> ~0.0192 kg per scf of pure CH4
  CO2e (kg)        = CH4 x GWP (AR5 28 from the EPA Hub, or your own, stated)

The CH4 mole fraction is YOUR input (gas analysis): the Subpart W default fractions were not in the text retrieved, so none is assumed.
These are reporting defaults for equipment populations. Real emissions are heavy-tailed (a few super-emitters) and surveys routinely differ from inventories by large factors: treat as a floor-grade screening estimate, and replace with measurement (OGI/aerial/satellite) where available.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

from . import epa_factors as EF

_P = Path(__file__).resolve().parents[1] / "data" / "subpart_w_table_w1.json"
R_J_PER_MOL_K = 8.314462618
T_K = (60.0 - 32.0) * 5.0 / 9.0 + 273.15            # 60 degF
P_PA = 14.7 * 6894.757293168                          # 14.7 psia
MW_CH4_G_PER_MOL = 16.043
FT3_PER_M3 = 35.3146667


class MethaneError(ValueError):
    pass


@lru_cache(maxsize=1)
def lib() -> dict:
    return json.loads(_P.read_text(encoding="utf-8"))


def kg_ch4_per_scf_pure() -> float:
    """Mass of one standard cubic foot of pure CH4 from the ideal gas law at 60 degF and 14.7 psia."""
    vm_m3_per_mol = R_J_PER_MOL_K * T_K / P_PA
    return MW_CH4_G_PER_MOL / 1000.0 / (vm_m3_per_mol * FT3_PER_M3)


def components() -> list:
    return [f'{r["group"]}: {r["component"]}' for r in lib()["rows"]]


def population(group: str, component: str, units: float, hours: float, ch4_mole_fraction: float, gwp_ch4: Optional[float] = None) -> dict:
    if units < 0 or hours < 0:
        raise MethaneError("units and hours must be >= 0")
    if not 0 < ch4_mole_fraction <= 1:
        raise MethaneError("give the CH4 mole fraction of the gas (0-1) from a gas analysis; no default is assumed")
    row = next((r for r in lib()["rows"] if r["group"] == group and r["component"].lower() == component.lower()), None)
    if row is None:
        raise MethaneError(f"unknown entry; use group and component from: {'; '.join(components())}")
    scf = row["scf_whole_gas_per_hour_per_unit"] * units * hours
    ch4_kg = scf * ch4_mole_fraction * kg_ch4_per_scf_pure()
    g = gwp_ch4 if gwp_ch4 is not None else EF.lib()["gwp100_ar5"]["CH4"]
    return {"group": group, "component": row["component"], "units": units, "hours": hours, "factor_scf_per_hour_per_unit": row["scf_whole_gas_per_hour_per_unit"], "whole_gas_scf": scf,
            "ch4_mole_fraction": ch4_mole_fraction, "ch4_kg": ch4_kg, "gwp_ch4": g, "gwp_basis": "given by you" if gwp_ch4 is not None else "IPCC AR5 via the EPA Hub", "co2e_kg": ch4_kg * g,
            "source": lib()["source"], "standard_conditions": lib()["standard_conditions"],
            "caveat": "Regulatory default for an equipment population, not a measurement; super-emitters and venting are not captured. Only the components you list are counted."}


def facility(items: list, ch4_mole_fraction: float, gwp_ch4: Optional[float] = None) -> dict:
    """items: [{group, component, units, hours}]. Sums CH4 and CO2e; lists what was not counted."""
    if not items:
        raise MethaneError("give at least one component")
    rows = [population(i["group"], i["component"], i["units"], i["hours"], ch4_mole_fraction, gwp_ch4) for i in items]
    return {"lines": rows, "ch4_kg": sum(r["ch4_kg"] for r in rows), "co2e_kg": sum(r["co2e_kg"] for r in rows),
            "not_counted": ["every component you did not list", "flaring and venting from tanks, unloading, blowdowns, completions", "leaker factors (Table W-2) and measured leaks", "combustion CO2 (use the stationary-combustion factors)"],
            "source": lib()["source"]}
