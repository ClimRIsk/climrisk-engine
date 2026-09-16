"""
stranded_assets.py — IEA WEO-calibrated stranded asset NPV impairment model.

Computes the year-by-year NPV write-down for reserves, plants, or assets
that become uneconomic before their technical end-of-life under each NGFS
climate scenario.

Methodology
-----------
Step 1 — Unburnable carbon budget by scenario and sector:
    IEA WEO 2023 Net Zero by 2050 Annex A provides the remaining carbon
    budget (Gt CO2) for each fossil fuel type across scenarios. Combined
    with company reserve data, we compute the fraction stranded per year.

Step 2 — Asset stranding year:
    The year at which the carbon price under each scenario makes incremental
    production uneconomic = production cost / (1 − carbon_cost_fraction).
    For most oil assets in NZE: stranding begins 2032-2038.
    For coal: 2025-2030.

Step 3 — NPV impairment:
    Stranded fraction × carrying value, discounted from the stranding year
    to today at WACC.

Step 4 — Equity impact:
    NPV impairment as % of market cap (or EV proxy when market cap missing).

Covered sectors
---------------
- Oil and gas (upstream, midstream)
- Coal (thermal and metallurgical)
- Fossil-fuel-dependent utilities (gas-fired power)
- High-carbon industrial assets (blast furnace steel, wet-process cement)

Non-covered (returned as zero stranded, not applicable)
-------------------------------------------------------
- Renewables, technology, financials, agriculture

Sources
-------
IEA WEO 2023: Net Zero by 2050 Annex A (remaining fossil fuel demand)
IRENA 2023: Stranded assets and transition costs in the power sector
Carbon Tracker Initiative (2023): Unburnable carbon budgets by scenario
NGFS Phase 4 Phase-out schedules for coal-fired generation
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ── IEA WEO 2023 / NGFS Phase 4 — remaining demand fraction by year ─────────
# Fraction of 2023 production level remaining under each scenario.
# Source: IEA WEO 2023 Annex A — Fossil fuel demand by scenario.
# Format: {scenario: {fuel_type: {year: demand_fraction}}}
_REMAINING_DEMAND: dict[str, dict[str, dict[int, float]]] = {
    "nze": {
        "oil":          {2025: 0.95, 2030: 0.75, 2035: 0.55, 2040: 0.35, 2045: 0.18, 2050: 0.05},
        "gas":          {2025: 0.97, 2030: 0.80, 2035: 0.60, 2040: 0.38, 2045: 0.20, 2050: 0.08},
        "coal_thermal": {2025: 0.90, 2030: 0.55, 2035: 0.25, 2040: 0.08, 2045: 0.02, 2050: 0.00},
        "coal_met":     {2025: 0.95, 2030: 0.80, 2035: 0.60, 2040: 0.40, 2045: 0.22, 2050: 0.10},
        "gas_power":    {2025: 0.95, 2030: 0.65, 2035: 0.35, 2040: 0.12, 2045: 0.04, 2050: 0.01},
    },
    "delayed": {
        "oil":          {2025: 0.98, 2030: 0.92, 2035: 0.75, 2040: 0.55, 2045: 0.35, 2050: 0.20},
        "gas":          {2025: 0.99, 2030: 0.95, 2035: 0.80, 2040: 0.60, 2045: 0.40, 2050: 0.22},
        "coal_thermal": {2025: 0.95, 2030: 0.80, 2035: 0.55, 2040: 0.28, 2045: 0.10, 2050: 0.04},
        "coal_met":     {2025: 0.98, 2030: 0.90, 2035: 0.75, 2040: 0.58, 2045: 0.40, 2050: 0.25},
        "gas_power":    {2025: 0.98, 2030: 0.88, 2035: 0.68, 2040: 0.42, 2045: 0.22, 2050: 0.10},
    },
    "current_policies": {
        "oil":          {2025: 1.00, 2030: 1.02, 2035: 1.04, 2040: 1.03, 2045: 1.00, 2050: 0.96},
        "gas":          {2025: 1.00, 2030: 1.05, 2035: 1.10, 2040: 1.12, 2045: 1.10, 2050: 1.05},
        "coal_thermal": {2025: 1.00, 2030: 1.02, 2035: 1.01, 2040: 0.98, 2045: 0.92, 2050: 0.85},
        "coal_met":     {2025: 1.00, 2030: 1.02, 2035: 1.03, 2040: 1.02, 2045: 1.00, 2050: 0.98},
        "gas_power":    {2025: 1.00, 2030: 1.05, 2035: 1.08, 2040: 1.05, 2045: 1.00, 2050: 0.92},
    },
}

# ── Sector → fuel type mapping ───────────────────────────────────────────────
_SECTOR_FUEL_TYPE: dict[str, str] = {
    "oil_gas":       "oil",
    "oil":           "oil",
    "gas":           "gas",
    "lng":           "gas",
    "coal":          "coal_thermal",
    "coal_thermal":  "coal_thermal",
    "coal_metallurgical": "coal_met",
    "utilities":     "gas_power",
    "power":         "gas_power",
    "steel":         "coal_met",    # blast furnace steel is coal-dependent
    "cement":        "coal_thermal", # wet-process cement kilns use coal
}

# High-carbon sectors that have stranded asset exposure
_STRANDED_SECTORS = set(_SECTOR_FUEL_TYPE.keys())

# ── Production cost thresholds (USD/BOE or USD/tonne equivalent) ─────────────
# Below-the-line cost that determines when production becomes uneconomic.
# Source: Rystad UCube 2023 p10 production costs by asset class.
_BREAK_EVEN_COST: dict[str, float] = {
    "oil":          40.0,   # USD/bbl (deepwater ~$50, shale ~$35-45)
    "gas":          4.5,    # USD/MMBtu
    "coal_thermal": 45.0,   # USD/tonne (thermal)
    "coal_met":     80.0,   # USD/tonne (metallurgical — higher value)
    "gas_power":    55.0,   # USD/MWh LCOE equivalent
}

# ── Carbon cost per unit of production (tCO2e per BOE/tonne/MWh) ─────────────
# Used with carbon price to compute carbon cost burden per unit.
_CARBON_INTENSITY_PER_UNIT: dict[str, float] = {
    "oil":          0.43,   # tCO2e/bbl (IPCC AR6 WG3 Ch6 — lifecycle)
    "gas":          0.06,   # tCO2e/MMBtu
    "coal_thermal": 2.42,   # tCO2e/tonne (thermal coal combustion)
    "coal_met":     1.80,   # tCO2e/tonne (metallurgical coal)
    "gas_power":    0.43,   # tCO2e/MWh (gas CCGT, IPCC AR6)
}

# ── NGFS Phase 4 carbon price (USD/tCO2e) by scenario ────────────────────────
_NGFS_CARBON_PRICE: dict[str, dict[int, float]] = {
    "nze":     {2025: 75,  2030: 145, 2035: 265, 2040: 480, 2045: 680, 2050: 720},
    "delayed": {2025: 30,  2030: 80,  2035: 160, 2040: 280, 2045: 420, 2050: 580},
    "current_policies": {2025: 20, 2030: 40, 2035: 65, 2040: 95, 2045: 130, 2050: 170},
}


def _interp(schedule: dict[int, float], year: int) -> float:
    keys = sorted(schedule.keys())
    if year <= keys[0]:  return schedule[keys[0]]
    if year >= keys[-1]: return schedule[keys[-1]]
    for i in range(len(keys) - 1):
        y0, y1 = keys[i], keys[i+1]
        if y0 <= year <= y1:
            t = (year - y0) / (y1 - y0)
            return schedule[y0] + t * (schedule[y1] - schedule[y0])
    return schedule[keys[-1]]


def _demand_fraction(scenario: str, fuel_type: str, year: int) -> float:
    scen = _REMAINING_DEMAND.get(scenario, _REMAINING_DEMAND["current_policies"])
    fuel = scen.get(fuel_type, None)
    if fuel is None:
        return 1.0  # no stranding data → assume no stranding
    return _interp(fuel, year)


def _stranding_year(
    scenario: str,
    fuel_type: str,
    break_even_cost: float,
    carbon_price_coverage: float = 1.0,
) -> Optional[int]:
    """
    Find the first year where carbon-inclusive cost > break-even.
    Returns None if the asset stays economic through 2050 under this scenario.
    """
    intensity = _CARBON_INTENSITY_PER_UNIT.get(fuel_type, 0.5)
    for year in range(2025, 2051):
        cp     = _interp(_NGFS_CARBON_PRICE.get(scenario, {}), year)
        carbon_cost_per_unit = cp * intensity * carbon_price_coverage
        if carbon_cost_per_unit >= break_even_cost:
            return year
    return None


@dataclass
class StrandedAssetResult:
    """Stranded asset analysis output for one company across three scenarios."""
    company_name:          str
    sector:                str
    fuel_type:             Optional[str]        = None
    is_applicable:         bool                 = False   # False for non-fossil sectors
    scenarios:             dict[str, dict]      = field(default_factory=dict)
    methodology:           str                  = ""
    data_gaps:             list[str]            = field(default_factory=list)


def assess_stranded_assets(
    company_name:        str,
    sector:              str,
    carrying_value_usd_m: float,
    revenue_usd_m:       float,
    ev_usd_m:            Optional[float] = None,
    market_cap_usd_m:    Optional[float] = None,
    wacc:                float = 0.09,
    carbon_price_coverage: float = 1.0,  # fraction of emissions under carbon price
    free_allocation_pct:   float = 0.0,  # EU ETS free allowance fraction
) -> StrandedAssetResult:
    """
    Assess stranded asset risk for a company.

    Parameters
    ----------
    company_name          : Display name for output
    sector                : Sector slug (oil_gas, coal, utilities, steel, cement, etc.)
    carrying_value_usd_m  : Net book value of PP&E / reserves (USD M)
    revenue_usd_m         : Annual revenue (USD M)
    ev_usd_m              : Enterprise value for equity impact normalisation
    market_cap_usd_m      : Market cap for equity impact
    wacc                  : Discount rate for NPV impairment
    carbon_price_coverage : Fraction of Scope 1 covered by carbon price (0-1)
    free_allocation_pct   : EU ETS style free allowance as fraction of covered emissions
    """
    result = StrandedAssetResult(
        company_name=company_name,
        sector=sector,
    )

    fuel_type = _SECTOR_FUEL_TYPE.get(sector.lower())
    if fuel_type is None:
        result.is_applicable = False
        result.methodology = (
            f"Sector '{sector}' has no fossil fuel asset exposure — "
            "stranded asset risk is not applicable."
        )
        return result

    result.is_applicable = True
    result.fuel_type      = fuel_type
    eff_coverage = carbon_price_coverage * (1.0 - free_allocation_pct)
    break_even   = _BREAK_EVEN_COST.get(fuel_type, 50.0)
    normaliser   = ev_usd_m or (market_cap_usd_m * 1.3 if market_cap_usd_m else revenue_usd_m * 2.0)

    for scenario_key in ("nze", "delayed", "current_policies"):
        strand_year = _stranding_year(scenario_key, fuel_type, break_even, eff_coverage)

        # Build year-by-year stranded fraction and NPV impairment
        cumulative_strand_frac = 0.0
        impairment_npv         = 0.0
        yearly: list[dict] = []

        for year in range(2025, 2051):
            demand_frac = _demand_fraction(scenario_key, fuel_type, year)
            prev_demand = _demand_fraction(scenario_key, fuel_type, year - 1)
            # Incremental demand loss this year
            delta_strand = max(0.0, prev_demand - demand_frac)
            cumulative_strand_frac = min(1.0, cumulative_strand_frac + delta_strand)

            # Only count new stranding this year
            yearly_impairment = carrying_value_usd_m * delta_strand
            t = year - 2025
            discounted = yearly_impairment / ((1 + wacc) ** t)
            impairment_npv += discounted

            yearly.append({
                "year":                  year,
                "demand_fraction":       round(demand_frac, 3),
                "incremental_strand_pct": round(delta_strand * 100, 2),
                "cumulative_strand_pct":  round(cumulative_strand_frac * 100, 2),
                "annual_impairment_usd_m": round(yearly_impairment, 1),
                "discounted_impairment_usd_m": round(discounted, 1),
                "carbon_price_usd":      round(_interp(_NGFS_CARBON_PRICE.get(scenario_key, {}), year), 0),
            })

        strand_pct_ev    = (impairment_npv / normaliser * 100) if normaliser > 0 else 0.0
        strand_pct_carry = (impairment_npv / carrying_value_usd_m * 100) if carrying_value_usd_m > 0 else 0.0

        result.scenarios[scenario_key] = {
            "stranding_year":          strand_year,
            "npv_impairment_usd_m":    round(impairment_npv, 1),
            "impairment_pct_ev":       round(strand_pct_ev, 2),
            "impairment_pct_carrying_value": round(strand_pct_carry, 2),
            "years":                   yearly,
            "fuel_type":               fuel_type,
            "break_even_cost":         break_even,
            "carbon_price_coverage":   eff_coverage,
        }

    result.methodology = (
        f"IEA WEO 2023 Annex A fuel demand paths ({fuel_type}); "
        "stranding year = first year carbon-inclusive cost exceeds break-even "
        f"(break-even ${break_even}/unit, IPCC AR6 carbon intensity "
        f"{_CARBON_INTENSITY_PER_UNIT.get(fuel_type, 0.5):.2f} tCO2e/unit). "
        "NPV impairment = annual stranded fraction × carrying value, discounted at WACC. "
        "Carbon Tracker Initiative (2023) methodology reference."
    )
    return result
