"""
CRI Data Requirements Registry.

Defines exactly what a client firm must provide, what we can derive
from their data, and what we enrich from open-source data.

This is the authoritative source for the intake wizard, API docs,
and the Excel template generator.

Three tiers of fields
---------------------
REQUIRED    — analysis cannot run without this; hard error if missing
RECOMMENDED — improves accuracy; sector defaults used if missing; warning issued
ENRICHED    — we fetch/compute from open sources (WRI/NGFS/NASA/IEA); never required

Data that firms typically hold in:
  - Annual Report / 10-K / 20-F      → Financials, share data
  - Asset Register / ERP system       → Asset list, locations, carrying values
  - Sustainability Report / GHG Inv.  → Scope 1/2/3 emissions
  - Capital Planning deck             → Transition goals, capex plans
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class RequirementLevel(str, Enum):
    REQUIRED     = "required"
    RECOMMENDED  = "recommended"
    ENRICHED     = "enriched"   # we get this from open sources


@dataclass
class FieldSpec:
    name: str
    level: RequirementLevel
    description: str
    data_source: str            # where the firm finds this
    unit: str = ""
    example: Any = None
    default: Any = None         # used if RECOMMENDED and missing
    validation: str = ""        # human-readable rule


# ---------------------------------------------------------------------------
# Company-level fields
# ---------------------------------------------------------------------------

COMPANY_FIELDS: list[FieldSpec] = [
    FieldSpec(
        name="company_name",
        level=RequirementLevel.REQUIRED,
        description="Legal name of the entity being assessed",
        data_source="Company registration / Annual Report cover",
        example="Acme Mining Limited",
    ),
    FieldSpec(
        name="sector",
        level=RequirementLevel.REQUIRED,
        description="Primary industry sector",
        data_source="Company classification (GICS / ICB)",
        example="Mining",
        validation="One of: Mining, Oil & Gas, Utilities, Steel & Metals, Cement, Other",
    ),
    FieldSpec(
        name="hq_region",
        level=RequirementLevel.REQUIRED,
        description="Headquarters country/region (ISO region code)",
        data_source="Company registration / Annual Report",
        example="AU-WA",
        validation="CRI region code (see region reference list)",
    ),

    # Financials — from Annual Report
    FieldSpec(
        name="revenue_musd",
        level=RequirementLevel.REQUIRED,
        description="Total revenue, most recent full fiscal year (USD millions)",
        data_source="Annual Report / 10-K Income Statement",
        unit="USD M",
        example=5_200.0,
        validation="Must be > 0",
    ),
    FieldSpec(
        name="ebitda_musd",
        level=RequirementLevel.REQUIRED,
        description="EBITDA, most recent full fiscal year (USD millions)",
        data_source="Annual Report / 10-K Income Statement",
        unit="USD M",
        example=2_100.0,
        validation="Can be negative for early-stage companies",
    ),
    FieldSpec(
        name="capex_musd",
        level=RequirementLevel.REQUIRED,
        description="Total capital expenditure, most recent fiscal year (USD millions)",
        data_source="Annual Report / 10-K Cash Flow Statement",
        unit="USD M",
        example=850.0,
        validation="Must be ≥ 0",
    ),
    FieldSpec(
        name="net_debt_musd",
        level=RequirementLevel.REQUIRED,
        description="Net debt (total debt minus cash & equivalents) (USD millions)",
        data_source="Annual Report / 10-K Balance Sheet",
        unit="USD M",
        example=1_200.0,
        validation="Can be negative (net cash position)",
    ),
    FieldSpec(
        name="shares_outstanding_m",
        level=RequirementLevel.REQUIRED,
        description="Total shares outstanding (millions)",
        data_source="Annual Report / stock exchange filing",
        unit="M shares",
        example=2_400.0,
        validation="Must be > 0",
    ),
    FieldSpec(
        name="current_share_price_usd",
        level=RequirementLevel.RECOMMENDED,
        description="Current (or year-end) share price in USD",
        data_source="Stock exchange / Bloomberg / Reuters",
        unit="USD",
        example=45.20,
        default=None,   # used to compute implied market cap for validation
    ),
    FieldSpec(
        name="wacc_base_pct",
        level=RequirementLevel.RECOMMENDED,
        description="Company's internal WACC estimate (pre-climate premium) (%)",
        data_source="Treasury / capital planning team / investor relations",
        unit="%",
        example=8.0,
        default=8.0,   # sector default if not provided
        validation="Typically 6–12% for listed industrials",
    ),
    FieldSpec(
        name="tax_rate_pct",
        level=RequirementLevel.RECOMMENDED,
        description="Effective corporate tax rate (%)",
        data_source="Annual Report / 10-K tax note",
        unit="%",
        example=25.0,
        default=25.0,
    ),
    FieldSpec(
        name="maintenance_capex_share",
        level=RequirementLevel.RECOMMENDED,
        description="Fraction of total capex that is maintenance (not growth) (0–1)",
        data_source="Capital planning / investor presentations",
        example=0.60,
        default=0.60,
    ),
]


# ---------------------------------------------------------------------------
# Asset-level fields (one record per asset)
# ---------------------------------------------------------------------------

ASSET_FIELDS: list[FieldSpec] = [
    FieldSpec(
        name="asset_name",
        level=RequirementLevel.REQUIRED,
        description="Asset name as known internally (mine name, refinery, plant)",
        data_source="Asset register / ERP system",
        example="Pilbara Iron Ore Operations",
    ),
    FieldSpec(
        name="commodity",
        level=RequirementLevel.REQUIRED,
        description="Primary commodity produced by this asset",
        data_source="Asset register",
        example="iron_ore",
        validation="One of: iron_ore, copper, aluminium, coal_thermal, coal_metallurgical, "
                   "crude_oil, natural_gas, refined_products, cement, electricity",
    ),
    FieldSpec(
        name="latitude",
        level=RequirementLevel.REQUIRED,
        description="Asset latitude in decimal degrees (for physical hazard lookup)",
        data_source="Operations team / GIS system / Google Maps",
        unit="decimal degrees",
        example=-22.5,
        validation="-90 to +90",
    ),
    FieldSpec(
        name="longitude",
        level=RequirementLevel.REQUIRED,
        description="Asset longitude in decimal degrees",
        data_source="Operations team / GIS system / Google Maps",
        unit="decimal degrees",
        example=118.8,
        validation="-180 to +180",
    ),
    FieldSpec(
        name="baseline_production",
        level=RequirementLevel.REQUIRED,
        description="Annual production volume at current baseline year",
        data_source="Operations report / Sustainability Report",
        example=280.0,
        validation="Must be > 0",
    ),
    FieldSpec(
        name="production_unit",
        level=RequirementLevel.REQUIRED,
        description="Unit of production measurement",
        data_source="Operations team",
        example="Mt",
        validation="One of: Mt (megatonnes), kt, MMboe, bbl/d, GWh, t",
    ),
    FieldSpec(
        name="carrying_value_musd",
        level=RequirementLevel.RECOMMENDED,
        description="Book value of the asset on the balance sheet (USD millions)",
        data_source="Annual Report / Balance Sheet / Asset Register",
        unit="USD M",
        example=3_500.0,
        default=0.0,    # stranded-asset writedown calculation won't run without it
        validation="Must be ≥ 0",
    ),
    FieldSpec(
        name="remaining_life_years",
        level=RequirementLevel.RECOMMENDED,
        description="Remaining operational life of the asset (years from today)",
        data_source="Mine plan / operations team / reserve statement",
        unit="years",
        example=25,
        default=None,   # None means we assume operates through horizon
        validation="Must be > 0 if provided",
    ),
    FieldSpec(
        name="baseline_unit_cost_usd",
        level=RequirementLevel.RECOMMENDED,
        description="All-in sustaining cost per production unit (USD per unit)",
        data_source="Operations report / investor presentation",
        unit="USD per production unit",
        example=18.5,
        default=None,   # derived from ebitda_margin if missing
        validation="Must be ≥ 0",
    ),
    FieldSpec(
        name="energy_cost_share",
        level=RequirementLevel.RECOMMENDED,
        description="Fraction of unit operating cost attributable to energy (0–1)",
        data_source="Site-level cost breakdown / procurement team",
        example=0.30,
        default=0.30,   # sector average default
        validation="0 to 1; typically 0.15–0.50 for mining",
    ),

    # Emissions — from GHG Inventory / Sustainability Report
    FieldSpec(
        name="scope1_intensity",
        level=RequirementLevel.RECOMMENDED,
        description="Direct GHG emissions per unit of production (tCO₂e per production unit)",
        data_source="GHG Inventory / Sustainability Report / CDP submission",
        unit="tCO₂e per unit",
        example=0.025,
        default=None,   # sector benchmark used if missing
        validation="Must be ≥ 0",
    ),
    FieldSpec(
        name="scope2_intensity",
        level=RequirementLevel.RECOMMENDED,
        description="Purchased electricity/heat GHG per unit of production (tCO₂e per unit)",
        data_source="GHG Inventory / Sustainability Report",
        unit="tCO₂e per unit",
        example=0.015,
        default=None,
    ),
    FieldSpec(
        name="scope3_intensity",
        level=RequirementLevel.RECOMMENDED,
        description="Value-chain GHG per unit (tCO₂e per unit) — mainly product use-phase",
        data_source="GHG Inventory / industry average",
        unit="tCO₂e per unit",
        example=0.38,
        default=None,
        validation="Often 5–20× Scope 1 for fossil fuels",
    ),
    FieldSpec(
        name="carbon_price_coverage",
        level=RequirementLevel.RECOMMENDED,
        description="Fraction of Scope 1+2 emissions subject to a carbon price (0–1)",
        data_source="Legal / compliance / treasury team",
        example=0.40,
        default=0.30,   # 30% global average coverage
    ),
    FieldSpec(
        name="free_allocation_share",
        level=RequirementLevel.RECOMMENDED,
        description="Fraction of priced emissions covered by free allowances (0–1)",
        data_source="Regulator / compliance team (EU ETS, AUS SAFIS, etc.)",
        example=0.15,
        default=0.0,
    ),
]


# ---------------------------------------------------------------------------
# Transition goal fields (company-level)
# ---------------------------------------------------------------------------

TRANSITION_FIELDS: list[FieldSpec] = [
    FieldSpec(
        name="has_netzero_commitment",
        level=RequirementLevel.RECOMMENDED,
        description="Does the company have a public net-zero commitment?",
        data_source="Sustainability Report / company website",
        example=True,
        default=False,
    ),
    FieldSpec(
        name="netzero_target_year",
        level=RequirementLevel.RECOMMENDED,
        description="Target year for net-zero Scope 1+2 (e.g., 2050)",
        data_source="Sustainability Report",
        example=2050,
        default=None,
    ),
    FieldSpec(
        name="interim_target_2030_pct",
        level=RequirementLevel.RECOMMENDED,
        description="% reduction in Scope 1+2 by 2030 vs base year",
        data_source="Sustainability Report / CDP submission",
        unit="%",
        example=30.0,
        default=None,
    ),
    FieldSpec(
        name="uses_internal_carbon_price",
        level=RequirementLevel.RECOMMENDED,
        description="Does the company use an internal shadow carbon price for investment decisions?",
        data_source="Treasury / strategy team",
        example=True,
        default=False,
    ),
    FieldSpec(
        name="internal_carbon_price_usd",
        level=RequirementLevel.RECOMMENDED,
        description="Shadow carbon price used internally for investment appraisal (USD/tCO₂e)",
        data_source="Treasury / strategy team",
        unit="USD/tCO₂e",
        example=80.0,
        default=None,
    ),
]


# ---------------------------------------------------------------------------
# Enriched fields — we compute from open sources, never required
# ---------------------------------------------------------------------------

ENRICHED_FIELDS: list[FieldSpec] = [
    FieldSpec(
        name="water_stress_score",
        level=RequirementLevel.ENRICHED,
        description="WRI Aqueduct 4.0 water stress score for asset location (0–5)",
        data_source="WRI Aqueduct API (by lat/lon)",
    ),
    FieldSpec(
        name="flood_risk_score",
        level=RequirementLevel.ENRICHED,
        description="WRI Aqueduct 4.0 coastal + riverine flood risk score (0–5)",
        data_source="WRI Aqueduct API (by lat/lon)",
    ),
    FieldSpec(
        name="heat_stress_delta_c",
        level=RequirementLevel.ENRICHED,
        description="IPCC AR6 / NASA NEX-GDDP projected warming by 2050 (°C above baseline)",
        data_source="NASA NEX-GDDP CMIP6 / IPCC AR6 regional proxies",
    ),
    FieldSpec(
        name="carbon_price_path",
        level=RequirementLevel.ENRICHED,
        description="Annual carbon price path 2026–2050 by scenario (USD/tCO₂e)",
        data_source="NGFS Phase 4 (2023) — embedded lookup table",
    ),
    FieldSpec(
        name="commodity_demand_index",
        level=RequirementLevel.ENRICHED,
        description="Annual commodity demand index 2026–2050 (2025=100) by scenario",
        data_source="IEA WEO 2023 / Our World in Data",
    ),
    FieldSpec(
        name="sector_emission_intensity",
        level=RequirementLevel.ENRICHED,
        description="Sector-average Scope 1/2 intensity used when firm data not provided",
        data_source="IEA GHG / IPCC AR6 WG3 sector benchmarks",
    ),
    FieldSpec(
        name="wacc_sector_benchmark",
        level=RequirementLevel.ENRICHED,
        description="Sector-average WACC benchmark used when firm WACC not provided",
        data_source="Damodaran (NYU) sector cost of capital estimates",
    ),
]


# ---------------------------------------------------------------------------
# Sector defaults for emission intensities (tCO2e per unit)
# ---------------------------------------------------------------------------

SECTOR_EMISSION_DEFAULTS: dict[str, dict] = {
    # commodity → {scope1, scope2, scope3} tCO2e / tonne (or bbl/MWh as noted)
    "iron_ore":            {"scope1": 0.022, "scope2": 0.012, "scope3": 0.005},
    "copper":              {"scope1": 0.30,  "scope2": 0.25,  "scope3": 0.10},
    "aluminium":           {"scope1": 0.55,  "scope2": 8.50,  "scope3": 0.20},
    "coal_thermal":        {"scope1": 0.015, "scope2": 0.008, "scope3": 2.40},
    "coal_metallurgical":  {"scope1": 0.018, "scope2": 0.010, "scope3": 2.30},
    "crude_oil":           {"scope1": 0.045, "scope2": 0.012, "scope3": 0.43},  # tCO2e/bbl
    "natural_gas":         {"scope1": 0.020, "scope2": 0.005, "scope3": 0.19},  # tCO2e/MMbtu
    "refined_products":    {"scope1": 0.060, "scope2": 0.015, "scope3": 0.35},
    "cement":              {"scope1": 0.75,  "scope2": 0.10,  "scope3": 0.05},
    "electricity":         {"scope1": 0.45,  "scope2": 0.00,  "scope3": 0.02},  # tCO2e/MWh
}

SECTOR_WACC_DEFAULTS: dict[str, float] = {
    "Mining":       0.085,
    "Oil & Gas":    0.090,
    "Utilities":    0.065,
    "Steel":        0.080,
    "Cement":       0.080,
    "Other":        0.080,
}


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def required_fields() -> list[FieldSpec]:
    all_fields = COMPANY_FIELDS + ASSET_FIELDS + TRANSITION_FIELDS
    return [f for f in all_fields if f.level == RequirementLevel.REQUIRED]


def recommended_fields() -> list[FieldSpec]:
    all_fields = COMPANY_FIELDS + ASSET_FIELDS + TRANSITION_FIELDS
    return [f for f in all_fields if f.level == RequirementLevel.RECOMMENDED]


def enriched_fields() -> list[FieldSpec]:
    return ENRICHED_FIELDS


def intake_summary() -> dict:
    """Return a human-readable summary of data requirements."""
    req = required_fields()
    rec = recommended_fields()
    enr = enriched_fields()
    return {
        "required": {
            "count": len(req),
            "fields": [f.name for f in req],
            "sources": list({f.data_source for f in req}),
        },
        "recommended": {
            "count": len(rec),
            "fields": [f.name for f in rec],
            "note": "Sector defaults applied if not provided",
        },
        "enriched_from_open_data": {
            "count": len(enr),
            "fields": [f.name for f in enr],
            "sources": [f.data_source for f in enr],
        },
        "minimum_time_to_complete": "15–30 minutes (with annual report and asset list to hand)",
        "recommended_time_to_complete": "2–4 hours (full GHG inventory and emission intensities)",
    }
