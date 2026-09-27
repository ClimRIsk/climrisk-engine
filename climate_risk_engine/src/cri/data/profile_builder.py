"""CompanyProfileBuilder — parametric synthetic company generator.
═══════════════════════════════════════════════════════════════════════════════
Builds a Company object and matching Scenario objects from minimal inputs
(company name, sector, jurisdiction, approximate revenue).

All parameters are derived from `parameters.py` — no external API calls,
no geocoding, no network dependency.  Works for ANY company in ANY sector
in ANY country.

The generated Company has:
  • One representative asset in the company's HQ region
  • Emissions calibrated to sector IEA averages (CDP cross-referenced)
  • Financials from Damodaran sector medians
  • Hazard exposure derived from IPCC AR6 climate zone profiles

The generated Scenarios are copies of the standard three NGFS scenarios
with additional HazardPath entries for the company's climate zone,
so the ScenarioHazardProvider can resolve physical losses for the asset.

Usage
─────
    from cri.data.profile_builder import CompanyProfileBuilder

    result = CompanyProfileBuilder.build(
        name          = "HSBC Holdings",
        sector        = "financials",
        jurisdiction  = "GB",
        revenue_usd_m = 55_000.0,
    )
    company   = result.company
    scenarios = result.scenarios   # {"nze": Scenario, "delayed": Scenario, "cp": Scenario}
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

from ..data.schemas import (
    Asset,
    Commodity,
    Company,
    EmissionsProfile,
    Financials,
)
from ..data.parameters import (
    country_to_climate_zone,
    generate_hazard_paths,
    get_sector_financial_params,
    get_sector_transition_params,
    normalise_sector,
)

# ── Standard scenarios (imported lazily to avoid circular imports) ──────────

def _load_base_scenarios() -> Dict[str, object]:
    """Load the canonical NZE / Delayed / CP scenario objects."""
    from .. import scenarios as _scen
    return {
        "nze":     _scen.NZE_2050,
        "delayed": _scen.DELAYED_TRANSITION,
        "cp":      _scen.CURRENT_POLICIES,
    }


# ── Representative lat/lon per climate zone ─────────────────────────────────
# One representative point per zone, used so PhysicalHazardEngine can resolve
# region-level hazard data (temperature baselines, elevation, LULC etc.).
# These are CENTRAL points of the zone, not the company's actual location.
# Source: IPCC AR6 Reference Region centroids (WGI Ch1 Table 1.2).
_ZONE_LATLON: Dict[str, tuple] = {
    "tropical_humid":        (13.0, 80.0),    # South India / Chennai
    "arid_hot":              (24.0, 46.0),    # Saudi Arabia (Riyadh area)
    "temperate_maritime":    (51.5, -0.1),    # London, UK
    "temperate_continental": (48.8, 2.3),     # Paris, France
    "continental_cold":      (55.7, 37.6),    # Moscow, Russia
    "mediterranean":         (41.0, 28.9),    # Istanbul, Turkey
    "coastal_delta":         (23.7, 90.4),    # Dhaka, Bangladesh
}


# ── Commodity proxy mapping ─────────────────────────────────────────────────
# Maps canonical sector names to the closest available Commodity enum value.
# MANUFACTURING is used as a universal proxy for sectors with no dedicated curve.
_SECTOR_COMMODITY: Dict[str, str] = {
    "oil_gas":       "CRUDE_OIL",
    "coal":          "THERMAL_COAL",
    "utilities":     "THERMAL_COAL",     # mix; coal/gas proxy
    "renewable_energy": "MANUFACTURING", # revenue-based
    "steel":         "STEEL",
    "cement":        "MANUFACTURING",
    "chemicals":     "MANUFACTURING",
    "mining":        "IRON_ORE",
    "aluminium":     "ALUMINIUM",
    "aviation":      "MANUFACTURING",
    "shipping":      "MANUFACTURING",
    "automotive":    "MANUFACTURING",
    "logistics":     "MANUFACTURING",
    "agriculture":   "MANUFACTURING",
    "food_beverage": "MANUFACTURING",
    "real_estate":   "MANUFACTURING",
    "construction":  "MANUFACTURING",
    "infrastructure":"MANUFACTURING",
    "financials":    "MANUFACTURING",
    "insurance":     "MANUFACTURING",
    "technology":    "MANUFACTURING",
    "healthcare":    "MANUFACTURING",
    "consumer":      "MANUFACTURING",
    "retail":        "MANUFACTURING",
    "telecom":       "MANUFACTURING",
    "pharma":        "MANUFACTURING",
    "fishing":       "MANUFACTURING",
    "media":         "MANUFACTURING",
}

# Manufacturing proxy price ≈ $200/unit so production = revenue_usd_m / 200 = M-units
_MFG_PRICE = 200.0

# Approximate 2026 baseline prices for non-MANUFACTURING commodities (USD per unit).
# These match the NZE/CP 2026 starting values from the scenario price tables.
# Used to compute production_units and baseline_unit_cost for real-commodity sectors.
_COMMODITY_BASELINE_PRICE: Dict[str, float] = {
    "CRUDE_OIL":    78.0,   # $/bbl — IEA WEO 2023 reference
    "THERMAL_COAL": 130.0,  # $/t — IEA coal price 2026 proxy
    "IRON_ORE":     120.0,  # $/t — World Bank CMO 2026
    "STEEL":        750.0,  # $/t — Worldsteel 2026 proxy
    "ALUMINIUM":    2_400.0, # $/t — LME 2026 proxy
    "MANUFACTURING": _MFG_PRICE,  # $200/unit proxy
}


def _resolve_commodity(canonical_sector: str) -> "Commodity":
    from ..data.schemas import Commodity
    raw = _SECTOR_COMMODITY.get(canonical_sector, "MANUFACTURING")
    try:
        return Commodity[raw]
    except KeyError:
        return Commodity.MANUFACTURING


def _nominal_price_for_commodity(commodity_enum_name: str) -> float:
    """Return the 2026 baseline price for a commodity (USD/unit)."""
    return _COMMODITY_BASELINE_PRICE.get(commodity_enum_name, _MFG_PRICE)


# ════════════════════════════════════════════════════════════════════════════
# Result type
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class BuildResult:
    """Output of CompanyProfileBuilder.build()."""
    company: Company
    scenarios: Dict[str, object]   # 'nze' | 'delayed' | 'cp' → Scenario
    climate_zone: str
    canonical_sector: str
    data_notes: list[str]          # Provenance notes for transparency


# ════════════════════════════════════════════════════════════════════════════
# Builder
# ════════════════════════════════════════════════════════════════════════════

class CompanyProfileBuilder:
    """Builds a synthetic but realistic Company + Scenario set from minimal inputs.

    Calling `build()` is a pure in-memory operation — no network, no file I/O,
    no external dependencies.  Results are deterministic for the same inputs.
    """

    # Fixed asset region tag: {company_id}_primary_asset
    _ASSET_SUFFIX = "_hq_ops"

    @classmethod
    def build(
        cls,
        name: str,
        sector: str,
        jurisdiction: str,
        revenue_usd_m: float = 5_000.0,
        employee_count: Optional[int] = None,
    ) -> BuildResult:
        """Build a synthetic Company and its three NGFS-scenario counterparts.

        Args:
            name:          Company display name.
            sector:        Free-text sector description (will be normalised).
            jurisdiction:  ISO 2-letter country code or region code like 'GB',
                           'US', 'DE', 'IN-MH', 'AU-WA'.
            revenue_usd_m: Annual revenue in USD millions (default: 5 000).
            employee_count: Optional headcount for emissions scaling.

        Returns:
            BuildResult with company, scenarios, and data provenance notes.
        """
        canonical_sector = normalise_sector(sector)
        climate_zone     = country_to_climate_zone(jurisdiction)
        fin_p            = get_sector_financial_params(canonical_sector)
        trans_p          = get_sector_transition_params(canonical_sector)
        data_notes: list[str] = []

        # ── 1. Financials ─────────────────────────────────────────────────────
        ebitda    = revenue_usd_m * fin_p["ebitda_margin"]
        capex     = revenue_usd_m * fin_p["capex_pct_rev"]
        net_debt  = ebitda * fin_p["nd_ebitda"]
        market_cap = ebitda / fin_p["wacc_base"] * 0.85  # rough EV→equity

        financials = Financials(
            revenue              = revenue_usd_m,
            ebitda               = ebitda,
            capex                = capex,
            maintenance_capex_share = fin_p["maint_capex"],
            tax_rate             = fin_p["tax_rate"],
            wacc_base            = fin_p["wacc_base"],
            net_debt             = net_debt,
            shares_outstanding   = max(100.0, revenue_usd_m * 0.08),   # rough proxy
            market_cap           = market_cap,
        )
        data_notes.append(
            f"Financials: Damodaran sector medians ({canonical_sector}), "
            f"revenue={revenue_usd_m:,.0f} USD M"
        )

        # ── 2. Emissions ──────────────────────────────────────────────────────
        # Scope 1 intensity in tCO2e per $M revenue (IEA/CDP sector averages)
        # trans_p already has 'carbon_intensity_t_per_m_rev' as Scope 1+2 combined;
        # split roughly 60/40 scope1/scope2 for most sectors.
        total_intensity = trans_p["carbon_intensity_t_per_m_rev"]
        scope1_intensity_per_unit = (total_intensity * 0.60) / _MFG_PRICE
        scope2_intensity_per_unit = (total_intensity * 0.40) / _MFG_PRICE
        # Scope 3 ≈ 4× scope 1+2 for most sectors (CDP sectoral averages)
        scope3_intensity_per_unit = scope1_intensity_per_unit * 4.0

        data_notes.append(
            f"Emissions: IEA sector intensity {total_intensity:.0f} tCO2/$M rev; "
            f"Scope 1/2/3 split 60/40/4× estimated"
        )

        # ── 3. Asset ──────────────────────────────────────────────────────────
        # Region tag = canonical climate zone — this MUST match the HazardPath
        # region in the augmented scenarios built below.
        asset_region = climate_zone
        commodity    = _resolve_commodity(canonical_sector)

        # Determine the nominal (2026 baseline) price for this commodity.
        # For MANUFACTURING proxy sectors this is $200/unit; for real commodity
        # sectors (crude oil, coal, iron ore, etc.) use calibrated 2026 prices.
        nominal_price = _nominal_price_for_commodity(commodity.name)

        # Production volume: revenue / price_per_unit = M-units
        production_units = revenue_usd_m / nominal_price  # M-units

        # Baseline unit cost = nominal_price × (1 – EBITDA margin).
        # This guarantees a positive margin equal to ebitda_margin × nominal_price,
        # so physical_loss_cost (= vol × loss_frac × margin) is always non-zero
        # when physical hazards are present — across ALL sectors and commodities.
        # Source: Damodaran sector EBITDA margins (fin_p["ebitda_margin"]).
        baseline_unit_cost = nominal_price * (1.0 - fin_p["ebitda_margin"])

        # Carrying value ≈ revenue × 0.6 (asset-intensity varies, this is middle-ground)
        carrying_value = revenue_usd_m * 0.60

        # Representative coordinates for the climate zone so the
        # PhysicalHazardEngine can resolve region-level data (temperature
        # baseline, elevation, LULC, cyclone band, etc.)
        zone_lat, zone_lon = _ZONE_LATLON.get(climate_zone, (40.0, 10.0))

        asset = Asset(
            id                = f"{cls._make_id(name)}{cls._ASSET_SUFFIX}",
            name              = f"{name} — primary operations",
            commodity         = commodity,
            region            = asset_region,   # climate zone code
            baseline_production  = round(production_units, 2),
            production_unit   = "M-units (revenue proxy)",
            baseline_unit_cost = baseline_unit_cost,   # nominal × (1 – EBITDA margin)
            energy_cost_share = 0.12 + trans_p["carbon_intensity_t_per_m_rev"] / 20_000.0,
            carrying_value    = carrying_value,
            remaining_life_years = 20,
            emissions         = EmissionsProfile(
                scope1_intensity = scope1_intensity_per_unit,
                scope2_intensity = scope2_intensity_per_unit,
                scope3_intensity = scope3_intensity_per_unit,
                carbon_price_coverage = trans_p["carbon_price_coverage"],
                free_allocation  = 0.10,   # sector median free ETS allocation
            ),
            lat  = zone_lat,    # Representative coordinates for climate zone
            lon  = zone_lon,    # (not actual company location)
            equipment_type = "generic",
        )

        # ── 4. Company ────────────────────────────────────────────────────────
        company_id = cls._make_id(name)
        company = Company(
            id         = company_id,
            name       = name,
            sector     = canonical_sector.replace("_", " ").title(),
            hq_region  = jurisdiction.upper(),
            financials = financials,
            assets     = [asset],
            data_quality = "estimated",
        )

        # ── 5. Augmented Scenarios ────────────────────────────────────────────
        # Copy each base scenario and append HazardPaths for our climate zone.
        # This ensures ScenarioHazardProvider can resolve physical losses for
        # the synthetic asset (which has region == climate_zone).
        base_scenarios = _load_base_scenarios()
        scenarios = {}
        for scen_key in ("nze", "delayed", "cp"):
            scenarios[scen_key] = cls._augment_scenario(
                base_scenarios[scen_key], climate_zone, scen_key
            )

        data_notes.append(
            f"Physical risk: IPCC AR6 Ch12 {climate_zone} zone, "
            f"scenarios NZE/Delayed/CP"
        )
        data_notes.append(
            f"Transition risk: NGFS Phase 4 {canonical_sector} sector pathway"
        )

        return BuildResult(
            company        = company,
            scenarios      = scenarios,
            climate_zone   = climate_zone,
            canonical_sector = canonical_sector,
            data_notes     = data_notes,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _make_id(name: str) -> str:
        """Make a safe ASCII ID from a company name."""
        import re
        return re.sub(r"[^a-z0-9_]", "_", name.lower().strip())[:40]

    @staticmethod
    def _augment_scenario(base_scenario: object, climate_zone: str, scen_key: str) -> object:
        """Return a copy of base_scenario with HazardPaths added for climate_zone.

        Uses Pydantic model_copy (v2) or copy (v1) to avoid mutating shared
        scenario objects.
        """
        from ..data.schemas import HazardPath, HazardType
        from copy import deepcopy

        # Map our hazard name strings to HazardType enum values
        _HAZARD_MAP: Dict[str, str] = {
            "heat_stress":  "HEAT_STRESS",
            "water_stress": "WATER_STRESS",
            "flood":        "FLOOD",
            "cyclone":      "CYCLONE",
            "wildfire":     "WILDFIRE",
        }

        # Generate hazard paths for this zone × scenario
        raw_paths = generate_hazard_paths(climate_zone, climate_zone, scen_key)

        new_hazard_paths = []
        for hazard_name, region_code, path_dict in raw_paths:
            enum_name = _HAZARD_MAP.get(hazard_name)
            if enum_name is None:
                continue
            try:
                hazard_type = HazardType[enum_name]
            except KeyError:
                continue
            new_hazard_paths.append(
                HazardPath(
                    hazard=hazard_type,
                    region=region_code,
                    path=path_dict,
                )
            )

        # Skip if region already has paths in this scenario (avoid duplicates)
        existing_regions = {hp.region for hp in base_scenario.hazards}
        paths_to_add = [
            hp for hp in new_hazard_paths
            if hp.region not in existing_regions
        ]

        if not paths_to_add:
            return base_scenario   # already covered

        # Deep-copy so we don't mutate the shared scenario registry
        scenario_copy = deepcopy(base_scenario)
        scenario_copy.hazards.extend(paths_to_add)
        return scenario_copy


# ── Convenience function for use in API endpoints ────────────────────────────

def build_synthetic_company(
    name: str,
    sector: str,
    jurisdiction: str,
    revenue_usd_m: float = 5_000.0,
) -> BuildResult:
    """Shortcut: build a synthetic Company + Scenario set.

    Example (in api/main.py)::

        from ..data.profile_builder import build_synthetic_company

        result = build_synthetic_company("Tesco PLC", "retail", "GB", 70_000)
        nze_r = run_engine(company=result.company, scenario=result.scenarios["nze"])
    """
    return CompanyProfileBuilder.build(
        name          = name,
        sector        = sector,
        jurisdiction  = jurisdiction,
        revenue_usd_m = revenue_usd_m,
    )
