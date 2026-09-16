"""
profile_adapter.py — CompanyProfile → CRI engine Company schema converter.

Bridges the autonomous data acquisition layer (CompanyProfile with full
provenance tagging) into the engine's strict Pydantic Company schema so
that job_runner can call orchestrator.run_full() directly.

Design choices
--------------
1. Revenue is treated as the "production unit" when no asset-level production
   data is available.  scope1_intensity = scope1_mt_co2e × 1e6 / revenue_usd_m
   (tCO2e per USD M revenue). This preserves the engine's carbon_cost formula:
       carbon_cost = production × intensity × carbon_price
                   = revenue × (scope1 Mt × 1e6 / revenue) × price
                   = scope1_tco2e × price  ✓

2. Sector → Commodity mapping covers the full CRI commodity enum.  Unknown
   sectors fall back to MANUFACTURING (closest generic proxy).

3. CAPEX is sector-calibrated as a fraction of revenue when not directly
   reported (EDGAR rarely provides clean capex in XBRL).

4. EBITDA margin fallback: if EBITDA is missing, use sector median margin
   so the DCF doesn't receive zero operating income.

5. Data quality → engine data_quality: VERIFIED/REPORTED → "high",
   ESTIMATED → "medium", MISSING-dominant → "low".
"""
from __future__ import annotations

import re
import uuid
from typing import Optional

from ..data.schemas import (
    Asset,
    Commodity,
    Company,
    EmissionsProfile,
    Financials,
    SegmentBaseline,
)
from .company_profiler import CompanyProfile
from .provenance import ConfidenceTier


# ── Sector → primary Commodity mapping ───────────────────────────────────────
_SECTOR_COMMODITY: dict[str, Commodity] = {
    "oil_gas":            Commodity.CRUDE_OIL,
    "oil":                Commodity.CRUDE_OIL,
    "gas":                Commodity.NATURAL_GAS,
    "refining":           Commodity.REFINED_PRODUCTS,
    "coal":               Commodity.COAL_THERMAL,
    "coal_thermal":       Commodity.COAL_THERMAL,
    "coal_metallurgical": Commodity.COAL_METALLURGICAL,
    "utilities":          Commodity.ELECTRICITY,
    "power":              Commodity.ELECTRICITY,
    "electricity":        Commodity.ELECTRICITY,
    "steel":              Commodity.IRON_ORE,
    "iron_ore":           Commodity.IRON_ORE,
    "mining":             Commodity.COPPER,
    "copper":             Commodity.COPPER,
    "aluminium":          Commodity.ALUMINIUM,
    "aluminum":           Commodity.ALUMINIUM,
    "cement":             Commodity.CEMENT,
    "chemicals":          Commodity.CHEMICALS,
    "chemical":           Commodity.CHEMICALS,
    "agriculture":        Commodity.AGRICULTURE,
    "food":               Commodity.FOOD,
    "beverages":          Commodity.BEVERAGES,
    "retail":             Commodity.RETAIL,
    "financials":         Commodity.FINANCIAL_SERVICES,
    "banks":              Commodity.FINANCIAL_SERVICES,
    "insurance":          Commodity.FINANCIAL_SERVICES,
    "real_estate":        Commodity.REAL_ESTATE,
    "property":           Commodity.REAL_ESTATE,
    "technology":         Commodity.MANUFACTURING,
    "automotive":         Commodity.MANUFACTURING,
    "healthcare":         Commodity.MANUFACTURING,
    "manufacturing":      Commodity.MANUFACTURING,
    "shipping":           Commodity.MANUFACTURING,
    "aviation":           Commodity.MANUFACTURING,
    "default":            Commodity.MANUFACTURING,
}

# ── Sector CAPEX as % of revenue (median, used when XBRL capex missing) ─────
_SECTOR_CAPEX_RATIO: dict[str, float] = {
    "oil_gas":     0.20,
    "utilities":   0.18,
    "steel":       0.08,
    "mining":      0.15,
    "coal":        0.10,
    "cement":      0.08,
    "chemicals":   0.07,
    "agriculture": 0.05,
    "real_estate": 0.10,
    "technology":  0.05,
    "financials":  0.02,
    "default":     0.06,
}

# ── Sector EBITDA margins (median, fallback when not in profile) ─────────────
_SECTOR_EBITDA_MARGIN: dict[str, float] = {
    "oil_gas":     0.22,
    "utilities":   0.28,
    "steel":       0.12,
    "mining":      0.30,
    "coal":        0.18,
    "cement":      0.20,
    "chemicals":   0.16,
    "agriculture": 0.10,
    "real_estate": 0.35,
    "technology":  0.22,
    "financials":  0.30,
    "default":     0.15,
}


def _slugify(name: str) -> str:
    """Convert company name to valid ID slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug[:50] or str(uuid.uuid4())[:8]


def _commodity_for_sector(sector: Optional[str]) -> Commodity:
    if sector is None:
        return Commodity.MANUFACTURING
    s = sector.lower().strip()
    return _SECTOR_COMMODITY.get(s, Commodity.MANUFACTURING)


def _data_quality(profile: CompanyProfile) -> str:
    """
    Derive engine data_quality string from provenance confidence tiers.
    "high" = all key fields VERIFIED or REPORTED
    "medium" = some ESTIMATED fields
    "low"  = critical gaps (MISSING scope1 or revenue)
    """
    tiers = []
    for field in (
        profile.revenue_usd_m, profile.scope1_mt_co2e,
        profile.ebitda_usd_m, profile.lat,
    ):
        if field is not None:
            tiers.append(field.confidence_tier)

    missing_critical = (
        profile.revenue_usd_m is None
        or profile.scope1_mt_co2e is None
    )
    if missing_critical:
        return "low"
    if any(t == ConfidenceTier.ESTIMATED for t in tiers):
        return "medium"
    return "high"


def profile_to_company(profile: CompanyProfile) -> Company:
    """
    Convert a CompanyProfile (provenance-tagged acquisition output) to a
    CRI engine Company (Pydantic schema expected by orchestrator.run_full).

    Parameters
    ----------
    profile : CompanyProfile
        Output from build_company_profile() — may have missing fields.

    Returns
    -------
    Company
        Fully populated Company schema.  Where profile fields are missing,
        sector-median benchmarks are substituted and flagged in data_quality.
    """
    name    = profile.resolved_name or profile.input_name
    sector  = profile.sector or "default"
    company_id = profile.lei or _slugify(name)

    # ── Financials ───────────────────────────────────────────────────────────
    revenue_usd_m  = (profile.revenue_usd_m.value if profile.revenue_usd_m else None) or 5_000.0
    ebitda_usd_m   = (profile.ebitda_usd_m.value if profile.ebitda_usd_m else None)
    if ebitda_usd_m is None:
        margin = _SECTOR_EBITDA_MARGIN.get(sector, _SECTOR_EBITDA_MARGIN["default"])
        ebitda_usd_m = revenue_usd_m * margin

    capex_ratio = _SECTOR_CAPEX_RATIO.get(sector, _SECTOR_CAPEX_RATIO["default"])
    capex_usd_m = revenue_usd_m * capex_ratio

    net_debt_usd_m = (profile.total_debt_usd_m.value if profile.total_debt_usd_m else 0.0) or 0.0
    market_cap_usd_m = (profile.market_cap_usd_m.value if profile.market_cap_usd_m else None)

    # WACC: base 8% adjusted by beta if available; energy/utilities get a small premium
    beta = getattr(profile, "beta", None)  # optional field
    wacc_base = 0.08
    if sector in ("oil_gas", "coal", "utilities", "steel"):
        wacc_base = 0.09  # climate-exposed sectors get 100bps premium
    if beta is not None:
        # Simple Hamada un/re-levering proxy: just clip unreasonable betas
        wacc_base = max(0.06, min(0.14, wacc_base * float(beta) / 1.0))

    financials = Financials(
        revenue=revenue_usd_m,
        ebitda=ebitda_usd_m,
        capex=capex_usd_m,
        maintenance_capex_share=0.6,
        tax_rate=0.25,
        wacc_base=wacc_base,
        net_debt=net_debt_usd_m,
        shares_outstanding=1.0,  # not required for company-level risk
        market_cap=market_cap_usd_m,
    )

    # ── Emissions intensity (per USD M revenue as proxy production unit) ─────
    # Engine computes: carbon_cost = production × intensity × carbon_price
    # If production = revenue (USD M) and intensity = tCO2e / USD M:
    #   carbon_cost = revenue_usd_m × (scope1_tco2e / revenue_usd_m) × price = scope1_tco2e × price ✓
    scope1_tco2e = (
        (profile.scope1_mt_co2e.value * 1_000_000)
        if profile.scope1_mt_co2e and profile.scope1_mt_co2e.value
        else 0.0
    )
    scope2_tco2e = (
        (profile.scope2_mt_co2e.value * 1_000_000)
        if profile.scope2_mt_co2e and profile.scope2_mt_co2e.value
        else 0.0
    )
    scope3_tco2e = (
        (profile.scope3_mt_co2e.value * 1_000_000)
        if profile.scope3_mt_co2e and profile.scope3_mt_co2e.value
        else 0.0
    )

    scope1_intensity = scope1_tco2e / revenue_usd_m if revenue_usd_m > 0 else 0.0
    scope2_intensity = scope2_tco2e / revenue_usd_m if revenue_usd_m > 0 else 0.0
    scope3_intensity = scope3_tco2e / revenue_usd_m if revenue_usd_m > 0 else 0.0

    emissions = EmissionsProfile(
        scope1_intensity=scope1_intensity,
        scope2_intensity=scope2_intensity,
        scope3_intensity=scope3_intensity,
        carbon_price_coverage=1.0,
        free_allocation=0.0,
    )

    # ── Segments ─────────────────────────────────────────────────────────────
    commodity = _commodity_for_sector(sector)
    ebitda_margin = ebitda_usd_m / revenue_usd_m if revenue_usd_m > 0 else 0.15

    segments = [
        SegmentBaseline(
            commodity=commodity,
            revenue_baseline=revenue_usd_m,
            volume_baseline=revenue_usd_m,   # USD M as proxy production unit
            ebitda_margin_baseline=ebitda_margin,
            emissions=emissions,
        )
    ]

    # ── Assets (single proxy asset from HQ location) ─────────────────────────
    lat = profile.lat.value if profile.lat else None
    lon = profile.lon.value if profile.lon else None
    hq_region = profile.jurisdiction or "global"

    assets: list[Asset] = []
    if lat is not None and lon is not None:
        assets.append(
            Asset(
                id=f"{company_id}-hq",
                name=f"{name} HQ / primary operations",
                commodity=commodity,
                region=hq_region,
                baseline_production=revenue_usd_m,
                production_unit="USD_M_revenue",
                emissions=emissions,
                carrying_value=revenue_usd_m * 2.0,   # rough 2× revenue proxy for asset base
                remaining_life_years=30,
                baseline_unit_cost=1.0,
                energy_cost_share=0.15,
                lat=float(lat),
                lon=float(lon),
            )
        )

    return Company(
        id=company_id,
        name=name,
        sector=sector,
        hq_region=hq_region,
        financials=financials,
        segments=segments,
        assets=assets,
        data_quality=_data_quality(profile),
    )
