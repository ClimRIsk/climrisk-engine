"""
cbam.py — EU Carbon Border Adjustment Mechanism (CBAM) exposure calculator.

The EU CBAM entered its full implementation phase in 2026, requiring importers
of covered goods into the EU to purchase CBAM certificates equal to the
embedded carbon in imported products minus any carbon price paid in the
country of production.

This module computes:
  1. Whether the company is in a CBAM-covered sector
  2. The estimated embedded carbon in EU-destined exports (tCO2e/year)
  3. The annual CBAM certificate cost under each NGFS carbon price scenario
  4. The exposure trajectory 2026-2034 (transition period → full implementation)

Covered sectors (Regulation EU 2023/956, Annex I)
--------------------------------------------------
  1. Cement — CN codes 2507, 2523
  2. Iron and steel — CN codes 7206-7229, 7301-7318 (selected)
  3. Aluminium — CN codes 7601-7607, 7616 (selected)
  4. Fertilisers — CN codes 2808, 2814, 3102-3105
  5. Electricity — CN code 2716
  6. Hydrogen — CN code 2804

Methodological notes
--------------------
- CBAM applies only to direct (Scope 1-equivalent) embedded emissions
- Carbon price already paid in the country of origin is deductible
- Phase-in schedule: 100% from 2026 (transitional period ended 2025-12-31)
- Certificate price tracks EU ETS price (proxied by NGFS EU ETS estimate)
- Revenue exposure fraction (EU sales as % of total) must be provided
  or estimated from company disclosures / country mix.

Sources
-------
EU CBAM Regulation 2023/956: https://eur-lex.europa.eu/eli/reg/2023/956
EU ETS prices: NGFS Phase 4 + EMBER monthly data
Carbon-to-product intensity (tCO2e/tonne): IEA WEO 2023 sector reports,
  JRC EDGAR v8, European Commission CBAM Technical Guidance 2023.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── CBAM-covered sectors and their CN code groups ────────────────────────────
_CBAM_SECTORS = {
    "cement":      "Cement (CN 2507, 2523)",
    "steel":       "Iron & Steel (CN 7206-7229, 7301-7318)",
    "aluminium":   "Aluminium (CN 7601-7616)",
    "aluminum":    "Aluminium (CN 7601-7616)",
    "fertilizers": "Fertilisers (CN 2808, 2814, 3102-3105)",
    "chemicals":   "Hydrogen & Chemicals (CN 2804, selected)",
    "hydrogen":    "Hydrogen (CN 2804 10 00)",
    "electricity": "Electricity (CN 2716)",
}

# ── Sector aliases that map to CBAM-covered primary sectors ──────────────────
_SECTOR_ALIASES: dict[str, str] = {
    "iron_ore":    "steel",
    "mining":      "steel",
    "coal_met":    "steel",
    "agrochemical": "fertilizers",
    "agriculture": "fertilizers",
    "lng":         "chemicals",
    "gas":         "chemicals",
}

# ── Embedded carbon intensity (tCO2e per tonne of product) ─────────────────
# Source: IEA WEO 2023, European Commission CBAM Technical Guidance 2023
_EMBEDDED_CARBON_INTENSITY: dict[str, float] = {
    "cement":      0.83,    # tCO2e/tonne cement (EU average, process + energy)
    "steel":       1.85,    # tCO2e/tonne crude steel (BF-BOF route)
    "aluminium":   8.50,    # tCO2e/tonne aluminium (Scope 1+2 primary)
    "aluminum":    8.50,
    "fertilizers": 2.60,    # tCO2e/tonne ammonia-based (urea proxy)
    "chemicals":   12.00,   # tCO2e/tonne hydrogen (grey H2 steam reforming)
    "hydrogen":    12.00,
    "electricity": 0.45,    # tCO2e/MWh (coal-heavy grid proxy for CBAM coverage)
}

# ── EU ETS price trajectory (EUR/tCO2, proxying CBAM certificate price) ─────
# Source: NGFS Phase 4 (2023), EU ETS futures strip, EMBER monthly
# Note: CBAM price tracks EU ETS spot under Regulation Article 21.
_EU_ETS_PRICE: dict[int, float] = {
    2026: 60,   # EUR/tCO2 (post-Phase 3 reform)
    2027: 72,
    2028: 85,
    2029: 95,
    2030: 108,
    2031: 120,
    2032: 135,
    2033: 148,
    2034: 160,
    2035: 175,  # extrapolated, NGFS NZE-consistent EU ETS
}

# ── Carbon price already paid in origin countries (EUR/tCO2) ─────────────────
# Deductible from CBAM certificate cost per Article 9 of Regulation 2023/956.
# Source: World Bank Carbon Pricing Dashboard 2024.
_ORIGIN_CARBON_PRICE: dict[str, float] = {
    "CN": 8.0,    # China national ETS (2024 vintage, electricity sector only)
    "US": 0.0,    # No federal carbon price; regional RGGI ~$15 not universally applicable
    "IN": 0.0,    # India PAT scheme — not a cap-and-trade carbon price
    "TR": 5.0,    # Turkey ETS (2025 launch)
    "KR": 20.0,   # South Korea ETS
    "CA": 65.0,   # Canada Output-Based Pricing System (CAD)
    "AU": 0.0,    # No carbon price at national level
    "GB": 45.0,   # UK ETS (linked to EU ETS)
    "NO": 100.0,  # Norway (full EU ETS participant)
    "default": 0.0,
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


@dataclass
class CBAMResult:
    """CBAM exposure analysis for a single company."""
    company_name:      str
    sector:            str
    cbam_sector_name:  Optional[str]     = None
    is_applicable:     bool              = False
    embedded_carbon_intensity: float     = 0.0   # tCO2e / tonne product
    scenarios:         dict[str, dict]   = field(default_factory=dict)  # year → cost breakdown
    peak_annual_cost_usd_m: float        = 0.0
    methodology:       str               = ""
    data_gaps:         list[str]         = field(default_factory=list)


def assess_cbam_exposure(
    company_name:         str,
    sector:               str,
    revenue_usd_m:        float,
    eu_revenue_fraction:  float = 0.20,    # fraction of revenue from EU sales (default 20%)
    production_tonnes:    Optional[float] = None,  # optional: actual production volume (tonnes/yr)
    origin_country:       str   = "default",       # country of production (ISO-2)
    carbon_price_paid_eur: Optional[float] = None, # override origin carbon price
    eur_usd_rate:         float = 1.08,            # EUR/USD conversion
) -> CBAMResult:
    """
    Assess EU CBAM exposure for a company exporting to the EU.

    Parameters
    ----------
    company_name          : Display name
    sector                : Sector slug
    revenue_usd_m         : Total annual revenue (USD M)
    eu_revenue_fraction   : Fraction of revenue from EU-bound exports (0-1)
    production_tonnes     : Annual production volume (tonnes); if None, estimated from revenue
    origin_country        : ISO-2 country of production (for origin carbon price deduction)
    carbon_price_paid_eur : Override for origin carbon price (EUR/tCO2)
    eur_usd_rate          : EUR to USD conversion rate
    """
    result = CBAMResult(company_name=company_name, sector=sector)

    # Check CBAM applicability
    s = sector.lower()
    cbam_key = _CBAM_SECTORS.get(s) or _CBAM_SECTORS.get(_SECTOR_ALIASES.get(s, ""))
    if cbam_key is None:
        result.is_applicable = False
        result.methodology   = (
            f"Sector '{sector}' is not in CBAM Regulation EU 2023/956 Annex I. "
            "CBAM exposure is not applicable."
        )
        return result

    # Resolve primary sector key for intensity lookup
    primary_sector = _SECTOR_ALIASES.get(s, s)
    result.is_applicable    = True
    result.cbam_sector_name = cbam_key

    intensity = _EMBEDDED_CARBON_INTENSITY.get(primary_sector, 2.0)
    result.embedded_carbon_intensity = intensity

    # Estimate EU-bound production volume
    eu_revenue_usd_m   = revenue_usd_m * eu_revenue_fraction
    # Revenue per tonne proxy: $500/t for steel, $120/t cement, $2,500/t aluminium
    _REV_PER_TONNE: dict[str, float] = {
        "steel": 700, "cement": 80, "aluminium": 2200, "aluminum": 2200,
        "fertilizers": 350, "chemicals": 1500, "hydrogen": 2500,
        "electricity": 60,   # per MWh, not tonne
    }
    rev_per_unit = _REV_PER_TONNE.get(primary_sector, 500.0)
    eu_production = production_tonnes * eu_revenue_fraction if production_tonnes else (
        eu_revenue_usd_m * 1_000_000 / rev_per_unit   # USD M → USD → tonnes
    )

    # Embedded carbon in EU-bound exports
    eu_embedded_co2e = eu_production * intensity   # tCO2e/year

    # Origin carbon price already paid
    origin_cp = (
        carbon_price_paid_eur
        if carbon_price_paid_eur is not None
        else _ORIGIN_CARBON_PRICE.get(origin_country.upper(), _ORIGIN_CARBON_PRICE["default"])
    )

    # Data gaps
    if eu_revenue_fraction == 0.20:
        result.data_gaps.append(
            "eu_revenue_fraction defaulted to 20% — provide actual EU sales breakdown"
        )
    if production_tonnes is None:
        result.data_gaps.append(
            "production_tonnes estimated from revenue — provide actual volume for precision"
        )

    # Annual cost projection 2026-2034 (key CBAM window)
    yearly: list[dict] = []
    peak_cost = 0.0
    for year in range(2026, 2035):
        ets_price_eur  = _interp(_EU_ETS_PRICE, year)
        net_price_eur  = max(0.0, ets_price_eur - origin_cp)
        annual_cost_eur_m = eu_embedded_co2e * net_price_eur / 1_000_000
        annual_cost_usd_m = annual_cost_eur_m * eur_usd_rate

        # Phase-in: full implementation from 2026
        phase_in = 1.0  # 100% from 2026

        effective_cost_usd_m = annual_cost_usd_m * phase_in
        cost_pct_revenue     = (effective_cost_usd_m / revenue_usd_m * 100) if revenue_usd_m > 0 else 0

        entry = {
            "year":                       year,
            "eu_ets_price_eur_t":         round(ets_price_eur, 0),
            "origin_carbon_price_eur_t":  round(origin_cp, 0),
            "net_cbam_price_eur_t":       round(net_price_eur, 0),
            "eu_embedded_co2e_t":         round(eu_embedded_co2e, 0),
            "cbam_cost_eur_m":            round(annual_cost_eur_m, 2),
            "cbam_cost_usd_m":            round(effective_cost_usd_m, 2),
            "cbam_cost_pct_revenue":      round(cost_pct_revenue, 3),
            "phase_in_pct":               round(phase_in * 100, 0),
        }
        yearly.append(entry)
        peak_cost = max(peak_cost, effective_cost_usd_m)

    result.scenarios["cbam_trajectory"] = {"years": yearly}
    result.peak_annual_cost_usd_m = round(peak_cost, 2)

    result.methodology = (
        f"EU CBAM Regulation 2023/956 (Annex I: {cbam_key}). "
        f"Embedded carbon intensity: {intensity:.2f} tCO2e/unit "
        f"(IEA WEO 2023 / EC CBAM Technical Guidance). "
        f"EU-bound exports: {eu_revenue_fraction*100:.0f}% of revenue (~{eu_embedded_co2e:,.0f} tCO2e/yr). "
        f"Origin carbon price deduction: EUR {origin_cp:.0f}/tCO2 ({origin_country.upper()}). "
        "Certificate price tracks EU ETS (NGFS Phase 4 / EMBER)."
    )
    return result
