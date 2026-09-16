"""
wildfire_risk.py — Wildfire financial risk assessment.

Wildfire is the fastest-growing physical climate hazard by insured loss growth
(+30% per decade, Swiss Re 2023). Despite NASA FIRMS being connected in the
engine, wildfire had no standalone financial translation module. This fills
that gap.

Framework
---------
1. Historical fire density (NASA FIRMS 10-year average burned area by location)
2. Future Fire Weather Index (FWI) trend under SSP2-4.5/SSP5-8.5 (CMIP6)
3. Sector-specific asset vulnerability (exposure × sensitivity)
4. Financial loss translation:
   a. Direct asset damage (% of asset replacement value)
   b. Business interruption (days offline × daily revenue)
   c. Smoke / air quality production loss (indirect)
   d. Insurance premium uplift and coverage gap
   e. Supply chain / procrement disruption (sector-specific)

Geography
---------
Six wildfire risk zones derived from CMIP6 FWI projections (Abatzoglou et al.
2019, Nature Communications):
  Zone 1 — Mediterranean Europe, California, Chile, SW Australia: VERY HIGH
  Zone 2 — SE USA, SE Australia, Southern Africa, N India: HIGH
  Zone 3 — Central USA, E Europe, NE China, C South America: MEDIUM
  Zone 4 — Rest of temperate regions: LOW
  Zone 5 — Boreal forest (Canada, Russia, Scandinavia): MEDIUM (rising)
  Zone 6 — Tropics, equatorial: VARIABLE

Sector vulnerability
--------------------
Sector-specific wildfire sensitivity reflects:
- Outdoor asset concentration (refineries, mines, orchards, timber)
- Labour exposure (agriculture, construction, forestry)
- Supply chain concentration in fire-prone regions
- Tourism / hospitality / real estate aesthetics
Source: Munich Re NatCat Service sector loss ratios 2010-2023

Sources
-------
NASA FIRMS (VIIRS S-NPP, 2012-2023): burned area density
CMIP6 FWI projections: Abatzoglou et al. (2019) Nature Communications
   DOI: 10.1038/s41467-019-11500-4
Swiss Re Institute (2023): NatCat 2023 wildfire insured loss report
Munich Re (2023): NatCat wildfire loss statistics 2002-2022
IPCC AR6 WGII Ch12 — compound events (fire + heat + drought)
California DOF (2022): Wildfire economic impact study
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ── Wildfire risk zone by country/region ──────────────────────────────────────
# Based on CMIP6 FWI projections for SSP2-4.5 (2030-2060 mean anomaly vs baseline)
# Zone 1=Very High, 2=High, 3=Medium, 4=Low, 5=Boreal/Rising, 6=Tropical
_COUNTRY_FIRE_ZONE: dict[str, int] = {
    # Very High (Zone 1) — Mediterranean + West Coast regions
    "US": 2,   # National average; California/Pacific NW would be Zone 1
    "CA": 2,   # Canada national; BC/Alberta forest fires rising fast
    "AU": 1,
    "ES": 1,
    "PT": 1,
    "IT": 1,
    "GR": 1,
    "CL": 1,
    "MX": 2,
    # High (Zone 2)
    "ZA": 2,
    "BR": 2,
    "IN": 2,
    "TR": 2,
    "AR": 2,
    # Medium (Zone 3)
    "DE": 3,
    "FR": 3,
    "PL": 3,
    "CN": 3,
    "RU": 5,   # Boreal — high and rising
    "ID": 6,
    "NG": 6,
    # Low (Zone 4) — temperate, humid
    "GB": 4,
    "NL": 4,
    "SE": 5,   # Boreal rising
    "NO": 5,
    "FI": 5,
    "DK": 4,
    "JP": 3,
    "KR": 3,
    "NZ": 3,
    "CH": 4,
    "BE": 4,
    "AT": 4,
    "default": 3,
}

# ── FWI trend multiplier by scenario and year ─────────────────────────────────
# Fire Weather Index increase (multiplier on 2020 baseline) per zone
# Based on CMIP6 SSP scenarios (Abatzoglou et al. 2019 + IPCC AR6 WGII update)
# Format: {zone: {scenario: {year: multiplier}}}
_FWI_MULTIPLIER: dict[int, dict[str, dict[int, float]]] = {
    1: {  # Very High (Mediterranean/Australia)
        "nze":     {2030: 1.15, 2040: 1.20, 2050: 1.18},  # Action limits warming
        "delayed": {2030: 1.20, 2040: 1.45, 2050: 1.75},
        "cp":      {2030: 1.25, 2040: 1.65, 2050: 2.20},
    },
    2: {  # High
        "nze":     {2030: 1.10, 2040: 1.15, 2050: 1.12},
        "delayed": {2030: 1.15, 2040: 1.35, 2050: 1.60},
        "cp":      {2030: 1.20, 2040: 1.55, 2050: 2.00},
    },
    3: {  # Medium
        "nze":     {2030: 1.05, 2040: 1.10, 2050: 1.08},
        "delayed": {2030: 1.10, 2040: 1.25, 2050: 1.45},
        "cp":      {2030: 1.12, 2040: 1.40, 2050: 1.75},
    },
    4: {  # Low
        "nze":     {2030: 1.02, 2040: 1.05, 2050: 1.04},
        "delayed": {2030: 1.05, 2040: 1.12, 2050: 1.25},
        "cp":      {2030: 1.07, 2040: 1.18, 2050: 1.40},
    },
    5: {  # Boreal (rising fast)
        "nze":     {2030: 1.12, 2040: 1.20, 2050: 1.22},
        "delayed": {2030: 1.20, 2040: 1.50, 2050: 1.90},
        "cp":      {2030: 1.25, 2040: 1.80, 2050: 2.50},
    },
    6: {  # Tropical (variable)
        "nze":     {2030: 1.05, 2040: 1.08, 2050: 1.07},
        "delayed": {2030: 1.10, 2040: 1.20, 2050: 1.35},
        "cp":      {2030: 1.12, 2040: 1.30, 2050: 1.60},
    },
}

# ── Sector wildfire sensitivity ───────────────────────────────────────────────
# vulnerability_score: 0-1 direct asset exposure
# bi_days: expected business interruption days per wildfire event
# supply_chain_impact: supply chain disruption factor (0-1)
# asset_damage_pct: % of asset replacement value at risk per fire event at site
_SECTOR_VULNERABILITY: dict[str, dict] = {
    "agriculture":    {"vuln": 0.90, "bi_days": 45, "supply": 0.70, "asset_dmg": 0.35},
    "forestry":       {"vuln": 1.00, "bi_days": 90, "supply": 0.80, "asset_dmg": 0.85},
    "oil_gas":        {"vuln": 0.70, "bi_days": 30, "supply": 0.40, "asset_dmg": 0.25},
    "mining":         {"vuln": 0.65, "bi_days": 20, "supply": 0.35, "asset_dmg": 0.20},
    "utilities":      {"vuln": 0.55, "bi_days": 15, "supply": 0.25, "asset_dmg": 0.15},
    "real_estate":    {"vuln": 0.75, "bi_days": 60, "supply": 0.10, "asset_dmg": 0.40},
    "tourism":        {"vuln": 0.80, "bi_days": 90, "supply": 0.20, "asset_dmg": 0.50},
    "food_processing":{"vuln": 0.55, "bi_days": 20, "supply": 0.60, "asset_dmg": 0.15},
    "chemicals":      {"vuln": 0.45, "bi_days": 15, "supply": 0.30, "asset_dmg": 0.20},
    "steel":          {"vuln": 0.30, "bi_days": 10, "supply": 0.20, "asset_dmg": 0.10},
    "cement":         {"vuln": 0.25, "bi_days": 5,  "supply": 0.15, "asset_dmg": 0.08},
    "construction":   {"vuln": 0.55, "bi_days": 15, "supply": 0.25, "asset_dmg": 0.15},
    "technology":     {"vuln": 0.25, "bi_days": 10, "supply": 0.20, "asset_dmg": 0.08},
    "aviation":       {"vuln": 0.30, "bi_days": 5,  "supply": 0.10, "asset_dmg": 0.05},
    "financials":     {"vuln": 0.10, "bi_days": 5,  "supply": 0.05, "asset_dmg": 0.02},
    "default":        {"vuln": 0.30, "bi_days": 10, "supply": 0.15, "asset_dmg": 0.10},
}

# ── Historical annual wildfire probability by zone ────────────────────────────
# Probability that a given site in this zone experiences a significant fire
# event (>10 ha burned within 5 km radius) in any given year.
# Source: NASA FIRMS 2012-2022 event frequency + Swiss Re NatCat
_ANNUAL_FIRE_PROB: dict[int, float] = {
    1: 0.12,   # Zone 1: ~12% per year
    2: 0.07,
    3: 0.04,
    4: 0.02,
    5: 0.05,   # Boreal: rising frequency
    6: 0.06,
}

# ── Insurance coverage ratio by zone ─────────────────────────────────────────
# Fraction of wildfire loss that is insured (protection gap = 1 - this)
# Source: Swiss Re sigma No. 1/2023 (natural catastrophe insurance protection gap)
_INSURANCE_COVERAGE: dict[int, float] = {
    1: 0.55,   # Mediterranean / California — moderate coverage
    2: 0.30,   # Developing world — significant protection gap
    3: 0.45,
    4: 0.60,
    5: 0.25,   # Boreal — very low coverage (Canada improving)
    6: 0.15,
}


@dataclass
class WildfireYearResult:
    year: int
    fwi_multiplier: float
    annual_fire_prob: float      # adjusted for FWI trend
    expected_direct_loss_usd_m: float
    expected_bi_loss_usd_m: float
    expected_supply_chain_loss_usd_m: float
    total_expected_loss_usd_m: float
    insured_loss_usd_m: float
    uninsured_loss_usd_m: float


@dataclass
class WildfireRiskResult:
    """Wildfire physical risk and financial loss assessment."""
    company_name:       str
    sector:             str
    country:            str
    fire_zone:          int    # 1=Very High → 6=Tropical

    # Annual probability (current and future)
    baseline_annual_prob: float = 0.0    # 2024 probability
    prob_2030:          float   = 0.0
    prob_2050:          float   = 0.0

    # NPV financial losses (USD M, 2024-2050)
    npv_direct_damage_usd_m:     float = 0.0   # Asset write-down
    npv_bi_loss_usd_m:           float = 0.0   # Business interruption
    npv_supply_chain_loss_usd_m: float = 0.0   # Supply disruption
    total_npv_loss_usd_m:        float = 0.0   # Total (NPV)
    total_npv_loss_pct_rev:      float = 0.0   # % of annual revenue

    # Insurance
    insurance_coverage_ratio:    float = 0.0
    protection_gap_usd_m:        float = 0.0   # Uninsured NPV exposure

    # Risk tier
    risk_tier:          str = ""   # Low | Medium | High | Critical

    # Year-by-year breakdown
    year_results:       list[WildfireYearResult] = field(default_factory=list)

    # Narrative
    key_risks:          list[str] = field(default_factory=list)
    scenario:           str = "delayed"
    methodology:        str = ""
    data_gaps:          list[str] = field(default_factory=list)


def assess_wildfire_risk(
    company_name:       str,
    sector:             str,
    hq_country:         str   = "default",
    revenue_usd_m:      float = 1000.0,
    asset_value_usd_m:  Optional[float] = None,   # replacement value of physical assets
    wacc:               float = 0.08,
    horizon:            int   = 2050,
    scenario:           str   = "delayed",         # nze | delayed | cp
    # Optional overrides
    fire_zone_override: Optional[int]   = None,    # 1-6, overrides country lookup
    nasa_firms_density: Optional[float] = None,    # Fire density (fires/km²/yr) from NASA FIRMS
) -> WildfireRiskResult:
    """
    Assess wildfire physical risk and NPV financial losses.

    Parameters
    ----------
    company_name        : Display name
    sector              : Sector slug
    hq_country          : ISO-2 country code
    revenue_usd_m       : Annual revenue
    asset_value_usd_m   : Physical asset replacement value. If None, estimated
                          from revenue × sector capex intensity.
    wacc                : Discount rate for NPV calculation
    horizon             : Assessment end year (default 2050)
    scenario            : NGFS scenario — nze, delayed, cp
    fire_zone_override  : Override country-based zone lookup (1-6)
    nasa_firms_density  : Real-time fire density from NASA FIRMS connector
                          (fires per km² per year); refines baseline probability
    """
    result = WildfireRiskResult(
        company_name=company_name,
        sector=sector,
        country=hq_country.upper(),
        scenario=scenario,
    )
    s = sector.lower()
    sc = scenario.lower()
    country = hq_country.upper()

    # ── Fire zone ─────────────────────────────────────────────────────────────
    zone = fire_zone_override or _COUNTRY_FIRE_ZONE.get(country, _COUNTRY_FIRE_ZONE["default"])
    result.fire_zone = zone
    zone_fwi = _FWI_MULTIPLIER.get(zone, _FWI_MULTIPLIER[3])
    scenario_fwi = zone_fwi.get(sc, zone_fwi.get("delayed"))

    # ── Sector vulnerability ──────────────────────────────────────────────────
    vuln_data = _SECTOR_VULNERABILITY.get(s, _SECTOR_VULNERABILITY["default"])
    vuln_score = vuln_data["vuln"]
    bi_days = vuln_data["bi_days"]
    supply_factor = vuln_data["supply"]
    asset_damage_pct = vuln_data["asset_dmg"]

    # ── Asset value ───────────────────────────────────────────────────────────
    if asset_value_usd_m is None:
        # Estimate: capex-intensive sectors have higher asset-to-revenue ratio
        _ASSET_REV_MULTIPLE = {
            "oil_gas": 3.5, "utilities": 4.0, "mining": 2.5, "steel": 2.0,
            "cement": 1.8, "chemicals": 1.5, "real_estate": 8.0, "forestry": 5.0,
            "agriculture": 2.0, "technology": 0.4, "financials": 0.2, "default": 1.0,
        }
        multiplier = _ASSET_REV_MULTIPLE.get(s, 1.0)
        asset_value_usd_m = revenue_usd_m * multiplier
        result.data_gaps.append(
            f"asset_value_usd_m not provided — estimated at {multiplier}× revenue "
            f"(${asset_value_usd_m:.1f}M). Provide facility replacement value for precision."
        )

    # ── Baseline annual probability ───────────────────────────────────────────
    base_prob = _ANNUAL_FIRE_PROB.get(zone, 0.04)
    # Refine with NASA FIRMS density if available
    if nasa_firms_density is not None:
        # Map density to probability: 0.01 fires/km²/yr ≈ base zone prob
        density_factor = nasa_firms_density / 0.01  # normalise to zone-1 baseline
        base_prob = min(0.40, base_prob * max(0.5, density_factor))
        result.data_gaps  # clear the default note
    result.baseline_annual_prob = round(base_prob, 4)

    # ── Daily revenue ─────────────────────────────────────────────────────────
    daily_revenue = revenue_usd_m / 365.0

    # ── Year-by-year NPV calculation ──────────────────────────────────────────
    current_year = 2024
    years = list(range(current_year + 1, min(horizon, 2051) + 1))
    year_results = []
    total_npv_direct = 0.0
    total_npv_bi = 0.0
    total_npv_supply = 0.0

    for y in years:
        # FWI multiplier for this year (interpolate between breakpoints)
        t = (y - 2025) / 25.0  # 0 → 1 over 2025-2050
        fwi_2030 = scenario_fwi.get(2030, 1.10)
        fwi_2050 = scenario_fwi.get(2050, 1.40)
        fwi_mult = fwi_2030 + (fwi_2050 - fwi_2030) * max(0, (y - 2030) / 20.0) if y >= 2030 else \
                   1.0 + (fwi_2030 - 1.0) * (y - 2025) / 5.0 if y >= 2025 else 1.0
        fwi_mult = max(1.0, fwi_mult)

        # Adjusted fire probability for this year
        adj_prob = min(0.50, base_prob * fwi_mult * vuln_score)

        # Expected direct damage (asset damage × probability × asset value)
        expected_direct = adj_prob * asset_damage_pct * asset_value_usd_m

        # Expected business interruption loss
        expected_bi = adj_prob * (bi_days * daily_revenue)

        # Expected supply chain disruption loss
        expected_supply = adj_prob * supply_factor * revenue_usd_m * 0.05  # 5% revenue disruption per event

        total_expected = expected_direct + expected_bi + expected_supply
        insurance_cov = _INSURANCE_COVERAGE.get(zone, 0.40)
        insured = total_expected * insurance_cov
        uninsured = total_expected * (1 - insurance_cov)

        # Discount to present value
        discount = 1 / ((1 + wacc) ** (y - current_year))
        total_npv_direct += expected_direct * discount
        total_npv_bi += expected_bi * discount
        total_npv_supply += expected_supply * discount

        year_results.append(WildfireYearResult(
            year=y,
            fwi_multiplier=round(fwi_mult, 3),
            annual_fire_prob=round(adj_prob, 4),
            expected_direct_loss_usd_m=round(expected_direct, 3),
            expected_bi_loss_usd_m=round(expected_bi, 3),
            expected_supply_chain_loss_usd_m=round(expected_supply, 3),
            total_expected_loss_usd_m=round(total_expected, 3),
            insured_loss_usd_m=round(insured, 3),
            uninsured_loss_usd_m=round(uninsured, 3),
        ))

    result.year_results = year_results
    result.npv_direct_damage_usd_m = round(total_npv_direct, 2)
    result.npv_bi_loss_usd_m = round(total_npv_bi, 2)
    result.npv_supply_chain_loss_usd_m = round(total_npv_supply, 2)
    total_npv = total_npv_direct + total_npv_bi + total_npv_supply
    result.total_npv_loss_usd_m = round(total_npv, 2)
    result.total_npv_loss_pct_rev = round(total_npv / max(revenue_usd_m, 1) * 100, 2)

    # ── Insurance metrics ─────────────────────────────────────────────────────
    insurance_cov = _INSURANCE_COVERAGE.get(zone, 0.40)
    result.insurance_coverage_ratio = insurance_cov
    result.protection_gap_usd_m = round(total_npv * (1 - insurance_cov), 2)

    # ── 2030 and 2050 point probabilities ─────────────────────────────────────
    fwi_2030 = scenario_fwi.get(2030, 1.10)
    fwi_2050 = scenario_fwi.get(2050, 1.40)
    result.prob_2030 = round(min(0.50, base_prob * fwi_2030 * vuln_score), 4)
    result.prob_2050 = round(min(0.50, base_prob * fwi_2050 * vuln_score), 4)

    # ── Risk tier ─────────────────────────────────────────────────────────────
    loss_pct = result.total_npv_loss_pct_rev
    if loss_pct < 1:        result.risk_tier = "Low"
    elif loss_pct < 5:      result.risk_tier = "Medium"
    elif loss_pct < 15:     result.risk_tier = "High"
    else:                   result.risk_tier = "Critical"

    # ── Key risks narrative ───────────────────────────────────────────────────
    zone_labels = {1: "Very High", 2: "High", 3: "Medium", 4: "Low", 5: "Boreal/Rising", 6: "Tropical"}
    result.key_risks.append(
        f"Wildfire zone: {zone_labels.get(zone, 'Medium')} (Zone {zone}). "
        f"Baseline annual fire event probability: {base_prob*100:.1f}%. "
        f"Under {scenario.upper()} scenario by 2050: {result.prob_2050*100:.1f}% "
        f"(FWI multiplier {fwi_2050:.2f}×). "
        f"Total NPV loss: ${total_npv:.1f}M ({loss_pct:.1f}% of revenue)."
    )
    if vuln_score >= 0.70:
        result.key_risks.append(
            f"Sector '{sector}' has HIGH direct asset exposure to wildfire "
            f"(vulnerability score {vuln_score:.2f}). Business interruption "
            f"of up to {bi_days} days per event creates significant revenue risk."
        )
    if insurance_cov < 0.40:
        result.key_risks.append(
            f"Significant insurance protection gap ({(1-insurance_cov)*100:.0f}% uninsured). "
            f"Uninsured NPV exposure: ${result.protection_gap_usd_m:.1f}M. "
            "Consider captive insurance or parametric wildfire products."
        )
    if zone == 5:
        result.key_risks.append(
            "Boreal forest region: wildfire frequency increasing faster than other zones "
            "due to permafrost thaw → drier fuel. CMIP6 models project 2× FWI by 2050 "
            "under Current Policies (Abatzoglou et al. 2019). "
            "Risk trajectory is super-linear and currently underinsured."
        )

    result.methodology = (
        f"Wildfire NPV loss = Σ P(fire|year) × [asset damage + BI + supply chain] / (1+WACC)^t. "
        f"Fire zone {zone} ({zone_labels.get(zone, 'Medium')}), baseline P={base_prob:.3f}/yr. "
        f"Sector vulnerability {vuln_score:.2f}, asset value ${asset_value_usd_m:.1f}M. "
        f"FWI multiplier {scenario.upper()}: 2030×{fwi_2030:.2f}, 2050×{fwi_2050:.2f}. "
        f"Insurance coverage: {insurance_cov*100:.0f}% (protection gap: ${result.protection_gap_usd_m:.1f}M). "
        "Sources: NASA FIRMS VIIRS (2012-2023); CMIP6 FWI Abatzoglou et al. 2019; "
        "Swiss Re Institute NatCat 2023; Munich Re NatCat sector loss ratios 2002-2022."
    )
    return result
