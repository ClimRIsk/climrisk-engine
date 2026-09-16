"""
impact_materiality.py — CSRD ESRS E1 double materiality (impact side).

The EU Corporate Sustainability Reporting Directive (CSRD) and its first
European Sustainability Reporting Standard (ESRS E1, "Climate Change")
require companies to assess and report on TWO dimensions of materiality:

  1. Financial materiality  — how climate risks and opportunities affect the
                              company's finances (covered elsewhere in the engine).
  2. Impact materiality     — how the company's activities affect climate and
                              ecosystems (this module).

Impact materiality under EFRAG guidance
----------------------------------------
An impact is "material" for ESRS reporting when it is:
  (a) Significant in scale/scope (large contribution to climate change or
      biodiversity loss), OR
  (b) Severe in depth (irreversible / hard-to-remedy damage), OR
  (c) Widespread in geographic or temporal scope.

The EFRAG IRO (Impacts, Risks and Opportunities) framework scores each
impact dimension 1-5 and flags the impact as material if the highest
score ≥ 3 (EFRAG 2022 draft guidance, updated 2024 implementation guide).

This module quantifies four climate-relevant impact domains for ESRS E1:

  Domain E1-1 — GHG emissions (Scope 1, 2, 3)
  Domain E1-4 — Carbon removal (LULUCF, afforestation commitments)
  Domain E1-5 — Energy consumption and efficiency
  Domain E1-6 — Biodiversity and ecosystem services (via Scope 3 land use)

Sources
-------
ESRS E1 "Climate Change" (2023): https://www.efrag.org/en/projects/esrs
EFRAG IG 1 — Implementation Guidance on Materiality Assessment (2023)
IPCC AR6 WG3 Ch2 — Contribution of sectors to global temperature change
GHG Protocol Corporate Standard (2015) — Scope 1/2/3 boundaries
TNFD v1.0 (2023) — Nature-related impact metrics
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── EFRAG severity scale ──────────────────────────────────────────────────────
# Score 1 (negligible) → 5 (severe). Material if any domain ≥ 3.
_SEVERITY_LABELS = {1: "Negligible", 2: "Minor", 3: "Moderate", 4: "Significant", 5: "Severe"}

# ── Sector GHG contribution to global temperature (IPCC AR6 WG3 Ch2) ─────────
# Fraction of sectoral global GHG in 2024 emissions inventory.
# Used to contextualise company Scope 1 against global sectoral contribution.
_SECTOR_GLOBAL_FRACTION: dict[str, float] = {
    "oil_gas":         0.155,   # Extraction, processing, distribution
    "coal":            0.096,
    "utilities":       0.250,   # Electricity and heat (largest single sector)
    "cement":          0.040,
    "steel":           0.045,
    "chemicals":       0.035,
    "agriculture":     0.120,
    "aviation":        0.025,
    "shipping":        0.027,
    "automobiles":     0.060,
    "mining":          0.018,
    "real_estate":     0.060,   # Buildings Scope 1+2
    "technology":      0.015,
    "financials":      0.003,
    "default":         0.010,
}

# ── Scope 3 land-use intensity by sector (biodiversity impact proxy) ──────────
# Higher value → more land-use-related Scope 3 in value chain.
_LAND_USE_INTENSITY: dict[str, float] = {
    "agriculture":  1.00,
    "food_processing": 0.80,
    "beverages":    0.70,
    "forestry":     1.00,
    "mining":       0.50,
    "oil_gas":      0.35,
    "chemicals":    0.30,
    "real_estate":  0.40,
    "default":      0.10,
}

# ── Energy intensity proxy by sector (ESRS E1-5 disclosure) ──────────────────
# GJ per USD M revenue — sector medians from IEA Energy Efficiency 2023.
_ENERGY_INTENSITY_GJ_PER_USDM: dict[str, float] = {
    "steel":           8500,
    "cement":          7200,
    "chemicals":       4500,
    "utilities":       3000,
    "oil_gas":         2800,
    "mining":          2200,
    "agriculture":     1800,
    "manufacturing":   1200,
    "food_processing": 1000,
    "real_estate":     700,
    "aviation":        900,
    "technology":      200,
    "financials":      80,
    "default":         600,
}


@dataclass
class ImpactMaterialityResult:
    """ESRS E1 double materiality — impact side — for one company."""
    company_name:  str
    sector:        str

    # EFRAG IRO scores per domain (1-5)
    score_e1_1_ghg:          int   = 1   # GHG emissions Scope 1+2+3
    score_e1_4_carbon_removal: int = 1   # Carbon removal / LULUCF commitments
    score_e1_5_energy:       int   = 1   # Energy use and efficiency
    score_e1_6_biodiversity: int   = 1   # Biodiversity / land-use impact via S3

    # Materiality flag and max score
    is_material:             bool  = False
    max_score:               int   = 1
    dominant_impact_domain:  str   = ""

    # Quantitative supporting data
    scope1_contribution_pct_global: Optional[float] = None  # company's share of sectoral global
    energy_intensity_gj_per_usdm:  float = 0.0
    land_use_score:                 float = 0.0

    # Narrative
    material_impacts:     list[str]  = field(default_factory=list)
    disclosure_requirements: list[str] = field(default_factory=list)
    methodology:          str        = ""
    data_gaps:            list[str]  = field(default_factory=list)


def assess_impact_materiality(
    company_name:         str,
    sector:               str,
    scope1_mt_co2e:       Optional[float] = None,    # Scope 1 Mt CO2e/yr
    scope2_mt_co2e:       Optional[float] = None,    # Scope 2 Mt CO2e/yr
    scope3_mt_co2e:       Optional[float] = None,    # Total Scope 3 Mt CO2e/yr
    revenue_usd_m:        float = 1000.0,
    has_net_zero_target:  bool  = False,
    has_sbti_validated:   bool  = False,
    has_lulucf_offset:    bool  = False,    # uses nature-based removal credits
    renewable_energy_pct: Optional[float] = None,   # 0-100%
    has_eu_operations:    bool  = False,
) -> ImpactMaterialityResult:
    """
    Assess a company's material impacts on the climate system under ESRS E1.

    Parameters
    ----------
    company_name         : Display name
    sector               : Sector slug
    scope1_mt_co2e       : Verified or estimated annual Scope 1 emissions (Mt)
    scope2_mt_co2e       : Market- or location-based Scope 2 (Mt)
    scope3_mt_co2e       : Total Scope 3 (Mt)
    revenue_usd_m        : Annual revenue in USD M
    has_net_zero_target  : Has a public net-zero commitment
    has_sbti_validated   : SBTi Science-Based Target validated (1.5°C pathway)
    has_lulucf_offset    : Uses LULUCF or nature-based carbon removals
    renewable_energy_pct : Current renewable energy share (0-100)
    has_eu_operations    : Subject to mandatory CSRD reporting
    """
    result = ImpactMaterialityResult(company_name=company_name, sector=sector)
    s = sector.lower()

    # ── Domain E1-1: GHG emissions ────────────────────────────────────────────
    total_scope12 = (scope1_mt_co2e or 0.0) + (scope2_mt_co2e or 0.0)
    total_all     = total_scope12 + (scope3_mt_co2e or 0.0)

    # Contextualise against global: annual global GHG ≈ 57 Gt CO2e (2023, IPCC AR6)
    _GLOBAL_GHG_MT = 57_000.0
    if total_all > 0:
        pct_global = (total_all / _GLOBAL_GHG_MT) * 100
        result.scope1_contribution_pct_global = round(pct_global, 6)
        if total_all > 50:        e1_1 = 5  # >50 Mt: among top 200 global emitters
        elif total_all > 10:      e1_1 = 4
        elif total_all > 1:       e1_1 = 3
        elif total_all > 0.1:     e1_1 = 2
        else:                     e1_1 = 1
    else:
        # Sector-based default if no emissions data
        sec_frac = _SECTOR_GLOBAL_FRACTION.get(s, _SECTOR_GLOBAL_FRACTION["default"])
        if sec_frac > 0.10:       e1_1 = 4
        elif sec_frac > 0.04:     e1_1 = 3
        else:                     e1_1 = 2
        result.data_gaps.append(
            "GHG emissions (Scope 1/2/3) not provided — E1-1 score derived from "
            "sector-level global contribution. Provide CDP or EUTL data for precision."
        )

    # Adjust upward if no abatement plan
    if not has_net_zero_target and e1_1 >= 3:
        e1_1 = min(5, e1_1 + 1)
        result.material_impacts.append(
            "ESRS E1-1: Material GHG impact — no net-zero commitment increases "
            "impact severity. EFRAG IRO guidance classifies 'lack of strategy' as "
            "a severity multiplier."
        )
    elif has_sbti_validated and e1_1 >= 3:
        e1_1 = max(1, e1_1 - 1)   # SBTi-validated reduces assessed severity

    result.score_e1_1_ghg = e1_1

    # ── Domain E1-4: Carbon removal ───────────────────────────────────────────
    # Companies with no removal commitments and high Scope 1 → higher impact
    if total_scope12 > 10 or (e1_1 >= 4 and not has_sbti_validated):
        if has_sbti_validated and has_lulucf_offset:
            e1_4 = 2
        elif has_net_zero_target:
            e1_4 = 3
        else:
            e1_4 = 4
            result.material_impacts.append(
                "ESRS E1-4: No verified carbon removal pathway despite material Scope 1 "
                "emissions. ESRS E1 §35-37 requires disclosure of carbon removal activities "
                "and planned GHG neutralisation measures."
            )
    else:
        e1_4 = 1 if has_sbti_validated else 2

    result.score_e1_4_carbon_removal = e1_4

    # ── Domain E1-5: Energy consumption and efficiency ────────────────────────
    sec_energy_intensity = _ENERGY_INTENSITY_GJ_PER_USDM.get(s, _ENERGY_INTENSITY_GJ_PER_USDM["default"])
    result.energy_intensity_gj_per_usdm = sec_energy_intensity
    # Convert to absolute energy estimate
    energy_total_gj = sec_energy_intensity * revenue_usd_m

    if energy_total_gj > 5_000_000:   e1_5 = 5     # >5 PJ: major energy-intensive site
    elif energy_total_gj > 1_000_000: e1_5 = 4
    elif energy_total_gj > 200_000:   e1_5 = 3
    elif energy_total_gj > 50_000:    e1_5 = 2
    else:                              e1_5 = 1

    # Renewable energy adoption reduces severity
    if renewable_energy_pct is not None:
        if renewable_energy_pct >= 80 and e1_5 >= 3:
            e1_5 -= 1
        elif renewable_energy_pct < 20 and e1_5 >= 3:
            e1_5 = min(5, e1_5 + 1)
            result.material_impacts.append(
                f"ESRS E1-5: Low renewable energy share (<20%) for an energy-intensive "
                f"sector. ESRS E1 §41 requires disclosure of energy intensity and "
                "renewable proportion. EU Taxonomy alignment requires >80% renewable energy "
                "for manufacturing activities."
            )

    result.score_e1_5_energy = max(1, min(5, e1_5))

    # ── Domain E1-6: Biodiversity (Scope 3 land-use proxy) ────────────────────
    land_score = _LAND_USE_INTENSITY.get(s, _LAND_USE_INTENSITY["default"])
    result.land_use_score = land_score
    # Combine with Scope 3 magnitude
    scope3_val = scope3_mt_co2e or 0.0
    if land_score >= 0.70 or scope3_val > 50:      e1_6 = 4
    elif land_score >= 0.40 or scope3_val > 10:    e1_6 = 3
    elif land_score >= 0.20 or scope3_val > 1:     e1_6 = 2
    else:                                           e1_6 = 1

    if e1_6 >= 3:
        result.material_impacts.append(
            "ESRS E1-6 / ESRS E4: Significant land-use-related Scope 3 creates material "
            "biodiversity impact. TNFD v1.0 §4.3 and ESRS E4 require disclosure of "
            "Nature-Related Impacts via the LEAP approach (Locate, Evaluate, Assess, Prepare)."
        )

    result.score_e1_6_biodiversity = e1_6

    # ── Materiality determination ─────────────────────────────────────────────
    scores = {
        "E1-1 GHG Emissions":                 e1_1,
        "E1-4 Carbon Removal":                e1_4,
        "E1-5 Energy Consumption":            e1_5,
        "E1-6 Biodiversity / Land Use":       e1_6,
    }
    max_score = max(scores.values())
    result.max_score             = max_score
    result.is_material           = max_score >= 3
    result.dominant_impact_domain = max(scores, key=scores.get)

    # ── Disclosure requirements ───────────────────────────────────────────────
    if result.is_material or has_eu_operations:
        reqs = []
        if e1_1 >= 3:
            reqs.append(
                "ESRS E1-6: Report GHG emissions (Scope 1/2/3) with breakdown by "
                "source category. Mandatory for CSRD in-scope companies from 2025 (large) "
                "or 2026 (mid-cap)."
            )
        if e1_4 >= 3:
            reqs.append(
                "ESRS E1-7: Disclose carbon removal activities and planned neutralisation "
                "measures by 2050. SBTi Net-Zero Standard requires net-zero target by 2050 "
                "for 'Validated' status."
            )
        if e1_5 >= 3:
            reqs.append(
                "ESRS E1-5: Report total energy consumption (GJ), energy intensity, "
                "renewable energy share. EU Taxonomy requires DNSH ('Do No Significant Harm') "
                "climate mitigation criteria for sustainable finance classification."
            )
        if e1_6 >= 3:
            reqs.append(
                "ESRS E4 / TNFD v1.0: Assess and disclose biodiversity-related impacts "
                "via LEAP framework. Dual materiality required for CSRD in-scope companies "
                "from FY2025 reporting."
            )
        result.disclosure_requirements = reqs

    if not result.material_impacts and result.is_material:
        result.material_impacts.append(
            f"Company has material climate impacts (max EFRAG score: {max_score}/5 on "
            f"{result.dominant_impact_domain}). ESRS E1 mandatory disclosure triggered "
            "for CSRD in-scope entities."
        )

    result.methodology = (
        "EFRAG IRO Materiality Assessment (2023 / 2024 implementation guidance). "
        "Scored on ESRS E1 domains: E1-1 GHG (scope 1+2+3 magnitude), "
        "E1-4 Carbon Removal (net-zero pathway), E1-5 Energy (intensity × revenue), "
        "E1-6 Biodiversity (land-use intensity × Scope 3). "
        f"Material threshold: any domain ≥ 3/5 (EFRAG IG1 §4.2). "
        f"Result: {'MATERIAL' if result.is_material else 'NOT MATERIAL'} "
        f"(max={max_score}/5 on {result.dominant_impact_domain}). "
        "Sources: ESRS E1 (2023), EFRAG IG1 (2023), TNFD v1.0 (2023), IPCC AR6 WG3."
    )
    return result
