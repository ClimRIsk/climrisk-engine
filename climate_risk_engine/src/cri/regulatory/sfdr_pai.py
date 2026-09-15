"""
SFDR Principal Adverse Impact (PAI) Indicators — Article 7 / RTS Annex I.

Computes all 18 mandatory PAI indicators for corporate investee companies
from ClimRisk engine inputs. Where underlying data is unavailable, the
indicator is flagged with a data-quality score (1 = estimated, 5 = reported)
consistent with PCAF methodology.

References
----------
SFDR Delegated Regulation (EU) 2022/1288, Annex I Table 1 (mandatory PAIs)
ESMA Q&A on SFDR (2023 consolidated)
PCAF Global GHG Accounting and Reporting Standard, Part A (2022)

Mandatory PAI indicators (18 total)
------------------------------------
Climate & environment:
  1.  GHG emissions (Scope 1+2+3, tCO2e)
  2.  Carbon footprint (tCO2e / EUR m invested)
  3.  GHG intensity of investee (tCO2e / EUR m revenue)
  4.  Exposure to fossil fuel companies (% portfolio)
  5.  Share non-renewable energy consumption (%)
  6.  Energy consumption intensity — high-impact sectors (MWh / EUR m revenue)
  7.  Activities negatively affecting biodiversity-sensitive areas (Y/N)
  8.  Emissions to water (tonne / EUR m revenue)
  9.  Hazardous waste ratio (tonne / EUR m revenue)

Social & governance:
  10. UNGC / OECD violations (Y/N)
  11. Lack of UNGC / OECD compliance policy (Y/N)
  12. Unadjusted gender pay gap (%)
  13. Board gender diversity (% female board members)
  14. Exposure to controversial weapons (Y/N)

Physical risk (climate-specific):
  15. Real estate energy performance (share EPC A/B)        [real estate only]
  16. Real estate fossil fuel exposure                       [real estate only]
  17. Companies with carbon reduction targets (Y/N)
  18. Exposure to fossil fuel sector (% revenue)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── Sector classification helpers ────────────────────────────────────────────

_FOSSIL_FUEL_SECTORS: frozenset[str] = frozenset({
    "oil_gas", "coal", "mining", "chemicals",
})
_CONTROVERSIAL_WEAPONS_SECTORS: frozenset[str] = frozenset({
    # Proxy: no CRI sector maps cleanly; flag as requires_verification
})
_HIGH_IMPACT_SECTORS: frozenset[str] = frozenset({
    # NACE sections A, B, C, D, E, F, G, H, L per SFDR RTS Annex I footnote 5
    "agriculture", "mining", "industrials", "energy", "utilities",
    "construction", "transport", "shipping", "aviation", "real_estate",
    "chemicals", "oil_gas", "metals_mining", "cement", "food_beverage",
})
_BIODIVERSITY_SENSITIVE_SECTORS: frozenset[str] = frozenset({
    "agriculture", "mining", "metals_mining", "cement", "chemicals", "oil_gas",
})

# Estimated energy intensity by sector (MWh per EUR m revenue, 2023 averages)
_ENERGY_INTENSITY_MWH_PER_EURAM: dict[str, float] = {
    "agriculture":    850.0,
    "mining":        3200.0,
    "metals_mining": 4100.0,
    "cement":        5500.0,
    "chemicals":     2800.0,
    "oil_gas":       1900.0,
    "energy":        1200.0,
    "utilities":     1100.0,
    "industrials":    620.0,
    "construction":   310.0,
    "transport":      480.0,
    "shipping":       690.0,
    "aviation":       920.0,
    "food_beverage":  540.0,
    "automotive":     430.0,
    "real_estate":    280.0,
    "technology":      95.0,
    "financials":      42.0,
    "healthcare":     185.0,
    "consumer":       210.0,
    "default":        300.0,
}

# Non-renewable energy share by sector (%, estimated)
_NON_RENEWABLE_PCT: dict[str, float] = {
    "oil_gas":       92.0,
    "coal":          98.0,
    "chemicals":     78.0,
    "cement":        85.0,
    "metals_mining": 72.0,
    "mining":        75.0,
    "industrials":   65.0,
    "energy":        45.0,   # mix — depends on utility type
    "utilities":     50.0,
    "transport":     88.0,
    "shipping":      90.0,
    "aviation":      96.0,
    "agriculture":   60.0,
    "construction":  70.0,
    "food_beverage": 55.0,
    "automotive":    62.0,
    "real_estate":   58.0,
    "technology":    35.0,
    "financials":    30.0,
    "healthcare":    45.0,
    "consumer":      50.0,
    "default":       60.0,
}


@dataclass
class PAIIndicator:
    """Single PAI indicator result."""
    pai_id: int
    name: str
    value: Optional[float]          # None = not applicable
    unit: str
    data_quality: int               # 1 (estimated/proxy) → 5 (audited reported)
    flag: Optional[bool] = None     # for Y/N indicators
    note: str = ""
    requires_additional_data: bool = False


@dataclass
class SFDRPAIReport:
    """Full SFDR PAI report for one investee company."""
    company_id: str
    company_name: str
    sector: str
    reference_period: str           # e.g. "2024"
    currency: str = "EUR"
    indicators: list[PAIIndicator] = field(default_factory=list)
    data_limitations: list[str] = field(default_factory=list)
    methodology_note: str = (
        "PAI indicators derived from ClimRisk engine inputs. "
        "Emissions data from company disclosure where available; "
        "sector proxies applied where not (data quality 1-2). "
        "Physical risk indicators from WRI Aqueduct + JRC flood maps. "
        "Reference: SFDR Delegated Regulation (EU) 2022/1288 Annex I Table 1."
    )


def compute_sfdr_pai(
    company_id: str,
    company_name: str,
    sector_key: str,
    revenue_eur_m: float,
    ev_eur_m: float,
    outstanding_loan_eur_m: float,          # investor's exposure
    scope1_tco2e: Optional[float] = None,
    scope2_tco2e: Optional[float] = None,
    scope3_tco2e: Optional[float] = None,
    has_carbon_reduction_target: bool = False,
    board_female_pct: Optional[float] = None,
    gender_pay_gap_pct: Optional[float] = None,
    ungc_compliant: Optional[bool] = None,
    reference_period: str = "2024",
) -> SFDRPAIReport:
    """
    Compute SFDR mandatory PAI indicators for one investee company.

    Parameters
    ----------
    company_id, company_name : str
    sector_key : str
        CRI internal sector key.
    revenue_eur_m : float
        Annual revenue in EUR millions.
    ev_eur_m : float
        Enterprise value in EUR millions (used as denominator for PAI 2/3).
    outstanding_loan_eur_m : float
        Investor's outstanding exposure in EUR millions (attribution factor).
    scope1/2/3_tco2e : float, optional
        Absolute GHG emissions in tonnes CO2e. Estimated from sector intensity
        if not provided (data quality 1).
    has_carbon_reduction_target : bool
        Whether the company has a published carbon reduction target (SBTi or equivalent).
    board_female_pct : float, optional
        % female board members. If None, flagged as requires additional data.
    gender_pay_gap_pct : float, optional
        Unadjusted gender pay gap %. If None, flagged as requires additional data.
    ungc_compliant : bool, optional
        Whether the company is UNGC signatory and compliant.
    reference_period : str
        Reporting year, e.g. "2024".

    Returns
    -------
    SFDRPAIReport
    """
    indicators: list[PAIIndicator] = []
    data_limitations: list[str] = []

    # ── Estimate emissions if not provided ────────────────────────────────────
    # Sector-average GHG intensity (tCO2e per EUR m revenue) from IEA/Eurostat
    _SCOPE12_INTENSITY: dict[str, float] = {
        "oil_gas": 2800.0, "coal": 4200.0, "chemicals": 1800.0,
        "cement": 3500.0, "metals_mining": 2200.0, "mining": 1600.0,
        "energy": 900.0, "utilities": 850.0, "industrials": 520.0,
        "agriculture": 780.0, "transport": 420.0, "shipping": 650.0,
        "aviation": 880.0, "food_beverage": 310.0, "automotive": 380.0,
        "construction": 290.0, "real_estate": 180.0, "technology": 65.0,
        "financials": 28.0, "healthcare": 120.0, "consumer": 190.0,
        "default": 300.0,
    }
    _SCOPE3_MULTIPLIER: dict[str, float] = {
        "oil_gas": 8.5, "coal": 9.2, "chemicals": 3.1, "cement": 1.8,
        "metals_mining": 2.4, "mining": 2.1, "energy": 2.0, "utilities": 1.9,
        "industrials": 3.8, "agriculture": 4.2, "transport": 2.9,
        "shipping": 2.6, "aviation": 2.2, "food_beverage": 5.1,
        "automotive": 6.8, "construction": 3.5, "real_estate": 2.8,
        "technology": 4.5, "financials": 1.5, "healthcare": 3.2,
        "consumer": 4.8, "default": 3.0,
    }

    s12_intensity = _SCOPE12_INTENSITY.get(sector_key, _SCOPE12_INTENSITY["default"])
    s3_mult = _SCOPE3_MULTIPLIER.get(sector_key, _SCOPE3_MULTIPLIER["default"])

    s1_reported = scope1_tco2e is not None
    s3_reported = scope3_tco2e is not None

    s1 = scope1_tco2e if s1_reported else revenue_eur_m * s12_intensity * 0.4
    s2 = scope2_tco2e if scope2_tco2e is not None else revenue_eur_m * s12_intensity * 0.6
    s3 = scope3_tco2e if s3_reported else (s1 + s2) * s3_mult
    total_ghg = s1 + s2 + s3

    emissions_dq = 5 if (s1_reported and s3_reported) else (3 if s1_reported else 1)
    if not s1_reported:
        data_limitations.append("Scope 1/2 estimated from sector intensity — data quality 1")
    if not s3_reported:
        data_limitations.append("Scope 3 estimated from Scope 1+2 multiplier — data quality 2")

    # Attribution factor = outstanding loan / EV (PCAF method)
    attribution = min(1.0, outstanding_loan_eur_m / max(ev_eur_m, 1.0))

    # ── PAI 1: GHG emissions ─────────────────────────────────────────────────
    indicators.append(PAIIndicator(
        pai_id=1,
        name="GHG emissions (Scope 1 + 2 + 3)",
        value=round(total_ghg * attribution, 0),
        unit="tCO2e (attributed)",
        data_quality=emissions_dq,
        note=f"Attribution factor: {attribution:.4f} (loan {outstanding_loan_eur_m}m / EV {ev_eur_m}m)",
    ))

    # ── PAI 2: Carbon footprint ───────────────────────────────────────────────
    carbon_footprint = (total_ghg * attribution) / max(outstanding_loan_eur_m, 1.0) * 1e6
    indicators.append(PAIIndicator(
        pai_id=2,
        name="Carbon footprint",
        value=round(carbon_footprint, 2),
        unit="tCO2e / EUR m invested",
        data_quality=emissions_dq,
    ))

    # ── PAI 3: GHG intensity of investee ─────────────────────────────────────
    ghg_intensity = total_ghg / max(revenue_eur_m, 1.0)
    indicators.append(PAIIndicator(
        pai_id=3,
        name="GHG intensity of investee companies",
        value=round(ghg_intensity, 1),
        unit="tCO2e / EUR m revenue",
        data_quality=emissions_dq,
    ))

    # ── PAI 4: Exposure to fossil fuel companies ──────────────────────────────
    is_fossil = sector_key in _FOSSIL_FUEL_SECTORS
    indicators.append(PAIIndicator(
        pai_id=4,
        name="Exposure to companies active in fossil fuel sector",
        value=100.0 if is_fossil else 0.0,
        unit="% of investment",
        data_quality=4,
        flag=is_fossil,
        note="Based on primary sector classification per SFDR RTS Art. 2(1)",
    ))

    # ── PAI 5: Share of non-renewable energy ─────────────────────────────────
    non_renew = _NON_RENEWABLE_PCT.get(sector_key, _NON_RENEWABLE_PCT["default"])
    indicators.append(PAIIndicator(
        pai_id=5,
        name="Share of non-renewable energy consumption and production",
        value=round(non_renew, 1),
        unit="%",
        data_quality=2,
        note="Sector proxy — replace with reported energy mix where available",
    ))

    # ── PAI 6: Energy consumption intensity (high-impact sectors) ────────────
    if sector_key in _HIGH_IMPACT_SECTORS:
        energy_intensity = _ENERGY_INTENSITY_MWH_PER_EURAM.get(
            sector_key, _ENERGY_INTENSITY_MWH_PER_EURAM["default"]
        )
        indicators.append(PAIIndicator(
            pai_id=6,
            name="Energy consumption intensity per high-impact climate sector",
            value=round(energy_intensity, 0),
            unit="MWh / EUR m revenue",
            data_quality=2,
            note="Sector proxy from IEA/Eurostat 2023 intensity tables",
        ))
    else:
        indicators.append(PAIIndicator(
            pai_id=6,
            name="Energy consumption intensity per high-impact climate sector",
            value=None,
            unit="MWh / EUR m revenue",
            data_quality=5,
            note="Not applicable — company not in high-impact climate sector per SFDR RTS",
        ))

    # ── PAI 7: Activities negatively affecting biodiversity-sensitive areas ───
    biodiversity_risk = sector_key in _BIODIVERSITY_SENSITIVE_SECTORS
    indicators.append(PAIIndicator(
        pai_id=7,
        name="Activities negatively affecting biodiversity-sensitive areas",
        value=None,
        unit="Y/N",
        data_quality=2,
        flag=biodiversity_risk,
        note="Proxy from sector classification; asset-level KBA/protected area screening recommended",
        requires_additional_data=True,
    ))

    # ── PAI 8: Emissions to water ─────────────────────────────────────────────
    # Proxy: chemical/mining sectors have material water emissions
    water_emitting = sector_key in {"chemicals", "oil_gas", "mining", "metals_mining",
                                     "agriculture", "food_beverage"}
    indicators.append(PAIIndicator(
        pai_id=8,
        name="Emissions to water",
        value=None,
        unit="tonne / EUR m revenue",
        data_quality=1,
        flag=water_emitting,
        note="Requires company-reported water discharge data; sector proxy flag only",
        requires_additional_data=True,
    ))

    # ── PAI 9: Hazardous waste ratio ──────────────────────────────────────────
    hazwaste_sector = sector_key in {"chemicals", "oil_gas", "mining", "metals_mining",
                                      "pharmaceuticals", "automotive"}
    indicators.append(PAIIndicator(
        pai_id=9,
        name="Hazardous waste and radioactive waste ratio",
        value=None,
        unit="tonne / EUR m revenue",
        data_quality=1,
        flag=hazwaste_sector,
        note="Requires company waste disclosure; sector proxy flag only",
        requires_additional_data=True,
    ))

    # ── PAI 10: UNGC / OECD violations ───────────────────────────────────────
    ungc_violation = False if ungc_compliant else None
    indicators.append(PAIIndicator(
        pai_id=10,
        name="Violations of UN Global Compact principles and OECD Guidelines",
        value=None,
        unit="Y/N",
        data_quality=2 if ungc_compliant is not None else 1,
        flag=ungc_violation,
        note="Based on UNGC participant status; controversy screening recommended",
        requires_additional_data=(ungc_compliant is None),
    ))

    # ── PAI 11: Lack of UNGC / OECD processes ────────────────────────────────
    no_policy = not bool(ungc_compliant)
    indicators.append(PAIIndicator(
        pai_id=11,
        name="Lack of processes and compliance mechanisms to monitor UNGC/OECD",
        value=None,
        unit="Y/N",
        data_quality=2 if ungc_compliant is not None else 1,
        flag=no_policy,
        note="Inferred from UNGC signatory status",
    ))

    # ── PAI 12: Gender pay gap ────────────────────────────────────────────────
    indicators.append(PAIIndicator(
        pai_id=12,
        name="Unadjusted gender pay gap",
        value=gender_pay_gap_pct,
        unit="%",
        data_quality=5 if gender_pay_gap_pct is not None else 1,
        note="Requires company-reported pay gap data" if gender_pay_gap_pct is None else "",
        requires_additional_data=(gender_pay_gap_pct is None),
    ))

    # ── PAI 13: Board gender diversity ───────────────────────────────────────
    indicators.append(PAIIndicator(
        pai_id=13,
        name="Board gender diversity",
        value=board_female_pct,
        unit="% female board members",
        data_quality=5 if board_female_pct is not None else 1,
        note="Requires board composition data from annual report" if board_female_pct is None else "",
        requires_additional_data=(board_female_pct is None),
    ))

    # ── PAI 14: Controversial weapons ────────────────────────────────────────
    controversial = sector_key in _CONTROVERSIAL_WEAPONS_SECTORS
    indicators.append(PAIIndicator(
        pai_id=14,
        name="Exposure to controversial weapons (antipersonnel mines, cluster munitions, etc.)",
        value=None,
        unit="Y/N",
        data_quality=2,
        flag=controversial,
        note="Requires controversy screening against SIPRI / RepRisk lists",
        requires_additional_data=True,
    ))

    # ── PAI 15-16: Real estate (not applicable for corporate investees) ───────
    for pid, name in [
        (15, "Real estate assets — energy performance (EPC label share)"),
        (16, "Real estate assets — exposure to fossil fuel heating systems"),
    ]:
        indicators.append(PAIIndicator(
            pai_id=pid,
            name=name,
            value=None,
            unit="N/A",
            data_quality=5,
            note="Applicable to real estate funds only — not applicable for corporate investee",
        ))

    # ── PAI 17: Carbon reduction targets ─────────────────────────────────────
    indicators.append(PAIIndicator(
        pai_id=17,
        name="Companies without carbon emission reduction initiatives",
        value=None,
        unit="Y/N",
        data_quality=4 if has_carbon_reduction_target else 2,
        flag=not has_carbon_reduction_target,
        note=("SBTi or equivalent target confirmed" if has_carbon_reduction_target
              else "No verified carbon reduction target found"),
    ))

    # ── PAI 18: Fossil fuel revenue exposure ─────────────────────────────────
    fossil_revenue_pct = 100.0 if sector_key in _FOSSIL_FUEL_SECTORS else 0.0
    indicators.append(PAIIndicator(
        pai_id=18,
        name="Exposure to fossil fuels through real assets",
        value=round(fossil_revenue_pct, 1),
        unit="% revenue from fossil fuel activities",
        data_quality=3,
        flag=fossil_revenue_pct > 0,
        note="Estimated from primary sector; refine with revenue segment disclosure",
    ))

    return SFDRPAIReport(
        company_id=company_id,
        company_name=company_name,
        sector=sector_key,
        reference_period=reference_period,
        indicators=indicators,
        data_limitations=data_limitations,
    )


def pai_to_dict(report: SFDRPAIReport) -> dict:
    """Serialise SFDRPAIReport to a JSON-ready dict."""
    return {
        "company_id":       report.company_id,
        "company_name":     report.company_name,
        "sector":           report.sector,
        "reference_period": report.reference_period,
        "currency":         report.currency,
        "methodology_note": report.methodology_note,
        "data_limitations": report.data_limitations,
        "indicators": [
            {
                "pai_id":                  i.pai_id,
                "name":                    i.name,
                "value":                   i.value,
                "unit":                    i.unit,
                "flag":                    i.flag,
                "data_quality_score":      i.data_quality,
                "data_quality_label":      _dq_label(i.data_quality),
                "note":                    i.note,
                "requires_additional_data": i.requires_additional_data,
            }
            for i in report.indicators
        ],
        "data_quality_summary": {
            "avg_score": round(
                sum(i.data_quality for i in report.indicators) / len(report.indicators), 2
            ),
            "indicators_requiring_data": sum(
                1 for i in report.indicators if i.requires_additional_data
            ),
        },
    }


def _dq_label(score: int) -> str:
    return {1: "Estimated/proxy", 2: "Model-based", 3: "Third-party",
            4: "Unaudited reported", 5: "Audited reported"}.get(score, "Unknown")
