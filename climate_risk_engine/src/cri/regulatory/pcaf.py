"""
PCAF-Standard Financed Emissions.

Implements the Partnership for Carbon Accounting Financials (PCAF) Global
GHG Accounting and Reporting Standard, Part A (2022) for financed emissions
across six asset classes.

Asset classes covered
---------------------
1. Listed equity and corporate bonds
2. Business loans and unlisted equity
3. Project finance
4. Commercial real estate
5. Mortgages
6. Motor vehicle loans

Key methodology
---------------
Financed emissions = Σ  (Attribution factor_i × GHG emissions_i)

Attribution factor depends on asset class:
  - Equity / corporate bonds : outstanding amount / (EV + total debt)
  - Business loans           : outstanding loan / EV
  - Project finance          : outstanding loan / total project value
  - Real estate              : outstanding mortgage / property value

Data quality score (1 = best, 5 = worst) per PCAF Standard Table 1:
  1 : Verified/audited reported Scope 1+2+3
  2 : Unaudited reported Scope 1+2 (Scope 3 estimated)
  3 : Third-party data (e.g. CDP, Bloomberg ESG)
  4 : Physical/economic intensity-based model
  5 : Revenue-based proxy (sector average)

References
----------
PCAF Global GHG Accounting and Reporting Standard, Part A (2022).
  https://carbonaccountingfinancials.com/standard#the-global-ghg-accounting-standard
TCFD Recommendations (2023 update).
Science Based Targets initiative (SBTi) Financial Sector Guidance.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PCAFAssetClass(str, Enum):
    LISTED_EQUITY_BONDS   = "listed_equity_bonds"
    BUSINESS_LOANS        = "business_loans"
    PROJECT_FINANCE       = "project_finance"
    COMMERCIAL_REAL_ESTATE = "commercial_real_estate"
    MORTGAGES             = "mortgages"
    MOTOR_VEHICLE_LOANS   = "motor_vehicle_loans"


@dataclass
class PCAFResult:
    """PCAF financed emissions result for one exposure."""
    counterparty_id: str
    counterparty_name: str
    asset_class: PCAFAssetClass
    outstanding_amount_eur_m: float
    attribution_factor: float               # dimensionless [0,1]
    scope1_absolute_tco2e: float
    scope2_absolute_tco2e: float
    scope3_absolute_tco2e: float
    financed_scope12_tco2e: float
    financed_scope3_tco2e: float
    financed_total_tco2e: float
    financed_intensity_tco2e_per_eur_m: float  # per EUR m outstanding
    data_quality_scope12: int               # 1-5
    data_quality_scope3: int
    weighted_data_quality: float            # portfolio-weighted avg
    methodology_note: str


# ── Sector GHG intensity tables (tCO2e per EUR m revenue) ────────────────────
# Source: IEA Tracking Clean Energy Progress 2023 + Eurostat env_ac_aibrid

_SCOPE12_INTENSITY_EUR: dict[str, float] = {
    "oil_gas":       2800.0,
    "coal":          4200.0,
    "chemicals":     1800.0,
    "cement":        3500.0,
    "metals_mining": 2200.0,
    "mining":        1600.0,
    "energy":         900.0,
    "utilities":      850.0,
    "industrials":    520.0,
    "agriculture":    780.0,
    "transport":      420.0,
    "shipping":       650.0,
    "aviation":       880.0,
    "food_beverage":  310.0,
    "automotive":     380.0,
    "construction":   290.0,
    "real_estate":    180.0,
    "technology":      65.0,
    "financials":      28.0,
    "healthcare":     120.0,
    "consumer":       190.0,
    "default":        300.0,
}

_SCOPE3_MULT: dict[str, float] = {
    "oil_gas": 8.5, "coal": 9.2, "chemicals": 3.1, "cement": 1.8,
    "metals_mining": 2.4, "mining": 2.1, "energy": 2.0, "utilities": 1.9,
    "industrials": 3.8, "agriculture": 4.2, "transport": 2.9,
    "shipping": 2.6, "aviation": 2.2, "food_beverage": 5.1,
    "automotive": 6.8, "construction": 3.5, "real_estate": 2.8,
    "technology": 4.5, "financials": 1.5, "healthcare": 3.2,
    "consumer": 4.8, "default": 3.0,
}


def _estimate_emissions(
    sector_key: str,
    revenue_eur_m: float,
    scope1_reported: Optional[float],
    scope2_reported: Optional[float],
    scope3_reported: Optional[float],
) -> tuple[float, float, float, int, int]:
    """
    Return (scope1, scope2, scope3, dq_s12, dq_s3).
    Data quality: 1=audited, 2=unaudited reported, 3=third-party, 4=model, 5=proxy.
    """
    s12_intensity = _SCOPE12_INTENSITY_EUR.get(sector_key, _SCOPE12_INTENSITY_EUR["default"])
    s3_mult = _SCOPE3_MULT.get(sector_key, _SCOPE3_MULT["default"])

    if scope1_reported is not None and scope2_reported is not None:
        s1, s2 = scope1_reported, scope2_reported
        dq_s12 = 2  # unaudited reported (PCAF DQ 2)
    else:
        s1 = revenue_eur_m * s12_intensity * 0.4
        s2 = revenue_eur_m * s12_intensity * 0.6
        dq_s12 = 5  # revenue proxy (PCAF DQ 5)

    if scope3_reported is not None:
        s3 = scope3_reported
        dq_s3 = 2
    else:
        s3 = (s1 + s2) * s3_mult
        dq_s3 = 5 if dq_s12 == 5 else 4

    return s1, s2, s3, dq_s12, dq_s3


def compute_financed_emissions(
    counterparty_id: str,
    counterparty_name: str,
    sector_key: str,
    asset_class: PCAFAssetClass,
    outstanding_amount_eur_m: float,
    # Valuation denominators
    enterprise_value_eur_m: Optional[float] = None,
    total_debt_eur_m: Optional[float] = None,
    total_project_value_eur_m: Optional[float] = None,
    property_value_eur_m: Optional[float] = None,
    # Revenue for intensity-based estimation
    revenue_eur_m: float = 100.0,
    # Reported emissions (override sector proxy if provided)
    scope1_tco2e: Optional[float] = None,
    scope2_tco2e: Optional[float] = None,
    scope3_tco2e: Optional[float] = None,
) -> PCAFResult:
    """
    Compute PCAF-standard financed emissions for one counterparty / exposure.

    Parameters
    ----------
    counterparty_id, counterparty_name : str
    sector_key : str
        CRI sector key.
    asset_class : PCAFAssetClass
        PCAF asset class determines attribution factor formula.
    outstanding_amount_eur_m : float
        Outstanding loan / bond / equity position in EUR millions.
    enterprise_value_eur_m : float, optional
        EV incl. cash (EVIC = Equity market cap + total debt). Required for
        listed equity/bonds and business loans.
    total_debt_eur_m : float, optional
        Total borrowings. Used in EVIC calculation for equity/bonds.
    total_project_value_eur_m : float, optional
        Total project cost for project finance asset class.
    property_value_eur_m : float, optional
        Property value for real estate / mortgage asset classes.
    revenue_eur_m : float
        Annual revenue — base for sector intensity estimation.
    scope1/2/3_tco2e : float, optional
        Reported absolute emissions — improves data quality score.

    Returns
    -------
    PCAFResult
    """
    s1, s2, s3, dq_s12, dq_s3 = _estimate_emissions(
        sector_key, revenue_eur_m, scope1_tco2e, scope2_tco2e, scope3_tco2e
    )

    # ── Attribution factor by asset class ─────────────────────────────────────
    if asset_class == PCAFAssetClass.LISTED_EQUITY_BONDS:
        # PCAF: outstanding amount / (equity market cap + total debt)
        evic = (enterprise_value_eur_m or revenue_eur_m * 2.5) + (total_debt_eur_m or 0.0)
        attr = outstanding_amount_eur_m / max(evic, 1.0)
        note = f"PCAF Part A §3.1: attr = {outstanding_amount_eur_m:.1f}m / EVIC {evic:.1f}m"

    elif asset_class == PCAFAssetClass.BUSINESS_LOANS:
        # PCAF: outstanding loan / enterprise value (EV excl. cash)
        ev = enterprise_value_eur_m or revenue_eur_m * 2.0
        attr = outstanding_amount_eur_m / max(ev, 1.0)
        note = f"PCAF Part A §3.2: attr = {outstanding_amount_eur_m:.1f}m / EV {ev:.1f}m"

    elif asset_class == PCAFAssetClass.PROJECT_FINANCE:
        # PCAF: outstanding loan / total project value
        tpv = total_project_value_eur_m or outstanding_amount_eur_m * 1.5
        attr = outstanding_amount_eur_m / max(tpv, 1.0)
        note = f"PCAF Part A §3.3: attr = {outstanding_amount_eur_m:.1f}m / TPV {tpv:.1f}m"

    elif asset_class in (PCAFAssetClass.COMMERCIAL_REAL_ESTATE, PCAFAssetClass.MORTGAGES):
        # PCAF: outstanding loan / property value
        pv = property_value_eur_m or outstanding_amount_eur_m * 1.4
        attr = outstanding_amount_eur_m / max(pv, 1.0)
        note = f"PCAF Part A §3.4/3.5: attr = {outstanding_amount_eur_m:.1f}m / property {pv:.1f}m"

    elif asset_class == PCAFAssetClass.MOTOR_VEHICLE_LOANS:
        # PCAF: outstanding loan / vehicle value (proxy: loan × 1.1)
        vehicle_val = outstanding_amount_eur_m * 1.1
        attr = outstanding_amount_eur_m / max(vehicle_val, 1.0)
        note = f"PCAF Part A §3.6: attr = {outstanding_amount_eur_m:.1f}m / vehicle value {vehicle_val:.1f}m"

    else:
        attr = outstanding_amount_eur_m / max(revenue_eur_m * 2.0, 1.0)
        note = "Attribution factor via EV proxy"

    attr = min(1.0, max(0.0, attr))

    # ── Financed emissions ────────────────────────────────────────────────────
    fin_s12 = (s1 + s2) * attr
    fin_s3  = s3 * attr
    fin_tot = fin_s12 + fin_s3
    fin_intensity = fin_tot / max(outstanding_amount_eur_m, 1.0)

    # Weighted DQ (Scope 3 typically 60-80% of total)
    s3_share = fin_s3 / max(fin_tot, 1.0)
    wdq = dq_s12 * (1 - s3_share) + dq_s3 * s3_share

    return PCAFResult(
        counterparty_id=counterparty_id,
        counterparty_name=counterparty_name,
        asset_class=asset_class,
        outstanding_amount_eur_m=outstanding_amount_eur_m,
        attribution_factor=round(attr, 6),
        scope1_absolute_tco2e=round(s1, 0),
        scope2_absolute_tco2e=round(s2, 0),
        scope3_absolute_tco2e=round(s3, 0),
        financed_scope12_tco2e=round(fin_s12, 0),
        financed_scope3_tco2e=round(fin_s3, 0),
        financed_total_tco2e=round(fin_tot, 0),
        financed_intensity_tco2e_per_eur_m=round(fin_intensity, 2),
        data_quality_scope12=dq_s12,
        data_quality_scope3=dq_s3,
        weighted_data_quality=round(wdq, 2),
        methodology_note=note,
    )


def pcaf_portfolio_summary(results: list[PCAFResult]) -> dict:
    """
    Aggregate PCAF results across a portfolio of exposures.

    Returns portfolio-level financed emissions totals, intensity,
    and weighted average data quality score.
    """
    total_outstanding = sum(r.outstanding_amount_eur_m for r in results)
    total_fin_s12 = sum(r.financed_scope12_tco2e for r in results)
    total_fin_s3  = sum(r.financed_scope3_tco2e for r in results)
    total_fin_tot = sum(r.financed_total_tco2e for r in results)

    # Portfolio-weighted average data quality
    wdq = (
        sum(r.weighted_data_quality * r.outstanding_amount_eur_m for r in results)
        / max(total_outstanding, 1.0)
    )

    # By asset class
    by_class: dict[str, dict] = {}
    for r in results:
        k = r.asset_class.value
        if k not in by_class:
            by_class[k] = {"outstanding_eur_m": 0.0, "financed_tco2e": 0.0, "count": 0}
        by_class[k]["outstanding_eur_m"] += r.outstanding_amount_eur_m
        by_class[k]["financed_tco2e"]    += r.financed_total_tco2e
        by_class[k]["count"]             += 1

    return {
        "portfolio_financed_emissions": {
            "scope_1_2_tco2e":   round(total_fin_s12, 0),
            "scope_3_tco2e":     round(total_fin_s3, 0),
            "total_tco2e":       round(total_fin_tot, 0),
            "intensity_tco2e_per_eur_m_outstanding": round(
                total_fin_tot / max(total_outstanding, 1.0), 2
            ),
        },
        "total_outstanding_eur_m":     round(total_outstanding, 1),
        "weighted_avg_data_quality":   round(wdq, 2),
        "data_quality_label": _dq_label(round(wdq)),
        "by_asset_class": {
            k: {
                "outstanding_eur_m": round(v["outstanding_eur_m"], 1),
                "financed_tco2e":    round(v["financed_tco2e"], 0),
                "count":             v["count"],
            }
            for k, v in by_class.items()
        },
        "methodology": "PCAF Global GHG Accounting and Reporting Standard, Part A (2022)",
    }


def pcaf_to_dict(result: PCAFResult) -> dict:
    return {
        "counterparty_id":       result.counterparty_id,
        "counterparty_name":     result.counterparty_name,
        "asset_class":           result.asset_class.value,
        "outstanding_eur_m":     result.outstanding_amount_eur_m,
        "attribution_factor":    result.attribution_factor,
        "absolute_emissions": {
            "scope_1_tco2e": result.scope1_absolute_tco2e,
            "scope_2_tco2e": result.scope2_absolute_tco2e,
            "scope_3_tco2e": result.scope3_absolute_tco2e,
        },
        "financed_emissions": {
            "scope_1_2_tco2e":   result.financed_scope12_tco2e,
            "scope_3_tco2e":     result.financed_scope3_tco2e,
            "total_tco2e":       result.financed_total_tco2e,
            "intensity_tco2e_per_eur_m": result.financed_intensity_tco2e_per_eur_m,
        },
        "data_quality": {
            "scope_1_2_score":   result.data_quality_scope12,
            "scope_3_score":     result.data_quality_scope3,
            "weighted_score":    result.weighted_data_quality,
            "label":             _dq_label(round(result.weighted_data_quality)),
        },
        "methodology_note": result.methodology_note,
    }


def _dq_label(score: int) -> str:
    return {
        1: "Audited reported",
        2: "Unaudited reported",
        3: "Third-party data",
        4: "Intensity model",
        5: "Revenue proxy",
    }.get(score, "Unknown")
