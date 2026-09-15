"""
IC Deal Brief Generator.

Takes the output of a full engine run (physical + transition + portfolio
risk scores) and produces an investment-committee-ready memo:

    - Go / No-Go recommendation with rationale
    - 3 ranked risk bullets (material → manageable → monitoring)
    - Specific covenant language
    - KPI monitoring list

This is the output layer for practitioner workflow — the engine does
the analysis; this module translates it into language a credit
committee can act on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timezone


# ── Decision thresholds ───────────────────────────────────────────────────────
_PROCEED = "PROCEED WITH CONDITIONS"
_DEFER   = "DEFER PENDING FURTHER DUE DILIGENCE"
_PASS    = "PASS — MATERIAL CLIMATE RISK"

# Score bands for composite recommendation
# All scores on 0–1 or 0–100 scales normalised internally
_PASS_THRESHOLD    = 0.72   # any composite above this → PASS
_DEFER_THRESHOLD   = 0.48   # above this → DEFER, else PROCEED


@dataclass
class DealBrief:
    company_name:       str
    generated_at:       str
    recommendation:     str   # PROCEED / DEFER / PASS
    recommendation_color: str  # green / amber / red
    executive_summary:  str
    risk_bullets:       list[dict]   # [{rank, label, detail, severity}]
    covenant_language:  list[str]
    kpi_monitoring:     list[dict]   # [{metric, frequency, threshold, action}]
    composite_score:    float
    score_breakdown:    dict


def generate_deal_brief(
    company_name: str,
    sector_key: str,
    country: str,
    # Physical risk inputs
    flood_var_pct: float = 0.0,       # % of EV at risk from flood
    slr_risk_tier: str = "negligible", # critical/high/medium/low/negligible
    biodiversity_score: float = 0.0,  # 0–1
    # Transition risk inputs
    transition_var_pct: float = 0.0,  # % revenue at risk from carbon price
    scope1_mt_co2e: float = 0.0,      # absolute Scope 1
    carbon_price_sensitivity: float = 0.0,  # USD M per USD/tCO2e
    # Portfolio / credit inputs
    ev_usd_m: float = 0.0,
    revenue_usd_m: float = 0.0,
    ltv_pct: float = 0.0,             # loan-to-value %
    tenor_years: int = 5,
    deal_type: str = "term_loan",     # term_loan | bond | equity | pe
    # Override scores if already computed
    physical_composite: Optional[float] = None,
    transition_composite: Optional[float] = None,
) -> DealBrief:
    """
    Generate an IC-ready deal brief from engine outputs.

    All percentage inputs are 0–100 (e.g., 8.5 means 8.5%).
    All score overrides are 0–1.
    """
    now = datetime.now(timezone.utc).isoformat()

    # ── Normalise inputs ──────────────────────────────────────────────────────
    flood_pct      = flood_var_pct / 100.0 if flood_var_pct > 1 else flood_var_pct
    trans_pct      = transition_var_pct / 100.0 if transition_var_pct > 1 else transition_var_pct
    ltv            = ltv_pct / 100.0 if ltv_pct > 1 else ltv_pct

    # ── Physical composite ────────────────────────────────────────────────────
    _slr_tier_score = {"critical": 1.0, "high": 0.75, "medium": 0.45,
                       "low": 0.20, "negligible": 0.05}
    slr_score = _slr_tier_score.get(slr_risk_tier.lower(), 0.10)

    if physical_composite is not None:
        phys_composite = physical_composite
    else:
        phys_composite = (
            0.50 * min(1.0, flood_pct * 5)    # 20% flood VaR → score 1.0
            + 0.30 * slr_score
            + 0.20 * min(1.0, biodiversity_score)
        )

    # ── Transition composite ──────────────────────────────────────────────────
    if transition_composite is not None:
        trans_composite = transition_composite
    else:
        # Scope 1 intensity: tCO2e per USD M revenue
        intensity = (scope1_mt_co2e * 1_000_000 / revenue_usd_m) if revenue_usd_m > 0 else 0
        intensity_score = min(1.0, intensity / 5000)  # 5000 tCO2e/M USD = score 1.0

        trans_composite = (
            0.50 * min(1.0, trans_pct * 3)     # 33% revenue at risk → score 1.0
            + 0.30 * intensity_score
            + 0.20 * min(1.0, (carbon_price_sensitivity / ev_usd_m) if ev_usd_m > 0 else 0)
        )

    # ── Composite score ───────────────────────────────────────────────────────
    # Weight physical and transition equally; tenor adds a longevity premium
    tenor_mult = 1.0 + max(0, (tenor_years - 3) * 0.03)  # +3% per year beyond 3Y
    composite  = min(1.0, (0.5 * phys_composite + 0.5 * trans_composite) * tenor_mult)

    # ── Recommendation ────────────────────────────────────────────────────────
    if composite >= _PASS_THRESHOLD:
        rec   = _PASS
        color = "red"
    elif composite >= _DEFER_THRESHOLD:
        rec   = _DEFER
        color = "amber"
    else:
        rec   = _PROCEED
        color = "green"

    # ── Executive summary ─────────────────────────────────────────────────────
    exec_summary = (
        f"{company_name} ({sector_key.replace('_', ' ').title()}, {country}) — "
        f"Composite climate risk score: {composite:.2f}/1.00. "
        f"Physical exposure: {phys_composite:.2f} | Transition exposure: {trans_composite:.2f}. "
        f"Recommendation: {rec}."
    )

    # ── Risk bullets (ranked by materiality) ─────────────────────────────────
    risk_bullets = _build_risk_bullets(
        flood_pct=flood_pct,
        slr_risk_tier=slr_risk_tier,
        biodiversity_score=biodiversity_score,
        trans_pct=trans_pct,
        scope1_mt_co2e=scope1_mt_co2e,
        carbon_price_sensitivity=carbon_price_sensitivity,
        ev_usd_m=ev_usd_m,
        phys_composite=phys_composite,
        trans_composite=trans_composite,
        sector_key=sector_key,
        country=country,
    )

    # ── Covenant language ─────────────────────────────────────────────────────
    covenants = _build_covenants(
        rec=rec,
        flood_pct=flood_pct,
        trans_pct=trans_pct,
        slr_risk_tier=slr_risk_tier,
        scope1_mt_co2e=scope1_mt_co2e,
        sector_key=sector_key,
        tenor_years=tenor_years,
        deal_type=deal_type,
        ltv=ltv,
    )

    # ── KPI monitoring ────────────────────────────────────────────────────────
    kpis = _build_kpis(
        rec=rec,
        flood_pct=flood_pct,
        trans_pct=trans_pct,
        slr_risk_tier=slr_risk_tier,
        scope1_mt_co2e=scope1_mt_co2e,
        sector_key=sector_key,
        company_name=company_name,
    )

    return DealBrief(
        company_name=company_name,
        generated_at=now,
        recommendation=rec,
        recommendation_color=color,
        executive_summary=exec_summary,
        risk_bullets=risk_bullets[:3],   # top 3 only
        covenant_language=covenants,
        kpi_monitoring=kpis,
        composite_score=round(composite, 3),
        score_breakdown={
            "physical_composite":    round(phys_composite, 3),
            "transition_composite":  round(trans_composite, 3),
            "tenor_multiplier":      round(tenor_mult, 3),
            "flood_var_pct":         round(flood_pct * 100, 2),
            "slr_risk_tier":         slr_risk_tier,
            "biodiversity_score":    round(biodiversity_score, 3),
            "transition_var_pct":    round(trans_pct * 100, 2),
        },
    )


def _build_risk_bullets(
    flood_pct, slr_risk_tier, biodiversity_score,
    trans_pct, scope1_mt_co2e, carbon_price_sensitivity,
    ev_usd_m, phys_composite, trans_composite,
    sector_key, country,
) -> list[dict]:
    bullets = []

    # Physical bullets
    if flood_pct >= 0.10:
        bullets.append({
            "severity": "material",
            "category": "Physical — Flood",
            "label":    f"Flood VaR: {flood_pct*100:.1f}% of enterprise value",
            "detail":   (
                f"Climate-adjusted flood scenario indicates {flood_pct*100:.1f}% EV at risk "
                f"under the tail event. Recommend site-level LISFLOOD analysis and review of "
                f"existing flood insurance/NATCAT cover."
            ),
        })
    elif flood_pct >= 0.04:
        bullets.append({
            "severity": "manageable",
            "category": "Physical — Flood",
            "label":    f"Elevated flood exposure ({flood_pct*100:.1f}% EV)",
            "detail":   "Moderate flood tail risk; verify asset-level protection and insurance.",
        })

    if slr_risk_tier in ("critical", "high"):
        bullets.append({
            "severity": "material",
            "category": "Physical — Sea Level Rise",
            "label":    f"SLR risk tier: {slr_risk_tier.upper()}",
            "detail":   (
                f"IPCC AR6 SSP2-4.5 projections indicate high-to-critical inundation "
                f"probability for coastal assets by 2050–2075. Structural adaptation capex "
                f"and/or relocation planning should be covenanted."
            ),
        })

    if biodiversity_score >= 0.55:
        bullets.append({
            "severity": "manageable",
            "category": "Nature — Biodiversity",
            "label":    f"Biodiversity sensitivity: {biodiversity_score:.2f}/1.00",
            "detail":   (
                f"Operations near protected areas or in high-dependency sector. "
                f"TNFD disclosure likely required by lenders' nature policy. "
                f"Include SBTN target covenant."
            ),
        })

    # Transition bullets
    if trans_pct >= 0.12:
        bullets.append({
            "severity": "material",
            "category": "Transition — Carbon Exposure",
            "label":    f"Transition VaR: {trans_pct*100:.1f}% of revenue",
            "detail":   (
                f"Under Net Zero 2050 carbon pricing, Scope 1 cost burden reaches "
                f"{trans_pct*100:.1f}% of annual revenue. Material refinancing risk "
                f"if decarbonisation capex is not committed."
            ),
        })
    elif trans_pct >= 0.05:
        bullets.append({
            "severity": "manageable",
            "category": "Transition — Carbon Exposure",
            "label":    f"Moderate transition exposure ({trans_pct*100:.1f}% revenue)",
            "detail":   "Include decarbonisation pathway KPI in facility terms.",
        })

    if scope1_mt_co2e > 1_000_000:
        bullets.append({
            "severity": "material",
            "category": "Transition — Absolute Emissions",
            "label":    f"Scope 1: {scope1_mt_co2e/1e6:.1f} Mt CO₂e/yr",
            "detail":   (
                f"Absolute emissions of {scope1_mt_co2e/1e6:.1f} Mt CO₂e place this "
                f"borrower in the top emitter category. CBAM exposure (if EU cross-border) "
                f"and ETS/carbon tax costs must be stress-tested."
            ),
        })

    # Sort by severity (material first)
    order = {"material": 0, "manageable": 1, "monitoring": 2}
    bullets.sort(key=lambda b: order.get(b["severity"], 3))

    if not bullets:
        bullets.append({
            "severity": "monitoring",
            "category": "Climate Risk",
            "label":    "Low standalone climate risk",
            "detail":   (
                f"{sector_key.replace('_',' ').title()} in {country} — no material "
                f"physical or transition risk flags. Standard annual climate monitoring "
                f"reporting sufficient."
            ),
        })

    return bullets


def _build_covenants(
    rec, flood_pct, trans_pct, slr_risk_tier, scope1_mt_co2e,
    sector_key, tenor_years, deal_type, ltv,
) -> list[str]:
    covenants = []

    # Standard climate clause (always included)
    covenants.append(
        "Climate Risk Information Covenant: Borrower shall, within 90 days of each "
        "fiscal year end, deliver to Agent a Climate Risk Report covering (i) Scope 1 "
        "and Scope 2 GHG emissions for the prior year, (ii) material climate events "
        "affecting operations, and (iii) progress against any agreed decarbonisation targets."
    )

    if flood_pct >= 0.08:
        covenants.append(
            "Flood Insurance Covenant: Borrower shall maintain property and casualty "
            "insurance with NATCAT/flood cover on all material operating assets in "
            "flood-exposed locations, with minimum insured value no less than "
            "replacement cost, and shall provide evidence of renewal within 30 days "
            "of each anniversary."
        )

    if slr_risk_tier in ("critical", "high"):
        covenants.append(
            "Coastal Adaptation Covenant: Borrower shall, within 12 months of closing, "
            "deliver to Agent an independently prepared Coastal Adaptation Plan covering "
            "assets within 10 km of the coastline, outlining estimated protection costs "
            "and a commitment schedule. Annual progress updates required."
        )

    if trans_pct >= 0.10 or scope1_mt_co2e > 500_000:
        target_yr = min(2030, 2024 + tenor_years)
        covenants.append(
            f"Decarbonisation KPI Covenant: Borrower shall achieve a minimum "
            f"{min(25, int(trans_pct * 150))}% reduction in absolute Scope 1 emissions "
            f"by {target_yr} (vs. FY2023 baseline), as verified by a third-party "
            f"assurer. Failure triggers a margin ratchet of +25 bps per annum until "
            f"the target is met or a remediation plan is agreed."
        )

    if rec == _DEFER:
        covenants.append(
            "Pre-Close Conditions Precedent: Borrower shall, prior to first drawdown, "
            "deliver (i) a Phase I Environmental & Climate Assessment from an approved "
            "assessor, (ii) evidence of board-level climate risk governance, and "
            "(iii) confirmation of alignment with the borrower's sector decarbonisation "
            "trajectory as published by the IEA Net Zero 2050 scenario."
        )

    if rec == _PASS:
        covenants.append(
            "Material Adverse Change — Climate: Transaction should not proceed in current "
            "form. If committee elects to override, a Climate Condition Package must be "
            "agreed, including: (i) maximum LTV covenant with step-down linked to "
            "emissions reductions, (ii) climate-linked springing cash sweep, and "
            "(iii) mandatory refinancing trigger if sector transition risk score exceeds "
            "threshold at next annual review."
        )

    if deal_type in ("bond", "green_bond") and scope1_mt_co2e > 100_000:
        covenants.append(
            "Use of Proceeds / Sustainability-Linked Structure: If instrument is to be "
            "marketed as sustainability-linked, Borrower must agree KPIs with Agent no "
            "later than 60 days pre-launch. KPIs must include absolute Scope 1 reduction "
            "targets consistent with a 1.5°C pathway per SBTi methodology."
        )

    return covenants


def _build_kpis(
    rec, flood_pct, trans_pct, slr_risk_tier, scope1_mt_co2e,
    sector_key, company_name,
) -> list[dict]:
    kpis = [
        {
            "metric":    "Scope 1 GHG emissions (Mt CO₂e)",
            "frequency": "Annual (fiscal year end)",
            "threshold": f"Baseline FY2023 | Target: SBTi 1.5°C trajectory",
            "action":    "Margin ratchet +25 bps if annual reduction < 5% vs prior year for 2 consecutive years",
            "source":    "Annual CDP/GRI disclosure + third-party assurance",
        },
        {
            "metric":    "Climate risk score (ClimRisk engine)",
            "frequency": "Annual re-run at covenant review",
            "threshold": "Alert if composite score increases by ≥ 0.10",
            "action":    "Trigger enhanced DD review; potential LTV covenant step-down",
            "source":    "ClimRisk engine — annual re-run with updated IPCC projections",
        },
    ]

    if flood_pct >= 0.05:
        kpis.append({
            "metric":    "Flood insurance coverage ratio",
            "frequency": "Annual at policy renewal",
            "threshold": "Minimum: replacement cost of flood-exposed assets",
            "action":    "Default event if coverage lapses; margin increase of 50 bps if under-insured",
            "source":    "Insurance certificate + independent asset valuation",
        })

    if trans_pct >= 0.08:
        kpis.append({
            "metric":    "Carbon cost as % of EBITDA",
            "frequency": "Semi-annual",
            "threshold": f"Alert at {min(15, int(trans_pct*80))}%; covenant breach at {min(25, int(trans_pct*130))}%",
            "action":    "Mandatory management presentation to credit committee",
            "source":    "Company accounts + relevant ETS/carbon tax filings",
        })

    if slr_risk_tier in ("critical", "high"):
        kpis.append({
            "metric":    "Coastal adaptation plan progress (%)",
            "frequency": "Annual",
            "threshold": "Minimum 25% of plan milestones met by Year 2",
            "action":    "Covenant cure period 90 days; if uncured, mandatory prepayment of 10% of outstanding",
            "source":    "Borrower delivery + independent coastal engineer sign-off",
        })

    if scope1_mt_co2e > 1_000_000:
        kpis.append({
            "metric":    "Science-Based Target (SBTi) validation status",
            "frequency": "Annual",
            "threshold": "SBTi target to be submitted within 24 months of close",
            "action":    "Margin ratchet +15 bps if no SBTi target filed by deadline",
            "source":    "SBTi public register",
        })

    return kpis


def deal_brief_to_dict(brief: DealBrief) -> dict:
    return {
        "company_name":       brief.company_name,
        "generated_at":       brief.generated_at,
        "recommendation":     brief.recommendation,
        "recommendation_color": brief.recommendation_color,
        "composite_score":    brief.composite_score,
        "executive_summary":  brief.executive_summary,
        "risk_bullets":       brief.risk_bullets,
        "covenant_language":  brief.covenant_language,
        "kpi_monitoring":     brief.kpi_monitoring,
        "score_breakdown":    brief.score_breakdown,
    }
