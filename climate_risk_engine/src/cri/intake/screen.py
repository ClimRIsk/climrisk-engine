"""
Fast Climate Risk Triage Screening.

Takes {company_name, sector, country} and returns a traffic-light
(RED / AMBER / GREEN) with a one-sentence rationale in < 100 ms —
no engine run, no lat/lon, no financial data required.

Use this to triage a pipeline of 30+ deals before deciding which 5
to send through the full ClimRisk engine.

Scoring
-------
Physical risk score (0–1):
    Region × sector hazard exposure, calibrated from IPCC AR6 regional
    projections and EM-DAT frequency data.

Transition risk score (0–1):
    Sector carbon intensity tier × jurisdiction policy ambition score.
    High-carbon sector in a high-ambition jurisdiction = highest risk.

Combined score = 0.5 × physical + 0.5 × transition
    RED   ≥ 0.60 — Investigate before commitment; material climate risk likely
    AMBER 0.35–0.60 — Standard climate DD; flag key hazards for site visit
    GREEN < 0.35 — Routine monitoring; low standalone climate risk

References
----------
IPCC AR6 WG2 Chapter 16 — Key Risks across sectors and regions.
EM-DAT International Disaster Database (2024).
IEA World Energy Outlook 2023 — sector carbon intensities.
Climate Policy Initiative (2023) — jurisdiction policy ambition index.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── Transition risk: sector carbon intensity tier (0 = lowest, 1 = highest) ──

_SECTOR_TRANSITION_SCORE: dict[str, float] = {
    # Fossil fuel / heavy industry — extremely exposed to carbon pricing
    "coal":          1.00,
    "oil_gas":       0.90,
    "cement":        0.85,
    "chemicals":     0.75,
    "metals_mining": 0.70,
    "mining":        0.65,
    "utilities":     0.70,   # mostly fossil generation
    "energy":        0.60,   # mixed renewables/fossil
    "shipping":      0.65,
    "aviation":      0.70,
    # Medium exposure
    "agriculture":   0.55,
    "industrials":   0.50,
    "automotive":    0.50,
    "transport":     0.45,
    "construction":  0.45,
    "food_beverage": 0.40,
    "real_estate":   0.35,
    "consumer":      0.35,
    # Low exposure
    "healthcare":    0.20,
    "financials":    0.15,
    "technology":    0.15,
    "default":       0.40,
}

# ── Physical risk: country/region exposure score (0 = lowest, 1 = highest) ───
# Based on IPCC AR6 regional risk assessments + EM-DAT multi-hazard frequency

_COUNTRY_PHYSICAL_SCORE: dict[str, float] = {
    # Extreme physical risk — multiple severe hazards, low adaptive capacity
    "Bangladesh":          0.95,
    "Pakistan":            0.90,
    "Philippines":         0.90,
    "Vietnam":             0.88,
    "Myanmar":             0.85,
    "Haiti":               0.90,
    "Mozambique":          0.88,
    "Sudan":               0.85,
    "Somalia":             0.87,
    "Yemen":               0.88,
    "Nepal":               0.82,
    "Cambodia":            0.80,
    "Laos":                0.78,
    "Maldives":            0.95,  # SLR existential
    "Tuvalu":              0.97,
    "Kiribati":            0.96,
    "Marshall Islands":    0.95,
    # High physical risk — significant hazards, moderate capacity
    "India":               0.72,
    "Indonesia":           0.75,
    "China":               0.65,
    "Thailand":            0.68,
    "Malaysia":            0.62,
    "Sri Lanka":           0.70,
    "Nigeria":             0.72,
    "Ethiopia":            0.70,
    "Kenya":               0.65,
    "Tanzania":            0.67,
    "Ghana":               0.60,
    "Egypt":               0.65,
    "Mexico":              0.62,
    "Brazil":              0.58,
    "Colombia":            0.62,
    "Peru":                0.60,
    "Bolivia":             0.58,
    "Iran":                0.65,
    "Iraq":                0.68,
    "Turkey":              0.60,
    "South Africa":        0.58,
    "Morocco":             0.55,
    "Algeria":             0.55,
    "Saudi Arabia":        0.60,  # heat + water
    "UAE":                 0.58,
    "Qatar":               0.60,
    "Kuwait":              0.62,
    # Medium physical risk — moderate hazards, good/mixed capacity
    "United States":       0.48,
    "Australia":           0.52,
    "Spain":               0.48,
    "Italy":               0.45,
    "Greece":              0.48,
    "Portugal":            0.45,
    "Japan":               0.50,  # high hazard but very high capacity
    "South Korea":         0.40,
    "Poland":              0.38,
    "Czech Republic":      0.35,
    "Hungary":             0.37,
    "Romania":             0.42,
    "Bulgaria":            0.40,
    "Argentina":           0.45,
    "Chile":               0.42,
    "New Zealand":         0.40,
    "Israel":              0.45,
    # Lower physical risk — moderate/low hazards, high adaptive capacity
    "United Kingdom":      0.35,
    "France":              0.33,
    "Germany":             0.30,
    "Netherlands":         0.35,  # flood managed
    "Belgium":             0.30,
    "Denmark":             0.28,
    "Sweden":              0.22,
    "Norway":              0.20,
    "Finland":             0.18,
    "Switzerland":         0.22,
    "Austria":             0.25,
    "Canada":              0.30,
    "Ireland":             0.25,
    "Luxembourg":          0.20,
    "Singapore":           0.38,   # SLR + heat but high capacity
    "default":             0.50,
}

# Jurisdiction policy ambition — multiplier on transition score
# High ambition = higher near-term transition cost (carbon pricing, CBAM, etc.)
_JURISDICTION_POLICY_MULT: dict[str, float] = {
    # Very high ambition — EU, UK, progressive jurisdictions
    "EU":              1.30,
    "Germany":         1.30,
    "France":          1.25,
    "Netherlands":     1.30,
    "Denmark":         1.35,
    "Sweden":          1.35,
    "Norway":          1.30,
    "Switzerland":     1.25,
    "United Kingdom":  1.20,
    "Belgium":         1.20,
    "Austria":         1.20,
    "Finland":         1.25,
    "Ireland":         1.20,
    # High ambition
    "Canada":          1.15,
    "Japan":           1.15,
    "South Korea":     1.10,
    "New Zealand":     1.15,
    "Singapore":       1.10,
    "Chile":           1.05,
    # Moderate ambition
    "United States":   1.00,
    "Australia":       1.00,
    "China":           1.05,
    "Brazil":          0.95,
    "India":           0.90,
    "Mexico":          0.85,
    "Indonesia":       0.85,
    # Low ambition — higher near-term physical but lower near-term transition cost
    "Saudi Arabia":    0.65,
    "UAE":             0.70,
    "Qatar":           0.65,
    "Russia":          0.60,
    "Iran":            0.55,
    "Iraq":            0.55,
    "default":         0.85,
}

# Sector-specific physical risk amplifiers (hazard particularly relevant to sector)
_SECTOR_PHYSICAL_AMP: dict[str, float] = {
    "agriculture":   1.40,   # directly exposed to drought, flood, heat
    "real_estate":   1.20,   # fixed assets, can't relocate
    "utilities":     1.15,   # water cooling, flood exposure
    "mining":        1.10,
    "metals_mining": 1.10,
    "oil_gas":       1.05,
    "transport":     1.10,
    "food_beverage": 1.20,
    "construction":  1.10,
    "default":       1.00,
}


@dataclass
class TriageResult:
    company_name:       str
    sector_key:         str
    country:            str
    physical_score:     float
    transition_score:   float
    combined_score:     float
    traffic_light:      str   # "RED" | "AMBER" | "GREEN"
    rationale:          str
    top_risks:          list[str]
    recommended_action: str
    full_engine_priority: int  # 1 = run first, 3 = run last in batch


def screen_company(
    company_name: str,
    sector_key: str,
    country: str,
    override_physical: Optional[float] = None,
    override_transition: Optional[float] = None,
) -> TriageResult:
    """
    Fast climate risk triage for one company.

    Parameters
    ----------
    company_name : str
    sector_key : str   CRI sector key or closest match
    country : str      Country name (English)
    override_physical : float, optional   Force physical score 0–1
    override_transition : float, optional  Force transition score 0–1

    Returns
    -------
    TriageResult
    """
    sk  = sector_key.lower().replace(" ", "_").replace("-", "_")
    ctr = country.strip()

    # ── Physical score ────────────────────────────────────────────────────────
    phys_base = override_physical if override_physical is not None else (
        _COUNTRY_PHYSICAL_SCORE.get(ctr, _COUNTRY_PHYSICAL_SCORE["default"])
    )
    phys_amp  = _SECTOR_PHYSICAL_AMP.get(sk, _SECTOR_PHYSICAL_AMP["default"])
    phys      = min(1.0, phys_base * phys_amp)

    # ── Transition score ──────────────────────────────────────────────────────
    trans_base = override_transition if override_transition is not None else (
        _SECTOR_TRANSITION_SCORE.get(sk, _SECTOR_TRANSITION_SCORE["default"])
    )
    pol_mult   = _JURISDICTION_POLICY_MULT.get(ctr, _JURISDICTION_POLICY_MULT["default"])
    trans      = min(1.0, trans_base * pol_mult)

    # ── Combined score ────────────────────────────────────────────────────────
    combined = round(0.5 * phys + 0.5 * trans, 3)

    # ── Traffic light ─────────────────────────────────────────────────────────
    if combined >= 0.60:
        light = "RED"
    elif combined >= 0.35:
        light = "AMBER"
    else:
        light = "GREEN"

    # ── Rationale ─────────────────────────────────────────────────────────────
    phys_label  = "high" if phys >= 0.65 else ("moderate" if phys >= 0.40 else "low")
    trans_label = "high" if trans >= 0.65 else ("moderate" if trans >= 0.40 else "low")

    rationale = (
        f"{company_name} ({sk.replace('_',' ')}, {ctr}) shows "
        f"{phys_label} physical exposure (score {phys:.2f}) and "
        f"{trans_label} transition risk (score {trans:.2f})."
    )

    # ── Top risks ─────────────────────────────────────────────────────────────
    risks: list[str] = []
    if phys >= 0.70:
        risks.append(f"Severe multi-hazard physical exposure in {ctr} — full site-level analysis required")
    elif phys >= 0.45:
        risks.append(f"Moderate physical risk in {ctr} — flood/heat/drought probabilities above global median")

    if trans >= 0.75:
        risks.append(f"{sk.replace('_',' ').title()} sector faces high carbon cost — CBAM/ETS/carbon tax material")
    elif trans >= 0.50:
        risks.append(f"Moderate carbon transition exposure — scenario analysis recommended pre-commitment")

    if pol_mult >= 1.20:
        risks.append(f"{ctr} is a high-ambition jurisdiction — accelerated regulatory timeline increases near-term cost")

    if phys_amp > 1.10:
        risks.append(f"Sector {sk.replace('_',' ')} has above-average physical asset sensitivity (fixed/immovable assets)")

    if not risks:
        risks.append("No dominant risk driver — routine climate monitoring sufficient")

    # ── Recommended action ────────────────────────────────────────────────────
    if light == "RED":
        action = "Run full ClimRisk engine analysis before investment committee. Request company climate data and site coordinates."
    elif light == "AMBER":
        action = "Run full engine analysis during standard DD. Flag key hazards for site visit agenda."
    else:
        action = "Include standard climate clause in documentation. Schedule annual monitoring review."

    priority = {"RED": 1, "AMBER": 2, "GREEN": 3}[light]

    return TriageResult(
        company_name=company_name,
        sector_key=sk,
        country=ctr,
        physical_score=round(phys, 3),
        transition_score=round(trans, 3),
        combined_score=combined,
        traffic_light=light,
        rationale=rationale,
        top_risks=risks,
        recommended_action=action,
        full_engine_priority=priority,
    )


def screen_batch(
    companies: list[dict],
) -> dict:
    """
    Triage a batch of companies. Input: list of {company_name, sector_key, country}.
    Returns results sorted by priority (RED first).
    """
    results = [
        screen_company(
            company_name=c.get("company_name", "Unknown"),
            sector_key=c.get("sector_key", "default"),
            country=c.get("country", "default"),
        )
        for c in companies
    ]
    results.sort(key=lambda r: r.full_engine_priority)

    red    = [r for r in results if r.traffic_light == "RED"]
    amber  = [r for r in results if r.traffic_light == "AMBER"]
    green  = [r for r in results if r.traffic_light == "GREEN"]

    return {
        "summary": {
            "total": len(results),
            "RED":   len(red),
            "AMBER": len(amber),
            "GREEN": len(green),
        },
        "results": [
            {
                "company_name":         r.company_name,
                "sector_key":           r.sector_key,
                "country":              r.country,
                "traffic_light":        r.traffic_light,
                "combined_score":       r.combined_score,
                "physical_score":       r.physical_score,
                "transition_score":     r.transition_score,
                "rationale":            r.rationale,
                "top_risks":            r.top_risks,
                "recommended_action":   r.recommended_action,
                "full_engine_priority": r.full_engine_priority,
            }
            for r in results
        ],
        "methodology": (
            "Physical score: country hazard exposure (EM-DAT / IPCC AR6) × sector physical amplifier. "
            "Transition score: sector carbon intensity tier × jurisdiction policy ambition index. "
            "Combined = 0.5 × physical + 0.5 × transition. "
            "RED ≥ 0.60, AMBER 0.35–0.60, GREEN < 0.35."
        ),
    }
