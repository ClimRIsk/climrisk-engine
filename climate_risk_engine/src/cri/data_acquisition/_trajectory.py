"""
_trajectory.py — 2025–2050 predictive climate risk trajectories.

Produces annual projections across three scenarios:
  • SSP1-2.6 / NGFS Net Zero 2050   (optimistic, 1.5°C)
  • SSP2-4.5 / NGFS Delayed         (central, ~2°C)
  • SSP5-8.5 / NGFS Current Policies (pessimistic, 3°C+)

For each year and each scenario:
  • Scope 1 emissions trajectory (company decarbonisation path)
  • Carbon cost (USD M) and transition VaR (% revenue)
  • Physical flood VaR (% EV) amplified by IPCC warming
  • Composite ClimRisk score (0–1)
  • Implied credit rating migration (AAA→CCC+)

Inflection point detection:
  Flags years where triage band changes (GREEN→AMBER, AMBER→RED),
  where carbon cost crosses EBITDA thresholds, or where flood VaR
  breaches common covenant levels.

Confidence bands:
  Widen with time horizon — inner 50% CI and outer 90% CI shown.
  Based on IPCC AR6 scenario uncertainty ranges.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ── NGFS Carbon price paths (USD/tCO2e) ──────────────────────────────────────
# Source: NGFS Phase 4 scenarios (2023 vintage)
_CARBON_PRICE: dict[str, dict[int, float]] = {
    "nz2050": {
        2025: 75,   2026: 90,   2027: 105,  2028: 120,  2029: 130,
        2030: 145,  2031: 165,  2032: 185,  2033: 210,  2034: 235,
        2035: 265,  2036: 300,  2037: 340,  2038: 385,  2039: 430,
        2040: 480,  2041: 520,  2042: 560,  2043: 610,  2044: 650,
        2045: 680,  2046: 700,  2047: 710,  2048: 715,  2049: 718,  2050: 720,
    },
    "delayed": {
        2025: 30,   2030: 80,   2035: 160,  2040: 280,  2045: 420,  2050: 580,
    },
    "current_policies": {
        2025: 20,   2030: 40,   2035: 65,   2040: 95,   2045: 130,  2050: 170,
    },
}

# ── IPCC AR6 temperature increase by scenario (°C above 1850–1900) ─────────
_WARMING_2050: dict[str, float] = {
    "nz2050":          1.5,
    "delayed":         2.0,
    "current_policies": 3.0,
}

# ── Physical hazard amplification (flood VaR multiplier per °C warming) ──────
# Source: IPCC AR6 WG1 – flood frequency increases ~15-20% per °C in most regions
_FLOOD_AMP_PER_DEG: float = 0.17   # +17% flood VaR per °C warming above current

# Current warming baseline (°C above pre-industrial) as of 2025
_BASELINE_WARMING_2025: float = 1.2

# ── Emission reduction pathways ────────────────────────────────────────────
_SCOPE1_REDUCTION_RATE: dict[str, float] = {
    "nz2050":          0.088,   # ~8.8%/yr  — aligned with IPCC 1.5°C pathway
    "delayed":         0.045,   # ~4.5%/yr  — late-start transition
    "current_policies": 0.012,  # ~1.2%/yr  — very slow decarbonisation
}

# ── Implied credit ratings by composite score ────────────────────────────────
def _score_to_rating(score: float) -> str:
    if score < 0.10: return "AAA"
    if score < 0.18: return "AA"
    if score < 0.26: return "A"
    if score < 0.35: return "BBB"
    if score < 0.45: return "BB"
    if score < 0.55: return "B"
    if score < 0.65: return "CCC+"
    if score < 0.75: return "CCC"
    return "CC/D"


def _interpolate_carbon_price(scenario: str, year: int) -> float:
    """Linear interpolation between known carbon price schedule points."""
    schedule = _CARBON_PRICE[scenario]
    known_years = sorted(schedule.keys())
    if year <= known_years[0]:
        return schedule[known_years[0]]
    if year >= known_years[-1]:
        return schedule[known_years[-1]]
    for i in range(len(known_years) - 1):
        y0, y1 = known_years[i], known_years[i + 1]
        if y0 <= year <= y1:
            t = (year - y0) / (y1 - y0)
            return schedule[y0] + t * (schedule[y1] - schedule[y0])
    return schedule[known_years[-1]]


def _warming_at_year(scenario: str, year: int) -> float:
    """Linear interpolation of global warming from 2025 to 2050."""
    t = max(0, min(1, (year - 2025) / 25))
    return _BASELINE_WARMING_2025 + t * (_WARMING_2050[scenario] - _BASELINE_WARMING_2025)


def _scope1_at_year(scope1_base: float, scenario: str, year: int) -> float:
    rate = _SCOPE1_REDUCTION_RATE[scenario]
    yrs  = year - 2025
    return scope1_base * ((1 - rate) ** yrs)


def _transition_var(scope1: float, carbon_price: float, revenue: float) -> float:
    if revenue <= 0:
        return 0.0
    cost_usd_m = scope1 * carbon_price / 1_000_000
    return (cost_usd_m / revenue) * 100


def _physical_var(base_flood_var_pct: float, warming: float) -> float:
    """Physical VaR amplified by warming above 2025 baseline."""
    delta_warming  = max(0, warming - _BASELINE_WARMING_2025)
    amplification  = 1 + _FLOOD_AMP_PER_DEG * delta_warming
    return base_flood_var_pct * amplification


def _composite(trans_var_pct: float, phys_var_pct: float) -> float:
    phys_score  = min(1.0, phys_var_pct / 20.0)
    trans_score = min(1.0, trans_var_pct / 33.0)
    return round(0.5 * phys_score + 0.5 * trans_score, 4)


def _confidence_band(score: float, year: int, scenario: str) -> dict:
    """
    Uncertainty band widens with time.
    Rule of thumb: ±5% in 2025, ±25% in 2050 (relative width).
    """
    yrs = year - 2025
    width_50 = score * (0.05 + 0.008 * yrs)    # 50% CI half-width
    width_90 = score * (0.08 + 0.015 * yrs)    # 90% CI half-width
    return {
        "p5":    max(0, round(score - width_90, 4)),
        "p25":   max(0, round(score - width_50, 4)),
        "p50":   score,
        "p75":   round(min(1, score + width_50), 4),
        "p95":   round(min(1, score + width_90), 4),
    }


@dataclass
class TrajectoryPoint:
    year:                   int
    scenario:               str
    scope1_mt_co2e:         float
    carbon_price_usd:       float
    carbon_cost_usd_m:      float
    transition_var_pct:     float
    flood_var_pct:          float
    composite_score:        float
    triage:                 str
    implied_rating:         str
    confidence_band:        dict
    warming_deg_c:          float


@dataclass
class InflectionPoint:
    year:       int
    scenario:   str
    event:      str
    from_triage: str
    to_triage:  str
    score:      float
    driver:     str


@dataclass
class TrajectoryResult:
    company_name:   str
    scenarios:      dict[str, list[dict]]   # scenario_key → list of yearly dicts
    inflections:    list[dict]
    methodology:    str
    covenant_alerts: list[dict]             # when metrics breach common thresholds


def _multi_hazard_var_pct(physical: dict, ev_usd_m: float) -> float:
    """
    Derive a single base physical VaR % from the full engine's per-hazard output.

    If the engine returned physical_loss_by_hazard (USD M per hazard accumulated
    over all projection years), we sum all hazards and express as % of EV.
    Falls back to flood_var_pct scalar for the shallow-fallback path.
    """
    if not physical:
        return 5.0
    by_hazard = physical.get("physical_loss_by_hazard")
    if by_hazard and isinstance(by_hazard, dict) and by_hazard:
        total_loss_usd_m = sum(by_hazard.values())
        if ev_usd_m and ev_usd_m > 0:
            return (total_loss_usd_m / ev_usd_m) * 100
    # Scalar fallback (shallow path or flood-only legacy assessor)
    return physical.get("flood_var_pct") or 5.0


async def build_trajectory(profile, risk_result: dict) -> dict:
    """
    Build 2025–2050 trajectories for three NGFS/IPCC scenarios.

    Parameters
    ----------
    profile     : CompanyProfile from company_profiler
    risk_result : Output from job_runner._run_risk_assessment (full or shallow)
    """
    company_name   = profile.resolved_name or profile.input_name
    scope1_base    = profile.scope1_mt_co2e.value if profile.scope1_mt_co2e else 1.0
    revenue        = profile.revenue_usd_m.value if profile.revenue_usd_m else 5_000.0
    ebitda         = profile.ebitda_usd_m.value if profile.ebitda_usd_m else revenue * 0.15

    # Prefer EV from engine scenarios (NZE run), fall back to profile EV
    ev_usd_m = (
        (risk_result.get("scenarios") or {})
        .get("nze", {})
        .get("enterprise_value_usd_m")
        or (profile.ev_usd_m.value if profile.ev_usd_m else revenue * 2.0)
        or revenue * 2.0
    )

    # Multi-hazard VaR — use CP (worst physical) scenario if full engine ran
    cp_physical = (risk_result.get("scenarios") or {}).get("cp", {}).get("physical") or {}
    physical    = cp_physical or (risk_result.get("physical") or {})
    base_physical_var = _multi_hazard_var_pct(physical, ev_usd_m)

    # Per-hazard breakdown for enriching trajectory data points
    hazard_breakdown: dict[str, float] = (
        physical.get("physical_loss_by_hazard") or {}
    )

    # Dominant non-flood hazard names for narrative inflection drivers
    _HAZARD_LABELS = {
        "heat_stress":      "rising heat stress",
        "water_stress":     "intensifying water stress",
        "drought":          "worsening drought",
        "wildfire":         "expanding wildfire",
        "cyclone":          "cyclone frequency",
        "flood_riverine":   "riverine flooding",
        "flood_coastal":    "coastal flooding",
        "sea_level_rise":   "sea level rise",
        "landslide":        "landslide risk",
        "saltwater_intrusion": "saltwater intrusion",
    }
    dominant_hazard_label = "increasing physical hazard"
    if hazard_breakdown:
        top = max(hazard_breakdown, key=hazard_breakdown.get)
        dominant_hazard_label = _HAZARD_LABELS.get(top, top.replace("_", " "))

    scenarios_out: dict[str, list[dict]] = {}
    all_inflections: list[InflectionPoint] = []
    covenant_alerts: list[dict] = []

    scenario_labels = {
        "nz2050":           "NGFS Net Zero 2050 / SSP1-2.6 (1.5°C)",
        "delayed":          "NGFS Delayed Transition / SSP2-4.5 (~2°C)",
        "current_policies": "NGFS Current Policies / SSP5-8.5 (3°C+)",
    }

    for scenario in ("nz2050", "delayed", "current_policies"):
        points: list[dict] = []
        prev_triage = None

        for year in range(2025, 2051):
            cp      = _interpolate_carbon_price(scenario, year)
            s1      = _scope1_at_year(scope1_base, scenario, year)
            warming = _warming_at_year(scenario, year)
            tv_pct  = _transition_var(s1, cp, revenue)
            pv_pct  = _physical_var(base_physical_var, warming)  # uses full multi-hazard base
            comp    = _composite(tv_pct, pv_pct)
            rating  = _score_to_rating(comp)
            band    = _confidence_band(comp, year, scenario)
            triage  = "GREEN" if comp < 0.35 else ("AMBER" if comp < 0.60 else "RED")

            carbon_cost_m = s1 * cp / 1_000_000

            pt = {
                "year":                year,
                "scenario":            scenario,
                "scope1_mt_co2e":      round(s1, 3),
                "carbon_price_usd":    round(cp, 0),
                "carbon_cost_usd_m":   round(carbon_cost_m, 2),
                "transition_var_pct":  round(tv_pct, 3),
                "flood_var_pct":       round(pv_pct, 3),
                "warming_deg_c":       round(warming, 2),
                "composite_score":     comp,
                "triage":              triage,
                "implied_rating":      rating,
                "confidence_band":     band,
            }
            points.append(pt)

            # ── Triage inflection detection ────────────────────────────────
            if prev_triage and prev_triage != triage:
                driver = (
                    "rising carbon price" if tv_pct > pv_pct
                    else dominant_hazard_label  # now uses actual dominant hazard
                )
                all_inflections.append(InflectionPoint(
                    year=year,
                    scenario=scenario,
                    event=f"{scenario_labels[scenario]}: triage {prev_triage} → {triage}",
                    from_triage=prev_triage,
                    to_triage=triage,
                    score=comp,
                    driver=driver,
                ))

            # ── Covenant threshold alerts ──────────────────────────────────
            # Carbon cost > 15% EBITDA
            if ebitda > 0 and carbon_cost_m / ebitda > 0.15:
                existing = any(
                    a["scenario"] == scenario and a["metric"] == "carbon_cost_ebitda"
                    for a in covenant_alerts
                )
                if not existing:
                    covenant_alerts.append({
                        "year":     year,
                        "scenario": scenario,
                        "metric":   "carbon_cost_ebitda",
                        "event":    (
                            f"Carbon cost exceeds 15% EBITDA threshold "
                            f"(${carbon_cost_m:.0f}M / ${ebitda:.0f}M EBITDA) in {year} "
                            f"under {scenario_labels[scenario]}"
                        ),
                        "severity": "HIGH",
                    })

            # Flood VaR > 10% EV
            if pv_pct > 10.0:
                existing = any(
                    a["scenario"] == scenario and a["metric"] == "flood_var_ev"
                    for a in covenant_alerts
                )
                if not existing:
                    covenant_alerts.append({
                        "year":     year,
                        "scenario": scenario,
                        "metric":   "flood_var_ev",
                        "event":    (
                            f"Flood VaR exceeds 10% EV covenant threshold "
                            f"({pv_pct:.1f}%) in {year} "
                            f"under {scenario_labels[scenario]}"
                        ),
                        "severity": "MEDIUM",
                    })

            prev_triage = triage

        scenarios_out[scenario] = points

    methodology = (
        "Three NGFS Phase 4 scenarios (2023 vintage): "
        "Net Zero 2050 (1.5°C), Delayed Transition (~2°C), Current Policies (3°C+). "
        "Carbon price schedule: NGFS published pathways, interpolated annually. "
        "Scope 1 decarbonisation: NZ2050 −8.8%/yr, Delayed −4.5%/yr, Current −1.2%/yr. "
        "Physical hazard: base flood VaR amplified +17% per °C above 2025 baseline "
        "(IPCC AR6 WG1 flood frequency sensitivity). "
        "Composite: 0.5×physical (normalised at 20% EV) + 0.5×transition (normalised at 33% revenue). "
        "Confidence bands: ±5% in 2025 widening to ±25% in 2050 (relative). "
        "Implied credit rating: proprietary mapping from composite score."
    )

    return {
        "company_name":  company_name,
        "base_year":     2025,
        "scenarios": {
            k: {"label": scenario_labels[k], "points": v}
            for k, v in scenarios_out.items()
        },
        "inflections":    [
            {
                "year":       inf.year,
                "scenario":   inf.scenario,
                "event":      inf.event,
                "from_triage": inf.from_triage,
                "to_triage":  inf.to_triage,
                "score":      inf.score,
                "driver":     inf.driver,
            }
            for inf in sorted(all_inflections, key=lambda x: x.year)
        ],
        "covenant_alerts": sorted(covenant_alerts, key=lambda a: a["year"]),
        "methodology":    methodology,
        "data_provenance": {
            "scope1_source":    (
                profile.scope1_mt_co2e.source
                if profile.scope1_mt_co2e else "MISSING — using 1.0 Mt placeholder"
            ),
            "revenue_source":   (
                profile.revenue_usd_m.source
                if profile.revenue_usd_m else "MISSING — using $5,000M placeholder"
            ),
            "physical_base_var_pct": round(base_physical_var, 3),
            "physical_source": (
                "Full CRI engine (25-hazard) — per-hazard NPV VaR summed as % of EV"
                if hazard_breakdown
                else "Flood VaR scalar (shallow fallback)"
            ),
            "hazard_breakdown_usd_m": hazard_breakdown,
            "dominant_hazard":  dominant_hazard_label,
        },
    }
