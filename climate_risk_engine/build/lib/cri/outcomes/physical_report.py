"""Physical Climate Risk Report builder.

Produces a standalone PhysicalRiskReport from three sets of operational
simulation results (one per NGFS scenario). This is the deliverable for
clients who commission a physical-risk-only scope (e.g. Morelli Consulting,
real-estate lenders, insurers, infrastructure operators).

No valuation opinion is produced. No transition risk drivers. The output
is purely:
  - Asset-level hazard exposure trajectories (2026–2050)
  - Expected annual production loss cost (USD)
  - Adaptation capex requirement
  - Physical risk score (0–100, normalised)
  - TCFD-aligned narrative summary

Usage
-----
    from cri.engine.scope import ReportScope
    from cri.outcomes.physical_report import build_physical_report

    report = build_physical_report(
        company=company,
        ops_nze=ops_nze,
        ops_delayed=ops_delayed,
        ops_cp=ops_cp,
        run_id=run_id,
    )
"""

from __future__ import annotations

import statistics
import uuid
from typing import Optional

from ..data.schemas import Company
from ..engine.scope import PhysicalRiskReport, PhysicalYear
from ..operations.company import OperationalYear

# ---------------------------------------------------------------------------
# Score normalisation constants
# ---------------------------------------------------------------------------
# Physical loss cost is normalised as a fraction of baseline annual revenue.
# Benchmarks (approximate industry ranges):
#   Low      < 0.5% of revenue   → score 0–20
#   Moderate  0.5–2% of revenue   → score 20–40
#   Elevated  2–5% of revenue     → score 40–60
#   High      5–8% of revenue     → score 60–80
#   Critical  > 8% of revenue     → score 80–100
PHYSICAL_SCORE_REV_CAP = 0.08   # 8% annual revenue loss → score 100

_LABEL_THRESHOLDS = [
    (20,  "Low"),
    (40,  "Moderate"),
    (60,  "Elevated"),
    (80,  "High"),
    (101, "Critical"),
]


def _score_label(score: float) -> str:
    for threshold, label in _LABEL_THRESHOLDS:
        if score < threshold:
            return label
    return "Critical"


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _ops_to_physical_years(
    ops: list[OperationalYear],
    baseline_revenue: float,
) -> list[PhysicalYear]:
    """Convert simulation output to physical-risk-only year objects."""
    years = []
    for o in ops:
        loss_fraction = (
            o.physical_loss_cost / baseline_revenue
            if baseline_revenue > 0 else 0.0
        )
        years.append(PhysicalYear(
            year=o.year,
            physical_loss_cost=round(o.physical_loss_cost, 0),
            adaptation_capex=round(o.adaptation_capex, 0),
            physical_loss_by_hazard={
                k: round(v, 0) for k, v in o.physical_loss_by_hazard.items()
            },
            total_loss_fraction=round(loss_fraction, 4),
        ))
    return years


def _weighted_physical_score(
    ops_nze: list[OperationalYear],
    ops_dly: list[OperationalYear],
    ops_cp: list[OperationalYear],
    baseline_revenue: float,
) -> float:
    """Scenario-weighted average physical loss → normalised 0–100 score.

    Weights: NZE 0.30 (best case), Delayed 0.40, Current Policies 0.30.
    These mirror the scenario weights in the full CRI composite.
    Each scenario's contribution is its mean annual physical loss fraction.
    """
    def mean_loss(ops: list[OperationalYear]) -> float:
        if not ops or baseline_revenue <= 0:
            return 0.0
        return statistics.mean(
            o.physical_loss_cost / baseline_revenue for o in ops
        )

    weighted = (
        0.30 * mean_loss(ops_nze) +
        0.40 * mean_loss(ops_dly) +
        0.30 * mean_loss(ops_cp)
    )
    return min(100.0, max(0.0, (weighted / PHYSICAL_SCORE_REV_CAP) * 100.0))


def _find_peak(
    ops_cp: list[OperationalYear],
    baseline_revenue: float,
) -> tuple[int, float, str]:
    """Find the year + hazard with the highest physical loss under CP scenario."""
    if not ops_cp:
        return 0, 0.0, "unknown"

    peak_op = max(ops_cp, key=lambda o: o.physical_loss_cost)
    peak_year = peak_op.year
    peak_usd = peak_op.physical_loss_cost

    # Dominant hazard = largest contributor at peak year
    hazards = peak_op.physical_loss_by_hazard
    if hazards:
        peak_hazard = max(hazards, key=lambda h: hazards[h])
    else:
        peak_hazard = "combined"

    return peak_year, round(peak_usd, 0), peak_hazard


def _hazard_breakdown_at(
    ops: list[OperationalYear],
    target_year: int,
) -> dict[str, float]:
    """Return per-hazard loss breakdown (USD) for a specific year."""
    for o in ops:
        if o.year == target_year:
            return {k: round(v, 0) for k, v in o.physical_loss_by_hazard.items()}
    return {}


def _build_narrative(
    company_name: str,
    score: float,
    label: str,
    peak_year: int,
    peak_usd: float,
    peak_hazard: str,
    total_adapt_cp: float,
    hazard_2035: dict[str, float],
) -> str:
    """Generate a TCFD-aligned one-paragraph physical risk narrative."""
    # Dominant hazard label mapping
    hazard_labels = {
        # Core 10
        "heat_stress":            "heat stress",
        "flood_riverine":         "riverine flooding",
        "flood_coastal":          "coastal flooding",
        "sea_level_rise":         "sea-level rise",
        "saltwater_intrusion":    "saltwater intrusion",
        "landslide":              "landslides",
        "wildfire":               "wildfire",
        "cyclone":                "tropical cyclones",
        "drought":                "drought",
        "water_stress":           "water stress",
        # New 15 (v0.4)
        "extreme_cold":           "extreme cold and frost",
        "blade_icing":            "blade and infrastructure icing",
        "extratropical_cyclone":  "extratropical cyclones",
        "flash_flood":            "flash flooding",
        "permafrost_thaw":        "permafrost thaw",
        "dust_storm":             "dust storms",
        "hail":                   "large hail",
        "lightning":              "lightning",
        "subsidence":             "ground subsidence",
        "freeze_thaw_cycle":      "freeze-thaw cycling",
        "compound_flood":         "compound flooding",
        "avalanche":              "avalanche",
        "marine_heatwave":        "marine heatwaves",
        "glof":                   "glacial lake outburst floods",
        "tornado":                "tornadoes",
        "combined":               "compounding climate hazards",
    }
    hazard_str = hazard_labels.get(peak_hazard, peak_hazard.replace("_", " "))

    # Format dollar figures — inputs are in USD millions (matching revenue scale)
    def fmt_usd(v: float) -> str:
        if v >= 1_000:                 # >= $1B
            return f"USD {v/1_000:.1f}B"
        if v >= 1:                     # >= $1M
            return f"USD {v:.0f}M"
        if v >= 0.001:                 # >= $1K
            return f"USD {v*1_000:.0f}K"
        return f"USD {v*1_000_000:,.0f}"

    # List top hazards at 2035
    top_hazards = sorted(hazard_2035.items(), key=lambda x: x[1], reverse=True)[:3]
    top_str = ", ".join(
        hazard_labels.get(h, h.replace("_", " ")) for h, _ in top_hazards if _ > 0
    ) or "various hazards"

    narrative = (
        f"{company_name} is assessed as facing {label.lower()} physical climate risk "
        f"(score {score:.0f}/100) across the 2026–2050 projection horizon. "
        f"Under the Current Policies scenario — representing the highest physical "
        f"risk trajectory — projected annual production losses peak in {peak_year} "
        f"at approximately {fmt_usd(peak_usd)}, primarily driven by {hazard_str}. "
        f"By 2035, the dominant physical risk drivers are {top_str}. "
        f"Across the 25-year horizon, cumulative adaptation capital expenditure "
        f"requirements are estimated at {fmt_usd(total_adapt_cp)} under the "
        f"Current Policies trajectory. These estimates are derived from asset-level "
        f"hazard assessments aligned with IPCC AR6 warming projections and "
        f"WRI Aqueduct 4.0 water risk data, and should be interpreted as "
        f"directional risk indicators for TCFD physical risk disclosure purposes."
    )
    return narrative


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_physical_report(
    company: Company,
    ops_nze: list[OperationalYear],
    ops_delayed: list[OperationalYear],
    ops_cp: list[OperationalYear],
    run_id: Optional[str] = None,
    model_version: str = "0.3.0",
    input_hash: str = "",
    scenario_version: str = "",
) -> PhysicalRiskReport:
    """Build a standalone PhysicalRiskReport from three scenario simulations.

    Parameters
    ----------
    company       : Company profile (needed for baseline revenue + name).
    ops_nze       : Operational simulation output under NZE 2050 scenario.
    ops_delayed   : Operational simulation output under Delayed Transition.
    ops_cp        : Operational simulation output under Current Policies.
    run_id        : Optional run identifier (auto-generated if omitted).
    model_version : Engine version string.
    """
    run_id = run_id or str(uuid.uuid4())[:8]
    baseline_revenue = company.financials.revenue

    # Build per-scenario year lists
    years_nze     = _ops_to_physical_years(ops_nze,     baseline_revenue)
    years_delayed = _ops_to_physical_years(ops_delayed, baseline_revenue)
    years_cp      = _ops_to_physical_years(ops_cp,      baseline_revenue)

    # Score
    score = _weighted_physical_score(ops_nze, ops_delayed, ops_cp, baseline_revenue)
    label = _score_label(score)

    # Peak loss under worst-case (Current Policies)
    peak_year, peak_usd, peak_hazard = _find_peak(ops_cp, baseline_revenue)

    # Adaptation capex totals
    total_adapt_nze = sum(y.adaptation_capex for y in years_nze)
    total_adapt_cp  = sum(y.adaptation_capex for y in years_cp)

    # Hazard breakdown at 2035 under Current Policies
    hazard_2035 = _hazard_breakdown_at(ops_cp, 2035)

    # TCFD narrative
    narrative = _build_narrative(
        company_name=company.name,
        score=score,
        label=label,
        peak_year=peak_year,
        peak_usd=peak_usd,
        peak_hazard=peak_hazard,
        total_adapt_cp=total_adapt_cp,
        hazard_2035=hazard_2035,
    )

    return PhysicalRiskReport(
        company_id=company.id,
        company_name=company.name,
        run_id=run_id,
        model_version=model_version,
        years_nze=years_nze,
        years_delayed=years_delayed,
        years_cp=years_cp,
        physical_score=round(score, 1),
        physical_label=label,
        peak_loss_year=peak_year,
        peak_loss_usd=peak_usd,
        peak_loss_hazard=peak_hazard,
        total_adaptation_capex_nze=round(total_adapt_nze, 0),
        total_adaptation_capex_cp=round(total_adapt_cp, 0),
        hazard_breakdown_2035=hazard_2035,
        narrative=narrative,
        input_hash=input_hash,
        scenario_version=scenario_version,
    )
