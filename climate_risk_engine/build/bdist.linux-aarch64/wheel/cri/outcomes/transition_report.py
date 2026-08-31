"""Transition Climate Risk Report builder.

Produces a standalone TransitionRiskReport covering carbon cost trajectory,
commodity demand shifts, and EBITDA compression across NGFS scenarios.
No asset-level physical hazard detail. No valuation DCF.

Typical client: equity analyst, fixed-income ESG team, policy team,
portfolio manager running TCFD transition risk screening.
"""

from __future__ import annotations

import statistics
import uuid
from typing import Optional

from ..data.schemas import Company
from ..engine.scope import TransitionRiskReport, TransitionYear
from ..operations.company import OperationalYear

# ---------------------------------------------------------------------------
# Score normalisation constants
# ---------------------------------------------------------------------------
# Transition score is driven by EBITDA compression under NZE 2050 (worst
# transition scenario for most fossil-heavy companies).
# Compression benchmarks:
#   Low:      < 5% EBITDA compression by 2035   → 0–20
#   Moderate:  5–15%                             → 20–40
#   Elevated: 15–30%                             → 40–60
#   High:     30–50%                             → 60–80
#   Critical: > 50%                              → 80–100
TRANSITION_SCORE_CAP = 0.50   # 50% EBITDA compression → score 100


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
# Helpers
# ---------------------------------------------------------------------------

def _ops_to_transition_years(
    ops: list[OperationalYear],
    baseline_ebitda: float,
) -> list[TransitionYear]:
    years = []
    for o in ops:
        ebitda = o.revenue - o.opex - o.carbon_cost
        carbon_pct = (o.carbon_cost / ebitda) if ebitda > 0 else 0.0
        years.append(TransitionYear(
            year=o.year,
            carbon_cost=round(o.carbon_cost, 0),
            carbon_cost_pct_ebitda=round(carbon_pct, 4),
            revenue_by_commodity={k: round(v, 0) for k, v in o.revenue_by_commodity.items()},
            emissions_scope1=round(o.emissions_scope1, 1),
            emissions_scope2=round(o.emissions_scope2, 1),
            emissions_scope3=round(o.emissions_scope3, 1),
        ))
    return years


def _ebitda_compression(
    ops: list[OperationalYear],
    baseline_ebitda: float,
    target_year: int,
) -> Optional[float]:
    """EBITDA compression fraction at target_year vs. baseline."""
    if baseline_ebitda <= 0:
        return None
    for o in ops:
        if o.year == target_year:
            ebitda = o.revenue - o.opex - o.carbon_cost
            return round(ebitda / baseline_ebitda - 1.0, 4)
    return None


def _carbon_pct_ebitda_at(
    ops: list[OperationalYear],
    target_year: int,
) -> Optional[float]:
    for o in ops:
        if o.year == target_year:
            ebitda = o.revenue - o.opex - o.carbon_cost
            if ebitda <= 0:
                return None
            return round(o.carbon_cost / ebitda, 4)
    return None


def _transition_score(
    ops_nze: list[OperationalYear],
    baseline_ebitda: float,
) -> float:
    """Score from EBITDA compression under NZE (most demanding transition)."""
    comp_2035 = _ebitda_compression(ops_nze, baseline_ebitda, 2035)
    if comp_2035 is None:
        return 50.0
    # compression is negative (e.g. -0.18 = 18% drop)
    severity = max(0.0, -comp_2035)
    return min(100.0, (severity / TRANSITION_SCORE_CAP) * 100.0)


def _build_narrative(
    company_name: str,
    score: float,
    label: str,
    comp_2030: Optional[float],
    comp_2040: Optional[float],
    carbon_pct_2030_nze: Optional[float],
    peak_carbon_year: int,
    peak_carbon_usd: float,
) -> str:
    def fmt_pct(v: Optional[float]) -> str:
        if v is None:
            return "N/A"
        return f"{v*100:+.1f}%"

    def fmt_usd(v: float) -> str:
        if v >= 1e9:
            return f"USD {v/1e9:.1f}B"
        if v >= 1e6:
            return f"USD {v/1e6:.1f}M"
        return f"USD {v:,.0f}"

    cp_2030 = fmt_pct(comp_2030)
    cp_2040 = fmt_pct(comp_2040)
    cc_pct  = f"{carbon_pct_2030_nze*100:.1f}%" if carbon_pct_2030_nze else "N/A"

    return (
        f"{company_name} faces {label.lower()} transition climate risk "
        f"(score {score:.0f}/100) under a Net Zero 2050 pathway. "
        f"Under the NZE 2050 scenario, EBITDA is projected to compress "
        f"{cp_2030} by 2030 and {cp_2040} by 2040, driven by rising carbon "
        f"costs and demand-side shifts away from high-carbon commodities. "
        f"Carbon costs represent approximately {cc_pct} of EBITDA by 2030 "
        f"under the NZE pathway. Carbon costs peak at approximately "
        f"{fmt_usd(peak_carbon_usd)} in {peak_carbon_year} under the Delayed "
        f"Transition scenario, reflecting the emergency repricing dynamic of "
        f"a late but abrupt policy response. These projections are aligned "
        f"with NGFS Phase 4 carbon price pathways and IEA WEO 2024 commodity "
        f"demand forecasts, and support TCFD transition risk disclosure and "
        f"IFRS S2 quantitative risk assessment."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_transition_report(
    company: Company,
    ops_nze: list[OperationalYear],
    ops_delayed: list[OperationalYear],
    ops_cp: list[OperationalYear],
    run_id: Optional[str] = None,
    model_version: str = "0.3.0",
    input_hash: str = "",
    scenario_version: str = "",
) -> TransitionRiskReport:
    """Build a standalone TransitionRiskReport from three scenario simulations."""
    run_id = run_id or str(uuid.uuid4())[:8]
    baseline_ebitda = company.financials.ebitda

    years_nze     = _ops_to_transition_years(ops_nze,     baseline_ebitda)
    years_delayed = _ops_to_transition_years(ops_delayed, baseline_ebitda)
    years_cp      = _ops_to_transition_years(ops_cp,      baseline_ebitda)

    score = _transition_score(ops_nze, baseline_ebitda)
    label = _score_label(score)

    comp_2030_nze = _ebitda_compression(ops_nze, baseline_ebitda, 2030)
    comp_2040_nze = _ebitda_compression(ops_nze, baseline_ebitda, 2040)
    carbon_pct_2030_nze = _carbon_pct_ebitda_at(ops_nze, 2030)
    carbon_pct_2030_cp  = _carbon_pct_ebitda_at(ops_cp,  2030)

    # Peak carbon cost under Delayed Transition (emergency repricing spike)
    peak_delayed = max(ops_delayed, key=lambda o: o.carbon_cost, default=None)
    peak_carbon_year = peak_delayed.year if peak_delayed else 2050
    peak_carbon_usd  = peak_delayed.carbon_cost if peak_delayed else 0.0

    narrative = _build_narrative(
        company_name=company.name,
        score=score,
        label=label,
        comp_2030=comp_2030_nze,
        comp_2040=comp_2040_nze,
        carbon_pct_2030_nze=carbon_pct_2030_nze,
        peak_carbon_year=peak_carbon_year,
        peak_carbon_usd=peak_carbon_usd,
    )

    return TransitionRiskReport(
        company_id=company.id,
        company_name=company.name,
        run_id=run_id,
        model_version=model_version,
        years_nze=years_nze,
        years_delayed=years_delayed,
        years_cp=years_cp,
        transition_score=round(score, 1),
        transition_label=label,
        ebitda_compression_2030_nze=comp_2030_nze,
        ebitda_compression_2040_nze=comp_2040_nze,
        carbon_pct_ebitda_2030_nze=carbon_pct_2030_nze,
        carbon_pct_ebitda_2030_cp=carbon_pct_2030_cp,
        narrative=narrative,
        input_hash=input_hash,
        scenario_version=scenario_version,
    )
