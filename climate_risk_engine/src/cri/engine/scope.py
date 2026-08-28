"""Report scope — controls which pillars the engine computes and returns.

A client chooses a scope when they commission a CRI analysis. The scope
determines:
  - Which computational paths are executed (skipping unneeded work)
  - Which fields are populated in the result
  - Which report template is used for the disclosure output

Scope catalogue
---------------
PHYSICAL
  What:  Asset-level hazard assessment + production loss + adaptation capex.
  For:   Real-estate lenders, insurers, physical asset owners, TCFD Chapter 2.
  Skip:  Carbon pricing, demand-shift, valuation DCF.
  → Morelli Consulting use case.

TRANSITION
  What:  Carbon cost trajectory, commodity demand shifts, EBITDA compression,
         stranded-asset exposure under NGFS carbon price paths.
  For:   Equity analysts, bond investors, climate policy teams.
  Skip:  Asset-level physical hazard detail, full DCF.

FINANCIAL
  What:  Full DCF enterprise valuation under all three scenarios, WACC uplift,
         equity value at risk, NPV impact vs. baseline.
  For:   M&A, credit committees, portfolio risk (ING use case).
  Includes: requires physical + transition as inputs to FCF; they are computed
  internally but not exposed as standalone pillar outputs.

PHYSICAL_TRANSITION
  What:  Physical + Transition combined — no full DCF valuation.
  For:   ESG teams who need both risk dimensions but not a valuation opinion.

FULL_CRI  (default)
  What:  All three pillars + composite A–E rating.
  For:   Full TCFD / CSRD / IFRS S2 disclosure packages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ReportScope(str, Enum):
    PHYSICAL           = "physical"
    TRANSITION         = "transition"
    FINANCIAL          = "financial"
    PHYSICAL_TRANSITION = "physical_transition"
    FULL_CRI           = "full_cri"

    # -----------------------------------------------------------------------
    # Convenience predicates — used by orchestrator to gate computation
    # -----------------------------------------------------------------------

    @property
    def needs_physical(self) -> bool:
        return self in (
            ReportScope.PHYSICAL,
            ReportScope.PHYSICAL_TRANSITION,
            ReportScope.FULL_CRI,
            ReportScope.FINANCIAL,   # physical feeds FCF
        )

    @property
    def needs_transition(self) -> bool:
        return self in (
            ReportScope.TRANSITION,
            ReportScope.PHYSICAL_TRANSITION,
            ReportScope.FULL_CRI,
            ReportScope.FINANCIAL,   # transition feeds FCF
        )

    @property
    def needs_valuation(self) -> bool:
        return self in (
            ReportScope.FINANCIAL,
            ReportScope.FULL_CRI,
        )

    @property
    def needs_composite_rating(self) -> bool:
        return self == ReportScope.FULL_CRI

    @property
    def label(self) -> str:
        return {
            ReportScope.PHYSICAL:            "Physical Climate Risk Report",
            ReportScope.TRANSITION:          "Transition Risk Report",
            ReportScope.FINANCIAL:           "Financial Climate Risk Report",
            ReportScope.PHYSICAL_TRANSITION: "Physical & Transition Risk Report",
            ReportScope.FULL_CRI:            "Full CRI Climate Rating",
        }[self]


# ---------------------------------------------------------------------------
# Per-year physical summary (used in PhysicalRiskReport)
# ---------------------------------------------------------------------------

@dataclass
class PhysicalYear:
    """One year of physical-risk outputs."""
    year: int
    physical_loss_cost: float          # USD — total production loss cost
    adaptation_capex: float            # USD — capex to harden assets
    physical_loss_by_hazard: dict[str, float] = field(default_factory=dict)
    total_loss_fraction: float = 0.0   # 0–1 fraction of baseline revenue lost


@dataclass
class PhysicalRiskReport:
    """Standalone physical climate risk output — Morelli / insurer use case.

    Covers the full 2026–2050 horizon under all three NGFS scenarios.
    No valuation opinion. No transition risk. Pure asset-level hazard exposure.
    """
    company_id: str
    company_name: str
    run_id: str
    model_version: str

    # Per-scenario trajectories (each list = 25 years)
    years_nze: list[PhysicalYear]
    years_delayed: list[PhysicalYear]
    years_cp: list[PhysicalYear]

    # Physical risk score (0–100; higher = more risk) — normalised across scenarios
    physical_score: float
    physical_label: str   # "Low" / "Moderate" / "Elevated" / "High" / "Critical"

    # Peak loss year under worst-case (Current Policies)
    peak_loss_year: int
    peak_loss_usd: float
    peak_loss_hazard: str   # dominant hazard at peak

    # Adaptation cost summary
    total_adaptation_capex_nze: float        # USD total over 25 yrs
    total_adaptation_capex_cp: float         # USD total under worst-case

    # Hazard breakdown at 2035 under Current Policies (for disclosure table)
    hazard_breakdown_2035: dict[str, float]  # hazard → USD cost

    # TCFD-aligned narrative summary (1 paragraph)
    narrative: str = ""

    # Input provenance
    input_hash: str = ""
    scenario_version: str = ""


# ---------------------------------------------------------------------------
# Per-year transition summary
# ---------------------------------------------------------------------------

@dataclass
class TransitionYear:
    year: int
    carbon_cost: float             # USD
    carbon_cost_pct_ebitda: float  # carbon cost as % of EBITDA
    revenue_by_commodity: dict[str, float] = field(default_factory=dict)
    emissions_scope1: float = 0.0
    emissions_scope2: float = 0.0
    emissions_scope3: float = 0.0


@dataclass
class TransitionRiskReport:
    """Standalone transition risk output.

    Covers carbon cost trajectory, commodity demand shifts, and EBITDA
    compression under each NGFS scenario. No valuation opinion.
    """
    company_id: str
    company_name: str
    run_id: str
    model_version: str

    years_nze: list[TransitionYear]
    years_delayed: list[TransitionYear]
    years_cp: list[TransitionYear]

    transition_score: float
    transition_label: str

    # EBITDA compression summary
    ebitda_compression_2030_nze: Optional[float] = None   # fraction, e.g. -0.18 = -18%
    ebitda_compression_2040_nze: Optional[float] = None

    # Carbon cost as % of EBITDA at key dates
    carbon_pct_ebitda_2030_nze: Optional[float] = None
    carbon_pct_ebitda_2030_cp:  Optional[float] = None

    narrative: str = ""
    input_hash: str = ""
    scenario_version: str = ""


# ---------------------------------------------------------------------------
# Combined scoped result — envelope returned by orchestrator.run_scoped()
# ---------------------------------------------------------------------------

@dataclass
class ScopedResult:
    """Envelope that carries whichever pillar reports were requested.

    Fields are None if their scope was not requested.
    """
    scope: ReportScope
    run_id: str

    physical:   Optional[PhysicalRiskReport]   = None
    transition: Optional[TransitionRiskReport] = None

    # Financial / Full CRI results use the existing RunResults / RatingResult
    # types — populated here when scope includes valuation.
    valuation_results: Optional[dict] = None    # RunResults per scenario (nze/dly/cp)
    rating_result: Optional[object]  = None    # RatingResult when FULL_CRI

    @property
    def scope_label(self) -> str:
        return self.scope.label
