"""API response models and request contracts for the CRI FastAPI."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator

from ..data.schemas import RunResults, YearResult


class ScenarioResponse(BaseModel):
    """Scenario metadata for GET /scenarios."""

    id: str
    name: str
    description: str
    family: str
    version: str


class CompanyResponse(BaseModel):
    """Company metadata for GET /companies."""

    id: str
    name: str
    sector: str
    region: str


class CustomScenarioParams(BaseModel):
    """Full custom scenario specification supplied inline by the caller.

    Used when scenario_id == "custom" in POST /runs.
    All named NGFS scenarios (nze_2050, delayed_transition, current_policies)
    ignore this object entirely.

    carbon_price_path  — year → USD/tCO2e for every year in the horizon.
                         Missing years are interpolated from neighbours; years
                         before the first key use that first value.
    risk_premium_bps   — WACC add-on for this scenario (basis points). Default 100.
    abatement_targets  — milestone year → required cumulative abatement fraction.
                         e.g. {2030: 0.25, 2040: 0.55, 2050: 0.85}
                         If omitted, the CUSTOM family default is used
                         ({2030: 0.20, 2040: 0.50, 2050: 0.75}).
    name               — human-readable label surfaced in reports.
    description        — optional free-text description.
    """

    carbon_price_path: Dict[int, float]
    risk_premium_bps: int = 100
    abatement_targets: Optional[Dict[int, float]] = None
    name: str = "Custom Scenario"
    description: str = ""

    @field_validator("carbon_price_path")
    @classmethod
    def _at_least_one_year(cls, v: Dict[int, float]) -> Dict[int, float]:
        if not v:
            raise ValueError("carbon_price_path must contain at least one year entry.")
        for yr, price in v.items():
            if price < 0:
                raise ValueError(f"Carbon price for year {yr} cannot be negative.")
        return v

    @field_validator("abatement_targets")
    @classmethod
    def _valid_abatement(cls, v: Optional[Dict[int, float]]) -> Optional[Dict[int, float]]:
        if v is None:
            return v
        for yr, frac in v.items():
            if not (0.0 <= frac <= 1.0):
                raise ValueError(
                    f"Abatement fraction for year {yr} must be between 0 and 1, got {frac}."
                )
        return v


class RunRequest(BaseModel):
    """Request body for POST /runs.

    For named NGFS scenarios, supply only company_id and scenario_id.
    For a fully custom scenario, set scenario_id="custom" and supply
    the custom_scenario block with your own carbon price path, WACC premium,
    and abatement targets.
    """

    company_id: str
    scenario_id: str       # "nze_2050" | "delayed_transition" | "current_policies" | "custom"
    custom_scenario: Optional[CustomScenarioParams] = None


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str
    version: str


class RunResponse(RunResults):
    """Full run results (inherits from RunResults)."""

    pass


# ── Ratings ─────────────────────────────────────────────────────────────────

class RatingRequest(BaseModel):
    """Request body for POST /ratings."""

    company_id: str
    tier: str = "free"   # "free" | "analyst" | "professional" | "enterprise"

    # Firm-configurable weight profile for the composite CRI score.
    # Options: "equal" (default) | "physical_focus" | "transition_focus" |
    #          "financial_focus" | "custom"
    # When "custom", also supply custom_weights as [physical, transition, financial]
    # summing to 1.0.
    weight_profile: str = "equal"
    custom_weights: Optional[List[float]] = None


class PillarSummary(BaseModel):
    label: str
    score: Optional[float] = None
    key_drivers: Optional[List[str]] = None


class RatingResponse(BaseModel):
    """Rating response — content varies by tier."""

    company_id: str
    company_name: str
    rating: str                           # A–E
    rating_label: str
    confidence: str
    summary: str
    sector_rank: Optional[str] = None

    # Free tier: pillar labels only
    physical_risk_label: str
    transition_risk_label: str
    financial_impact_label: str

    # Paid tier fields (None for free)
    composite_score: Optional[float] = None
    physical_risk_score: Optional[float] = None
    transition_risk_score: Optional[float] = None
    financial_impact_score: Optional[float] = None
    physical_drivers: Optional[List[str]] = None
    transition_drivers: Optional[List[str]] = None
    financial_drivers: Optional[List[str]] = None

    tier: str
    locked_features: List[str]
    upgrade_prompt: Optional[str] = None

    # Weight transparency — always returned so the firm can audit their rating
    weight_profile_used: str = "equal"
    weights_applied: Optional[dict] = None   # {"physical": 0.33, "transition": 0.33, "financial": 0.33}


# ── Disclosure reports ───────────────────────────────────────────────────────

class DisclosureRequest(BaseModel):
    """Request body for POST /reports/{framework}."""

    company_id: str
    reporting_year: Optional[int] = None
    tier: str = "professional"


class DisclosureResponse(BaseModel):
    """Wrapper for any disclosure report."""

    framework: str
    framework_version: str
    generated_at: str
    company_id: str
    company_name: str
    reporting_year: int
    data_sources: List[str]
    caveats: List[str]
    sections: Dict[str, Any]


# ── Physical Hazard Report ────────────────────────────────────────────────────

class AssetInput(BaseModel):
    """Inline asset definition for POST /reports/physical.

    Clients who don't have a registered company_id can submit a single asset
    directly. Financial fields (EBITDA, WACC, net_debt) are NOT required —
    the physical report only needs location, production, and carrying value.
    """
    id: str = "custom_asset"
    name: str
    commodity: str                   # e.g. "iron_ore", "crude_oil", "copper"
    region: str                      # e.g. "AU-WA", "US-TX", "CL-02"
    baseline_production: float       # Mtonnes or Mbbl depending on commodity
    production_unit: str = "Mtonnes"
    baseline_unit_cost: float        # USD / tonne  (or USD / bbl)
    energy_cost_share: float = 0.30  # fraction of unit cost that is energy
    carrying_value: float            # USD millions (replacement cost)
    remaining_life_years: int = 25


class PhysicalReportRequest(BaseModel):
    """Request body for POST /reports/physical.

    Accepts EITHER:
      • company_id  — run the physical report for a registered seed company.
      • asset       — run on a single inline asset (no company registration needed).

    company_name is optional and used only for narrative output when
    supplying an inline asset.
    """
    company_id: Optional[str] = None
    company_name: Optional[str] = "Custom Asset"
    asset: Optional[AssetInput] = None

    @field_validator("company_id", "asset", mode="before")
    @classmethod
    def at_least_one(cls, v, info):
        # Pydantic calls field validators individually; cross-field check
        # is done in the endpoint. Just pass through here.
        return v


class HazardYearOut(BaseModel):
    """Per-year physical risk output."""
    year: int
    physical_loss_cost: float        # USD millions
    adaptation_capex: float          # USD millions
    physical_loss_by_hazard: Dict[str, float]
    total_loss_fraction: float       # 0–1  (fraction of baseline revenue)


class PhysicalHazardReportResponse(BaseModel):
    """Standalone physical climate risk report.

    Returned by POST /reports/physical. Contains no transition risk,
    no carbon cost, no valuation — purely asset-level hazard assessment.
    """
    # Identifiers
    company_id: str
    company_name: str
    run_id: str
    model_version: str
    generated_at: str

    # Summary scores
    physical_score: float            # 0–100
    physical_label: str              # Low / Moderate / Elevated / High / Critical

    # Peak-loss summary
    peak_loss_year: int
    peak_loss_usd: float             # USD millions
    peak_loss_hazard: str

    # Adaptation capex totals
    total_adaptation_capex_nze: float   # USD millions, 25-year sum
    total_adaptation_capex_cp: float

    # Dominant hazards at 2035 under Current Policies
    hazard_breakdown_2035: Dict[str, float]

    # TCFD-aligned narrative
    narrative: str

    # Per-year trajectories
    years_nze:     List[HazardYearOut]
    years_delayed: List[HazardYearOut]
    years_cp:      List[HazardYearOut]

    # Data provenance
    data_sources: List[str]
    caveats: List[str]
    scenario_set: str = "NGFS Phase 4"


# ── Tiers ────────────────────────────────────────────────────────────────────

class TierInfo(BaseModel):
    tier: str
    label: str
    price: str
    cta: str
    description: str
    features: Dict[str, Any]


class TiersResponse(BaseModel):
    tiers: List[TierInfo]


# ── Scoped / modular run ──────────────────────────────────────────────────────

VALID_SCOPES = {
    "physical",
    "transition",
    "financial",
    "physical_transition",
    "full_cri",
}


class ScopedRunRequest(BaseModel):
    """Request body for POST /runs/scoped.

    The firm selects exactly which analysis pillars they need.

    scope options
    -------------
    physical            Asset-level hazard assessment + production loss only.
                        No carbon pricing, no valuation.
    transition          Carbon cost trajectory, commodity demand shifts,
                        EBITDA compression under NGFS scenarios only.
    financial           Full DCF enterprise valuation across all scenarios
                        (physical + transition computed internally as inputs
                        but not returned as standalone outputs).
    physical_transition Physical AND transition combined; no DCF.
    full_cri            All three pillars + composite CRI rating.
    """

    company_id: str
    scope: str  # one of the VALID_SCOPES strings above

    @field_validator("scope")
    @classmethod
    def scope_must_be_valid(cls, v: str) -> str:
        v = v.lower()
        if v not in VALID_SCOPES:
            raise ValueError(
                f"Invalid scope '{v}'. "
                f"Choose from: {sorted(VALID_SCOPES)}"
            )
        return v


# -- Per-year sub-models ------------------------------------------------------

class PhysicalYearOut(BaseModel):
    year: int
    physical_loss_cost: float        # USD
    adaptation_capex: float          # USD
    physical_loss_by_hazard: Dict[str, float]
    total_loss_fraction: float       # 0–1


class TransitionYearOut(BaseModel):
    year: int
    carbon_cost: float               # USD
    carbon_cost_pct_ebitda: float
    revenue_by_commodity: Dict[str, float]
    emissions_scope1: float
    emissions_scope2: float
    emissions_scope3: float


# -- Pillar report models ------------------------------------------------------

class PhysicalRiskOut(BaseModel):
    """Physical risk pillar output — returned when scope includes 'physical'."""

    company_id: str
    company_name: str
    run_id: str
    model_version: str

    # Scores
    physical_score: float            # 0–100
    physical_label: str              # Low / Moderate / Elevated / High / Critical

    # Peak-loss summary
    peak_loss_year: int
    peak_loss_usd: float
    peak_loss_hazard: str

    # Adaptation capex totals over the 25-year horizon
    total_adaptation_capex_nze: float
    total_adaptation_capex_cp: float

    # Hazard cost breakdown at 2035 under Current Policies
    hazard_breakdown_2035: Dict[str, float]

    # TCFD-aligned narrative
    narrative: str

    # Per-year trajectories (25 years per scenario)
    years_nze:     List[PhysicalYearOut]
    years_delayed: List[PhysicalYearOut]
    years_cp:      List[PhysicalYearOut]


class TransitionRiskOut(BaseModel):
    """Transition risk pillar output — returned when scope includes 'transition'."""

    company_id: str
    company_name: str
    run_id: str
    model_version: str

    # Scores
    transition_score: float          # 0–100
    transition_label: str

    # EBITDA compression at key dates under NZE
    ebitda_compression_2030_nze: Optional[float] = None
    ebitda_compression_2040_nze: Optional[float] = None

    # Carbon cost as % of EBITDA
    carbon_pct_ebitda_2030_nze: Optional[float] = None
    carbon_pct_ebitda_2030_cp:  Optional[float] = None

    # Narrative
    narrative: str

    # Per-year trajectories
    years_nze:     List[TransitionYearOut]
    years_delayed: List[TransitionYearOut]
    years_cp:      List[TransitionYearOut]


# -- Top-level scoped response -------------------------------------------------

class ScopedRunResponse(BaseModel):
    """Response envelope for POST /runs/scoped.

    Only the pillars requested by the firm are populated.
    All other pillar fields are null.
    """

    scope: str
    scope_label: str
    run_id: str

    # Physical pillar — populated for scopes: physical, physical_transition, full_cri
    physical: Optional[PhysicalRiskOut] = None

    # Transition pillar — populated for scopes: transition, physical_transition, full_cri
    transition: Optional[TransitionRiskOut] = None

    # Valuation — populated for scopes: financial, full_cri
    # Keys: "nze" | "delayed" | "cp"  →  full RunResults dict
    valuation_results: Optional[Dict[str, Any]] = None

    # Composite rating — populated for scope: full_cri only
    rating_result: Optional[Dict[str, Any]] = None
