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


class AssetLocationResponse(BaseModel):
    """Asset-level metadata including precise coordinates."""

    id: str
    name: str
    commodity: str
    region: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    equipment_type: Optional[str] = None
    baseline_production: float
    production_unit: str
    carrying_value_usd_m: float


class FinancialsResponse(BaseModel):
    """Company-level baseline financials (USD millions)."""

    revenue_usd_m: float
    ebitda_usd_m: float
    capex_usd_m: float
    wacc_pct: float
    net_debt_usd_m: float


class CompanyResponse(BaseModel):
    """Company metadata for GET /companies — full profile including asset locations."""

    id: str
    name: str
    sector: str
    region: str
    financials: FinancialsResponse
    assets: List[AssetLocationResponse]
    data_quality: str


class CreateCompanyRequest(BaseModel):
    """POST /companies — register a new company for CRI assessment.

    Supply the core financials and at least one asset location.
    The engine derives country ISO2 from lat/lon and fills in any missing
    fields (EBITDA, CAPEX) from sector benchmarks.
    """
    # Identity
    id:              str    # unique slug, e.g. "rio-tinto-plc" — used as lookup key
    name:            str
    sector:          str    # free-text; matched to engine taxonomy (e.g. "Mining", "Chemicals")
    hq_region:       str = "global"

    # Financials (all USD millions)
    revenue_usd_m:               float
    ebitda_usd_m:                Optional[float] = None   # defaults to 20% of revenue
    net_debt_usd_m:              float = 0.0
    equity_value_usd_m:          Optional[float] = None   # enterprise value proxy
    wacc_pct:                    float = 0.10
    capex_usd_m:                 Optional[float] = None   # defaults to 8% of revenue

    # Emissions (tonnes CO2e / year)
    scope1_tco2:     float = 0.0
    scope2_tco2:     float = 0.0
    scope3_tco2:     float = 0.0

    # Primary asset (optional — enables geospatial hazard calibration)
    asset_lat:                   Optional[float] = None
    asset_lon:                   Optional[float] = None
    asset_replacement_cost_usd_m: Optional[float] = None  # Replacement Cost New
    asset_name:                  Optional[str]  = None

    # Data quality flag influences confidence multiplier in reports
    data_quality:    str = "medium"   # "low" | "medium" | "high"


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

    Gap 5 — RCN vs NBV: physical damage is computed on Replacement Cost New
    (replacement_cost_new_usd_m), NOT on Net Book Value (carrying_value).
    These diverge materially for older assets (NBV may be 30–50% of RCN after
    depreciation). Damage ratios from hazard intensity curves are applied to
    RCN and then compared to carrying_value to assess insurance adequacy.

    If replacement_cost_new_usd_m is not supplied it defaults to carrying_value
    (implying a new asset, NBV = RCN), but a warning is surfaced in the report.
    """
    id: str = "custom_asset"
    name: str
    commodity: str                   # e.g. "iron_ore", "crude_oil", "copper"
    region: str                      # e.g. "AU-WA", "US-TX", "CL-02"
    baseline_production: float       # Mtonnes or Mbbl depending on commodity
    production_unit: str = "Mtonnes"
    baseline_unit_cost: float        # USD / tonne  (or USD / bbl)
    energy_cost_share: float = 0.30  # fraction of unit cost that is energy
    carrying_value: float            # USD millions — Net Book Value (NBV) for balance sheet
    replacement_cost_new_usd_m: Optional[float] = None  # RCN for damage calculation (Gap 5)
    asset_elevation_m: Optional[float] = None   # metres above sea level for DEM check (Gap 6)
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

    Gap 7 — Horizon decoupling: near-term (2026–2030) outputs are reported
    separately from long-term (2030/2040/2050) to prevent institutional
    reviewers from confusing short-term operational risk with long-term
    strategic exposure. CSRD ESRS E1 requires this separation.
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

    # Gap 7 — Near-term block (2026–2030): separate from long-term for CSRD
    hazard_breakdown_2028: Dict[str, float] = {}   # snapshot at 2028 under CP
    near_term_physical_loss_nze: float = 0.0       # USD millions, 2026–2030 sum, NZE
    near_term_physical_loss_cp:  float = 0.0       # USD millions, 2026–2030 sum, CP

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

    # Gap 3 — Audit trail for CSRD / IFRS S2 compliance
    audit_trail: Dict[str, Any] = {}


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


# ── Aladdin-style Portfolio Risk ──────────────────────────────────────────────

class PortfolioPositionIn(BaseModel):
    """One position (company / borrower) in a portfolio risk request."""
    company_id:       str
    company_name:     str
    exposure_usd_m:   float           # gross exposure in USD millions
    sector:           str
    region:           str = "Global"
    physical_score:   float = 50.0    # 0–100; supply from a prior CRI run if available
    transition_score: float = 50.0
    # Task 53: Optional multi-factor transition sub-scores (0–100 each).
    # If omitted, decomposition is inferred from sector weights.
    policy_score:     Optional[float] = None
    technology_score: Optional[float] = None
    market_score:     Optional[float] = None
    litigation_score: Optional[float] = None

class PortfolioRiskRequest(BaseModel):
    """POST /portfolio/risk — aggregate portfolio climate VaR."""
    positions: List[PortfolioPositionIn]
    run_stress: bool = True           # include 1.5/2/3/4°C stress scenarios

class PortfolioRiskResponse(BaseModel):
    """Aladdin-style portfolio climate risk report (v0.9)."""
    run_id:             str
    n_positions:        int
    total_exposure_usd_m: float
    # Portfolio CVaR — Current Policies (worst case)
    portfolio_eal_cp_usd_m:    float
    portfolio_var95_cp_usd_m:  float
    portfolio_cvar95_cp_usd_m: float
    portfolio_var99_cp_usd_m:  float
    portfolio_cvar99_cp_usd_m: float
    # Portfolio CVaR — Net Zero (best case)
    portfolio_eal_nze_usd_m:   float
    portfolio_var99_nze_usd_m: float
    portfolio_cvar99_nze_usd_m:float
    # Diversification
    diversification_benefit_usd_m: float
    diversification_ratio:         float
    # Concentration
    herfindahl_index:       float
    top3_concentration_pct: float
    # Detailed output
    positions:              List[Dict[str, Any]] = []
    factor_attribution:     Dict[str, Any] = {}
    stress_scenarios:       List[Dict[str, Any]] = []
    # Gap 2: MC CVaR regime flags
    stress_regime_active:   bool = False
    stress_weight:          float = 0.0
    # Gap 4: macro factor summary
    macro_factor_summary:   Dict[str, Any] = {}
    # Gap 6: term structure (5/10/30-year EAL/VaR99)
    term_structure:         List[Dict[str, Any]] = []
    # Task 53: multi-factor transition risk decomposition
    transition_factor_cp:   Dict[str, Any] = {}


# ── Aladdin-style Credit Risk ─────────────────────────────────────────────────

class CounterpartyIn(BaseModel):
    """One borrower / bond issuer in a credit risk request."""
    company_id:       str
    company_name:     str
    sector:           str
    ead_usd_m:        float           # exposure at default (USD M)
    baseline_pd:      float           # annual PD, e.g. 0.03 = 3%
    baseline_lgd:     float = 0.45
    physical_score:   float = 50.0
    transition_score: float = 50.0
    collateral_type:  str   = "unsecured"
    scenario:         str   = "cp"
    region:           str   = "Global"

class PortfolioCreditRequest(BaseModel):
    """POST /portfolio/credit — climate-adjusted PD/LGD/ECL/RWA for loan book."""
    obligors: List[CounterpartyIn]

class PortfolioCreditResponse(BaseModel):
    """Climate-adjusted credit risk summary for a portfolio."""
    n_obligors:                int
    total_ead_usd_m:           float
    scenario:                  str
    baseline_ecl_usd_m:        float
    climate_ecl_usd_m:         float
    ecl_increment_usd_m:       float
    ecl_uplift_pct:            float
    baseline_rwa_usd_m:        float
    climate_rwa_usd_m:         float
    rwa_increment_usd_m:       float
    incremental_capital_usd_m: float
    sector_breakdown:          Dict[str, Any] = {}
    obligors:                  List[Dict[str, Any]] = []
    # Gap 6: portfolio-level IFRS 9 lifetime ECL term structure
    portfolio_term_structure:  List[Dict[str, Any]] = []
    calibration_method:        str = ""


# ── Gap 7: Real-time position monitor (SSE) ───────────────────────────────────

class PositionTickIn(BaseModel):
    """A single live position update (price/exposure change) for SSE stream."""
    company_id:       str
    company_name:     str
    sector:           str
    region:           str = "Global"
    exposure_usd_m:   float
    physical_score:   float = 50.0
    transition_score: float = 50.0

class PortfolioStreamRequest(BaseModel):
    """POST /portfolio/stream — initiate SSE real-time position monitor."""
    positions: List[PositionTickIn]
    interval_seconds: float = 5.0   # push frequency
    max_ticks: int = 60             # stop after N ticks (0 = run until client disconnects)


# ── Task 54: CVaR Optimiser ───────────────────────────────────────────────────

class PortfolioOptimiseRequest(BaseModel):
    """POST /portfolio/optimise — minimise CVaR₉₉ via SLSQP."""
    positions:       List[PortfolioPositionIn]
    sector_cap:      float = 0.40    # max weight in any single sector
    single_name_cap: float = 0.30    # max single-name weight
    hhi_limit:       float = 0.25    # max Herfindahl-Hirschman Index of weights
    w_min:           float = 0.01    # minimum position weight

class PortfolioOptimiseResponse(BaseModel):
    """CVaR optimisation result."""
    status:                  str
    message:                 str
    original_cvar99_usd_m:   float = 0.0
    optimised_cvar99_usd_m:  float = 0.0
    cvar99_reduction_pct:    float = 0.0
    original_eal_usd_m:      float = 0.0
    optimised_eal_usd_m:     float = 0.0
    hhi_original:            float = 0.0
    hhi_optimised:           float = 0.0
    positions:               List[Dict[str, Any]] = []
    binding_constraints:     List[str] = []
    n_iterations:            int = 0
    solver:                  str = "scipy-SLSQP"
    methodology:             str = ""


# ── Task 56: Rules engine ────────────────────────────────────────────────────

class RulesCheckRequest(BaseModel):
    """POST /portfolio/rules — check portfolio against configurable risk limits."""
    positions:              List[PortfolioPositionIn]
    cvar99_limit_usd_m:     Optional[float] = None   # hard CVaR99 limit
    sector_concentration_limit: float = 0.40         # max sector % of CVaR
    single_name_limit:      float = 0.30             # max single-name % of CVaR
    hhi_limit:              float = 0.30             # HHI of MC-CVaR weights
    eal_to_revenue_limit:   Optional[float] = None   # EAL / total exposure threshold

class RulesCheckResponse(BaseModel):
    """Rules engine output — breaches and alerts."""
    overall_status:   str               # "PASS" | "WARN" | "BREACH"
    checks:           List[Dict[str, Any]] = []
    breach_count:     int = 0
    warn_count:       int = 0
    portfolio_cvar99_usd_m: float = 0.0
    portfolio_eal_usd_m:    float = 0.0
    hhi:                    float = 0.0
    generated_at:           str = ""


# ── Task 57: Validation framework ────────────────────────────────────────────

class ValidationRequest(BaseModel):
    """POST /portfolio/validate — backtesting and model quality metrics."""
    # Realised losses (actuals) vs model predictions over a set of periods
    # Each entry: {"period": str, "predicted_eal": float, "realised_loss": float,
    #              "predicted_pd": float, "realised_default": int (0|1)}
    observations: List[Dict[str, Any]]
    # Which metrics to compute (all by default)
    metrics: List[str] = ["gini", "brier", "kupiec"]

class ValidationResponse(BaseModel):
    """Model validation metrics."""
    n_observations:  int = 0
    gini_coefficient: Optional[float] = None   # discriminatory power (0–1; 0.6+ good)
    brier_score:      Optional[float] = None   # PD calibration (lower = better; <0.25 good)
    kupiec_pof:       Optional[Dict[str, Any]] = None  # VaR coverage test
    hl_test:          Optional[Dict[str, Any]] = None  # Hosmer-Lemeshow calibration
    summary:          str = ""
    methodology:      str = (
        "Gini: AUROC×2-1 (Engelmann 2003). "
        "Brier: mean squared error of PD vs default indicator. "
        "Kupiec: Proportion-of-Failures test for VaR coverage (Basel III §MAR99). "
        "HL: Hosmer-Lemeshow χ² calibration test (10 bins)."
    )


# ── Task 72: Decision-support first — primary risk management API ─────────────
# The Decision Brief is the primary output for risk managers, credit committees,
# and portfolio teams. Compliance outputs (TCFD/ISSB/CSRD) are a derived layer.
#
# Design principle: answer "should I hold this exposure?" before answering
# "how do I disclose this?".

class DecisionRequest(BaseModel):
    """POST /decision — primary risk management assessment.

    For a company (by id) or a custom asset, returns a Decision Brief
    structured for risk managers and credit committees, not compliance teams.

    Scope options:
      "quick"     — physical hazard + transition signal only (fast, ~1s)
      "full"      — all pillars: physical + transition + valuation + credit
      "portfolio" — portfolio-level CVaR signal for a list of positions
    """
    company_id:       Optional[str]   = None
    scope:            str             = "full"    # "quick" | "full" | "portfolio"
    scenario:         str             = "cp"      # "nze" | "delayed" | "cp"
    horizon_year:     int             = 2035
    # Optional override fields for quick one-off assessments
    exposure_usd_m:   Optional[float] = None      # credit/investment exposure
    sector:           Optional[str]   = None
    region:           Optional[str]   = None
    # Portfolio mode: pass positions instead of company_id
    positions:        Optional[List[PortfolioPositionIn]] = None
    # Risk appetite controls signal thresholds — use "conservative" for DFIs/insurers,
    # "aggressive" for PE/trading desks with explicit high-risk mandates.
    risk_appetite:    str             = "moderate"  # "conservative" | "moderate" | "aggressive"


class RiskSignal(BaseModel):
    """Primary risk management signal — the first thing a risk manager reads."""
    action:            str    # "HOLD" | "WATCH" | "REDUCE" | "EXIT"
    confidence:        str    # "HIGH" | "MEDIUM" | "LOW"
    rationale:         str    # one-sentence justification
    urgency:           str    # "IMMEDIATE" | "NEAR_TERM" | "MEDIUM_TERM" | "LONG_TERM"


class FinancialImpact(BaseModel):
    """Quantified financial impact — what the risk manager cares about."""
    expected_annual_loss_usd_m:   float   # EAL under current policies
    worst_case_1pct_usd_m:        float   # CVaR₉₉ — the tail loss
    ebitda_impact_pct:            float   # EAL as % of company EBITDA
    revenue_at_risk_pct:          float   # revenue at risk under stress
    stranded_asset_value_usd_m:   float   # stranded value under NZE scenario
    credit_spread_widening_bps:   float   # implied spread widening (Moody's calibration)
    wacc_uplift_bps:              float   # climate WACC premium
    npv_haircut_pct:              float   # % enterprise value reduction vs unconstrained


class RiskDriver(BaseModel):
    """One material risk driver with quantified contribution."""
    factor:          str    # e.g. "Flood (riverine) — Queensland assets"
    category:        str    # "physical" | "transition" | "macro"
    contribution_pct: float # share of total EAL from this factor
    horizon:         str    # "2025–2030" | "2030–2040" | "2040–2050"
    severity:        str    # "LOW" | "MODERATE" | "HIGH" | "CRITICAL"


class RecommendedAction(BaseModel):
    """Concrete action for the risk manager — not a disclosure bullet."""
    priority:    int    # 1 = highest
    action:      str    # what to do
    rationale:   str    # why
    cost_usd_m:  Optional[float] = None   # estimated cost if applicable
    deadline:    Optional[str]   = None   # "Q3 2026", "Before 2030", etc.


class DecisionBrief(BaseModel):
    """Decision-first risk output for risk managers and credit committees.

    Structured so the first field answers the decision, the second quantifies it,
    and compliance outputs are in the final `compliance_layer` field.

    This is the primary output of the CRI engine. TCFD/ISSB reports are generated
    from this — not the other way around.
    """
    # ── 1. The signal (read first) ────────────────────────────────────────────
    signal:              RiskSignal

    # ── 2. Quantified financial impact ────────────────────────────────────────
    financial_impact:    FinancialImpact

    # ── 3. What's driving the risk ────────────────────────────────────────────
    top_drivers:         List[RiskDriver]      # ranked by contribution

    # ── 4. What the risk manager should do ───────────────────────────────────
    recommended_actions: List[RecommendedAction]

    # ── 5. Position context ───────────────────────────────────────────────────
    company_id:          Optional[str]   = None
    company_name:        Optional[str]   = None
    sector:              str             = ""
    region:              str             = ""
    scenario_label:      str             = ""   # "Current Policies (SSP3-7.0)"
    horizon_year:        int             = 2035
    data_quality:        str             = ""   # "LIVE_API" | "MODELLED" | "FALLBACK"
    model_confidence:    str             = ""   # "HIGH" | "MEDIUM" | "LOW"

    # ── 6. Compliance layer (secondary — derive TCFD/ISSB from this) ──────────
    compliance_layer:    Optional[Dict[str, Any]] = None   # {tcfd, issb, csrd} on demand

    run_id:              str             = ""
    generated_at:        str             = ""
    methodology_note:    str             = (
        "Decision brief produced by CRI Engine. Physical risk: WRI Aqueduct 4.0 + "
        "IPCC AR6 log-logistic hazard fragility. Transition: NGFS Phase 4 + IEA NZE 2023. "
        "Financial: climate-adjusted DCF with regime-based terminal value. "
        "Compliance outputs (TCFD/ISSB/CSRD) are derived from this brief on request."
    )


# ── Pillar 2: Standardized Asset Data Schema (batch onboarding) ───────────────
# Clients supply raw asset-level data — no company registration required.
# Engine runs physical + transition scoring across all supplied assets and
# returns per-asset results plus portfolio aggregates.

class StandardAssetIn(BaseModel):
    """Canonical asset record for batch ingestion.

    Covers all data a risk manager needs to supply for a full CRI assessment.
    The engine derives country_iso2 from lat/lon if not provided.
    Missing financial fields are estimated from sector benchmarks.
    """
    asset_id:               str              # client's own reference (e.g. "AU-QLD-001")
    name:                   str
    lat:                    float            # decimal degrees  (e.g. -27.47)
    lon:                    float            # decimal degrees  (e.g. 153.02)
    sector:                 str              # matches engine sector taxonomy
    country_iso2:           Optional[str]  = None   # auto-derived from lat/lon if omitted
    # Financials
    replacement_cost_usd_m: float            # Replacement Cost New — for physical damage
    annual_revenue_usd_m:   float
    ebitda_usd_m:           Optional[float] = None   # defaults to 20% of revenue if omitted
    net_debt_usd_m:         Optional[float] = None   # defaults to 0
    # Emissions (tonnes CO2e / year)
    scope1_tco2:            float = 0.0
    scope2_tco2:            float = 0.0
    scope3_tco2:            Optional[float] = None
    # Optional qualitative features
    critical_features:      List[str] = []  # e.g. ["power_plant","water_cooling","rail_access"]
    asset_type:             str = "industrial"  # "industrial"|"office"|"data_center"|"infrastructure"
    # Gap 9: optional component dependency graph for critical-path BI
    # Supply as adjacency dict: {"nodes": {"boiler": {"hazard_sensitivities": {"flood": 0.6},
    # "recovery_days": 30, "has_redundancy": false}, ...},
    # "upstream": {"boiler": ["water_intake"], ...}, "description": "..."}
    # If omitted, engine auto-selects the sector default graph.
    component_graph:        Optional[Dict[str, Any]] = None


class BatchAssessmentRequest(BaseModel):
    """POST /assets/batch — assess one or many assets in a single call."""
    assets:          List[StandardAssetIn]
    horizon_year:    int  = 2035
    risk_appetite:   str  = "moderate"    # "conservative"|"moderate"|"aggressive"
    scenarios:       List[str] = ["nze", "cp"]


class AssetAssessmentResult(BaseModel):
    """Per-asset output from batch assessment."""
    asset_id:                   str
    name:                       str
    lat:                        float
    lon:                        float
    sector:                     str
    country_iso2:               str
    # Risk scores (0–100)
    physical_score:             float
    transition_score:           float
    composite_score:            float
    # Signal
    signal:                     str    # "HOLD"|"WATCH"|"REDUCE"|"EXIT"
    signal_confidence:          str    # "HIGH"|"MEDIUM"|"LOW"
    # Financial impact
    expected_annual_loss_usd_m: float
    eal_pct_revenue:            float
    eal_pct_rcv:                float
    worst_case_cvar99_usd_m:    float
    stranded_asset_value_usd_m: float
    credit_spread_widening_bps: float
    # Dominant hazard
    primary_hazard:             str
    primary_hazard_contribution_pct: float
    hazard_breakdown:           Dict[str, float] = {}
    # Transition
    carbon_cost_usd_m_2030:     float = 0.0
    ebitda_compression_pct:     float = 0.0
    # Data quality
    data_quality:               str   # "CALIBRATED"|"MODELLED"|"ESTIMATED"
    caveats:                    List[str] = []


class BatchAssessmentResponse(BaseModel):
    """Response from POST /assets/batch — per-asset results + portfolio aggregates."""
    run_id:                 str
    n_assets:               int
    horizon_year:           int
    risk_appetite:          str
    generated_at:           str
    # Portfolio totals
    total_rcv_usd_m:        float
    total_revenue_usd_m:    float
    total_eal_usd_m:        float
    portfolio_eal_pct_rcv:  float
    portfolio_cvar99_usd_m: float
    # Signal distribution
    signal_counts:          Dict[str, int]  # {"HOLD":2,"WATCH":1,"REDUCE":0,"EXIT":0}
    highest_risk_asset_id:  str
    highest_risk_asset_name: str
    # Breakdowns
    sector_breakdown:       Dict[str, Any] = {}
    # Per-asset results
    assets:                 List[AssetAssessmentResult]
    # Methodology
    data_sources:           List[str] = []
    methodology_note:       str = (
        "Physical EAL: INFORM Risk Index 2024 + WRI 2023 EAL fractions, 60/40 blended "
        "with log-logistic fragility curves (IPCC AR6). Transition: NGFS Phase 4 carbon "
        "price trajectories × sector emissions intensity. Financial: climate-adjusted DCF."
    )


# ── Pillar 3: LLM-generated content from engine data ─────────────────────────
# Articles, narratives, and summaries are produced by an LLM micro-agent
# seeded with deterministic engine output. No numbers are invented.

class ArticleRequest(BaseModel):
    """POST /generate/article — produce a grounded article from live engine data."""
    company_id:    str
    article_type:  str = "linkedin"   # "linkedin"|"investor_memo"|"case_study"|"regulatory_commentary"
    tone:          str = "professional"  # "professional"|"executive"|"accessible"
    word_count:    int = 600
    risk_appetite: str = "moderate"
    focus:         Optional[str] = None  # optional emphasis: "physical"|"transition"|"financial"


class ArticleResponse(BaseModel):
    """Generated article with full data provenance."""
    company_id:       str
    company_name:     str
    article_type:     str
    tone:             str
    content:          str             # the article text
    word_count_actual: int
    key_numbers_used: List[str]       # real engine numbers embedded in the article
    signal:           str             # HOLD/WATCH/REDUCE/EXIT that drove the framing
    data_source:      str = "CRI Engine — deterministic physics-based assessment"
    run_id:           str
    generated_at:     str
    disclaimer:       str = (
        "All figures are produced by the CRI Engine from deterministic climate risk models. "
        "Data provenance: CMIP6 IPCC AR6 / NGFS Phase 4 / INFORM Risk Index 2024 / WRI 2023. "
        "This content is generated for informational purposes and does not constitute investment advice."
    )


class NarrativeRequest(BaseModel):
    """POST /generate/narrative — produce one section of a risk narrative."""
    company_id: str
    section:    str   # "executive_summary"|"physical_risk"|"transition_risk"|"financial_impact"|"recommendations"
    audience:   str = "risk_manager"  # "board"|"risk_manager"|"auditor"|"investor"|"public"
    risk_appetite: str = "moderate"


class NarrativeResponse(BaseModel):
    """One narrative section with supporting metrics."""
    section:      str
    audience:     str
    content:      str
    key_metrics:  Dict[str, Any] = {}
    run_id:       str
    generated_at: str
