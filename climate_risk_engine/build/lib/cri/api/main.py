"""FastAPI REST API for the Climate Risk Intelligence engine."""

from __future__ import annotations

# ── Self-healing multipart install ───────────────────────────────────────────
# FastAPI calls ensure_multipart_is_installed() at *app startup* for any route
# that uses UploadFile.  On Render the cached venv sometimes omits it despite
# requirements.txt listing it, so we install it inline before FastAPI loads.
try:
    import multipart  # noqa: F401 — python-multipart
except ImportError:
    import subprocess, sys  # noqa: E401
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "python-multipart", "--quiet"],
        stderr=subprocess.DEVNULL,
    )
# ─────────────────────────────────────────────────────────────────────────────

import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import dataclasses

from ..data.companies_seed import all_seed as get_all_companies
from ..engine.orchestrator import run as run_engine
from ..engine.orchestrator import run_scoped as run_scoped_engine
from ..engine.scope import ReportScope
from ..data.schemas import (
    CarbonPricePath,
    Scenario,
    ScenarioFamily,
)
from ..scenarios import (
    CURRENT_POLICIES,
    DELAYED_TRANSITION,
    NZE_2050,
)
from .schemas import (
    AssetInput,
    CompanyResponse,
    DisclosureRequest,
    DisclosureResponse,
    HazardYearOut,
    HealthResponse,
    PhysicalHazardReportResponse,
    PhysicalReportRequest,
    PhysicalRiskOut,
    PhysicalYearOut,
    RatingRequest,
    RatingResponse,
    CustomScenarioParams,
    RunRequest,
    RunResponse,
    ScenarioResponse,
    ScopedRunRequest,
    ScopedRunResponse,
    TierInfo,
    TiersResponse,
    TransitionRiskOut,
    TransitionYearOut,
)


# Initialize FastAPI app
app = FastAPI(
    title="Climate Risk Intelligence API",
    description="REST API for climate financial risk modelling",
    version="0.3.0",
)

# Add CORS middleware (allow all origins for dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build registries
SCENARIO_REGISTRY = {
    "nze_2050": NZE_2050,
    "delayed_transition": DELAYED_TRANSITION,
    "current_policies": CURRENT_POLICIES,
}

COMPANY_REGISTRY = get_all_companies()

# Custom scenarios created via POST /scenarios are stored here at runtime.
# Not persisted across restarts; persistence is a Phase 2 feature.
CUSTOM_SCENARIO_REGISTRY: dict[str, Scenario] = {}


def _build_custom_scenario(params: CustomScenarioParams, scenario_id: str = 'custom') -> Scenario:
    """Build a Scenario from a CustomScenarioParams request body."""
    return Scenario(
        id=scenario_id,
        name=params.name,
        family=ScenarioFamily.CUSTOM,
        horizon=(2026, 2050),
        description=params.description,
        version='0.4.0',
        carbon_prices=[CarbonPricePath(region='global', path=params.carbon_price_path)],
        commodity_curves=CURRENT_POLICIES.commodity_curves,
        risk_premium_bps=params.risk_premium_bps,
        abatement_targets=params.abatement_targets,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok", version="0.3.0")


@app.get("/scenarios", response_model=list[ScenarioResponse])
def list_scenarios() -> list[ScenarioResponse]:
    """Return list of available scenarios."""
    scenarios = []
    for scenario_id, scenario in SCENARIO_REGISTRY.items():
        scenarios.append(
            ScenarioResponse(
                id=scenario_id,
                name=scenario.name,
                description=scenario.description,
                family=scenario.family.value,
                version=scenario.version,
            )
        )
    return scenarios


@app.get("/companies", response_model=list[CompanyResponse])
def list_companies() -> list[CompanyResponse]:
    """Return list of available companies."""
    companies = []
    for company_id, company in COMPANY_REGISTRY.items():
        companies.append(
            CompanyResponse(
                id=company_id,
                name=company.name,
                sector=company.sector,
                region=company.hq_region,
            )
        )
    return companies


@app.post("/runs", response_model=RunResponse)
def run_simulation(request: RunRequest) -> RunResponse:
    """Run the climate risk engine for a given company and scenario.

    Returns full RunResults including per-year trajectory and valuation metrics.
    """
    scenario_id = request.scenario_id.lower()
    company_id = request.company_id.lower()

    if company_id not in COMPANY_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Company '{request.company_id}' not found. "
            f"Available: {list(COMPANY_REGISTRY.keys())}",
        )

    # Resolve scenario: named NGFS | saved custom | inline custom
    if scenario_id in SCENARIO_REGISTRY:
        scenario = SCENARIO_REGISTRY[scenario_id]
    elif scenario_id in CUSTOM_SCENARIO_REGISTRY:
        scenario = CUSTOM_SCENARIO_REGISTRY[scenario_id]
    elif scenario_id == 'custom':
        if not request.custom_scenario:
            raise HTTPException(
                status_code=422,
                detail=(
                    "scenario_id='custom' requires a 'custom_scenario' block with "
                    "at minimum a 'carbon_price_path' dict mapping years to USD/tCO2e prices."
                ),
            )
        scenario = _build_custom_scenario(request.custom_scenario)
    else:
        available = list(SCENARIO_REGISTRY.keys()) + list(CUSTOM_SCENARIO_REGISTRY.keys()) + ['custom']
        raise HTTPException(
            status_code=404,
            detail=f"Scenario '{request.scenario_id}' not found. Available: {available}",
        )

    company = COMPANY_REGISTRY[company_id]
    results = run_engine(company=company, scenario=scenario)
    return RunResponse(**results.model_dump())



@app.post("/scenarios", status_code=201)
def create_custom_scenario(params: CustomScenarioParams) -> ScenarioResponse:
    """Persist a custom scenario in the runtime registry.

    After creation the scenario can be referenced by its auto-generated id in
    subsequent POST /runs calls without re-sending the full carbon_price_path.
    Registry is in-memory and resets on server restart (Phase 2: persistence).
    """
    import re as _re
    slug = _re.sub(r"[^a-z0-9]+", "_", params.name.lower()).strip("_")
    scenario_id = f"custom_{slug}"
    if scenario_id in CUSTOM_SCENARIO_REGISTRY:
        raise HTTPException(
            status_code=409,
            detail=f"Custom scenario '{scenario_id}' already exists. Use a different name.",
        )
    scenario = _build_custom_scenario(params, scenario_id=scenario_id)
    CUSTOM_SCENARIO_REGISTRY[scenario_id] = scenario
    return ScenarioResponse(
        id=scenario_id, name=scenario.name, description=scenario.description,
        family=scenario.family.value, version=scenario.version,
    )


@app.delete("/scenarios/{scenario_id}", status_code=204)
def delete_custom_scenario(scenario_id: str) -> None:
    """Remove a custom scenario from the runtime registry."""
    if scenario_id in SCENARIO_REGISTRY:
        raise HTTPException(status_code=403, detail=f"Cannot delete built-in NGFS scenario '{scenario_id}'.")
    if scenario_id not in CUSTOM_SCENARIO_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Custom scenario '{scenario_id}' not found.")
    del CUSTOM_SCENARIO_REGISTRY[scenario_id]

# ── Scoped / modular run ─────────────────────────────────────────────────────

def _physical_to_response(p) -> PhysicalRiskOut:
    """Convert PhysicalRiskReport dataclass → PhysicalRiskOut Pydantic model."""
    def _years(lst) -> list[PhysicalYearOut]:
        return [
            PhysicalYearOut(
                year=y.year,
                physical_loss_cost=y.physical_loss_cost,
                adaptation_capex=y.adaptation_capex,
                physical_loss_by_hazard=y.physical_loss_by_hazard,
                total_loss_fraction=y.total_loss_fraction,
            )
            for y in lst
        ]
    return PhysicalRiskOut(
        company_id=p.company_id,
        company_name=p.company_name,
        run_id=p.run_id,
        model_version=p.model_version,
        physical_score=p.physical_score,
        physical_label=p.physical_label,
        peak_loss_year=p.peak_loss_year,
        peak_loss_usd=p.peak_loss_usd,
        peak_loss_hazard=p.peak_loss_hazard,
        total_adaptation_capex_nze=p.total_adaptation_capex_nze,
        total_adaptation_capex_cp=p.total_adaptation_capex_cp,
        hazard_breakdown_2035=p.hazard_breakdown_2035,
        narrative=p.narrative,
        years_nze=_years(p.years_nze),
        years_delayed=_years(p.years_delayed),
        years_cp=_years(p.years_cp),
    )


def _transition_to_response(t) -> TransitionRiskOut:
    """Convert TransitionRiskReport dataclass → TransitionRiskOut Pydantic model."""
    def _years(lst) -> list[TransitionYearOut]:
        return [
            TransitionYearOut(
                year=y.year,
                carbon_cost=y.carbon_cost,
                carbon_cost_pct_ebitda=y.carbon_cost_pct_ebitda,
                revenue_by_commodity=y.revenue_by_commodity,
                emissions_scope1=y.emissions_scope1,
                emissions_scope2=y.emissions_scope2,
                emissions_scope3=y.emissions_scope3,
            )
            for y in lst
        ]
    return TransitionRiskOut(
        company_id=t.company_id,
        company_name=t.company_name,
        run_id=t.run_id,
        model_version=t.model_version,
        transition_score=t.transition_score,
        transition_label=t.transition_label,
        ebitda_compression_2030_nze=t.ebitda_compression_2030_nze,
        ebitda_compression_2040_nze=t.ebitda_compression_2040_nze,
        carbon_pct_ebitda_2030_nze=t.carbon_pct_ebitda_2030_nze,
        carbon_pct_ebitda_2030_cp=t.carbon_pct_ebitda_2030_cp,
        narrative=t.narrative,
        years_nze=_years(t.years_nze),
        years_delayed=_years(t.years_delayed),
        years_cp=_years(t.years_cp),
    )


@app.post("/runs/scoped", response_model=ScopedRunResponse)
def run_scoped_analysis(request: ScopedRunRequest) -> ScopedRunResponse:
    """Run only the analysis pillars selected by the firm.

    The ``scope`` field determines what is computed and returned:

    - **physical**            – Asset-level hazard + production loss.
                                No carbon pricing, no valuation.
    - **transition**          – Carbon cost trajectory, commodity demand shifts,
                                EBITDA compression under NGFS scenarios.
    - **financial**           – Full DCF enterprise valuation across all three
                                scenarios (physical + transition are computed
                                internally as inputs but not returned standalone).
    - **physical_transition** – Physical AND transition combined; no DCF.
    - **full_cri**            – All three pillars + composite CRI rating (A–E).

    Unselected pillar fields are ``null`` in the response.
    """
    if request.company_id.lower() not in COMPANY_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Company '{request.company_id}' not found. "
                   f"Available: {list(COMPANY_REGISTRY.keys())}",
        )

    company = COMPANY_REGISTRY[request.company_id.lower()]
    scope   = ReportScope(request.scope)

    scoped = run_scoped_engine(company=company, scope=scope)

    # Convert physical dataclass → Pydantic response model
    physical_out = _physical_to_response(scoped.physical) if scoped.physical else None

    # Convert transition dataclass → Pydantic response model
    transition_out = _transition_to_response(scoped.transition) if scoped.transition else None

    # Valuation: RunResults are Pydantic models — serialise each scenario
    valuation_out = None
    if scoped.valuation_results:
        valuation_out = {
            label: rr.model_dump()
            for label, rr in scoped.valuation_results.items()
        }

    # Rating: convert to plain dict if present
    rating_out = None
    if scoped.rating_result:
        try:
            rating_out = dataclasses.asdict(scoped.rating_result)
        except TypeError:
            rating_out = vars(scoped.rating_result)

    return ScopedRunResponse(
        scope=scoped.scope.value,
        scope_label=scoped.scope_label,
        run_id=scoped.run_id,
        physical=physical_out,
        transition=transition_out,
        valuation_results=valuation_out,
        rating_result=rating_out,
    )


# ── File upload endpoint ────────────────────────────────────────────────────

class FileRunSummaryRow(BaseModel):
    company: str
    scenario: str
    ev_bn: float
    equity_bn: float
    share_price: float
    wacc_pct: float
    npv_impact_pct: float | None              # transition risk NPV impact (NGFS carbon cost)
    physical_npv_impact_pct: float | None     # physical risk NPV drag (CMIP6+WRI or IPCC scaling)
    combined_npv_impact_pct: float | None     # transition + physical (TCFD double materiality)
    hazard_scores: dict | None = None         # hazard → annual prob % at 2038
    ssp_id: str | None = None                 # e.g. "ssp126", "ssp245", "ssp370"
    gmst_2050: float | None = None            # °C above 1995–2014 baseline
    data_sources: list[str] | None = None     # CMIP6 / WRI Aqueduct / PhysicalHazardEngine
    sector: str | None = None                 # normalised sector key
    physical_exposure_ratio: float | None = None  # φ — fraction of revenue exposed
    warnings: int


class FileRunResponse(BaseModel):
    source_file: str
    companies_processed: int
    scenarios_run: list[str]
    total_duration_s: float
    errors: list[str]
    results: list[FileRunSummaryRow]


@app.post("/runs/file", response_model=FileRunResponse)
async def run_from_file(
    file: UploadFile = File(..., description="Client intake Excel (.xlsx)"),
    scenarios: str = Form(
        default="Net Zero 2050,Delayed Transition,Current Policies",
        description="Comma-separated scenario names",
    ),
) -> FileRunResponse:
    """
    Upload a client intake Excel file and run the full pipeline.

    The file must match the CRI intake template (download via GET /template).
    Returns valuation results for every company × scenario combination.
    """
    from ..engine.pipeline import Pipeline

    if not file.filename or not file.filename.endswith(".xlsx"):
        raise HTTPException(
            status_code=400,
            detail="File must be a .xlsx Excel file. Download the template from GET /template.",
        )

    scenario_list = [s.strip() for s in scenarios.split(",") if s.strip()]

    # Save upload to temp file
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        pipeline = Pipeline()
        report = pipeline.run_file(tmp_path, scenario_names=scenario_list)
    finally:
        tmp_path.unlink(missing_ok=True)

    return FileRunResponse(
        source_file=file.filename,
        companies_processed=report.companies_processed,
        scenarios_run=report.scenarios_run,
        total_duration_s=round(report.total_duration_s, 2),
        errors=report.errors,
        results=[FileRunSummaryRow(**row) for row in report.summary_table()],
    )


@app.get("/template")
def download_template():
    """Download the blank client intake Excel template."""
    from fastapi.responses import FileResponse
    from ..intake.template import generate_template

    template_path = Path(__file__).parent.parent.parent.parent / "data" / "client_template.xlsx"
    if not template_path.exists():
        generate_template(str(template_path))

    return FileResponse(
        path=str(template_path),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="CRI_Client_Intake_Template.xlsx",
    )


@app.get("/tiers", response_model=TiersResponse)
def list_tiers() -> TiersResponse:
    """Return all CRI subscription tiers with pricing and feature matrix."""
    from ..outcomes.tiers import TIER_FEATURES, TIER_PRICING, Tier
    import dataclasses

    tiers = []
    for tier in [Tier.FREE, Tier.ANALYST, Tier.PROFESSIONAL, Tier.ENTERPRISE]:
        pricing = TIER_PRICING[tier]
        features = dataclasses.asdict(TIER_FEATURES[tier])
        tiers.append(TierInfo(
            tier=tier.value,
            label=pricing["label"],
            price=pricing["price"],
            cta=pricing["cta"],
            description=pricing["description"],
            features=features,
        ))
    return TiersResponse(tiers=tiers)


@app.post("/ratings", response_model=RatingResponse)
def rate_company(request: RatingRequest) -> RatingResponse:
    """
    Compute a climate risk rating for a company.

    Free tier: returns A–E rating, pillar labels, and summary narrative.
    Paid tiers: returns full numeric scores, key drivers, and peer context.

    Runs the company under all three canonical scenarios internally.
    """
    from ..outcomes.ratings import RatingEngine, WeightProfile
    from ..outcomes.tiers import Tier, TierGate

    if request.company_id.lower() not in COMPANY_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Company '{request.company_id}' not found.",
        )

    # Validate weight_profile
    try:
        wp = WeightProfile(request.weight_profile.lower())
    except ValueError:
        valid = [p.value for p in WeightProfile]
        raise HTTPException(
            status_code=422,
            detail=f"Invalid weight_profile '{request.weight_profile}'. "
                   f"Choose from: {valid}",
        )

    # Validate custom_weights when CUSTOM profile selected
    custom_weights_tuple = None
    if wp == WeightProfile.CUSTOM:
        if not request.custom_weights or len(request.custom_weights) != 3:
            raise HTTPException(
                status_code=422,
                detail="custom_weights must be a list of 3 floats [physical, transition, financial] "
                       "summing to 1.0 when weight_profile='custom'.",
            )
        w = request.custom_weights
        if abs(sum(w) - 1.0) > 1e-4:
            raise HTTPException(
                status_code=422,
                detail=f"custom_weights must sum to 1.0, got {sum(w):.4f}.",
            )
        custom_weights_tuple = (w[0], w[1], w[2])

    company = COMPANY_REGISTRY[request.company_id.lower()]
    tier = Tier.from_str(request.tier)

    # Run all three scenarios
    nze_results = run_engine(company=company, scenario=SCENARIO_REGISTRY["nze_2050"])
    dt_results  = run_engine(company=company, scenario=SCENARIO_REGISTRY["delayed_transition"])
    cp_results  = run_engine(company=company, scenario=SCENARIO_REGISTRY["current_policies"])

    engine = RatingEngine()
    rating_result = engine.rate(
        company_name=company.name,
        sector=company.sector,
        nze_results=nze_results,
        dt_results=dt_results,
        cp_results=cp_results,
        data_quality=company.data_quality,
        weight_profile=wp,
        custom_weights=custom_weights_tuple,
    )

    # Determine actual weights applied (for transparency in response)
    from ..outcomes.ratings import _PROFILE_WEIGHTS
    if wp == WeightProfile.CUSTOM and custom_weights_tuple:
        w_p, w_t, w_f = custom_weights_tuple
    else:
        w_p, w_t, w_f = _PROFILE_WEIGHTS[wp]
    weights_applied = {
        "physical":   round(w_p, 4),
        "transition": round(w_t, 4),
        "financial":  round(w_f, 4),
    }

    gate = TierGate(tier)
    show_scores = gate.features.pillar_scores
    show_drivers = gate.features.pillar_scores  # drivers visible from Analyst up

    return RatingResponse(
        company_id=company.id,
        company_name=company.name,
        rating=str(rating_result.rating),
        rating_label=rating_result.rating_label,
        confidence=rating_result.confidence,
        summary=rating_result.summary,
        sector_rank=rating_result.sector_rank,
        physical_risk_label=rating_result.physical.label,
        transition_risk_label=rating_result.transition.label,
        financial_impact_label=rating_result.financial.label,
        composite_score=round(rating_result.composite_score, 1) if show_scores else None,
        physical_risk_score=round(rating_result.physical.score, 1) if show_scores else None,
        transition_risk_score=round(rating_result.transition.score, 1) if show_scores else None,
        financial_impact_score=round(rating_result.financial.score, 1) if show_scores else None,
        physical_drivers=rating_result.physical.drivers if show_drivers else None,
        transition_drivers=rating_result.transition.drivers if show_drivers else None,
        financial_drivers=rating_result.financial.drivers if show_drivers else None,
        tier=tier.value,
        locked_features=gate._locked_list(),
        upgrade_prompt=(
            None if tier != Tier.FREE else
            "Unlock full scores, asset-level breakdown, and TCFD/ISSB S2 reports "
            "with CRI Analyst or Professional."
        ),
        weight_profile_used=wp.value,
        weights_applied=weights_applied,
    )


@app.post("/reports/tcfd", response_model=DisclosureResponse)
def generate_tcfd_report(request: DisclosureRequest) -> DisclosureResponse:
    """
    Generate a TCFD-aligned climate risk disclosure report.
    Requires Professional tier or above.
    """
    from ..outcomes.tiers import Tier, TierGate
    from ..outcomes.disclosure import generate_tcfd

    tier = Tier.from_str(request.tier)
    gate = TierGate(tier)
    if not gate.features.disclosure_tcfd:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "TCFD reports require Professional or Enterprise tier.",
                **TierGate.upgrade_prompt("disclosure_tcfd"),
            },
        )

    if request.company_id.lower() not in COMPANY_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Company '{request.company_id}' not found.")

    company = COMPANY_REGISTRY[request.company_id.lower()]
    nze = run_engine(company=company, scenario=SCENARIO_REGISTRY["nze_2050"])
    dt  = run_engine(company=company, scenario=SCENARIO_REGISTRY["delayed_transition"])
    cp  = run_engine(company=company, scenario=SCENARIO_REGISTRY["current_policies"])

    report = generate_tcfd(company, nze, dt, cp, reporting_year=request.reporting_year)
    return DisclosureResponse(**report.to_dict())


@app.post("/reports/issb", response_model=DisclosureResponse)
def generate_issb_report(request: DisclosureRequest) -> DisclosureResponse:
    """
    Generate an IFRS S2 (ISSB) climate disclosure metrics report.
    Requires Professional tier or above.
    """
    from ..outcomes.tiers import Tier, TierGate
    from ..outcomes.disclosure import generate_issb

    tier = Tier.from_str(request.tier)
    gate = TierGate(tier)
    if not gate.features.disclosure_issb:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "ISSB S2 reports require Professional or Enterprise tier.",
                **TierGate.upgrade_prompt("disclosure_issb"),
            },
        )

    if request.company_id.lower() not in COMPANY_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Company '{request.company_id}' not found.")

    company = COMPANY_REGISTRY[request.company_id.lower()]
    nze = run_engine(company=company, scenario=SCENARIO_REGISTRY["nze_2050"])
    dt  = run_engine(company=company, scenario=SCENARIO_REGISTRY["delayed_transition"])
    cp  = run_engine(company=company, scenario=SCENARIO_REGISTRY["current_policies"])

    report = generate_issb(company, nze, dt, cp, reporting_year=request.reporting_year)
    return DisclosureResponse(**report.to_dict())


@app.post("/reports/csrd", response_model=DisclosureResponse)
def generate_csrd_report(request: DisclosureRequest) -> DisclosureResponse:
    """
    Generate an EU CSRD ESRS E1 climate disclosure data point report.
    Requires Professional tier or above.
    """
    from ..outcomes.tiers import Tier, TierGate
    from ..outcomes.disclosure import generate_csrd

    tier = Tier.from_str(request.tier)
    gate = TierGate(tier)
    if not gate.features.disclosure_csrd:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "EU CSRD reports require Professional or Enterprise tier.",
                **TierGate.upgrade_prompt("disclosure_csrd"),
            },
        )

    if request.company_id.lower() not in COMPANY_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Company '{request.company_id}' not found.")

    company = COMPANY_REGISTRY[request.company_id.lower()]
    nze = run_engine(company=company, scenario=SCENARIO_REGISTRY["nze_2050"])
    dt  = run_engine(company=company, scenario=SCENARIO_REGISTRY["delayed_transition"])
    cp  = run_engine(company=company, scenario=SCENARIO_REGISTRY["current_policies"])

    report = generate_csrd(company, nze, dt, cp, reporting_year=request.reporting_year)
    return DisclosureResponse(**report.to_dict())


@app.post("/reports/physical", response_model=PhysicalHazardReportResponse)
def generate_physical_report(request: PhysicalReportRequest) -> PhysicalHazardReportResponse:
    """
    Generate a standalone Physical Climate Hazard Report.

    This endpoint requires **no financial data** (no EBITDA, WACC, or net debt).
    It is designed for clients who need asset-level physical climate risk assessment
    without a full transition or valuation analysis — e.g. real estate lenders,
    infrastructure operators, insurers, or due-diligence teams assessing a single site.

    Accepts EITHER:
    - `company_id` — runs against a registered seed company.
    - `asset`      — accepts a single inline asset definition (any region, any commodity).

    Returns:
    - Physical risk score (0–100) and label (Low → Critical)
    - 25-year production loss trajectory under NZE, Delayed Transition, and Current Policies
    - Per-hazard loss breakdown at 2035
    - Adaptation capex requirement (cumulative, 25-year)
    - TCFD-aligned narrative summary
    - Data sources and methodology caveats

    No carbon pricing. No DCF. No transition risk. Pure physical hazard output.
    Available from the Analyst tier and above.
    """
    from datetime import datetime, timezone
    from ..data.schemas import (
        Asset, Commodity, Company, EmissionsProfile, Financials,
    )
    from ..outcomes.physical_report import build_physical_report
    from ..operations.company import simulate

    # ── 1. Resolve company ────────────────────────────────────────────────────
    if request.company_id:
        # Use registered company
        cid = request.company_id.lower()
        if cid not in COMPANY_REGISTRY:
            raise HTTPException(
                status_code=404,
                detail=f"Company '{request.company_id}' not found. "
                       f"Available: {list(COMPANY_REGISTRY)}. "
                       f"To run on a custom asset, omit company_id and supply an 'asset' object."
            )
        company = COMPANY_REGISTRY[cid]

    elif request.asset:
        # Build a minimal Company wrapper around the inline asset
        a = request.asset
        try:
            commodity = Commodity(a.commodity.lower())
        except ValueError:
            valid = [c.value for c in Commodity]
            raise HTTPException(
                status_code=422,
                detail=f"Unknown commodity '{a.commodity}'. Valid values: {valid}"
            )

        company = Company(
            id=a.id,
            name=request.company_name or a.name,
            sector="Custom",
            hq_region=a.region,
            financials=Financials(
                # Physical report only — use production × unit cost as proxy revenue
                # Financial fields not used in physical calculations.
                revenue=a.baseline_production * a.baseline_unit_cost,
                ebitda=a.baseline_production * a.baseline_unit_cost * 0.40,
                capex=a.baseline_production * a.baseline_unit_cost * 0.10,
                maintenance_capex_share=0.60,
                tax_rate=0.28,
                wacc_base=0.08,
                net_debt=0.0,
                shares_outstanding=100.0,
                market_cap=a.carrying_value * 3.0,
            ),
            assets=[Asset(
                id=a.id,
                name=a.name,
                commodity=commodity,
                region=a.region,
                baseline_production=a.baseline_production,
                production_unit=a.production_unit,
                baseline_unit_cost=a.baseline_unit_cost,
                energy_cost_share=a.energy_cost_share,
                carrying_value=a.carrying_value,
                remaining_life_years=a.remaining_life_years,
                emissions=EmissionsProfile(
                    # Emissions not required for physical-only report
                    scope1_intensity=0.0,
                    scope2_intensity=0.0,
                    scope3_intensity=0.0,
                    carbon_price_coverage=0.0,
                    free_allocation=0.0,
                ),
            )],
            exposure_weight=0.5,
            transition_weight=0.5,
            data_quality="medium",
        )
    else:
        raise HTTPException(
            status_code=422,
            detail="Supply either 'company_id' (registered company) or "
                   "'asset' (inline asset definition). Both are absent."
        )

    # ── 2. Run physical simulation for all three scenarios ────────────────────
    ops_nze = simulate(company, NZE_2050)
    ops_dly = simulate(company, DELAYED_TRANSITION)
    ops_cp  = simulate(company, CURRENT_POLICIES)

    # ── 3. Build report ───────────────────────────────────────────────────────
    report = build_physical_report(
        company=company,
        ops_nze=ops_nze,
        ops_delayed=ops_dly,
        ops_cp=ops_cp,
    )

    # ── 4. Convert to response ────────────────────────────────────────────────
    def _years(year_list) -> list[HazardYearOut]:
        return [
            HazardYearOut(
                year=y.year,
                physical_loss_cost=y.physical_loss_cost,
                adaptation_capex=y.adaptation_capex,
                physical_loss_by_hazard=y.physical_loss_by_hazard,
                total_loss_fraction=y.total_loss_fraction,
            )
            for y in year_list
        ]

    return PhysicalHazardReportResponse(
        company_id=company.id,
        company_name=company.name,
        run_id=report.run_id,
        model_version=report.model_version,
        generated_at=datetime.now(timezone.utc).isoformat(),
        physical_score=report.physical_score,
        physical_label=report.physical_label,
        peak_loss_year=report.peak_loss_year,
        peak_loss_usd=report.peak_loss_usd,
        peak_loss_hazard=report.peak_loss_hazard,
        total_adaptation_capex_nze=report.total_adaptation_capex_nze,
        total_adaptation_capex_cp=report.total_adaptation_capex_cp,
        hazard_breakdown_2035=report.hazard_breakdown_2035,
        narrative=report.narrative,
        years_nze=_years(report.years_nze),
        years_delayed=_years(report.years_delayed),
        years_cp=_years(report.years_cp),
        data_sources=[
            "IPCC AR6 (2021) — warming trajectories and hazard intensity projections",
            "WRI Aqueduct 4.0 (methodology) — water stress and flood risk regional baselines",
            "NGFS Phase 4 (2023) — climate scenario pathways (NZE, Delayed Transition, Current Policies)",
            "ERA5 / Copernicus Climate Data Store — temperature and precipitation baselines",
            "CRI Engine v0.3.0 — asset-level hazard simulation",
        ],
        caveats=[
            "Physical loss costs are in USD millions, consistent with asset carrying values.",
            "Hazard paths are parameterised regional estimates; they do not incorporate "
            "site-specific GPS-resolved data unless a premium connector is active.",
            "This report does not constitute a financial valuation or insurance assessment.",
            "Adaptation capex estimates are indicative; actual costs depend on asset design "
            "and local engineering factors.",
            "Emissions data is not required for this report. Transition and carbon cost "
            "analysis is excluded. For full CRI assessment use POST /runs/scoped?scope=full_cri.",
        ],
        scenario_set="NGFS Phase 4",
    )


@app.get("/connectors/status")
def connector_status() -> dict:
    """Return the status and data sources available for enrichment."""
    from ..connectors.wri_aqueduct import WRIAqueductConnector
    from ..connectors.ngfs import NGFSConnector
    from ..connectors.owid import OWIDConnector

    ngfs = NGFSConnector()
    return {
        "wri_aqueduct": {
            "status": "active",
            "source": "WRI Aqueduct 4.0",
            "url": "https://aqueduct.wri.org",
            "coverage": "21 major mining/energy regions + lat/lon API",
            "hazards": ["water_stress", "flood_risk", "drought_risk"],
        },
        "ngfs_scenarios": {
            "status": "active",
            "source": "NGFS Phase 4 (2023)",
            "url": "https://www.ngfs.net",
            "scenarios": ngfs.list_scenarios(),
            "carbon_price_range": "$12–$250/tCO2e",
        },
        "nasa_gddp": {
            "status": "active",
            "source": "NASA NEX-GDDP / IPCC AR6 regional proxy",
            "url": "https://www.nccs.nasa.gov/services/data-collections/land-based-products/nex-gddp-cmip6",
            "coverage": "34 regions, 2026–2050",
            "hazards": ["heat_stress"],
        },
        "owid_energy": {
            "status": "active",
            "source": "Our World in Data / IEA WEO 2023-aligned",
            "url": "https://ourworldindata.org/energy",
            "coverage": "9 commodities × 4 scenarios",
            "hazards": ["demand_shift"],
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# SCENARIO CASCADE ENGINE
# Physical compound-event → sectoral causal chain → itemised financial impact
# ──────────────────────────────────────────────────────────────────────────────

class ScenarioRunRequest(BaseModel):
    """Request body for POST /scenarios/run.

    THREE-LAYER ARCHITECTURE
    ─────────────────────────
    Layer 1  Physical hazard assessment   — always runs (PhysicalHazardEngine)
    Layer 2  Scenario cascade             — always runs (ScenarioCascadeEngine)
             Translates compound physical events → itemised financial impact
    Layer 3  Transition risk overlay      — OPTIONAL (set include_transition_overlay=True)
             Applies NGFS carbon pricing / demand-shift pathway on top of the
             physical result to produce a combined physical + transition exposure
    """
    company_id: str = Field(
        ...,
        description="Registered company ID (case-insensitive). "
                    "Use GET /companies to list available IDs.",
    )
    event_id: str = Field(
        ...,
        description="Physical event identifier from GET /scenarios/events "
                    "(e.g. 'el_nino_super_drought', 'tropical_cyclone_cat4').",
    )
    year: int = Field(
        2026,
        ge=2024,
        le=2050,
        description="Reference year for the scenario (used to select the SSP "
                    "warming pathway and adjust hazard intensity). Default 2026.",
    )
    ssp: str = Field(
        "ssp370",
        description="SSP warming pathway for background hazard intensity. "
                    "One of: ssp126, ssp245, ssp370, ssp585. Default ssp370.",
    )
    include_transition_overlay: bool = Field(
        False,
        description=(
            "Layer 3 — optional transition risk overlay. "
            "When True, runs the NGFS carbon-pricing engine on top of the physical "
            "cascade result and appends a transition_overlay block to the response. "
            "The overlay includes carbon cost exposure, stranded-asset risk, "
            "demand-shift impact, and an implied additional credit spread from transition. "
            "Uses the NGFS Delayed Transition scenario by default (most credit-relevant). "
            "Set False (default) for pure physical-risk analysis."
        ),
    )
    transition_scenario_id: str = Field(
        "delayed_transition",
        description=(
            "NGFS scenario to use for the transition overlay (only used when "
            "include_transition_overlay=True). "
            "Options: 'nze_2050' (Net Zero Emissions by 2050), "
            "'delayed_transition' (default — most credit-relevant for investors), "
            "'current_policies' (no-transition baseline). "
            "Use GET /scenarios to list all available NGFS scenario IDs."
        ),
    )


@app.get(
    "/scenarios/events",
    summary="List physical climate events",
    tags=["Scenario Cascade"],
)
def list_physical_events() -> list[dict]:
    """
    Return all physical compound climate events in the event library.

    Each event has:
    - `id` — use this in POST /scenarios/run
    - `name`, `driver`, `context` — human-readable description
    - `duration_months` — expected duration
    - `acute` — whether this is a rapid-onset event (True) or chronic/seasonal (False)
    - `hazard_multipliers` — how baseline hazard severities are scaled
    - `hazard_floors` — minimum severity floor for named hazards
    - `historical_analogs` — real-world precedents used for calibration
    - `affected_regions` — geographic scope

    Events are drawn from the CRI v0.4 physical event library, calibrated against
    IPCC AR6, EM-DAT historical loss data, and academic literature.
    """
    from ..climate.scenarios.physical_events import list_events
    return list_events()


@app.post(
    "/scenarios/run",
    summary="Run physical scenario cascade",
    tags=["Scenario Cascade"],
)
def run_scenario_cascade(request: ScenarioRunRequest) -> dict:
    """
    Run the physical climate scenario cascade engine for a registered company.

    **Pipeline:**
    1. Resolve baseline asset-level hazard profiles via the five-layer physical
       hazard engine (WRI baseline → GIS elevation → live APIs → CMIP6 projections).
    2. Apply the selected compound physical event (hazard multipliers + floors).
    3. Route each asset through its sector-specific damage chain, generating
       granular itemised cost lines (physical damage, inventory loss, production
       halt, emergency response, recovery capex, etc.).
    4. Aggregate across assets into company-wide financials: total direct loss,
       EBITDA haircut, revenue impact, capex burden, and an implied credit spread
       proxy (bps) calibrated against Moody's historical credit migration data.

    **Output includes:**
    - Per-asset cost breakdown with source assumptions for every line item
    - Company-wide EBITDA impact (%), revenue impact (%), capex burden (%)
    - Implied credit spread widening (basis points)
    - Recovery timeline (months)
    - Structured investor-grade narrative
    - Historical analogs used for calibration
    - Key vulnerability statements for each asset

    **Sector coverage:** Beverages, Agriculture, Mining/Extractives, Real Estate.
    Other sectors fall back to a proportional damage approximation.

    **Data quality:** all hazard inputs are cited with their provenance tier
    (LIVE / REGIONAL_BASELINE / GLOBAL_FALLBACK). Satellite observations from
    NASA FIRMS (fire) and GDACS (floods/cyclones) are incorporated when active
    events are detected.
    """
    from ..climate.scenario_engine import ScenarioCascadeEngine

    cid = request.company_id.lower()
    if cid not in COMPANY_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"Company '{request.company_id}' not found.",
                "available_companies": list(COMPANY_REGISTRY.keys()),
                "hint": "Use GET /companies to browse registered companies.",
            },
        )

    # Validate SSP
    valid_ssps = {"ssp126", "ssp245", "ssp370", "ssp585"}
    if request.ssp not in valid_ssps:
        raise HTTPException(
            status_code=422,
            detail={
                "error": f"Invalid SSP pathway '{request.ssp}'.",
                "valid_values": sorted(valid_ssps),
            },
        )

    company = COMPANY_REGISTRY[cid]

    try:
        engine = ScenarioCascadeEngine()
        result = engine.run(
            company=company,
            event_id=request.event_id,
            year=request.year,
            ssp=request.ssp,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"Physical event '{request.event_id}' not found.",
                "hint": "Use GET /scenarios/events to list valid event IDs.",
            },
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Scenario cascade engine error.",
                "detail": str(exc),
                "hint": "Check that the company has assets with lat/lon coordinates "
                        "and valid commodity types for full sector chain resolution.",
            },
        ) from exc

    output = result.model_dump()

    # ── Layer 3: Optional transition risk overlay ─────────────────────────────
    if request.include_transition_overlay:
        try:
            transition_overlay = _run_transition_overlay(
                company=company,
                scenario_id=request.transition_scenario_id,
                physical_ebitda_impact_pct=result.ebitda_impact_pct,
            )
            output["transition_overlay"] = transition_overlay
            output["layer_architecture"] = {
                "layer_1": "physical_hazard_assessment",
                "layer_2": "scenario_cascade_financial_impact",
                "layer_3": f"transition_overlay:{request.transition_scenario_id}",
                "note": (
                    "Layer 3 is additive. Transition costs compound on top of "
                    "the physical impact. Combined EBITDA impact = physical + transition "
                    "(with 0.7 correlation factor applied to transition component)."
                ),
            }
        except Exception as exc:
            # Transition overlay failure should NOT fail the whole request
            output["transition_overlay"] = {
                "status": "error",
                "error": str(exc),
                "note": "Physical cascade result is complete. Transition overlay failed.",
            }
    else:
        output["layer_architecture"] = {
            "layer_1": "physical_hazard_assessment",
            "layer_2": "scenario_cascade_financial_impact",
            "layer_3": "not_requested — set include_transition_overlay=true to enable",
        }

    return output


def _run_transition_overlay(
    company,
    scenario_id: str,
    physical_ebitda_impact_pct: float,
) -> dict:
    """
    Run a lightweight transition risk overlay on top of a physical cascade result.

    Executes the NGFS engine for the named scenario and extracts:
    - Carbon cost exposure (USD millions)
    - Stranded-asset risk estimate
    - Demand-shift revenue impact
    - Implied transition credit spread (bps)
    - Combined (physical + transition) EBITDA haircut

    This is Layer 3 of the three-layer CRI architecture.
    Physical risk always precedes transition risk; transition is optional.
    """
    # Map API scenario slug → NGFS scenario registry key
    SCENARIO_MAP = {
        "nze_2050":           "nze_2050",
        "delayed_transition":  "delayed_transition",
        "current_policies":    "current_policies",
    }
    resolved_id = SCENARIO_MAP.get(scenario_id, scenario_id)

    if resolved_id not in SCENARIO_REGISTRY:
        available = list(SCENARIO_REGISTRY.keys())
        raise ValueError(
            f"Transition scenario '{scenario_id}' not found. "
            f"Available: {available}"
        )

    from ..engine.orchestrator import run as run_full_engine

    scenario = SCENARIO_REGISTRY[resolved_id]
    transition_result = run_full_engine(company=company, scenario=scenario)

    # Extract transition-specific metrics from the full run
    # Peak transition year is the year with the highest carbon cost burden
    peak_transition_year = None
    peak_carbon_cost_usd_m = 0.0
    peak_demand_shift_pct = 0.0

    for yr in transition_result.years:
        # Carbon cost contribution approximated from revenue impact
        carbon_cost = getattr(yr, "carbon_cost_usd_m", 0.0)
        demand_shift = getattr(yr, "demand_shock_pct", 0.0)
        if carbon_cost > peak_carbon_cost_usd_m:
            peak_carbon_cost_usd_m = carbon_cost
            peak_transition_year = yr.year

    # EBITDA impact from transition alone (remove physical component)
    transition_ebitda_impact_pct = abs(
        getattr(transition_result, "peak_ebitda_compression_pct", 0.0)
    )

    # Combined exposure
    combined_ebitda_impact_pct = min(
        100.0,
        physical_ebitda_impact_pct + transition_ebitda_impact_pct * 0.7
        # Apply 0.7 correlation factor — not all transition costs
        # hit simultaneously with physical event
    )

    # Stranded-asset risk: high for fossil fuels under NZE; low for beverages/RE
    stranded_asset_risk = "low"
    if hasattr(company, "assets"):
        for asset in company.assets:
            if hasattr(asset, "commodity"):
                commodity_str = str(asset.commodity).lower()
                if any(kw in commodity_str for kw in ["coal", "oil", "gas", "lng", "crude"]):
                    stranded_asset_risk = "high"
                    break
                elif any(kw in commodity_str for kw in ["iron", "mining", "mineral"]):
                    stranded_asset_risk = "medium"

    # Transition credit spread proxy (bps)
    # Calibrated against Moody's ESG Solutions transition risk spread estimates
    spread_map = {
        "low":    15,
        "medium": 45,
        "high":   120,
    }
    transition_credit_spread_bps = spread_map.get(stranded_asset_risk, 30)

    return {
        "scenario_id":                  resolved_id,
        "scenario_name":                scenario.name,
        "scenario_family":              scenario.family.value if hasattr(scenario, "family") else "unknown",
        "peak_carbon_cost_usd_m":       round(peak_carbon_cost_usd_m, 2),
        "peak_transition_year":         peak_transition_year,
        "transition_ebitda_impact_pct": round(transition_ebitda_impact_pct, 1),
        "combined_ebitda_impact_pct":   round(combined_ebitda_impact_pct, 1),
        "stranded_asset_risk":          stranded_asset_risk,
        "transition_credit_spread_bps": transition_credit_spread_bps,
        "company_revenue_usd_m":        round(company.financials.revenue, 1),
        "data_source":                  "NGFS Phase 4 (2023) via CRI engine v0.4",
        "methodology": (
            "Transition overlay runs the NGFS carbon-price pathway through the "
            "existing financial engine (DCF + working capital model). "
            "Combined EBITDA impact applies a 0.7 correlation factor between physical "
            "and transition costs, reflecting that both do not necessarily peak simultaneously. "
            "Credit spread estimate based on Moody's ESG Solutions transition risk "
            "spread calibration (2023)."
        ),
    }


@app.post(
    "/scenarios/worst-case",
    summary="Worst-case scenario across multiple events",
    tags=["Scenario Cascade"],
)
def run_worst_case_scenario(
    company_id: str,
    event_ids: list[str],
    year: int = 2026,
) -> dict:
    """
    Run multiple physical scenario cascades and return the worst-case result
    by EBITDA impact.

    Useful for stress-testing or identifying which compound event poses the
    greatest financial threat to a specific company. All events are run in
    parallel using ThreadPoolExecutor.

    Returns the full ScenarioCascadeResult for the single most severe event,
    with `event_id` identifying which event drove the worst case.
    """
    from ..climate.scenario_engine import ScenarioCascadeEngine

    cid = company_id.lower()
    if cid not in COMPANY_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"Company '{company_id}' not found.",
                "available_companies": list(COMPANY_REGISTRY.keys()),
            },
        )

    if not event_ids:
        raise HTTPException(
            status_code=422,
            detail={"error": "event_ids must be a non-empty list."},
        )

    company = COMPANY_REGISTRY[cid]

    try:
        engine = ScenarioCascadeEngine()
        result = engine.worst_case(
            company=company,
            event_ids=event_ids,
            year=year,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": "Worst-case scenario engine error.", "detail": str(exc)},
        ) from exc

    return result.model_dump()


# ──────────────────────────────────────────────────────────────────────────────
# HISTORICAL SCENARIO CALIBRATION
# Real-world event database + model validation
# ──────────────────────────────────────────────────────────────────────────────

class CalibrateRequest(BaseModel):
    """Request body for POST /scenarios/calibrate."""
    historical_event_id: str = Field(
        ...,
        description=(
            "Historical event ID from GET /scenarios/historical "
            "(e.g. 'thailand_floods_2011', 'cape_town_drought_2017_18'). "
            "The engine runs the mapped physical event with calibrated multipliers "
            "and compares predicted losses against the documented historical figures."
        ),
    )
    company_id: Optional[str] = Field(
        None,
        description=(
            "Registered company ID to use as the calibration subject. "
            "If omitted, a synthetic single-asset proxy matching the primary sector "
            "of the historical event is used. "
            "Using a real company gives sector-specific results but note that "
            "historical losses are economy-wide — see methodology_note in the response."
        ),
    )
    sector_filter: Optional[str] = Field(
        None,
        description=(
            "Restrict calibration comparison to a single sector "
            "(e.g. 'beverages', 'agriculture', 'mining', 'real_estate'). "
            "If omitted, all sectors with documented historical loss data are compared."
        ),
    )
    ssp: str = Field(
        "ssp370",
        description="SSP warming pathway for the calibration run. Default ssp370.",
    )
    year: int = Field(
        2025,
        ge=2020,
        le=2050,
        description="Reference year for the calibration engine run. Default 2025.",
    )


@app.get(
    "/scenarios/historical",
    summary="List historical real-world climate events",
    tags=["Scenario Calibration"],
)
def list_historical_events() -> list[dict]:
    """
    Return all historical real-world climate events in the calibration database.

    Each event has:
    - `id` — use this in POST /scenarios/calibrate
    - `name`, `year_start`, `year_end` — event identification
    - `region` — primary geographic scope
    - `physical_event_id` — the nearest matching event in the physical event library
    - `total_loss_usd_m` — verified total economic loss (nominal USD millions, event year)
    - `insured_loss_usd_m` — insured portion (null if unavailable)
    - `source_total_loss` — primary citation for the loss figure
    - `sectors_with_data` — which sectors have disaggregated loss data
    - `affected_countries` — ISO 3166-1 alpha-2 country codes
    - `key_impacts` — plain-language summary of primary financial channels

    The calibration database contains 16 events spanning 1997–2023 across
    all major climate hazard categories (El Niño/La Niña, floods, drought,
    wildfire, cyclone, heat dome).

    Data sources: Munich Re NatCatSERVICE, Swiss Re sigma, EM-DAT, World Bank
    GFDRR, NOAA NCEI Billion-Dollar Disasters, and peer-reviewed literature.
    All figures are nominally reported in event-year USD millions.
    """
    from ..climate.scenarios.historical_events import list_historical_events as _list
    return _list()


@app.post(
    "/scenarios/calibrate",
    summary="Calibrate engine against a historical event",
    tags=["Scenario Calibration"],
)
def calibrate_against_historical(request: CalibrateRequest) -> dict:
    """
    Run the CRI cascade engine against a historical real-world event and
    compute predicted-vs-actual calibration error statistics.

    **How it works:**
    1. Fetches the HistoricalClimateEvent record (verified losses + sector data).
    2. Constructs a calibrated PhysicalEvent by scaling the mapped event's
       hazard multipliers by the historical event's calibration_scale factors.
       (e.g. the 2011 Thai floods were 1.35× more severe than the baseline
       river flood event's riverine flood hazard multiplier.)
    3. Runs the ScenarioCascadeEngine with the calibrated event on the specified
       company (or a synthetic proxy if none supplied).
    4. Normalises both predicted and historical losses to loss-as-fraction-of-revenue
       to correct for the scale difference between a single company and the economy.
    5. Computes absolute and relative error per sector and overall.

    **Calibration status thresholds:**
    - CALIBRATED  — ≤ 20% relative error
    - ACCEPTABLE  — 20–50% relative error (within typical model uncertainty)
    - NEEDS_REVIEW — > 50% relative error (systematic bias suspected)

    **Important caveats:**
    - Historical losses are economy-wide / industry-wide; engine prediction is
      single-company. Comparison is via normalised percentages, not raw USD.
    - Documented losses often include indirect economic effects (multiplier,
      supply chain) that the engine does not model.
    - The calibration is a transparency tool, not a tuning system — it does
      not modify any engine parameters.

    **Response includes:**
    - `overall_status` and `overall_relative_error_pct` — top-level verdict
    - `sector_results` — per-sector error breakdown with company examples
    - `summary`, `methodology_note`, `caveats` — interpretation guidance
    - `total_historical_loss_usd_m` + `source_total_loss` — the benchmark
    """
    from ..climate.scenarios.calibration import (
        run_calibration,
        calibration_report_to_dict,
    )

    # Resolve optional company
    company = None
    if request.company_id:
        cid = request.company_id.lower()
        if cid not in COMPANY_REGISTRY:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": f"Company '{request.company_id}' not found.",
                    "available_companies": list(COMPANY_REGISTRY.keys()),
                    "hint": "Leave company_id blank to use a synthetic proxy.",
                },
            )
        company = COMPANY_REGISTRY[cid]

    try:
        report = run_calibration(
            historical_event_id=request.historical_event_id,
            company=company,
            sector_filter=request.sector_filter,
            ssp=request.ssp,
            year=request.year,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail={
                "error": str(exc),
                "hint": "Use GET /scenarios/historical to list valid historical event IDs.",
            },
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Calibration engine error.",
                "detail": str(exc),
            },
        ) from exc

    return calibration_report_to_dict(report)
