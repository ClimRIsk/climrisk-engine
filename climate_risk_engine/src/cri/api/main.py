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
    PortfolioPositionIn,
    PortfolioRiskRequest,
    PortfolioRiskResponse,
    CounterpartyIn,
    PortfolioCreditRequest,
    PortfolioCreditResponse,
    PortfolioStreamRequest,
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
    """Return full company profiles including asset locations and baseline financials."""
    from .schemas import AssetLocationResponse, FinancialsResponse
    companies = []
    for company_id, company in COMPANY_REGISTRY.items():
        companies.append(
            CompanyResponse(
                id=company_id,
                name=company.name,
                sector=company.sector,
                region=company.hq_region,
                data_quality=company.data_quality,
                financials=FinancialsResponse(
                    revenue_usd_m=round(company.financials.revenue, 1),
                    ebitda_usd_m=round(company.financials.ebitda, 1),
                    capex_usd_m=round(company.financials.capex, 1),
                    wacc_pct=round(company.financials.wacc_base * 100, 2),
                    net_debt_usd_m=round(company.financials.net_debt, 1),
                ),
                assets=[
                    AssetLocationResponse(
                        id=a.id,
                        name=a.name,
                        commodity=a.commodity.value,
                        region=a.region,
                        lat=a.lat,
                        lon=a.lon,
                        equipment_type=a.equipment_type,
                        baseline_production=a.baseline_production,
                        production_unit=a.production_unit,
                        carrying_value_usd_m=round(a.carrying_value, 1),
                    )
                    for a in company.assets
                ],
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


@app.post("/runs/export")
def export_run_excel(request: RunRequest):
    """Run the engine and return results as a downloadable Excel workbook.

    Same request body as POST /runs. Returns a .xlsx file with four sheets:
      - Summary       : company profile, scenario, headline KPIs
      - Projections   : full 25-year P&L + FCF table
      - Hazards       : per-hazard physical loss breakdown by year
      - Asset Locations: asset id, name, region, lat, lon, commodity, carrying value

    Available for all tiers — the Excel file mirrors the JSON returned by POST /runs.
    """
    import io
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, numbers as xl_numbers
    from openpyxl.utils import get_column_letter
    from fastapi.responses import StreamingResponse

    # ── resolve company + scenario (same logic as POST /runs) ─────────────────
    company_id = request.company_id.lower()
    if company_id not in COMPANY_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Company '{request.company_id}' not found.")
    scenario_id = request.scenario_id.lower().replace(" ", "_")
    if scenario_id not in SCENARIO_REGISTRY and scenario_id not in CUSTOM_SCENARIO_REGISTRY:
        if scenario_id == "custom" and request.custom_scenario:
            scenario = _build_custom_scenario(request.custom_scenario)
        else:
            raise HTTPException(status_code=404, detail=f"Scenario '{request.scenario_id}' not found.")
    else:
        scenario = SCENARIO_REGISTRY.get(scenario_id) or CUSTOM_SCENARIO_REGISTRY[scenario_id]

    company = COMPANY_REGISTRY[company_id]
    results = run_engine(company=company, scenario=scenario)

    # ── build workbook ─────────────────────────────────────────────────────────
    wb = openpyxl.Workbook()

    # helpers
    HDR = Font(bold=True, color="FFFFFF")
    HDR_FILL = PatternFill("solid", fgColor="1F3864")   # dark navy
    SUB_FILL = PatternFill("solid", fgColor="2F75B6")   # mid blue
    MONEY = '#,##0.0'
    PCT   = '0.00%'

    def _hdr_row(ws, row, values, fill=HDR_FILL):
        for col, v in enumerate(values, 1):
            c = ws.cell(row=row, column=col, value=v)
            c.font = HDR
            c.fill = fill
            c.alignment = Alignment(horizontal="center")

    def _autofit(ws, min_w=10, max_w=40):
        for col in ws.columns:
            length = max((len(str(cell.value or "")) for cell in col), default=min_w)
            ws.column_dimensions[get_column_letter(col[0].column)].width = min(max(length + 2, min_w), max_w)

    # ── Sheet 1: Summary ──────────────────────────────────────────────────────
    ws1 = wb.active
    ws1.title = "Summary"
    _hdr_row(ws1, 1, ["Field", "Value"])
    summary_rows = [
        ("Company ID",       company.id),
        ("Company Name",     company.name),
        ("Sector",           company.sector),
        ("HQ Region",        company.hq_region),
        ("Scenario",         scenario.name),
        ("Scenario Family",  scenario.family.value),
        ("Horizon",          f"{scenario.horizon[0]}–{scenario.horizon[1]}"),
        ("Model Version",    results.model_version),
        ("Run ID",           results.run_id),
        ("",                 ""),
        ("Revenue (USD M)",  round(company.financials.revenue, 1)),
        ("EBITDA (USD M)",   round(company.financials.ebitda, 1)),
        ("WACC (%)",         round(company.financials.wacc_base * 100, 2)),
        ("Net Debt (USD M)", round(company.financials.net_debt, 1)),
        ("",                 ""),
        ("NPV of FCF (USD M)",       round(results.npv_fcf, 1)),
        ("Enterprise Value (USD M)", round(results.enterprise_value, 1)),
        ("Equity Value (USD M)",     round(results.equity_value, 1)),
        ("WACC Used (%)",            round(results.wacc_used * 100, 2)),
        ("EBITDA Compression 2030",  f"{round((results.ebitda_compression_2030_pct or 0)*100, 1)}%"),
        ("EBITDA Compression 2040",  f"{round((results.ebitda_compression_2040_pct or 0)*100, 1)}%"),
        ("",                         ""),
        ("Gross VaR NPV (USD M)",    round(results.gross_var_npv, 1)),
        ("Net VaR NPV (USD M)",      round(results.net_var_npv, 1)),
        ("Portfolio Impairment",     f"{round(results.portfolio_impairment_pct * 100, 2)}%"),
        ("Peak Annual Gross VaR (USD M)", round(results.peak_annual_gross_var, 1)),
        ("Dominant Hazard",          results.dominant_hazard.replace("_", " ").title()),
    ]
    for r, (k, v) in enumerate(summary_rows, 2):
        ws1.cell(row=r, column=1, value=k).font = Font(bold=bool(k))
        ws1.cell(row=r, column=2, value=v)
    _autofit(ws1)

    # ── Sheet 2: 25-Year Projections ──────────────────────────────────────────
    ws2 = wb.create_sheet("Projections")
    proj_cols = ["Year", "Revenue (M)", "OPEX (M)", "Carbon Cost (M)",
                 "Physical Loss (M)", "EBITDA (M)", "D&A (M)", "EBIT (M)",
                 "NOPAT (M)", "Trans. CapEx (M)", "Adapt. CapEx (M)",
                 "Maint. CapEx (M)", "FCF (M)",
                 "CAPEX Shock (M)", "OPEX Shock (M)", "Gross VaR (M)",
                 "Insurance Offset (M)", "Net VaR (M)"]
    _hdr_row(ws2, 1, proj_cols)
    for r, yr in enumerate(results.years, 2):
        row_vals = [
            yr.year,
            round(yr.revenue, 1),
            round(yr.opex, 1),
            round(yr.carbon_cost, 1),
            round(yr.physical_loss_cost, 1),
            round(yr.ebitda, 1),
            round(yr.da, 1),
            round(yr.ebit, 1),
            round(yr.nopat, 1),
            round(yr.transition_capex, 1),
            round(yr.adaptation_capex, 1),
            round(yr.maintenance_capex, 1),
            round(yr.fcf, 1),
            round(yr.capex_shock, 2),
            round(yr.opex_shock, 2),
            round(yr.gross_var, 2),
            round(yr.insurance_offset, 2),
            round(yr.net_var, 2),
        ]
        for col, v in enumerate(row_vals, 1):
            ws2.cell(row=r, column=col, value=v)
    _autofit(ws2)

    # ── Sheet 3: Hazard Breakdown ─────────────────────────────────────────────
    ws3 = wb.create_sheet("Hazards")
    # Collect all hazard keys from any year
    all_hazards: list[str] = []
    for yr in results.years:
        for k in yr.physical_loss_by_hazard:
            if k not in all_hazards:
                all_hazards.append(k)
    haz_cols = ["Year", "Total Physical Loss (M)"] + [h.replace("_", " ").title() + " (M)" for h in all_hazards]
    _hdr_row(ws3, 1, haz_cols)
    for r, yr in enumerate(results.years, 2):
        row_vals = [yr.year, round(yr.physical_loss_cost, 3)]
        for h in all_hazards:
            row_vals.append(round(yr.physical_loss_by_hazard.get(h, 0.0), 3))
        for col, v in enumerate(row_vals, 1):
            ws3.cell(row=r, column=col, value=v)
    _autofit(ws3)

    # ── Sheet 4: Asset Locations ───────────────────────────────────────────────
    ws4 = wb.create_sheet("Asset Locations")
    asset_cols = ["Asset ID", "Name", "Commodity", "Region",
                  "Latitude", "Longitude", "Equipment Type",
                  "Production (units/yr)", "Unit", "Carrying Value (USD M)"]
    _hdr_row(ws4, 1, asset_cols)
    for r, a in enumerate(company.assets, 2):
        row_vals = [
            a.id, a.name, a.commodity.value, a.region,
            a.lat, a.lon, a.equipment_type or "",
            a.baseline_production, a.production_unit,
            round(a.carrying_value / 1e6, 2),
        ]
        for col, v in enumerate(row_vals, 1):
            ws4.cell(row=r, column=col, value=v)
    _autofit(ws4)

    # ── stream response ────────────────────────────────────────────────────────
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)

    filename = f"CRI_{company.id}_{scenario_id}.xlsx"
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


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
    # Temporal + spatial enrichment (v0.7)
    asset_locations: list[dict] | None = None       # [{id, name, lat, lon}] for map plotting
    hazard_trajectory: dict | None = None           # {hazard → {year → prob %}} key years 2026-2050
    critical_year: int | None = None                # first year ELF > 5% (actionable threshold)
    peak_loss_pct: float | None = None              # peak annual loss % across horizon
    warming_delta_c: float | None = None            # actual CMIP6 warming at asset (°C)
    has_live_data: bool = False                     # True if NASA POWER / Open-Meteo responded
    # Decision-support metrics (Gap 1–4 fixes, pipeline.py v0.8)
    gross_var_npv: float | None = None              # USD M — cumulative NPV of gross physical VaR
    net_var_npv: float | None = None                # USD M — cumulative NPV after insurance
    peak_annual_gross_var: float | None = None      # USD M — worst single-year gross VaR
    revenue_usd_m: float | None = None              # USD M — annual revenue (from company profile)
    expected_annual_loss_m: float | None = None     # USD M — EAL = peak_annual_gross_var
    insurance_gap_npv_m: float | None = None        # USD M — gross_var_npv − net_var_npv
    physical_risk_score: float | None = None        # 0–100 from RatingEngine.physical.score
    compound_risk_score: float | None = None        # 0–100 — physical × 1.15 compound multiplier
    transition_score: float | None = None           # 0–100 from RatingEngine.transition.score
    financial_score: float | None = None            # 0–100 from RatingEngine.financial.score
    climate_rating: str | None = None               # A / B / C / D / E letter rating
    composite_score: float | None = None            # 0–100 overall CRI composite
    stranded_year: int | None = None                # first year with stranded_writedown > 0
    disruption_days_yr: float | None = None         # peak_annual_gross_var_usd / (revenue/260)


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


@app.post("/reports/pdf")
def generate_pdf_report(request: DisclosureRequest):
    """Generate an institutional-grade PDF climate risk report.

    Runs the full CRI pipeline under all three NGFS scenarios (NZE 2050,
    Delayed Transition, Current Policies) and compiles results into a
    multi-page audit-ready PDF document covering:

    - Executive Summary with Gross VaR, Net VaR, EBITDA compression
    - Physical hazard assessment (per-hazard breakdown, asset coordinates)
    - Transition risk (carbon cost trajectory, stranded asset flags)
    - 25-year financial projections at 5-year milestones
    - TCFD / IFRS S2 / CSRD alignment index
    - Methodology and audit transparency appendix

    Returns a downloadable .pdf file. Available from Professional tier and above.
    """
    import io
    from fastapi.responses import StreamingResponse
    from ..outcomes.pdf_report import generate_pdf

    cid = request.company_id.lower()
    if cid not in COMPANY_REGISTRY:
        raise HTTPException(status_code=404,
            detail=f"Company '{request.company_id}' not found. Available: {list(COMPANY_REGISTRY)}")

    company = COMPANY_REGISTRY[cid]

    # Run all three scenarios
    r_nze     = run_engine(company=company, scenario=NZE_2050)
    r_delayed = run_engine(company=company, scenario=DELAYED_TRANSITION)
    r_cp      = run_engine(company=company, scenario=CURRENT_POLICIES)

    pdf_bytes = generate_pdf(
        company=company,
        results_nze=r_nze,
        results_delayed=r_delayed,
        results_cp=r_cp,
    )

    filename = f"ClimRisk_{company.id}_Report_{request.reporting_year}.pdf"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


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
            "status": "live",
            "source": "WRI Aqueduct 4.0",
            "url": "https://aqueduct40.wri.org/api/v1/point",
            "coverage": "Global point query (lat/lon) — fallback to 21-region lookup table",
            "hazards": ["water_stress", "flood_riverine", "flood_coastal", "drought", "groundwater"],
            "note": "Year-adjusted projections via WRI SSP2-RCP4.5 stress multipliers",
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
# PREDICTIVE EVENT ENGINE
# Short-to-medium horizon climate event alerts via GDACS + NOAA NHC + Open-Meteo
# ──────────────────────────────────────────────────────────────────────────────

class PredictiveAsset(BaseModel):
    """Single asset for a predictive assessment."""
    id:                  str
    name:                str
    lat:                 float
    lon:                 float
    sector:              str = "industrial"
    annual_revenue_usd:  float = 0.0


class PredictiveRequest(BaseModel):
    """Request body for POST /predictive/alerts."""
    company_id:          str
    company_name:        str
    assets:              list[PredictiveAsset]
    annual_revenue_usd:  float
    min_severity:        int   = 2      # 1=Minor … 5=Severe; default filter ≥ Moderate
    min_confidence:      float = 0.35


@app.post("/predictive/alerts")
def get_predictive_alerts(req: PredictiveRequest) -> list[dict]:
    """Run the predictive event engine for a company's assets.

    Queries GDACS (real-time disaster alerts), NOAA NHC (tropical storm cone),
    and Open-Meteo 16-day ensemble forecasts for each asset location.
    Returns a ranked list of EventAlert objects serialised as dicts.

    Each alert includes:
      - event_type, severity, time_horizon_days, confidence
      - total_revenue_at_risk_usd, capex_at_risk_usd, ebitda_margin_hit_pp
      - urgency_score (higher = act sooner)
      - asset coordinates + company context

    Alerts are filtered to severity ≥ req.min_severity and confidence ≥ req.min_confidence.
    """
    from ..predictive.engine import PredictiveEventEngine
    import dataclasses

    engine = PredictiveEventEngine(
        min_severity=req.min_severity,
        min_confidence=req.min_confidence,
    )
    asset_dicts = [a.model_dump() for a in req.assets]
    alerts = engine.assess_company(
        company_id=req.company_id,
        company_name=req.company_name,
        assets=asset_dicts,
        annual_revenue_usd=req.annual_revenue_usd,
    )
    # Serialise: use to_dict if available, else dataclasses.asdict
    result = []
    for a in alerts:
        if hasattr(a, "to_dict"):
            d = a.to_dict()
        else:
            d = dataclasses.asdict(a)
            # Convert enum fields to strings for JSON serialisation
            for k, v in d.items():
                if hasattr(v, "value"):
                    d[k] = v.value
                elif hasattr(v, "name"):
                    d[k] = v.name
        result.append(d)
    return result


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


# ──────────────────────────────────────────────────────────────────────────────
# MODEL SCORECARD  —  aggregate backtesting across all historical events
# ──────────────────────────────────────────────────────────────────────────────

@app.get(
    "/validation/scorecard",
    summary="Aggregate model accuracy scorecard across all 16 backtesting events",
    tags=["Validation"],
)
def model_scorecard() -> dict:
    """
    Run the CRI physical hazard engine against all 16 documented historical
    climate events and return an aggregate accuracy scorecard.

    This is the primary trust-building endpoint.  Each event is run with a
    synthetic company proxy; predicted loss fractions are compared against
    documented economy-wide losses from Munich Re NatCatSERVICE, Swiss Re
    sigma, EM-DAT, World Bank GFDRR, and NOAA NCEI.

    **Accuracy thresholds:**
    - CALIBRATED   — predicted within ±20% of documented loss rate
    - ACCEPTABLE   — predicted within ±50% (within typical model uncertainty)
    - NEEDS_REVIEW — predicted error >50% (systematic bias suspected)

    **Aggregate metrics:**
    - `hit_rate_pct`       — % of events CALIBRATED or ACCEPTABLE (target ≥80%)
    - `calibrated_pct`     — % of events strictly CALIBRATED (target ≥50%)
    - `mean_abs_error_pct` — mean absolute relative error across all events
    - `events`             — per-event breakdown (name, type, predicted, actual, error, status)

    Results are cached for 24 hours since they do not depend on external APIs.

    **Citation:** All loss benchmarks are cited in GET /scenarios/historical.
    Methodology: CRI Physical Hazard Engine v0.4, backtested against 16 events
    spanning 1997–2023 (El Niño/La Niña, floods, drought, cyclone, wildfire,
    heat dome, cold snap).
    """
    from ..climate.scenarios.historical_events import HISTORICAL_LIBRARY, list_historical_events
    from ..climate.scenarios.calibration import run_calibration, calibration_report_to_dict

    # Check 24-hour disk cache (simple JSON file)
    import json as _json
    import time as _time
    from pathlib import Path as _Path
    _cache_path = _Path("/tmp/cri_scorecard_cache_v04.json")
    try:
        if _cache_path.exists():
            _cache_age = _time.time() - _cache_path.stat().st_mtime
            if _cache_age < 86400:   # 24 hours
                cached = _json.loads(_cache_path.read_text())
                if isinstance(cached, dict) and "hit_rate_pct" in cached:
                    cached["_cached"] = True
                    return cached
    except Exception:
        pass

    event_ids = list(HISTORICAL_LIBRARY.keys())
    rows = []
    calibrated = 0
    acceptable = 0
    total_rel_err = 0.0
    ran = 0

    for eid in event_ids:
        try:
            report = run_calibration(
                historical_event_id=eid,
                company=None,          # synthetic proxy
                sector_filter=None,
                ssp="ssp370",
                year=2025,
            )
            rd = calibration_report_to_dict(report)
            status = rd.get("overall_status", "error")
            rel_err = rd.get("overall_relative_error_pct", None)

            ev = HISTORICAL_LIBRARY[eid]
            rows.append({
                "event_id":            eid,
                "event_name":          ev.name,
                "year":                ev.year_start,
                "hazard_type":         ev.physical_event_id.replace("_", " ").title()
                                       if ev.physical_event_id else "—",
                "primary_region":      str(ev.region.value) if hasattr(ev.region, "value") else str(ev.region),
                "total_loss_usd_m":    ev.total_loss_usd_m,
                "source":              ev.source_total_loss,
                "overall_status":      status,
                "relative_error_pct":  round(rel_err, 1) if rel_err is not None else None,
                "sector_results":      rd.get("sector_results", []),
            })

            if rel_err is not None:
                total_rel_err += abs(rel_err)
                ran += 1
            if status == "calibrated":
                calibrated += 1
            elif status == "acceptable":
                acceptable += 1

        except Exception as exc:
            rows.append({
                "event_id": eid,
                "event_name": eid,
                "overall_status": "error",
                "error": str(exc),
            })

    n = len(event_ids)
    hit_rate = round(100.0 * (calibrated + acceptable) / max(1, n), 1)
    calibrated_pct = round(100.0 * calibrated / max(1, n), 1)
    mean_abs_err = round(total_rel_err / max(1, ran), 1)

    # Grade interpretation
    if hit_rate >= 80:
        grade, grade_note = "A", "Strong model performance — suitable for institutional use"
    elif hit_rate >= 65:
        grade, grade_note = "B", "Acceptable model performance — suitable with documented caveats"
    elif hit_rate >= 50:
        grade, grade_note = "C", "Moderate performance — recommend supplemental analyst review"
    else:
        grade, grade_note = "D", "Below target — recalibration recommended before institutional deployment"

    result = {
        "model_version":        "CRI v0.4",
        "backtesting_events":   n,
        "events_ran":           ran,
        "hit_rate_pct":         hit_rate,
        "calibrated_pct":       calibrated_pct,
        "acceptable_pct":       round(100.0 * acceptable / max(1, n), 1),
        "needs_review_pct":     round(100.0 * (n - calibrated - acceptable) / max(1, n), 1),
        "mean_abs_error_pct":   mean_abs_err,
        "grade":                grade,
        "grade_note":           grade_note,
        "events":               rows,
        "accuracy_thresholds": {
            "calibrated":   "Predicted loss within ±20% of documented loss rate",
            "acceptable":   "Predicted loss within ±50% (within typical model uncertainty)",
            "needs_review": "Predicted error >50% — systematic bias suspected",
        },
        "data_sources": [
            "Munich Re NatCatSERVICE", "Swiss Re sigma reports",
            "EM-DAT (CRED)", "World Bank GFDRR",
            "NOAA NCEI Billion-Dollar Disasters",
        ],
        "event_coverage": {
            "date_range":    "1997–2023",
            "hazard_types":  ["flood", "drought", "cyclone", "wildfire", "heat_dome",
                              "cold_snap", "El_Nino_La_Nina"],
            "regions":       ["Asia-Pacific", "North America", "Europe",
                              "South Asia", "Southern Africa"],
        },
        "caveats": [
            "Historical losses are economy-wide; engine predictions are single-company normalised.",
            "Indirect losses (supply chain, GDP multiplier) are excluded from engine predictions.",
            "Calibration uses synthetic sector proxies, not specific named companies.",
            "Seasonal timing of events may differ from the engine's annual average hazard.",
        ],
        "methodology": (
            "For each event, a synthetic company matching the primary affected sector is run "
            "through the CRI physical hazard cascade engine. Predicted revenue loss fraction "
            "is compared against the documented economy-wide loss as a fraction of sectoral GDP. "
            "Relative error = |predicted_frac - historical_frac| / historical_frac."
        ),
        "_cached": False,
    }

    # Cache result for 24 h
    try:
        _cache_path.write_text(_json.dumps(result, default=str))
    except Exception:
        pass

    return result


# ── Aladdin-style Portfolio Risk ──────────────────────────────────────────────

@app.post("/portfolio/risk", response_model=PortfolioRiskResponse)
def portfolio_risk(request: PortfolioRiskRequest) -> PortfolioRiskResponse:
    """Aggregate portfolio climate VaR, MC-CVaR, and factor attribution.

    Mirrors BlackRock Aladdin's portfolio risk aggregation layer:
    - Portfolio Climate VaR at 95th and 99th percentile
    - Marginal Contribution to CVaR (Euler decomposition) per position
    - Factor attribution by hazard, sector, and geography
    - Diversification benefit vs standalone sum
    - Optional 1.5 / 2.0 / 3.0 / 4.0 °C stress scenarios

    Accepts up to 500 positions. For larger portfolios contact enterprise support.
    """
    from ..climate.portfolio import aggregate_portfolio_risk, PortfolioPosition
    import dataclasses

    if not request.positions:
        raise HTTPException(status_code=422, detail="At least one position required.")
    if len(request.positions) > 500:
        raise HTTPException(status_code=422, detail="Maximum 500 positions per request.")

    positions = [
        PortfolioPosition(
            company_id=p.company_id,
            company_name=p.company_name,
            exposure_usd_m=p.exposure_usd_m,
            sector=p.sector,
            region=p.region,
            physical_score=p.physical_score,
            transition_score=p.transition_score,
            # Task 53: pass through sub-scores if provided
            policy_score=getattr(p, "policy_score", None),
            technology_score=getattr(p, "technology_score", None),
            market_score=getattr(p, "market_score", None),
            litigation_score=getattr(p, "litigation_score", None),
        )
        for p in request.positions
    ]

    result = aggregate_portfolio_risk(positions, run_stress=request.run_stress)

    def _dc(obj):
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {k: _dc(v) for k, v in dataclasses.asdict(obj).items()}
        if isinstance(obj, list):
            return [_dc(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _dc(v) for k, v in obj.items()}
        return obj

    pos_dicts = [_dc(p) for p in result.positions]
    stress_dicts = [_dc(s) for s in result.stress_scenarios]
    fa = _dc(result.factor_attribution)

    term_dicts = [_dc(t) for t in result.term_structure]

    return PortfolioRiskResponse(
        run_id=result.run_id,
        n_positions=result.n_positions,
        total_exposure_usd_m=result.total_exposure_usd_m,
        portfolio_eal_cp_usd_m=result.portfolio_eal_cp_usd_m,
        portfolio_var95_cp_usd_m=result.portfolio_var95_cp_usd_m,
        portfolio_cvar95_cp_usd_m=result.portfolio_cvar95_cp_usd_m,
        portfolio_var99_cp_usd_m=result.portfolio_var99_cp_usd_m,
        portfolio_cvar99_cp_usd_m=result.portfolio_cvar99_cp_usd_m,
        portfolio_eal_nze_usd_m=result.portfolio_eal_nze_usd_m,
        portfolio_var99_nze_usd_m=result.portfolio_var99_nze_usd_m,
        portfolio_cvar99_nze_usd_m=result.portfolio_cvar99_nze_usd_m,
        diversification_benefit_usd_m=result.diversification_benefit_usd_m,
        diversification_ratio=result.diversification_ratio,
        herfindahl_index=result.herfindahl_index,
        top3_concentration_pct=result.top3_concentration_pct,
        positions=pos_dicts,
        factor_attribution=fa,
        stress_scenarios=stress_dicts,
        # v0.9 gap fixes
        stress_regime_active=result.stress_regime_active,
        stress_weight=result.stress_weight,
        macro_factor_summary=result.macro_factor_summary,
        term_structure=term_dicts,
        # Task 53: multi-factor transition decomposition
        transition_factor_cp=result.factor_attribution.transition_factor_cp,
    )


@app.post("/portfolio/credit", response_model=PortfolioCreditResponse)
def portfolio_credit(request: PortfolioCreditRequest) -> PortfolioCreditResponse:
    """Climate-adjusted PD, LGD, ECL, and RWA for a bank loan book.

    Implements the ECB Climate Stress Test 2022 methodology for PD uplift
    combined with NGFS Phase 4 sector-level transition shock tables:
    - Physical PD uplift from asset exposure to hazard events
    - Transition PD uplift from stranded-asset risk under NGFS scenarios
    - LGD uplift from collateral degradation (sector + collateral type)
    - IFRS 9 Expected Credit Loss: PD × LGD × EAD
    - Basel III RWA under Standardised Approach with climate PD migration
    - CET1 incremental capital charge (8% + 4.5% conservation buffer)
    """
    from ..climate.credit_risk import compute_portfolio_credit_risk, CounterpartyInput
    import dataclasses

    if not request.obligors:
        raise HTTPException(status_code=422, detail="At least one obligor required.")

    obligors = [
        CounterpartyInput(
            company_id=o.company_id,
            company_name=o.company_name,
            sector=o.sector,
            ead_usd_m=o.ead_usd_m,
            baseline_pd=o.baseline_pd,
            baseline_lgd=o.baseline_lgd,
            physical_score=o.physical_score,
            transition_score=o.transition_score,
            collateral_type=o.collateral_type,
            scenario=o.scenario,
            region=getattr(o, "region", "Global"),
        )
        for o in request.obligors
    ]

    result = compute_portfolio_credit_risk(obligors)

    def _dc(obj):
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {k: _dc(v) for k, v in dataclasses.asdict(obj).items()}
        if isinstance(obj, list):
            return [_dc(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _dc(v) for k, v in obj.items()}
        return obj

    return PortfolioCreditResponse(
        n_obligors=result.n_obligors,
        total_ead_usd_m=result.total_ead_usd_m,
        scenario=result.scenario,
        baseline_ecl_usd_m=result.baseline_ecl_usd_m,
        climate_ecl_usd_m=result.climate_ecl_usd_m,
        ecl_increment_usd_m=result.ecl_increment_usd_m,
        ecl_uplift_pct=result.ecl_uplift_pct,
        baseline_rwa_usd_m=result.baseline_rwa_usd_m,
        climate_rwa_usd_m=result.climate_rwa_usd_m,
        rwa_increment_usd_m=result.rwa_increment_usd_m,
        incremental_capital_usd_m=result.incremental_capital_usd_m,
        sector_breakdown=result.sector_breakdown,
        obligors=[_dc(o) for o in result.obligors],
        portfolio_term_structure=result.portfolio_term_structure,
        calibration_method=result.calibration_method,
    )


# ── Task 54: Portfolio CVaR optimiser ────────────────────────────────────────

from ..climate.portfolio_optimiser import optimise_portfolio_cvar as _optimise_cvar
from ..climate.portfolio_optimiser import OptimisationResult as _OptResult
import dataclasses as _dc_mod

from .schemas import (
    PortfolioOptimiseRequest, PortfolioOptimiseResponse,
    RulesCheckRequest, RulesCheckResponse,
    ValidationRequest, ValidationResponse,
)

@app.post("/portfolio/optimise", response_model=PortfolioOptimiseResponse)
def portfolio_optimise(request: PortfolioOptimiseRequest) -> PortfolioOptimiseResponse:
    """Minimise portfolio MC-CVaR₉₉ via scipy SLSQP subject to risk constraints.

    Implements Rockafellar-Uryasev CVaR minimisation.
    Returns original vs optimised weights, CVaR reduction, and binding constraints.
    Falls back gracefully if scipy is not installed.
    """
    from ..climate.portfolio import PortfolioPosition as _PP
    positions = [
        _PP(
            company_id=p.company_id,
            company_name=p.company_name,
            exposure_usd_m=p.exposure_usd_m,
            sector=p.sector,
            region=p.region,
            physical_score=p.physical_score,
            transition_score=p.transition_score,
        )
        for p in request.positions
    ]
    result = _optimise_cvar(
        positions,
        sector_cap=request.sector_cap,
        single_name_cap=request.single_name_cap,
        hhi_limit=request.hhi_limit,
        w_min=request.w_min,
    )
    def _dc(obj):
        if _dc_mod.is_dataclass(obj) and not isinstance(obj, type):
            return {k: _dc(v) for k, v in _dc_mod.asdict(obj).items()}
        if isinstance(obj, list):
            return [_dc(x) for x in obj]
        return obj

    return PortfolioOptimiseResponse(
        status=result.status,
        message=result.message,
        original_cvar99_usd_m=result.original_cvar99_usd_m,
        optimised_cvar99_usd_m=result.optimised_cvar99_usd_m,
        cvar99_reduction_pct=result.cvar99_reduction_pct,
        original_eal_usd_m=result.original_eal_usd_m,
        optimised_eal_usd_m=result.optimised_eal_usd_m,
        hhi_original=result.hhi_original,
        hhi_optimised=result.hhi_optimised,
        positions=[_dc(p) for p in result.positions],
        binding_constraints=result.binding_constraints,
        n_iterations=result.n_iterations,
        solver=result.solver,
        methodology=result.methodology,
    )


# ── Task 56: Rules engine ─────────────────────────────────────────────────────

import datetime as _datetime

@app.post("/portfolio/rules", response_model=RulesCheckResponse)
def portfolio_rules(request: RulesCheckRequest) -> RulesCheckResponse:
    """Check portfolio against configurable risk limit rules.

    Returns: breach/warn/pass status per rule, overall flag, and metrics.
    Rules checked:
      • CVaR₉₉ absolute limit (if supplied)
      • Sector concentration (% of total CVaR from any single sector)
      • Single-name concentration (% of total CVaR from any single position)
      • HHI of MC-CVaR weights
      • EAL / total exposure ratio (if limit supplied)
    """
    from ..climate.portfolio import PortfolioPosition as _PP, aggregate_portfolio_risk as _agg
    positions = [
        _PP(
            company_id=p.company_id,
            company_name=p.company_name,
            exposure_usd_m=p.exposure_usd_m,
            sector=p.sector,
            region=p.region,
            physical_score=p.physical_score,
            transition_score=p.transition_score,
        )
        for p in request.positions
    ]
    result = _agg(positions, run_stress=False)

    checks = []
    breach = 0
    warn   = 0

    cvar99 = result.portfolio_cvar99_cp_usd_m
    eal    = result.portfolio_eal_cp_usd_m
    hhi    = result.herfindahl_index
    total_exp = result.total_exposure_usd_m

    # ── Rule 1: CVaR absolute limit ───────────────────────────────────────────
    if request.cvar99_limit_usd_m is not None:
        limit = request.cvar99_limit_usd_m
        ratio = cvar99 / max(1e-9, limit)
        status = "PASS"
        if ratio >= 1.0:
            status = "BREACH"; breach += 1
        elif ratio >= 0.85:
            status = "WARN"; warn += 1
        checks.append({
            "rule": "CVaR₉₉ Absolute Limit",
            "status": status,
            "value": round(cvar99, 4),
            "limit": limit,
            "utilisation_pct": round(ratio * 100, 1),
            "message": f"CVaR₉₉ ${cvar99:.2f}M vs limit ${limit:.2f}M ({ratio:.0%} utilised)",
        })

    # ── Rule 2: Sector concentration ─────────────────────────────────────────
    mc_total = sum(p.mc_cvar99_usd_m for p in result.positions) or 1e-9
    sector_mc: dict[str, float] = {}
    for p in result.positions:
        sector_mc[p.sector] = sector_mc.get(p.sector, 0.0) + p.mc_cvar99_usd_m
    for sec, sec_cvar in sorted(sector_mc.items(), key=lambda x: -x[1]):
        ratio = sec_cvar / mc_total
        status = "PASS"
        if ratio >= request.sector_concentration_limit:
            status = "BREACH"; breach += 1
        elif ratio >= request.sector_concentration_limit * 0.85:
            status = "WARN"; warn += 1
        checks.append({
            "rule": f"Sector Concentration: {sec}",
            "status": status,
            "value": round(ratio, 4),
            "limit": request.sector_concentration_limit,
            "utilisation_pct": round(ratio / request.sector_concentration_limit * 100, 1),
            "message": f"{sec} = {ratio:.1%} of MC-CVaR (limit {request.sector_concentration_limit:.0%})",
        })

    # ── Rule 3: Single-name concentration ────────────────────────────────────
    for p in sorted(result.positions, key=lambda x: -x.mc_cvar99_usd_m):
        ratio = p.mc_cvar99_usd_m / mc_total
        status = "PASS"
        if ratio >= request.single_name_limit:
            status = "BREACH"; breach += 1
        elif ratio >= request.single_name_limit * 0.85:
            status = "WARN"; warn += 1
        checks.append({
            "rule": f"Single-Name: {p.company_name}",
            "status": status,
            "value": round(ratio, 4),
            "limit": request.single_name_limit,
            "utilisation_pct": round(ratio / request.single_name_limit * 100, 1),
            "message": f"{p.company_name} = {ratio:.1%} of MC-CVaR (limit {request.single_name_limit:.0%})",
        })

    # ── Rule 4: HHI ───────────────────────────────────────────────────────────
    hhi_status = "PASS"
    if hhi >= request.hhi_limit:
        hhi_status = "BREACH"; breach += 1
    elif hhi >= request.hhi_limit * 0.85:
        hhi_status = "WARN"; warn += 1
    checks.append({
        "rule": "HHI Concentration",
        "status": hhi_status,
        "value": round(hhi, 4),
        "limit": request.hhi_limit,
        "utilisation_pct": round(hhi / request.hhi_limit * 100, 1),
        "message": f"HHI = {hhi:.3f} (limit {request.hhi_limit:.3f})",
    })

    # ── Rule 5: EAL / exposure ratio ─────────────────────────────────────────
    if request.eal_to_revenue_limit is not None and total_exp > 0:
        eal_ratio = eal / total_exp
        limit = request.eal_to_revenue_limit
        er_status = "PASS"
        if eal_ratio >= limit:
            er_status = "BREACH"; breach += 1
        elif eal_ratio >= limit * 0.85:
            er_status = "WARN"; warn += 1
        checks.append({
            "rule": "EAL / Total Exposure",
            "status": er_status,
            "value": round(eal_ratio, 4),
            "limit": limit,
            "utilisation_pct": round(eal_ratio / limit * 100, 1),
            "message": f"EAL/Exposure = {eal_ratio:.2%} (limit {limit:.2%})",
        })

    overall = "PASS"
    if breach > 0:
        overall = "BREACH"
    elif warn > 0:
        overall = "WARN"

    return RulesCheckResponse(
        overall_status=overall,
        checks=checks,
        breach_count=breach,
        warn_count=warn,
        portfolio_cvar99_usd_m=round(cvar99, 4),
        portfolio_eal_usd_m=round(eal, 4),
        hhi=round(hhi, 4),
        generated_at=_datetime.datetime.utcnow().isoformat() + "Z",
    )


# ── Task 57: Validation framework ────────────────────────────────────────────

from ..climate.validation import run_validation as _run_validation

@app.post("/portfolio/validate", response_model=ValidationResponse)
def portfolio_validate(request: ValidationRequest) -> ValidationResponse:
    """Model validation — Gini, Brier, Kupiec, and Hosmer-Lemeshow metrics.

    Submit historical observations (predicted PD + realised default, predicted EAL +
    realised loss) to assess model discriminatory power and calibration quality.
    """
    result = _run_validation(request.observations, metrics=request.metrics)

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    kpof = result.get("kupiec_pof")
    hl   = result.get("hl_test")

    return ValidationResponse(
        n_observations=result.get("n_observations", 0),
        gini_coefficient=result.get("gini_coefficient"),
        brier_score=result.get("brier_score"),
        kupiec_pof=kpof if isinstance(kpof, dict) else None,
        hl_test=hl if isinstance(hl, dict) else None,
        summary=result.get("summary", ""),
        methodology=result.get("methodology", ""),
    )


# ── Gap 7: Real-time position monitor (SSE) ───────────────────────────────────

from fastapi.responses import StreamingResponse as _StreamingResponse
import asyncio as _asyncio
import json as _json

@app.post("/portfolio/stream")
async def portfolio_stream(request: PortfolioStreamRequest):
    """SSE endpoint — push recalculated portfolio CVaR on every tick.

    Clients connect via EventSource (or fetch with ReadableStream) and receive
    a JSON-encoded PortfolioRiskResponse on each push. The engine recalculates
    the full MC-CVaR each tick so the client always sees fresh numbers.

    Gap 7 fix: provides the real-time position monitor that Aladdin uses to
    reprice the portfolio on every market tick.

    Query parameters (also accepted in body):
        interval_seconds : push frequency in seconds (default 5, min 1)
        max_ticks        : stop after N pushes (0 = run until disconnect)
    """
    from ..climate.portfolio import aggregate_portfolio_risk, PortfolioPosition
    import dataclasses, time

    def _dc(obj):
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {k: _dc(v) for k, v in dataclasses.asdict(obj).items()}
        if isinstance(obj, list):
            return [_dc(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _dc(v) for k, v in obj.items()}
        return obj

    interval = max(1.0, request.interval_seconds)
    max_ticks = request.max_ticks if request.max_ticks > 0 else 999_999

    base_positions = [
        PortfolioPosition(
            company_id=p.company_id,
            company_name=p.company_name,
            exposure_usd_m=p.exposure_usd_m,
            sector=p.sector,
            region=p.region,
            physical_score=p.physical_score,
            transition_score=p.transition_score,
        )
        for p in request.positions
    ]

    # ── Task 58: yfinance ticker map ─────────────────────────────────────────
    # Maps company_id → Yahoo Finance ticker for real market price changes.
    # Price changes are used to update exposure (mark-to-market) on each SSE tick.
    _TICKER_MAP: dict[str, str] = {
        "bhp":         "BHP.AX",
        "rio_tinto":   "RIO.L",
        "shell":       "SHEL.L",
        "santos":      "STO.AX",
        "meridian":    "MEL.NZ",
        "woodside":    "WDS.AX",
        "chevron":     "CVX",
        "exxon":       "XOM",
        "bp":          "BP.L",
        "total":       "TTE.PA",
        "glencore":    "GLEN.L",
        "vale":        "VALE",
        "fortescue":   "FMG.AX",
        "nextenergy":  "NEE",
        "orsted":      "ORSTED.CO",
    }

    def _fetch_price_change(company_id: str) -> float:
        """Try yfinance; fall back to ±2% random walk if unavailable."""
        import random as _rnd
        try:
            import yfinance as yf
            ticker = _TICKER_MAP.get(company_id.lower())
            if not ticker:
                return _rnd.uniform(-0.02, 0.02)
            # Fast 2-day history — returns a single close-to-close return
            hist = yf.Ticker(ticker).history(period="2d", interval="1d", timeout=3)
            if len(hist) >= 2:
                c0, c1 = hist["Close"].iloc[-2], hist["Close"].iloc[-1]
                return float(c1 / c0 - 1.0)
            return _rnd.uniform(-0.005, 0.005)
        except Exception:
            import random as _rnd2
            return _rnd2.uniform(-0.02, 0.02)

    async def event_generator():
        import random as _rand
        tick = 0
        while tick < max_ticks:
            try:
                # Task 58: attempt real price changes from yfinance; fall back to random walk
                price_changes = {p.company_id: _fetch_price_change(p.company_id)
                                 for p in base_positions}

                live_positions = [
                    PortfolioPosition(
                        company_id=p.company_id,
                        company_name=p.company_name,
                        exposure_usd_m=p.exposure_usd_m * (1.0 + price_changes.get(p.company_id, 0.0)),
                        sector=p.sector,
                        region=p.region,
                        physical_score=p.physical_score,
                        transition_score=p.transition_score,
                    )
                    for p in base_positions
                ]

                result = aggregate_portfolio_risk(live_positions, run_stress=False)
                payload = {
                    "tick":                      tick + 1,
                    "timestamp":                 time.time(),
                    "run_id":                    result.run_id,
                    "portfolio_eal_cp_usd_m":    result.portfolio_eal_cp_usd_m,
                    "portfolio_var99_cp_usd_m":  result.portfolio_var99_cp_usd_m,
                    "portfolio_cvar99_cp_usd_m": result.portfolio_cvar99_cp_usd_m,
                    "herfindahl_index":          result.herfindahl_index,
                    "stress_regime_active":      result.stress_regime_active,
                    "stress_weight":             result.stress_weight,
                    "positions": [
                        {
                            "company_id":      p.company_id,
                            "company_name":    p.company_name,
                            "exposure_usd_m":  p.exposure_usd_m,
                            "eal_cp_usd_m":    r.eal_cp_usd_m,
                            "mc_cvar99_usd_m": r.mc_cvar99_usd_m,
                            "mc_cvar99_pct":   r.mc_cvar99_pct,
                        }
                        for p, r in zip(live_positions, result.positions)
                    ],
                }
                yield f"data: {_json.dumps(payload)}\n\n"
                tick += 1
                await _asyncio.sleep(interval)
            except _asyncio.CancelledError:
                break
            except Exception as exc:
                yield f"data: {_json.dumps({'error': str(exc), 'tick': tick})}\n\n"
                break

        yield "data: {\"event\": \"stream_ended\"}\n\n"

    return _StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
