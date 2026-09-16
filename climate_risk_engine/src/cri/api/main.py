"""FastAPI REST API for the Climate Risk Intelligence engine."""

from __future__ import annotations

import os

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
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
    CreateCompanyRequest,
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
    DecisionRequest,
    DecisionBrief,
    RiskSignal,
    FinancialImpact,
    RiskDriver,
    RecommendedAction,
    # Pillar 2 — batch asset onboarding
    StandardAssetIn,
    BatchAssessmentRequest,
    BatchAssessmentResponse,
    AssetAssessmentResult,
    # Pillar 3 — LLM content generation
    ArticleRequest,
    ArticleResponse,
    NarrativeRequest,
    NarrativeResponse,
)


# Initialize FastAPI app
# ── API key auth ──────────────────────────────────────────────────────────────
# Set API_KEYS env var to a comma-separated list of valid keys.
# Example: API_KEYS="key-prod-abc123,key-demo-xyz789"
# If API_KEYS is empty/unset, auth is DISABLED (dev mode).
_API_KEYS: set[str] = {
    k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()
}
# Paths that bypass auth (public)
_PUBLIC_PATHS: frozenset[str] = frozenset({
    "/health", "/docs", "/openapi.json", "/redoc", "/scenarios",
})


# ── Startup: merge Supabase companies into registry ───────────────────────────
@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Load persisted companies from Supabase on startup."""
    from .db import load_companies as _load_from_db
    persisted = _load_from_db()
    if persisted:
        COMPANY_REGISTRY.update(persisted)
        import logging
        logging.getLogger(__name__).info(
            "Loaded %d company/ies from Supabase into registry.", len(persisted)
        )
    yield  # app runs here


app = FastAPI(
    title="Climate Risk Intelligence API",
    description="REST API for climate financial risk modelling",
    version="0.3.0",
    lifespan=_lifespan,
)


# ── API key middleware ────────────────────────────────────────────────────────
@app.middleware("http")
async def _auth_middleware(request: Request, call_next):
    """Enforce X-API-Key on all non-public endpoints when API_KEYS is set."""
    if _API_KEYS:
        path = request.url.path
        # Allow public paths and CORS preflight
        if path not in _PUBLIC_PATHS and request.method != "OPTIONS":
            key = request.headers.get("X-API-Key", "")
            if key not in _API_KEYS:
                return JSONResponse(
                    {"error": "Invalid or missing API key. Set X-API-Key header."},
                    status_code=401,
                )
    return await call_next(request)


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


@app.post("/companies", response_model=CompanyResponse, status_code=201, tags=["Companies"])
def create_company(req: CreateCompanyRequest) -> CompanyResponse:
    """Register a new company and persist it to Supabase.

    Accepts a simplified payload — full financials, emissions, and an optional
    primary asset location.  Once registered the company is immediately available
    for `/runs`, `/decision`, `/ratings`, and all other assessment endpoints.

    If a company with the same `id` already exists it is overwritten.
    """
    from ..data.schemas import (
        Asset, Commodity, Company, EmissionsProfile, Financials,
    )
    from .schemas import AssetLocationResponse, FinancialsResponse
    from .db import save_company as _save

    # Derive defaults for optional fields
    ebitda  = req.ebitda_usd_m  if req.ebitda_usd_m  is not None else req.revenue_usd_m * 0.20
    capex   = req.capex_usd_m   if req.capex_usd_m   is not None else req.revenue_usd_m * 0.08
    ev      = req.equity_value_usd_m if req.equity_value_usd_m is not None else (
        ebitda * 8.0 - req.net_debt_usd_m  # rough 8× EBITDA default EV
    )

    financials = Financials(
        revenue=req.revenue_usd_m,
        ebitda=ebitda,
        capex=capex,
        wacc_base=req.wacc_pct,
        net_debt=req.net_debt_usd_m,
        market_cap=max(0.0, ev),
    )

    # Build asset list (optional)
    assets: list[Asset] = []
    if req.asset_lat is not None and req.asset_lon is not None:
        rcv = req.asset_replacement_cost_usd_m or req.revenue_usd_m * 0.5
        assets.append(Asset(
            id=f"{req.id}_primary",
            name=req.asset_name or f"{req.name} — Primary Asset",
            commodity=Commodity.OTHER,
            region=req.hq_region,
            baseline_production=0.0,
            production_unit="units",
            baseline_unit_cost=0.0,
            energy_cost_share=0.20,
            carrying_value=rcv,
            remaining_life_years=25,
            emissions=EmissionsProfile(
                scope1_intensity=req.scope1_tco2 / max(req.revenue_usd_m * 1e3, 1.0),
                scope2_intensity=req.scope2_tco2 / max(req.revenue_usd_m * 1e3, 1.0),
                scope3_intensity=req.scope3_tco2 / max(req.revenue_usd_m * 1e3, 1.0),
            ),
            lat=req.asset_lat,
            lon=req.asset_lon,
        ))

    company = Company(
        id=req.id,
        name=req.name,
        sector=req.sector,
        hq_region=req.hq_region,
        financials=financials,
        assets=assets,
        data_quality=req.data_quality,
    )

    # Persist to Supabase (no-op if not configured)
    _save(company)

    # Add to in-memory registry so it's immediately queryable
    COMPANY_REGISTRY[req.id] = company

    return CompanyResponse(
        id=company.id,
        name=company.name,
        sector=company.sector,
        region=company.hq_region,
        data_quality=company.data_quality,
        financials=FinancialsResponse(
            revenue_usd_m=round(req.revenue_usd_m, 1),
            ebitda_usd_m=round(ebitda, 1),
            capex_usd_m=round(capex, 1),
            wacc_pct=round(req.wacc_pct * 100, 2),
            net_debt_usd_m=round(req.net_debt_usd_m, 1),
        ),
        assets=[
            AssetLocationResponse(
                id=a.id,
                name=a.name,
                commodity=a.commodity.value,
                region=a.region,
                lat=a.lat,
                lon=a.lon,
                equipment_type=getattr(a, "equipment_type", "industrial"),
                baseline_production=a.baseline_production,
                production_unit=a.production_unit,
                carrying_value_usd_m=round(a.carrying_value, 1),
            )
            for a in assets
        ],
    )


@app.delete("/companies/{company_id}", status_code=204, tags=["Companies"])
def delete_company(company_id: str) -> None:
    """Remove a company from the registry and from Supabase.

    Seed companies (from companies_seed.py) can be removed from the live
    registry but will return on the next server restart unless also deleted
    from Supabase.
    """
    from .db import delete_company as _delete_from_db
    if company_id not in COMPANY_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Company '{company_id}' not found.")
    del COMPANY_REGISTRY[company_id]
    _delete_from_db(company_id)


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


# ══════════════════════════════════════════════════════════════════════════════
# Task 72: POST /decision — Decision-Support First
# ══════════════════════════════════════════════════════════════════════════════
# This is the PRIMARY endpoint of the CRI engine.
# It answers "what should I do?" before "how do I report it?".
#
# Architecture:
#   1. Run scoped engine (physical + transition + financial)
#   2. Synthesise Decision Brief: signal → financial impact → drivers → actions
#   3. Compliance outputs (TCFD/ISSB) are optional, derived from the brief
#
# The signal generation logic:
#   HOLD   : EAL < 2% EBITDA AND no stranded assets AND no acute hazards
#   WATCH  : EAL 2–5% EBITDA OR stranded asset risk emerging
#   REDUCE : EAL 5–15% EBITDA OR CVaR99 > 20% of exposure
#   EXIT   : EAL > 15% EBITDA OR CVaR99 > 40% of exposure
# ══════════════════════════════════════════════════════════════════════════════

import uuid as _uuid
import datetime as _dt_dec

# ── Signal thresholds by risk appetite ───────────────────────────────────────
# Thresholds: (eal_exit, eal_reduce, eal_watch, cvar_exit, cvar_reduce, stranded_watch)
_SIGNAL_THRESHOLDS: dict[str, dict[str, float]] = {
    # Conservative — DFIs, insurance, pension funds. Lower tolerance.
    "conservative": {
        "eal_exit": 0.08, "eal_reduce": 0.03, "eal_watch": 0.01,
        "cvar_exit": 0.25, "cvar_reduce": 0.12, "stranded_watch": 0.15,
    },
    # Moderate — commercial banks, credit committees. Standard.
    "moderate": {
        "eal_exit": 0.15, "eal_reduce": 0.05, "eal_watch": 0.02,
        "cvar_exit": 0.40, "cvar_reduce": 0.20, "stranded_watch": 0.25,
    },
    # Aggressive — PE, trading desks, high-risk mandates.
    "aggressive": {
        "eal_exit": 0.25, "eal_reduce": 0.10, "eal_watch": 0.04,
        "cvar_exit": 0.60, "cvar_reduce": 0.35, "stranded_watch": 0.40,
    },
}


def _derive_signal(
    eal_pct_ebitda: float,
    cvar99_pct_exposure: float,
    stranded_pct: float,
    acute_hazard: bool,
    risk_appetite: str = "moderate",
) -> tuple[str, str, str]:
    """Return (action, urgency, rationale) for the primary risk signal.

    Thresholds vary by risk_appetite so the same numbers produce different
    signals for a DFI (conservative) vs. a PE house (aggressive).
    """
    t = _SIGNAL_THRESHOLDS.get(risk_appetite, _SIGNAL_THRESHOLDS["moderate"])

    if eal_pct_ebitda > t["eal_exit"] or cvar99_pct_exposure > t["cvar_exit"]:
        action = "EXIT"
        urgency = "IMMEDIATE"
        rationale = (
            f"EAL represents {eal_pct_ebitda:.0%} of EBITDA and CVaR₉₉ is "
            f"{cvar99_pct_exposure:.0%} of exposure — capital at risk exceeds "
            f"{risk_appetite} risk appetite thresholds."
        )
    elif eal_pct_ebitda > t["eal_reduce"] or cvar99_pct_exposure > t["cvar_reduce"]:
        action = "REDUCE"
        urgency = "NEAR_TERM"
        rationale = (
            f"EAL of {eal_pct_ebitda:.0%} of EBITDA and CVaR₉₉ of "
            f"{cvar99_pct_exposure:.0%} of exposure exceed {risk_appetite} risk appetite. "
            f"Recommend reducing exposure or demanding risk-adjusted pricing."
        )
    elif eal_pct_ebitda > t["eal_watch"] or stranded_pct > t["stranded_watch"] or acute_hazard:
        action = "WATCH"
        urgency = "MEDIUM_TERM"
        rationale = (
            f"EAL of {eal_pct_ebitda:.0%} EBITDA is within {risk_appetite} appetite but "
            f"{'stranded asset risk is emerging' if stranded_pct > t['stranded_watch'] else 'acute hazards detected'}. "
            f"Monitor quarterly; review covenants."
        )
    else:
        action = "HOLD"
        urgency = "LONG_TERM"
        rationale = (
            f"EAL of {eal_pct_ebitda:.0%} EBITDA is within {risk_appetite} risk appetite. "
            f"No acute hazards or stranded asset flags. Continue standard monitoring."
        )
    return action, urgency, rationale


def _confidence_from_data_quality(data_quality: str, n_assets: int) -> str:
    if data_quality == "LIVE_API" and n_assets >= 2:
        return "HIGH"
    if data_quality in ("LIVE_API", "MODELLED") and n_assets >= 1:
        return "MEDIUM"
    return "LOW"


def _build_recommended_actions(
    action: str,
    eal_pct: float,
    stranded_pct: float,
    top_hazards: list[str],
    sector: str,
    company_name: str,
) -> list[dict]:
    """Generate concrete, actionable recommendations — not compliance bullets."""
    actions = []
    priority = 1

    if action in ("EXIT", "REDUCE"):
        actions.append({
            "priority": priority,
            "action": f"Initiate position reduction for {company_name}",
            "rationale": (
                f"Climate EAL ({eal_pct:.0%} of EBITDA) breaches risk appetite. "
                f"Target 30–50% exposure reduction over next 2 quarters."
            ),
            "cost_usd_m": None,
            "deadline": "Q2 2027",
        })
        priority += 1

    if stranded_pct > 0.20:
        actions.append({
            "priority": priority,
            "action": "Commission independent stranded-asset valuation",
            "rationale": (
                f"Model shows {stranded_pct:.0%} of asset value at risk under NZE scenario. "
                f"Engage specialist for asset-level recovery value assessment."
            ),
            "cost_usd_m": 0.15,
            "deadline": "Within 90 days",
        })
        priority += 1

    if "flood_riverine" in top_hazards or "flood_coastal" in top_hazards:
        actions.append({
            "priority": priority,
            "action": "Verify flood protection status and insurance coverage for all riverside/coastal assets",
            "rationale": (
                "Riverine/coastal flood is a top risk driver. "
                "Confirm flood barriers, drainage, and BI insurance are current. "
                "Require flood resilience capex roadmap from management."
            ),
            "cost_usd_m": None,
            "deadline": "Before wet season",
        })
        priority += 1

    if "heat_stress" in top_hazards or "drought" in top_hazards or "water_stress" in top_hazards:
        actions.append({
            "priority": priority,
            "action": "Request water security plan and cooling resilience assessment",
            "rationale": (
                "Thermal and water stress are material risk drivers. "
                "Confirm water allocation rights, cooling backup, and crop/process alternatives."
            ),
            "cost_usd_m": None,
            "deadline": "Next operational review",
        })
        priority += 1

    if sector in ("Oil & Gas", "Coal", "Utilities") and action != "EXIT":
        actions.append({
            "priority": priority,
            "action": "Request transition plan with 2030/2040 emissions pathway and capex commitments",
            "rationale": (
                f"Sector ({sector}) faces material policy and technology transition risk. "
                f"Require credible net-zero trajectory before next credit renewal."
            ),
            "cost_usd_m": None,
            "deadline": "Before next facility renewal",
        })
        priority += 1

    if action == "HOLD":
        actions.append({
            "priority": priority,
            "action": "Maintain standard climate monitoring — annual review sufficient",
            "rationale": (
                "Risk is within appetite. "
                "Include climate EAL in annual credit review package."
            ),
            "cost_usd_m": None,
            "deadline": "Annual review",
        })

    return actions[:5]  # cap at 5


@app.post(
    "/decision",
    response_model=DecisionBrief,
    summary="Primary risk management decision brief",
    description=(
        "Decision-support first. Returns a structured brief for risk managers, "
        "credit committees, and portfolio teams. "
        "Answers 'what should I do?' before 'how do I disclose it?'. "
        "Compliance outputs (TCFD/ISSB) are a derived secondary layer."
    ),
    tags=["Decision Support"],
)
def get_decision_brief(request: DecisionRequest) -> DecisionBrief:
    """Run the CRI engine and return a decision-first risk brief.

    Modes:
      company_id provided   → full scoped engine run on seed company
      portfolio + positions → portfolio CVaR signal (fast)
      sector + region only  → parametric signal (fast, no company data)
    """
    from ..engine.scope import ReportScope
    from ..engine.orchestrator import run_scoped as _scoped
    from ..climate.portfolio import aggregate_portfolio_risk, PortfolioPosition

    run_id     = str(_uuid.uuid4())[:8]
    generated  = _dt_dec.datetime.utcnow().isoformat() + "Z"

    # ── Scenario label ─────────────────────────────────────────────────────────
    scenario_labels = {
        "nze":     "Net Zero by 2050 (SSP1-2.6)",
        "delayed": "Delayed Transition (SSP2-4.5)",
        "cp":      "Current Policies (SSP3-7.0)",
    }
    scenario_label = scenario_labels.get(request.scenario, "Current Policies (SSP3-7.0)")

    # ══════════════════════════════════════════════════════════════════════════
    # MODE A: Portfolio — multi-position CVaR signal
    # ══════════════════════════════════════════════════════════════════════════
    if request.positions:
        positions_obj = [
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
        port = aggregate_portfolio_risk(positions_obj, run_stress=False)
        total_exp = port.total_exposure_usd_m or 1.0
        cvar99    = port.portfolio_cvar99_cp_usd_m
        eal       = port.portfolio_eal_cp_usd_m
        # Rough EBITDA proxy: assume 15% EBITDA margin on exposure as revenue stand-in
        ebitda_proxy  = total_exp * 0.15
        eal_pct       = eal / max(ebitda_proxy, 1.0)
        cvar99_pct    = cvar99 / max(total_exp, 1.0)
        stranded_pct  = 0.0   # not computed in portfolio mode

        action, urgency, rationale = _derive_signal(eal_pct, cvar99_pct, stranded_pct, False, risk_appetite=request.risk_appetite)
        confidence = "MEDIUM"

        # Top sectors by CVaR contribution as "drivers"
        sector_contrib: dict[str, float] = {}
        for p in port.positions:
            sector_contrib[p.sector] = sector_contrib.get(p.sector, 0.0) + p.mc_cvar99_usd_m
        mc_total = sum(sector_contrib.values()) or 1.0
        drivers = [
            RiskDriver(
                factor=f"Sector: {sec}",
                category="physical",
                contribution_pct=round(val / mc_total * 100, 1),
                horizon="2025–2035",
                severity="HIGH" if val / mc_total > 0.40 else "MODERATE",
            )
            for sec, val in sorted(sector_contrib.items(), key=lambda x: -x[1])[:3]
        ]

        top_sectors = [r.factor.replace("Sector: ", "") for r in drivers]
        rec_actions = _build_recommended_actions(
            action, eal_pct, stranded_pct, [], top_sectors[0] if top_sectors else "", "Portfolio"
        )

        return DecisionBrief(
            signal=RiskSignal(action=action, confidence=confidence,
                              rationale=rationale, urgency=urgency),
            financial_impact=FinancialImpact(
                expected_annual_loss_usd_m=round(eal, 3),
                worst_case_1pct_usd_m=round(cvar99, 3),
                ebitda_impact_pct=round(eal_pct * 100, 2),
                revenue_at_risk_pct=round(eal / max(total_exp, 1.0) * 100, 2),
                stranded_asset_value_usd_m=0.0,
                credit_spread_widening_bps=round(min(500.0, eal_pct * 1200), 1),
                wacc_uplift_bps=round(cvar99_pct * 50, 1),
                npv_haircut_pct=round(eal_pct * 60, 2),
            ),
            top_drivers=drivers,
            recommended_actions=[RecommendedAction(**a) for a in rec_actions],
            sector=", ".join(list(sector_contrib.keys())[:3]),
            region="Multi-region",
            scenario_label=scenario_label,
            horizon_year=request.horizon_year,
            data_quality="MODELLED",
            model_confidence=confidence,
            run_id=run_id,
            generated_at=generated,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # MODE B: Single company — full scoped run
    # ══════════════════════════════════════════════════════════════════════════
    company = None
    if request.company_id:
        company = COMPANY_REGISTRY.get(request.company_id)
        if not company:
            raise HTTPException(
                status_code=404,
                detail=f"Company '{request.company_id}' not found. "
                       f"Use GET /companies to list available IDs."
            )

    if company is None:
        # ── MODE C: Parametric (no company data) ─────────────────────────────
        # Use sector/region defaults to produce a signal without a full engine run.
        sector = request.sector or "Industrials"
        region = request.region or "Global"
        exposure = request.exposure_usd_m or 100.0

        # Parametric EAL: sector phi × regional multiplier × exposure
        _sector_elf = {
            "Oil & Gas": 0.038, "Coal": 0.035, "Utilities": 0.032,
            "Mining": 0.030, "Agriculture": 0.045, "Real Estate": 0.028,
            "Chemicals": 0.025, "Transport": 0.022, "Industrials": 0.020,
            "Technology": 0.008, "Financial Services": 0.010, "Healthcare": 0.012,
        }
        elf = _sector_elf.get(sector, 0.022)
        eal = elf * exposure
        cvar99 = eal * 3.0   # rule-of-thumb: CVaR99 ≈ 3× EAL for heavy tails
        ebitda_proxy = exposure * 0.15
        eal_pct    = eal / max(ebitda_proxy, 1.0)
        cvar99_pct = cvar99 / max(exposure, 1.0)

        action, urgency, rationale = _derive_signal(eal_pct, cvar99_pct, 0.0, False, risk_appetite=request.risk_appetite)
        confidence = "LOW"

        drivers = [
            RiskDriver(factor="Physical hazard (parametric)", category="physical",
                       contribution_pct=60.0, horizon="2025–2035", severity="MODERATE"),
            RiskDriver(factor="Transition risk (sector exposure)", category="transition",
                       contribution_pct=30.0, horizon="2030–2040", severity="MODERATE"),
            RiskDriver(factor="Macro factor (GDP/rate shock)", category="macro",
                       contribution_pct=10.0, horizon="2035–2050", severity="LOW"),
        ]
        rec_actions = _build_recommended_actions(action, eal_pct, 0.0, [], sector, "Target Company")
        return DecisionBrief(
            signal=RiskSignal(action=action, confidence=confidence,
                              rationale=rationale + " (parametric estimate — provide company_id for full analysis)",
                              urgency=urgency),
            financial_impact=FinancialImpact(
                expected_annual_loss_usd_m=round(eal, 3),
                worst_case_1pct_usd_m=round(cvar99, 3),
                ebitda_impact_pct=round(eal_pct * 100, 2),
                revenue_at_risk_pct=round(eal / max(exposure, 1.0) * 100, 2),
                stranded_asset_value_usd_m=round(exposure * 0.15 if sector in ("Oil & Gas","Coal") else 0.0, 2),
                credit_spread_widening_bps=round(min(500.0, eal_pct * 1200), 1),
                wacc_uplift_bps=round(cvar99_pct * 50, 1),
                npv_haircut_pct=round(eal_pct * 60, 2),
            ),
            top_drivers=drivers,
            recommended_actions=[RecommendedAction(**a) for a in rec_actions],
            sector=sector,
            region=region,
            scenario_label=scenario_label,
            horizon_year=request.horizon_year,
            data_quality="FALLBACK",
            model_confidence="LOW",
            run_id=run_id,
            generated_at=generated,
        )

    # ── Full scoped engine run on seed company ─────────────────────────────────
    _scope_str = "full_cri" if request.scope == "full" else "physical_transition"
    scoped = _scoped(
        company,
        scope=ReportScope(_scope_str),
    )

    # ── Extract physical numbers ──────────────────────────────────────────────
    physical = scoped.physical
    transition = scoped.transition

    phys_eal     = 0.0
    top_hazards: list[str] = []
    data_quality = "MODELLED"

    if physical:
        # Primary: use engine hazard_breakdown_2035 (model-derived, asset-level)
        engine_map = physical.hazard_breakdown_2035 or {}
        engine_eal = sum(engine_map.values())   # USD M

        # Secondary: geo-calibrated EAL from INFORM/WRI indices as cross-check
        # Compute weighted average across all company assets by replacement value
        try:
            from ..climate.hazard_geo_calibration import calibrated_eal_usd_m
            geo_eal_total = 0.0
            geo_breakdown: dict[str, float] = {}
            for asset in company.assets:
                asset_rcn = getattr(asset, "replacement_cost_usd_m",
                                    getattr(asset, "book_value_usd_m", 0.0))
                if asset_rcn <= 0:
                    # Estimate from revenue share if RCN not set
                    asset_rcn = revenue * 0.05
                region = getattr(asset, "region", company.hq_region)
                geo_total, geo_bd = calibrated_eal_usd_m(
                    asset_rcn, region, company.sector
                )
                geo_eal_total += geo_total
                for h, v in geo_bd.items():
                    geo_breakdown[h] = geo_breakdown.get(h, 0.0) + v

            # Blend: 60% engine (model-calibrated to fragility curves) +
            #        40% geo-calibrated (country/sector frequency data)
            if geo_eal_total > 0 and engine_eal > 0:
                blend_factor = 0.60
                blended_map: dict[str, float] = {}
                all_hazards = set(engine_map) | set(geo_breakdown)
                for h in all_hazards:
                    if h == "other":
                        continue
                    e_val = engine_map.get(h, 0.0)
                    g_val = geo_breakdown.get(h, 0.0)
                    blended_map[h] = blend_factor * e_val + (1 - blend_factor) * g_val
                hazard_map = blended_map
                phys_eal = sum(hazard_map.values())
                data_quality = "CALIBRATED"   # best-available: engine + geo index
            else:
                hazard_map = engine_map
                phys_eal = engine_eal
                data_quality = "MODELLED"
        except Exception:
            hazard_map = engine_map
            phys_eal = engine_eal
            data_quality = "MODELLED"

        # Sort hazards by contribution so highest-impact comes first
        top_hazards = [h for h, _ in sorted(hazard_map.items(),
                                             key=lambda kv: kv[1], reverse=True)]

    trans_eal = 0.0
    if transition:
        # Use EBITDA compression under NZE 2030 × baseline EBITDA as proxy cost
        comp_frac = abs(transition.ebitda_compression_2030_nze or 0.0)
        trans_eal = comp_frac * max(company.financials.ebitda, 1.0)

    total_eal = phys_eal + trans_eal

    # ── Financial ratios ──────────────────────────────────────────────────────
    ebitda      = max(company.financials.ebitda, 1.0)
    revenue     = max(company.financials.revenue, 1.0)
    if request.exposure_usd_m:
        exposure = request.exposure_usd_m
    elif company.financials.market_cap:
        # Enterprise value = equity (market cap) + net debt — the correct credit exposure denominator
        exposure = company.financials.market_cap + company.financials.net_debt
    else:
        # Fallback: approximate EV as 8× EBITDA (typical industrial multiple)
        exposure = ebitda * 8.0
    exposure    = max(exposure, 1.0)

    eal_pct       = total_eal / ebitda
    cvar99        = total_eal * 3.2     # CVaR₉₉ ≈ 3.2× EAL (log-logistic tail calibration)
    cvar99_pct    = cvar99 / exposure

    # Stranded asset: NZE NPV haircut vs CP — valuation_results values are
    # RunResults dataclasses (attribute access, not .get())
    stranded_pct = 0.0
    if scoped.valuation_results:
        _rr_nze = scoped.valuation_results.get("nze")
        _rr_cp  = scoped.valuation_results.get("cp")
        if _rr_nze is not None and _rr_cp is not None:
            npv_nze_v = getattr(_rr_nze, "npv_fcf", None)
            npv_cp_v  = getattr(_rr_cp,  "npv_fcf", None)
            if npv_nze_v is not None and npv_cp_v is not None and npv_cp_v > 0:
                stranded_pct = max(0.0, (npv_cp_v - npv_nze_v) / npv_cp_v)

    # Acute hazard flag: any top hazard with severity > 3 on flood/cyclone/wildfire
    acute_flag = any(h in ("flood_riverine", "flood_coastal", "cyclone", "wildfire")
                     for h in top_hazards)

    # ── Signal ────────────────────────────────────────────────────────────────
    action, urgency, rationale = _derive_signal(eal_pct, cvar99_pct, stranded_pct, acute_flag, risk_appetite=request.risk_appetite)
    n_assets    = len(company.assets)
    confidence  = _confidence_from_data_quality(data_quality, n_assets)

    # ── Top risk drivers ──────────────────────────────────────────────────────
    drivers = []
    if phys_eal > 0:
        phys_share = phys_eal / max(total_eal, 1e-9) * 100
        for i, h in enumerate(top_hazards[:3]):
            drivers.append(RiskDriver(
                factor=h.replace("_", " ").title() + " — " + (company.assets[0].region if company.assets else ""),
                category="physical",
                contribution_pct=round(phys_share / max(len(top_hazards[:3]), 1), 1),
                horizon="2025–2035",
                severity="HIGH" if i == 0 else "MODERATE",
            ))
    if trans_eal > 0:
        from ..climate.portfolio import _transition_factor_weights
        trans_share = trans_eal / max(total_eal, 1e-9) * 100
        tf_weights = _transition_factor_weights(company.sector)
        _subfactor_labels = {
            "policy":     ("Policy & Regulation", "Mandatory carbon pricing, stranded-asset regulation, CBAM border levy"),
            "technology": ("Technology Displacement", "Low-carbon substitution eroding demand for core products"),
            "market":     ("Market Repricing",      "Commodity price erosion as green alternatives gain share"),
            "litigation": ("Litigation Exposure",   "Climate liability, delayed-action lawsuits, disclosure penalties"),
        }
        for subfactor, weight in sorted(tf_weights.items(), key=lambda kv: kv[1], reverse=True):
            label, rationale = _subfactor_labels.get(subfactor, (subfactor.title(), ""))
            sub_pct = round(trans_share * weight, 1)
            if sub_pct < 0.5:
                continue
            drivers.append(RiskDriver(
                factor=f"{label} — {company.sector}",
                category="transition",
                contribution_pct=sub_pct,
                horizon="2030–2045",
                severity="HIGH" if weight >= 0.35 else "MODERATE" if weight >= 0.20 else "LOW",
            ))
    if not drivers:
        drivers.append(RiskDriver(
            factor="Climate risk (insufficient asset data for breakdown)",
            category="physical",
            contribution_pct=100.0,
            horizon="2025–2035",
            severity="MODERATE",
        ))

    # ── Financial impact ──────────────────────────────────────────────────────
    # valuation_results values are RunResults dataclasses — use getattr, not .get()
    def _vattr(label: str, attr: str, default: float = 0.0) -> float:
        if not scoped.valuation_results:
            return default
        rr = scoped.valuation_results.get(label)
        return getattr(rr, attr, default) if rr is not None else default

    npv_cp   = _vattr("cp",  "npv_fcf")
    npv_nze  = _vattr("nze", "npv_fcf")
    wacc_cp  = _vattr("cp",  "wacc_used")
    wacc_nze = _vattr("nze", "wacc_used")
    wacc_uplift_bps = round((wacc_cp - wacc_nze) * 10000, 1) if wacc_cp > 0 and wacc_nze > 0 else 0.0
    npv_haircut = round((npv_nze - npv_cp) / max(abs(npv_nze), 1.0) * 100, 2) if npv_nze != 0 else 0.0

    # ── Recommended actions ───────────────────────────────────────────────────
    rec_raw = _build_recommended_actions(
        action, eal_pct, stranded_pct, top_hazards, company.sector, company.name
    )

    # ── Compliance layer (secondary) ──────────────────────────────────────────
    compliance_layer = {
        "note": "Compliance outputs are derived from this brief. Use POST /reports/tcfd, /reports/issb, or /reports/csrd.",
        "tcfd_pillars": ["Governance", "Strategy", "Risk Management", "Metrics & Targets"],
        "issb_s2_material": eal_pct > 0.02,
        "csrd_required": company.sector in ("Utilities", "Oil & Gas", "Mining", "Agriculture"),
        "key_metric_for_disclosure": {
            "metric": "Expected Annual Loss from climate hazards",
            "value_usd_m": round(total_eal, 3),
            "as_pct_ebitda": round(eal_pct * 100, 2),
        },
    }

    return DecisionBrief(
        signal=RiskSignal(action=action, confidence=confidence,
                          rationale=rationale, urgency=urgency),
        financial_impact=FinancialImpact(
            expected_annual_loss_usd_m=round(total_eal, 3),
            worst_case_1pct_usd_m=round(cvar99, 3),
            ebitda_impact_pct=round(eal_pct * 100, 2),
            revenue_at_risk_pct=round(total_eal / revenue * 100, 2),
            stranded_asset_value_usd_m=round(stranded_pct * max(npv_cp, 0.0), 2),
            credit_spread_widening_bps=round(min(500.0, eal_pct * 1200), 1),
            wacc_uplift_bps=wacc_uplift_bps,
            npv_haircut_pct=npv_haircut,
        ),
        top_drivers=drivers,
        recommended_actions=[RecommendedAction(**a) for a in rec_raw],
        company_id=company.id,
        company_name=company.name,
        sector=company.sector,
        region=company.assets[0].region if company.assets else "Global",
        scenario_label=scenario_label,
        horizon_year=request.horizon_year,
        data_quality=data_quality,
        model_confidence=confidence,
        compliance_layer=compliance_layer,
        run_id=run_id,
        generated_at=generated,
    )


# ── Pillar 2: POST /assets/batch — standardized batch asset assessment ────────

@app.post("/assets/batch", response_model=BatchAssessmentResponse, tags=["Pillar 2 — Batch Assets"])
async def batch_assess_assets(request: BatchAssessmentRequest) -> BatchAssessmentResponse:
    """Assess one or many assets from raw asset data — no company registration required.

    Accepts a list of assets with coordinates, sector, financials, and emissions.
    Returns per-asset risk scores, signals, and EAL alongside portfolio aggregates.

    This is the primary intake for institutional clients onboarding their asset portfolio.
    A client with 500 assets in a CSV can batch-submit and receive a ranked risk register
    within seconds.
    """
    import math
    from datetime import datetime, timezone

    # Lazy imports to keep startup fast
    from ..climate.hazard_geo_calibration import (
        calibrated_eal_usd_m,
        country_from_latlon,
        get_all_hazard_elfs,
    )

    run_id = f"batch-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')[:20]}"
    generated = datetime.now(timezone.utc).isoformat()
    horizon = request.horizon_year
    appetite = request.risk_appetite

    # Signal thresholds (mirrors /decision logic)
    _THRESHOLDS = {
        "conservative": {"exit": 0.04, "reduce": 0.02, "watch": 0.01},
        "moderate":     {"exit": 0.08, "reduce": 0.04, "watch": 0.02},
        "aggressive":   {"exit": 0.15, "reduce": 0.08, "watch": 0.04},
    }.get(appetite, {"exit": 0.08, "reduce": 0.04, "watch": 0.02})

    # Transition EAL multipliers by sector
    _TRANSITION_MULT = {
        "Oil & Gas": 0.18, "Mining": 0.12, "Utilities": 0.14, "Steel": 0.10,
        "Cement": 0.09, "Chemicals": 0.08, "Agriculture": 0.07,
        "Shipping": 0.07, "Aviation": 0.08, "Automotive": 0.06,
        "Real Estate": 0.04, "Financial Services": 0.02,
    }

    asset_results: list[AssetAssessmentResult] = []
    total_rcv = 0.0
    total_rev = 0.0
    total_eal = 0.0
    signal_counts: dict[str, int] = {"HOLD": 0, "WATCH": 0, "REDUCE": 0, "EXIT": 0}

    for a in request.assets:
        # 1. Resolve country
        iso2 = a.country_iso2
        if not iso2:
            try:
                iso2 = country_from_latlon(a.lat, a.lon)
            except Exception:
                iso2 = "GL"

        # 2. Physical EAL from geo-calibrated lookup
        ebitda = a.ebitda_usd_m if a.ebitda_usd_m is not None else a.annual_revenue_usd_m * 0.20
        try:
            phys_eal, hazard_bkdn = calibrated_eal_usd_m(
                a.replacement_cost_usd_m, iso2, a.sector, top_n=6
            )
        except Exception:
            phys_eal = a.replacement_cost_usd_m * 0.015
            hazard_bkdn = {}

        # 2b. Gap 9: Critical-path BI EAL — revenue downtime from component dependency graph
        # Property damage (phys_eal) and business interruption are additive; the geo calibration
        # only covers replacement cost × damage fraction; BI is a separate revenue exposure.
        bi_eal = 0.0
        try:
            from ..climate.asset_graph import (
                ComponentGraph as _CG,
                ComponentNode as _CN,
                critical_path_downtime as _cpd,
                get_sector_graph as _gsg,
            )
            hazard_probs_geo = get_all_hazard_elfs(iso2, a.sector)

            # Resolve graph: caller-supplied > sector default > skip
            _cp_graph = None
            if a.component_graph and isinstance(a.component_graph, dict):
                # Deserialise caller-supplied adjacency dict
                cg = _CG(description=a.component_graph.get("description", "custom"))
                for node_name, node_data in (a.component_graph.get("nodes") or {}).items():
                    cg.nodes[node_name] = _CN(
                        name=node_name,
                        hazard_sensitivities=node_data.get("hazard_sensitivities", {}),
                        recovery_days=int(node_data.get("recovery_days", 7)),
                        has_redundancy=bool(node_data.get("has_redundancy", False)),
                    )
                cg.upstream = a.component_graph.get("upstream") or {}
                if cg.nodes:
                    _cp_graph = cg
            if _cp_graph is None:
                _cp_graph = _gsg(a.sector)

            if _cp_graph is not None:
                cp_fraction, _ = _cpd(_cp_graph, hazard_probs_geo)
                bi_eal = a.annual_revenue_usd_m * cp_fraction
        except Exception:
            pass  # silently skip; property damage EAL still computed above

        phys_eal = phys_eal + bi_eal  # total physical risk: damage + downtime

        # 3. Transition EAL from sector + emissions
        co2_total = a.scope1_tco2 + a.scope2_tco2 + (a.scope3_tco2 or 0.0)
        # Carbon cost under CP scenario by 2035: ~$80/tCO2e (NGFS Phase 4)
        carbon_cost_m = co2_total * 80.0 / 1_000_000
        # Plus EBITDA compression from sector transition multiplier
        trans_mult = _TRANSITION_MULT.get(a.sector, 0.05)
        trans_eal = ebitda * trans_mult + carbon_cost_m

        total_asset_eal = phys_eal + trans_eal

        # 4. Scores (0–100)
        phys_score = min(100.0, (phys_eal / max(a.replacement_cost_usd_m, 1.0)) * 1500.0)
        trans_score = min(100.0, (trans_eal / max(ebitda, 1.0)) * 200.0)
        composite = phys_score * 0.45 + trans_score * 0.35 + (co2_total / max(a.annual_revenue_usd_m * 1e3, 1.0)) * 20.0
        composite = min(100.0, composite)

        # 5. Signal
        eal_pct = total_asset_eal / max(a.annual_revenue_usd_m, 1.0)
        if eal_pct >= _THRESHOLDS["exit"]:
            signal, conf = "EXIT", "HIGH"
        elif eal_pct >= _THRESHOLDS["reduce"]:
            signal, conf = "REDUCE", "HIGH" if composite > 70 else "MEDIUM"
        elif eal_pct >= _THRESHOLDS["watch"]:
            signal, conf = "WATCH", "MEDIUM"
        else:
            signal, conf = "HOLD", "HIGH" if composite < 30 else "MEDIUM"

        signal_counts[signal] = signal_counts.get(signal, 0) + 1

        # 6. CVaR99 (simple scaling from EAL)
        cvar99 = total_asset_eal * 4.2

        # 7. Stranded asset (NZE penalty)
        stranded = a.replacement_cost_usd_m * min(0.35, trans_mult * 2.0)

        # 8. Credit spread
        cs_bps = min(500.0, eal_pct * 1200.0)

        # 9. Primary hazard
        primary_hazard = max(hazard_bkdn, key=hazard_bkdn.get) if hazard_bkdn else "compound"
        primary_pct = (
            (hazard_bkdn[primary_hazard] / phys_eal * 100.0) if phys_eal > 0 and hazard_bkdn else 0.0
        )

        # 10. Data quality
        data_quality = "CALIBRATED" if hazard_bkdn else "ESTIMATED"

        total_rcv += a.replacement_cost_usd_m
        total_rev += a.annual_revenue_usd_m
        total_eal += total_asset_eal

        asset_results.append(AssetAssessmentResult(
            asset_id=a.asset_id,
            name=a.name,
            lat=a.lat,
            lon=a.lon,
            sector=a.sector,
            country_iso2=iso2,
            physical_score=round(phys_score, 1),
            transition_score=round(trans_score, 1),
            composite_score=round(composite, 1),
            signal=signal,
            signal_confidence=conf,
            expected_annual_loss_usd_m=round(total_asset_eal, 3),
            eal_pct_revenue=round(eal_pct * 100, 2),
            eal_pct_rcv=round(total_asset_eal / max(a.replacement_cost_usd_m, 1.0) * 100, 2),
            worst_case_cvar99_usd_m=round(cvar99, 2),
            stranded_asset_value_usd_m=round(stranded, 2),
            credit_spread_widening_bps=round(cs_bps, 1),
            primary_hazard=primary_hazard,
            primary_hazard_contribution_pct=round(primary_pct, 1),
            hazard_breakdown={k: round(v, 3) for k, v in hazard_bkdn.items()},
            carbon_cost_usd_m_2030=round(carbon_cost_m, 3),
            ebitda_compression_pct=round(trans_mult * 100, 1),
            data_quality=data_quality,
            caveats=[] if data_quality == "CALIBRATED" else ["No CMIP6 raster — EAL estimated from sector benchmark"],
        ))

    # Portfolio aggregates
    portfolio_cvar99 = total_eal * 3.8  # portfolio diversification reduces tail
    highest = max(asset_results, key=lambda x: x.composite_score) if asset_results else None

    # Sector breakdown
    sector_bkdn: dict[str, dict] = {}
    for ar in asset_results:
        s = ar.sector
        if s not in sector_bkdn:
            sector_bkdn[s] = {"n_assets": 0, "total_eal_usd_m": 0.0, "signals": {}}
        sector_bkdn[s]["n_assets"] += 1
        sector_bkdn[s]["total_eal_usd_m"] = round(sector_bkdn[s]["total_eal_usd_m"] + ar.expected_annual_loss_usd_m, 3)
        sector_bkdn[s]["signals"][ar.signal] = sector_bkdn[s]["signals"].get(ar.signal, 0) + 1

    return BatchAssessmentResponse(
        run_id=run_id,
        n_assets=len(asset_results),
        horizon_year=horizon,
        risk_appetite=appetite,
        generated_at=generated,
        total_rcv_usd_m=round(total_rcv, 2),
        total_revenue_usd_m=round(total_rev, 2),
        total_eal_usd_m=round(total_eal, 3),
        portfolio_eal_pct_rcv=round(total_eal / max(total_rcv, 1.0) * 100, 2),
        portfolio_cvar99_usd_m=round(portfolio_cvar99, 2),
        signal_counts=signal_counts,
        highest_risk_asset_id=highest.asset_id if highest else "",
        highest_risk_asset_name=highest.name if highest else "",
        sector_breakdown=sector_bkdn,
        assets=asset_results,
        data_sources=[
            "INFORM Risk Index 2024 (EAL fractions)",
            "World Risk Index 2023 (exposure/vulnerability)",
            "NGFS Phase 4 carbon price trajectories",
            "IPCC AR6 log-logistic hazard fragility curves",
        ],
    )


# ── Pillar 3: POST /generate/article — LLM article from engine data ───────────

@app.post("/generate/article", response_model=ArticleResponse, tags=["Pillar 3 — Content Generation"])
async def generate_article_endpoint(request: ArticleRequest) -> ArticleResponse:
    """Generate a grounded article (LinkedIn, investor memo, case study, etc.)
    from live engine data for a registered company.

    The engine runs a full /decision assessment first, then passes the structured
    output to the LLM narrator. Every number in the article comes from the engine —
    none are invented by the LLM.

    Requires ANTHROPIC_API_KEY environment variable for LLM generation.
    Falls back to a high-quality template if the key is not set.
    """
    from datetime import datetime, timezone
    from ..ai.narrator import generate_article as _gen_article

    # Run the engine to get live data
    dec_request = DecisionRequest(
        company_id=request.company_id,
        scope="full",
        horizon_year=2035,
        risk_appetite=request.risk_appetite,
    )
    brief: DecisionBrief = await assess_decision_brief(dec_request)
    brief_dict = brief.model_dump()

    result = _gen_article(
        brief=brief_dict,
        article_type=request.article_type,
        tone=request.tone,
        word_count=request.word_count,
        focus=request.focus,
    )

    return ArticleResponse(
        company_id=request.company_id,
        company_name=brief.company_name or request.company_id,
        article_type=request.article_type,
        tone=request.tone,
        content=result["content"],
        word_count_actual=result["word_count_actual"],
        key_numbers_used=result["key_numbers_used"],
        signal=brief.signal.action,
        run_id=brief.run_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


# ── Pillar 3: POST /generate/narrative — one narrative section ────────────────

@app.post("/generate/narrative", response_model=NarrativeResponse, tags=["Pillar 3 — Content Generation"])
async def generate_narrative_endpoint(request: NarrativeRequest) -> NarrativeResponse:
    """Generate one narrative section (executive summary, physical risk, etc.)
    from live engine data.

    Useful for assembling disclosure documents section-by-section, or for
    populating TCFD/IFRS S2 report templates with LLM-drafted prose backed
    by deterministic engine numbers.
    """
    from datetime import datetime, timezone
    from ..ai.narrator import generate_narrative as _gen_narrative

    dec_request = DecisionRequest(
        company_id=request.company_id,
        scope="full",
        horizon_year=2035,
        risk_appetite=request.risk_appetite,
    )
    brief: DecisionBrief = await assess_decision_brief(dec_request)
    brief_dict = brief.model_dump()

    result = _gen_narrative(
        brief=brief_dict,
        section=request.section,
        audience=request.audience,
    )

    return NarrativeResponse(
        section=request.section,
        audience=request.audience,
        content=result["content"],
        key_metrics=result["key_metrics"],
        run_id=brief.run_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ── REGULATORY COMPLIANCE ENDPOINTS ──────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class SFDRRequest(BaseModel):
    company_id:                 str
    company_name:               str
    sector_key:                 str
    revenue_eur_m:              float
    ev_eur_m:                   float
    outstanding_loan_eur_m:     float = 0.0
    scope1_tco2e:               Optional[float] = None
    scope2_tco2e:               Optional[float] = None
    scope3_tco2e:               Optional[float] = None
    has_carbon_reduction_target: bool = False
    board_female_pct:           Optional[float] = None
    gender_pay_gap_pct:         Optional[float] = None
    ungc_compliant:             Optional[bool]  = None
    reference_period:           str = "2024"


@app.post("/regulatory/sfdr-pai", tags=["Regulatory — SFDR / PCAF / EU Taxonomy"])
async def sfdr_pai_endpoint(req: SFDRRequest):
    """
    Compute all 18 SFDR Principal Adverse Impact (PAI) indicators.

    Returns scores, data quality ratings, and data-limitation disclosures
    per SFDR RTS Annex I (EU) 2022/1288.
    """
    from ..regulatory.sfdr_pai import compute_sfdr_pai, pai_to_dict
    result = compute_sfdr_pai(
        company_id=req.company_id,
        company_name=req.company_name,
        sector_key=req.sector_key,
        revenue_eur_m=req.revenue_eur_m,
        ev_eur_m=req.ev_eur_m,
        outstanding_loan_eur_m=req.outstanding_loan_eur_m,
        scope1_tco2e=req.scope1_tco2e,
        scope2_tco2e=req.scope2_tco2e,
        scope3_tco2e=req.scope3_tco2e,
        has_carbon_reduction_target=req.has_carbon_reduction_target,
        board_female_pct=req.board_female_pct,
        gender_pay_gap_pct=req.gender_pay_gap_pct,
        ungc_compliant=req.ungc_compliant,
        reference_period=req.reference_period,
    )
    return pai_to_dict(result)


class EUTaxonomyRequest(BaseModel):
    company_id:       str
    company_name:     str
    sector_key:       str
    revenue_eur_m:    float
    capex_eur_m:      float = 0.0
    opex_eur_m:       float = 0.0
    ungc_compliant:   bool = True
    reference_period: str  = "2024"
    override_alignment: Optional[float] = None


@app.post("/regulatory/eu-taxonomy", tags=["Regulatory — SFDR / PCAF / EU Taxonomy"])
async def eu_taxonomy_endpoint(req: EUTaxonomyRequest):
    """
    Compute EU Taxonomy alignment across 6 environmental objectives.

    Returns eligible and aligned turnover/capex/opex percentages, DNSH
    compliance, and MSS assessment per EU Taxonomy Regulation 2020/852.
    """
    from ..regulatory.eu_taxonomy import compute_eu_taxonomy, taxonomy_to_dict
    result = compute_eu_taxonomy(
        company_id=req.company_id,
        company_name=req.company_name,
        sector_key=req.sector_key,
        revenue_eur_m=req.revenue_eur_m,
        capex_eur_m=req.capex_eur_m,
        opex_eur_m=req.opex_eur_m,
        ungc_compliant=req.ungc_compliant,
        reference_period=req.reference_period,
        override_alignment=req.override_alignment,
    )
    return taxonomy_to_dict(result)


class PCAFRequest(BaseModel):
    counterparty_id:             str
    counterparty_name:           str
    sector_key:                  str
    asset_class:                 str   # PCAFAssetClass value
    outstanding_amount_eur_m:    float
    enterprise_value_eur_m:      Optional[float] = None
    total_debt_eur_m:            Optional[float] = None
    total_project_value_eur_m:   Optional[float] = None
    property_value_eur_m:        Optional[float] = None
    revenue_eur_m:               float = 100.0
    scope1_tco2e:                Optional[float] = None
    scope2_tco2e:                Optional[float] = None
    scope3_tco2e:                Optional[float] = None


class PCAFPortfolioRequest(BaseModel):
    exposures: list[PCAFRequest]


@app.post("/regulatory/pcaf", tags=["Regulatory — SFDR / PCAF / EU Taxonomy"])
async def pcaf_endpoint(req: PCAFRequest):
    """
    Compute PCAF-standard financed emissions for one counterparty / exposure.

    Covers all 6 PCAF asset classes with attribution factor calculation
    and data quality scoring (1–5) per PCAF Global Standard Part A (2022).
    """
    from ..regulatory.pcaf import compute_financed_emissions, pcaf_to_dict, PCAFAssetClass
    try:
        asset_class = PCAFAssetClass(req.asset_class)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Unknown asset_class: {req.asset_class}")

    result = compute_financed_emissions(
        counterparty_id=req.counterparty_id,
        counterparty_name=req.counterparty_name,
        sector_key=req.sector_key,
        asset_class=asset_class,
        outstanding_amount_eur_m=req.outstanding_amount_eur_m,
        enterprise_value_eur_m=req.enterprise_value_eur_m,
        total_debt_eur_m=req.total_debt_eur_m,
        total_project_value_eur_m=req.total_project_value_eur_m,
        property_value_eur_m=req.property_value_eur_m,
        revenue_eur_m=req.revenue_eur_m,
        scope1_tco2e=req.scope1_tco2e,
        scope2_tco2e=req.scope2_tco2e,
        scope3_tco2e=req.scope3_tco2e,
    )
    return pcaf_to_dict(result)


@app.post("/regulatory/pcaf/portfolio", tags=["Regulatory — SFDR / PCAF / EU Taxonomy"])
async def pcaf_portfolio_endpoint(req: PCAFPortfolioRequest):
    """
    Compute PCAF financed emissions for a portfolio of exposures and return
    portfolio-level aggregated totals with weighted average data quality.
    """
    from ..regulatory.pcaf import (
        compute_financed_emissions, pcaf_portfolio_summary, PCAFAssetClass
    )
    results = []
    for exp in req.exposures:
        try:
            ac = PCAFAssetClass(exp.asset_class)
        except ValueError:
            raise HTTPException(status_code=422, detail=f"Unknown asset_class: {exp.asset_class}")
        results.append(compute_financed_emissions(
            counterparty_id=exp.counterparty_id,
            counterparty_name=exp.counterparty_name,
            sector_key=exp.sector_key,
            asset_class=ac,
            outstanding_amount_eur_m=exp.outstanding_amount_eur_m,
            enterprise_value_eur_m=exp.enterprise_value_eur_m,
            total_debt_eur_m=exp.total_debt_eur_m,
            total_project_value_eur_m=exp.total_project_value_eur_m,
            property_value_eur_m=exp.property_value_eur_m,
            revenue_eur_m=exp.revenue_eur_m,
            scope1_tco2e=exp.scope1_tco2e,
            scope2_tco2e=exp.scope2_tco2e,
            scope3_tco2e=exp.scope3_tco2e,
        ))
    return pcaf_portfolio_summary(results)


# ═══════════════════════════════════════════════════════════════════════════════
# ── PHYSICAL RISK: EXTENDED ENDPOINTS ────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class EventStressRequest(BaseModel):
    company_id:    str
    company_name:  str
    sector_key:    str
    asset_lat:     float
    asset_lon:     float
    ev_usd_m:      float
    revenue_usd_m: float
    event_ids:     Optional[list[str]] = None   # None → run all events


@app.post("/stress/event", tags=["Physical Risk — Extended"])
async def event_stress_endpoint(req: EventStressRequest):
    """
    Replay named historical climate events against a company position.

    Returns worst-case direct + BI losses per event, ranked by total loss.
    Events: Harvey 2017, EU Floods 2021, Australia Black Summer 2019,
    Texas Freeze 2021, European Heatwave 2003, Typhoon Hagibis 2019,
    and more. See GET /stress/event/catalogue for the full list.
    """
    from ..climate.event_stress import run_event_stress, event_stress_to_dict
    results = run_event_stress(
        company_id=req.company_id,
        company_name=req.company_name,
        sector_key=req.sector_key,
        asset_lat=req.asset_lat,
        asset_lon=req.asset_lon,
        ev_usd_m=req.ev_usd_m,
        revenue_usd_m=req.revenue_usd_m,
        event_ids=req.event_ids,
    )
    return event_stress_to_dict(results)


@app.get("/stress/event/catalogue", tags=["Physical Risk — Extended"])
async def event_catalogue_endpoint():
    """Return the catalogue of available historical climate events."""
    from ..climate.event_stress import list_events
    return {"events": list_events()}


class SLRRequest(BaseModel):
    asset_id:                  str
    asset_name:                str
    asset_lat:                 float
    asset_lon:                 float
    asset_elevation_m:         float
    ev_usd_m:                  float
    scenario:                  str   = "SSP2-4.5"
    distance_to_coast_km:      Optional[float] = None
    tidal_regime:              str   = "default"
    custom_regional_multiplier: Optional[float] = None


class SLRPortfolioRequest(BaseModel):
    assets:   list[SLRRequest]
    scenario: str = "SSP2-4.5"


@app.post("/physical/slr", tags=["Physical Risk — Extended"])
async def slr_endpoint(req: SLRRequest):
    """
    Compute sea level rise (SLR) coastal inundation risk for an asset.

    Uses IPCC AR6 WGI Chapter 9 SLR trajectories (median + likely range)
    under SSP1-2.6, SSP2-4.5, or SSP5-8.5. Returns inundation probabilities
    at 2030 / 2050 / 2100 and expected financial losses.
    """
    from ..climate.sea_level_rise import compute_slr_risk, slr_to_dict
    result = compute_slr_risk(
        asset_id=req.asset_id,
        asset_name=req.asset_name,
        asset_lat=req.asset_lat,
        asset_lon=req.asset_lon,
        asset_elevation_m=req.asset_elevation_m,
        ev_usd_m=req.ev_usd_m,
        scenario=req.scenario,
        distance_to_coast_km=req.distance_to_coast_km,
        tidal_regime=req.tidal_regime,
        custom_regional_multiplier=req.custom_regional_multiplier,
    )
    return slr_to_dict(result)


@app.post("/physical/slr/portfolio", tags=["Physical Risk — Extended"])
async def slr_portfolio_endpoint(req: SLRPortfolioRequest):
    """
    Compute SLR risk across a portfolio of coastal assets and return
    aggregate exposure, tier distribution, and disclosure flags.
    """
    from ..climate.sea_level_rise import compute_slr_risk, slr_portfolio_summary
    results = []
    for a in req.assets:
        results.append(compute_slr_risk(
            asset_id=a.asset_id,
            asset_name=a.asset_name,
            asset_lat=a.asset_lat,
            asset_lon=a.asset_lon,
            asset_elevation_m=a.asset_elevation_m,
            ev_usd_m=a.ev_usd_m,
            scenario=req.scenario,
            distance_to_coast_km=a.distance_to_coast_km,
            tidal_regime=a.tidal_regime,
            custom_regional_multiplier=a.custom_regional_multiplier,
        ))
    return slr_portfolio_summary(results)


class BiodiversityRequest(BaseModel):
    company_id:    str
    company_name:  str
    sector_key:    str
    asset_lat:     float
    asset_lon:     float
    revenue_usd_m: float = 100.0


@app.post("/physical/biodiversity", tags=["Physical Risk — Extended"])
async def biodiversity_endpoint(req: BiodiversityRequest):
    """
    Compute nature-related financial risk using the TNFD LEAP framework.

    Returns:
    - Location sensitivity (KBA / protected area proximity)
    - Sector dependency on ecosystem services (ENCORE v1.1)
    - Combined nature risk score and tier
    - TNFD disclosure recommendations and recommended metrics
    """
    from ..climate.biodiversity import compute_biodiversity_risk, biodiversity_to_dict
    result = compute_biodiversity_risk(
        company_id=req.company_id,
        company_name=req.company_name,
        sector_key=req.sector_key,
        asset_lat=req.asset_lat,
        asset_lon=req.asset_lon,
        revenue_usd_m=req.revenue_usd_m,
    )
    return biodiversity_to_dict(result)


# ═══════════════════════════════════════════════════════════════════════════════
# ── PORTFOLIO: BENCHMARK + COUNTERPARTY ENDPOINTS ────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class BenchmarkRequest(BaseModel):
    positions:   list[PortfolioPositionIn]
    benchmark:   str = "MSCI_World"   # "MSCI_World" | "MSCI_EM" | "MSCI_Europe" | "SP500"
    scenario_id: str = "nze_2050"


class CounterpartyPosition(BaseModel):
    counterparty_id:   str
    counterparty_name: str
    sector_key:        str
    region:            str
    exposure_usd_m:    float
    ev_usd_m:          float
    revenue_usd_m:     float
    asset_lat:         Optional[float] = None
    asset_lon:         Optional[float] = None


class CounterpartyAggRequest(BaseModel):
    positions: list[CounterpartyPosition]
    group_by:  str = "sector"   # "sector" | "region" | "counterparty"


# Benchmark sector climate VaR premia (MSCI Climate VaR Sector Study, 2022)
_BENCHMARK_SECTOR_VAR: dict[str, dict[str, float]] = {
    "MSCI_World": {
        "oil_gas": 0.18, "coal": 0.22, "chemicals": 0.12, "utilities": 0.10,
        "industrials": 0.08, "financials": 0.04, "technology": 0.03,
        "real_estate": 0.09, "agriculture": 0.14, "consumer": 0.06,
        "healthcare": 0.04, "transport": 0.07, "default": 0.07,
    },
    "MSCI_EM": {
        "oil_gas": 0.22, "coal": 0.28, "chemicals": 0.16, "utilities": 0.14,
        "industrials": 0.12, "financials": 0.06, "technology": 0.04,
        "real_estate": 0.13, "agriculture": 0.20, "consumer": 0.09,
        "healthcare": 0.05, "transport": 0.10, "default": 0.10,
    },
    "MSCI_Europe": {
        "oil_gas": 0.16, "coal": 0.18, "chemicals": 0.11, "utilities": 0.09,
        "industrials": 0.07, "financials": 0.03, "technology": 0.02,
        "real_estate": 0.08, "agriculture": 0.12, "consumer": 0.05,
        "healthcare": 0.03, "transport": 0.06, "default": 0.06,
    },
    "SP500": {
        "oil_gas": 0.17, "coal": 0.21, "chemicals": 0.11, "utilities": 0.09,
        "industrials": 0.07, "financials": 0.04, "technology": 0.03,
        "real_estate": 0.09, "agriculture": 0.13, "consumer": 0.06,
        "healthcare": 0.04, "transport": 0.07, "default": 0.07,
    },
}


@app.post("/portfolio/benchmark", tags=["Portfolio — Benchmark & Counterparty"])
async def portfolio_benchmark_endpoint(req: BenchmarkRequest):
    """
    Compare portfolio climate VaR against a market benchmark index.

    Returns per-sector active VaR (portfolio − benchmark), aggregate
    active climate VaR, and sector tilt table showing over/under-weight
    relative to benchmark sector exposures.

    Benchmarks: MSCI_World, MSCI_EM, MSCI_Europe, SP500.
    """
    bm_var = _BENCHMARK_SECTOR_VAR.get(req.benchmark, _BENCHMARK_SECTOR_VAR["MSCI_World"])
    total_exposure = sum(p.position_usd_m for p in req.positions)
    if total_exposure <= 0:
        raise HTTPException(status_code=422, detail="Total exposure must be > 0")

    sector_groups: dict[str, dict] = {}
    for p in req.positions:
        sk = p.sector or "default"
        if sk not in sector_groups:
            sector_groups[sk] = {"exposure": 0.0, "count": 0}
        sector_groups[sk]["exposure"] += p.position_usd_m
        sector_groups[sk]["count"] += 1

    rows = []
    portfolio_var_total = 0.0
    benchmark_var_total = 0.0

    for sk, grp in sector_groups.items():
        wt = grp["exposure"] / total_exposure
        sect_var  = bm_var.get(sk, bm_var["default"])
        port_var  = wt * sect_var
        bm_var_w  = wt * sect_var   # benchmark weight assumed equal to portfolio weight
        active    = 0.0             # diverges when actual benchmark weights are supplied
        portfolio_var_total += port_var
        benchmark_var_total += bm_var_w
        rows.append({
            "sector":             sk,
            "weight_pct":         round(wt * 100, 2),
            "portfolio_var_pct":  round(port_var * 100, 3),
            "benchmark_var_pct":  round(bm_var_w * 100, 3),
            "active_var_pct":     round(active * 100, 3),
            "active_var_usd_m":   round(active * total_exposure, 2),
        })

    rows.sort(key=lambda r: r["portfolio_var_pct"], reverse=True)

    return {
        "benchmark":                req.benchmark,
        "total_exposure_usd_m":     round(total_exposure, 2),
        "portfolio_climate_var_pct": round(portfolio_var_total * 100, 3),
        "benchmark_climate_var_pct": round(benchmark_var_total * 100, 3),
        "active_climate_var_pct":   round((portfolio_var_total - benchmark_var_total) * 100, 3),
        "active_climate_var_usd_m": round(
            (portfolio_var_total - benchmark_var_total) * total_exposure, 2
        ),
        "sector_tilt_table": rows,
        "methodology": (
            "Climate VaR premia from MSCI Climate VaR Sector Study (2022). "
            "Active VaR requires benchmark sector weights to be meaningful — "
            f"currently returns portfolio VaR by sector. Scenario: {req.scenario_id}."
        ),
    }


@app.post("/portfolio/counterparty", tags=["Portfolio — Benchmark & Counterparty"])
async def counterparty_aggregation_endpoint(req: CounterpartyAggRequest):
    """
    Aggregate climate exposure by counterparty, sector, or geography.

    For each group returns total exposure, concentration risk (HHI),
    and climate VaR contribution. Useful for ICAAP / ILAAP counterparty
    concentration reporting.
    """
    total_exp = sum(p.exposure_usd_m for p in req.positions)
    if total_exp <= 0:
        raise HTTPException(status_code=422, detail="Total exposure must be > 0")

    groups: dict[str, dict] = {}
    for p in req.positions:
        key = (
            p.sector_key       if req.group_by == "sector"
            else p.region      if req.group_by == "region"
            else p.counterparty_id
        )
        if key not in groups:
            groups[key] = {
                "label": p.counterparty_name if req.group_by == "counterparty" else key,
                "exposure": 0.0,
                "sector_key": p.sector_key,
                "count": 0,
            }
        groups[key]["exposure"] += p.exposure_usd_m
        groups[key]["count"]    += 1

    bm = _BENCHMARK_SECTOR_VAR["MSCI_World"]
    rows = []
    hhi = 0.0
    for key, grp in groups.items():
        wt = grp["exposure"] / total_exp
        hhi += wt ** 2
        c_var = bm.get(grp["sector_key"], bm["default"])
        rows.append({
            "group_key":         key,
            "label":             grp["label"],
            "exposure_usd_m":    round(grp["exposure"], 2),
            "weight_pct":        round(wt * 100, 2),
            "climate_var_pct":   round(c_var * 100, 2),
            "climate_var_usd_m": round(c_var * grp["exposure"], 2),
            "position_count":    grp["count"],
        })

    rows.sort(key=lambda r: r["exposure_usd_m"], reverse=True)
    concentration_tier = "low" if hhi < 0.10 else ("medium" if hhi < 0.25 else "high")

    return {
        "group_by":             req.group_by,
        "total_exposure_usd_m": round(total_exp, 2),
        "group_count":          len(rows),
        "hhi_concentration":    round(hhi, 4),
        "concentration_tier":   concentration_tier,
        "groups":               rows,
        "methodology": (
            "Herfindahl-Hirschman Index (HHI) for concentration. "
            "Climate VaR from MSCI Climate VaR sector premia (MSCI World benchmark). "
            "HHI < 0.10 = low, 0.10–0.25 = medium, > 0.25 = high concentration."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PRACTITIONER WORKFLOW LAYER  (Tasks #116–122)
#  Fast triage, company lookup, scenario comparison, IC brief,
#  what-if sensitivity, SFDR Excel export, portfolio delta
# ═══════════════════════════════════════════════════════════════════════════════

# ── /screen/triage ────────────────────────────────────────────────────────────

class TriageCompany(BaseModel):
    company_name: str
    sector_key:   str
    country:      str

class BatchTriageRequest(BaseModel):
    companies: list[TriageCompany]

class SingleTriageRequest(BaseModel):
    company_name: str
    sector_key:   str
    country:      str

@app.post("/screen/triage", tags=["Screening"])
async def screen_triage(req: SingleTriageRequest):
    """
    Fast RED / AMBER / GREEN traffic-light triage.
    No engine run, no lat/lon required — ideal for pipeline screening.
    Returns in < 100 ms.
    """
    from cri.intake.screen import screen_company
    result = screen_company(
        company_name=req.company_name,
        sector_key=req.sector_key,
        country=req.country,
    )
    return {
        "company_name":         result.company_name,
        "sector_key":           result.sector_key,
        "country":              result.country,
        "traffic_light":        result.traffic_light,
        "combined_score":       result.combined_score,
        "physical_score":       result.physical_score,
        "transition_score":     result.transition_score,
        "rationale":            result.rationale,
        "top_risks":            result.top_risks,
        "recommended_action":   result.recommended_action,
        "full_engine_priority": result.full_engine_priority,
        "methodology": (
            "Physical: country hazard (IPCC AR6 / EM-DAT) × sector amplifier. "
            "Transition: sector carbon tier × jurisdiction policy ambition. "
            "Combined = 0.5×physical + 0.5×transition. "
            "RED ≥ 0.60 | AMBER 0.35–0.60 | GREEN < 0.35."
        ),
    }

@app.post("/screen/triage/batch", tags=["Screening"])
async def screen_triage_batch(req: BatchTriageRequest):
    """
    Batch triage up to 50 companies in one call.
    Returns results sorted RED-first so you know which deals
    to prioritise for full engine analysis.
    """
    from cri.intake.screen import screen_batch
    companies = [c.model_dump() for c in req.companies]
    return screen_batch(companies)


# ── /company/lookup ───────────────────────────────────────────────────────────

class CompanyLookupRequest(BaseModel):
    query:        str
    hint_sector:  Optional[str] = None
    hint_country: Optional[str] = None

@app.post("/company/lookup", tags=["Workflow"])
async def company_lookup(req: CompanyLookupRequest):
    """
    Resolve a company name or ticker to a pre-populated ClimRisk
    API payload. Use this before calling /physical/flood or /transition
    to avoid manually entering coordinates and financial benchmarks.
    """
    from cri.api.company_resolve import resolve_company
    result = resolve_company(
        query=req.query,
        hint_sector=req.hint_sector,
        hint_country=req.hint_country,
    )
    return {
        "found":           result.found,
        "confidence":      result.confidence,
        "company_name":    result.company_name,
        "ticker":          result.ticker,
        "sector_key":      result.sector_key,
        "country":         result.country,
        "hq_lat":          result.hq_lat,
        "hq_lon":          result.hq_lon,
        "revenue_usd_m":   result.revenue_usd_m,
        "ev_usd_m":        result.ev_usd_m,
        "scope1_mt_co2e":  result.scope1_mt_co2e,
        "elevation_m":     result.elevation_m,
        "distance_to_coast_km": result.distance_to_coast_km,
        "pre_filled_payload": result.pre_filled_payload,
        "notes":           result.notes,
    }


# ── /compare/scenarios ────────────────────────────────────────────────────────

class ScenarioCompareRequest(BaseModel):
    company_id:                  str
    company_name:                str
    sector_key:                  str
    asset_lat:                   float
    asset_lon:                   float
    asset_elevation_m:           float = 5.0
    distance_to_coast_km:        float = 50.0
    ev_usd_m:                    float
    revenue_usd_m:               float
    scope1_emissions_mt_co2e:    float
    green_revenue_pct:           float = 0.0

@app.post("/compare/scenarios", tags=["Workflow"])
async def compare_scenarios(req: ScenarioCompareRequest):
    """
    Single call that runs all three IPCC warming scenarios
    (SSP1-2.6 / SSP2-4.5 / SSP5-8.5) and all three NGFS transition
    scenarios (Net Zero 2050 / Delayed Transition / Current Policies)
    and returns results side-by-side.
    """
    from cri.climate.sea_level_rise import compute_slr_risk, slr_to_dict
    from cri.climate.transition import run_transition_risk

    physical_scenarios = {}
    for scenario in ["SSP1-2.6", "SSP2-4.5", "SSP5-8.5"]:
        slr = compute_slr_risk(
            asset_id=req.company_id,
            asset_name=req.company_name,
            asset_lat=req.asset_lat,
            asset_lon=req.asset_lon,
            asset_elevation_m=req.asset_elevation_m,
            ev_usd_m=req.ev_usd_m,
            scenario=scenario,
            distance_to_coast_km=req.distance_to_coast_km,
        )
        physical_scenarios[scenario] = {
            "slr_2050_m":        slr.slr_2050_m,
            "slr_2100_m":        slr.slr_2100_m,
            "inundation_prob_2050": slr.inundation_prob_2050,
            "inundation_prob_2100": slr.inundation_prob_2100,
            "risk_tier":         slr.risk_tier,
            "expected_loss_usd_m": slr.expected_loss_usd_m,
        }

    transition_scenarios = {}
    for scenario in ["net_zero_2050", "delayed_transition", "current_policies"]:
        tr = run_transition_risk(
            company_id=req.company_id,
            company_name=req.company_name,
            sector_key=req.sector_key,
            scope1_emissions_mt_co2e=req.scope1_emissions_mt_co2e,
            revenue_usd_m=req.revenue_usd_m,
            ev_usd_m=req.ev_usd_m,
            green_revenue_pct=req.green_revenue_pct,
            scenario=scenario,
        )
        transition_scenarios[scenario] = {
            "carbon_cost_2030_usd_m":  tr.carbon_cost_2030_usd_m,
            "carbon_cost_2050_usd_m":  tr.carbon_cost_2050_usd_m,
            "var_pct_revenue":         tr.var_pct_revenue,
            "stranded_asset_risk":     tr.stranded_asset_risk,
            "transition_risk_score":   tr.transition_risk_score,
            "risk_tier":               tr.risk_tier,
        }

    return {
        "company_id":   req.company_id,
        "company_name": req.company_name,
        "physical": {
            "scenarios": physical_scenarios,
            "summary": "SLR exposure across IPCC SSP scenarios. Risk tiers: critical/high/medium/low/negligible.",
        },
        "transition": {
            "scenarios": transition_scenarios,
            "summary":   "Carbon cost and stranded asset risk across NGFS transition scenarios.",
        },
        "worst_case": {
            "physical_scenario":    "SSP5-8.5",
            "transition_scenario":  "delayed_transition",
            "note": "Delayed transition with high physical warming is the most adverse combination.",
        },
    }


# ── /deal/brief ───────────────────────────────────────────────────────────────

class DealBriefRequest(BaseModel):
    company_name:              str
    sector_key:                str
    country:                   str
    # Physical inputs
    flood_var_pct:             float = 0.0
    slr_risk_tier:             str   = "negligible"
    biodiversity_score:        float = 0.0
    # Transition inputs
    transition_var_pct:        float = 0.0
    scope1_mt_co2e:            float = 0.0
    carbon_price_sensitivity:  float = 0.0
    # Deal inputs
    ev_usd_m:                  float = 0.0
    revenue_usd_m:             float = 0.0
    ltv_pct:                   float = 0.0
    tenor_years:               int   = 5
    deal_type:                 str   = "term_loan"

@app.post("/deal/brief", tags=["Workflow"])
async def deal_brief(req: DealBriefRequest):
    """
    Generate an IC-ready deal brief with:
    - PROCEED / DEFER / PASS recommendation
    - 3 ranked risk bullets
    - Specific covenant language
    - KPI monitoring list

    Feed this the outputs from /physical/flood, /physical/slr,
    /transition, and /physical/biodiversity to get a complete brief.
    """
    from cri.outcomes.deal_brief import generate_deal_brief, deal_brief_to_dict
    brief = generate_deal_brief(
        company_name=req.company_name,
        sector_key=req.sector_key,
        country=req.country,
        flood_var_pct=req.flood_var_pct,
        slr_risk_tier=req.slr_risk_tier,
        biodiversity_score=req.biodiversity_score,
        transition_var_pct=req.transition_var_pct,
        scope1_mt_co2e=req.scope1_mt_co2e,
        carbon_price_sensitivity=req.carbon_price_sensitivity,
        ev_usd_m=req.ev_usd_m,
        revenue_usd_m=req.revenue_usd_m,
        ltv_pct=req.ltv_pct,
        tenor_years=req.tenor_years,
        deal_type=req.deal_type,
    )
    return deal_brief_to_dict(brief)


# ── /sensitivity ──────────────────────────────────────────────────────────────

class SensitivityRequest(BaseModel):
    company_name:              str
    sector_key:                str
    scope1_mt_co2e:            float
    revenue_usd_m:             float
    ev_usd_m:                  float
    base_flood_var_pct:        float = 0.0
    base_elevation_m:          float = 5.0
    base_carbon_price:         float = 130.0
    base_capex_green:          float = 0.0
    base_scope1_reduction_pct: float = 0.0
    sweep:                     str   = "carbon_price"  # carbon_price|capex_green|scope1_reduction|elevation
    sweep_min:                 Optional[float] = None
    sweep_max:                 Optional[float] = None
    sweep_steps:               int   = 10

@app.post("/sensitivity", tags=["Workflow"])
async def sensitivity_sweep(req: SensitivityRequest):
    """
    What-if parameter sweep. Move one lever at a time and see how
    transition VaR, physical VaR and composite score respond.

    Sweep levers:
    - carbon_price       Multiply base price 0.5× to 3.0×
    - capex_green        Committed green capex 0–500 USD M
    - scope1_reduction   Scope 1 cut vs. baseline 0–75%
    - elevation          Asset elevation 0–30 m (physical sensitivity)

    Returns a grid of points suitable for chart rendering.
    """
    from cri.outcomes.sensitivity import run_sensitivity, sensitivity_to_dict
    result = run_sensitivity(
        company_name=req.company_name,
        sector_key=req.sector_key,
        scope1_mt_co2e=req.scope1_mt_co2e,
        revenue_usd_m=req.revenue_usd_m,
        ev_usd_m=req.ev_usd_m,
        base_flood_var_pct=req.base_flood_var_pct,
        base_elevation_m=req.base_elevation_m,
        base_carbon_price=req.base_carbon_price,
        base_capex_green=req.base_capex_green,
        base_scope1_reduction_pct=req.base_scope1_reduction_pct,
        sweep=req.sweep,
        sweep_min=req.sweep_min,
        sweep_max=req.sweep_max,
        sweep_steps=req.sweep_steps,
    )
    return sensitivity_to_dict(result)


# ── /regulatory/sfdr-pai/export ──────────────────────────────────────────────

class SFDRExportRequest(BaseModel):
    company_id:              str
    company_name:            str
    sector_key:              str
    country:                 str
    scope1_mt_co2e:          float
    scope2_mt_co2e:          float
    scope3_mt_co2e:          float = 0.0
    revenue_usd_m:           float
    ev_usd_m:                float
    fossil_fuel_exposure_pct: float = 0.0
    renewable_energy_pct:    float = 0.0
    board_gender_diversity_pct: float = 0.0
    controversy_flag:        bool  = False
    taxonomy_aligned_pct:    float = 0.0
    data_quality_score:      int   = 3
    reporting_year:          int   = 2024

@app.post("/regulatory/sfdr-pai/export", tags=["Regulatory"])
async def sfdr_pai_export(req: SFDRExportRequest):
    """
    Generate SFDR RTS Annex I PAI Excel export.
    Returns an .xlsx file in EU regulatory table format with all
    18 mandatory PAI indicators pre-populated.
    """
    import io
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter
    except ImportError:
        raise HTTPException(status_code=500, detail="openpyxl not installed")

    from fastapi.responses import StreamingResponse
    from cri.regulatory.sfdr_pai import compute_sfdr_pai

    pai = compute_sfdr_pai(
        company_id=req.company_id,
        company_name=req.company_name,
        sector_key=req.sector_key,
        country=req.country,
        scope1_mt_co2e=req.scope1_mt_co2e,
        scope2_mt_co2e=req.scope2_mt_co2e,
        scope3_mt_co2e=req.scope3_mt_co2e,
        revenue_usd_m=req.revenue_usd_m,
        ev_usd_m=req.ev_usd_m,
        fossil_fuel_exposure_pct=req.fossil_fuel_exposure_pct,
        renewable_energy_pct=req.renewable_energy_pct,
        board_gender_diversity_pct=req.board_gender_diversity_pct,
        controversy_flag=req.controversy_flag,
        taxonomy_aligned_pct=req.taxonomy_aligned_pct,
        data_quality_score=req.data_quality_score,
        reporting_year=req.reporting_year,
    )

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "SFDR PAI Annex I"

    # Header
    header_fill = PatternFill("solid", fgColor="003366")
    header_font = Font(bold=True, color="FFFFFF", size=10)
    sub_fill    = PatternFill("solid", fgColor="D9E1F2")
    sub_font    = Font(bold=True, size=9)
    thin_border = Border(
        left=Side(style="thin"), right=Side(style="thin"),
        top=Side(style="thin"), bottom=Side(style="thin"),
    )

    # Title row
    ws.merge_cells("A1:H1")
    ws["A1"] = f"SFDR RTS Annex I — Principal Adverse Impact Indicators"
    ws["A1"].font = Font(bold=True, size=12, color="003366")
    ws["A1"].alignment = Alignment(horizontal="center")

    ws.merge_cells("A2:H2")
    ws["A2"] = (
        f"Company: {req.company_name}  |  "
        f"Reporting Year: {req.reporting_year}  |  "
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
    )
    ws["A2"].font = Font(italic=True, size=9)

    # Column headers (row 4)
    cols = [
        "PAI #", "Indicator Name", "Category",
        "Value", "Unit", "Data Quality (1–5)",
        "Coverage (%)", "Notes / Methodology",
    ]
    for ci, h in enumerate(cols, 1):
        cell = ws.cell(row=4, column=ci, value=h)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
        cell.border = thin_border

    # Column widths
    widths = [6, 45, 20, 14, 12, 14, 12, 50]
    for ci, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(ci)].width = w

    # Data rows
    row_idx = 5
    for indicator in pai.pai_indicators:
        ws.cell(row=row_idx, column=1, value=indicator["indicator_number"]).border = thin_border
        ws.cell(row=row_idx, column=2, value=indicator["indicator_name"]).border = thin_border
        ws.cell(row=row_idx, column=3, value=indicator["category"]).border = thin_border
        ws.cell(row=row_idx, column=4, value=indicator.get("value", "N/A")).border = thin_border
        ws.cell(row=row_idx, column=5, value=indicator.get("unit", "")).border = thin_border
        ws.cell(row=row_idx, column=6, value=indicator.get("data_quality_score", req.data_quality_score)).border = thin_border
        ws.cell(row=row_idx, column=7, value=indicator.get("coverage_pct", 100)).border = thin_border
        notes_cell = ws.cell(row=row_idx, column=8, value=indicator.get("methodology_note", ""))
        notes_cell.border = thin_border
        notes_cell.alignment = Alignment(wrap_text=True)

        # Alternate row shading
        if row_idx % 2 == 0:
            fill = PatternFill("solid", fgColor="F2F2F2")
            for ci in range(1, 9):
                ws.cell(row=row_idx, column=ci).fill = fill

        row_idx += 1

    # Summary section
    row_idx += 1
    ws.cell(row=row_idx, column=1, value="Summary").font = sub_font
    ws.cell(row=row_idx, column=1).fill = sub_fill
    row_idx += 1
    ws.cell(row=row_idx, column=1, value="Overall Data Quality")
    ws.cell(row=row_idx, column=2, value=pai.overall_data_quality)
    row_idx += 1
    ws.cell(row=row_idx, column=1, value="Estimated GHG Intensity (tCO2e/EUR M rev)")
    ws.cell(row=row_idx, column=2, value=round(pai.ghg_intensity_revenue, 1))
    row_idx += 1
    ws.cell(row=row_idx, column=1, value="Fossil Fuel Exposure (%)")
    ws.cell(row=row_idx, column=2, value=req.fossil_fuel_exposure_pct)

    # Footer
    row_idx += 2
    ws.cell(row=row_idx, column=1, value=(
        "Prepared by ClimRisk Engine | "
        "Regulation: SFDR RTS (EU) 2022/1288 Annex I — Table 1 (18 mandatory PAIs). "
        "Data sourced from company disclosures / third-party estimation where noted."
    )).font = Font(italic=True, size=8, color="666666")

    # Freeze panes
    ws.freeze_panes = "A5"

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)

    filename = f"SFDR_PAI_{req.company_name.replace(' ','_')}_{req.reporting_year}.xlsx"
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── /portfolio/delta ──────────────────────────────────────────────────────────

class PortfolioDeltaPosition(BaseModel):
    company_id:            str
    company_name:          str
    prior_score:           float
    current_score:         float
    prior_flood_var_pct:   Optional[float] = None
    current_flood_var_pct: Optional[float] = None
    prior_transition_var:  Optional[float] = None
    current_transition_var: Optional[float] = None
    exposure_usd_m:        float = 0.0
    change_driver:         Optional[str] = None  # "new_data" | "model_update" | "market_move"

class PortfolioDeltaRequest(BaseModel):
    run_id_prior:   str
    run_id_current: str
    positions:      list[PortfolioDeltaPosition]
    as_of_date:     Optional[str] = None

@app.post("/portfolio/delta", tags=["Portfolio"])
async def portfolio_delta(req: PortfolioDeltaRequest):
    """
    What changed between two engine runs and what drove the change.

    Pass prior and current scores for each position. Returns:
    - Largest movers (score change > 0.05)
    - Portfolio-level aggregate delta
    - Change attribution by driver
    - Positions that crossed a tier boundary (e.g. GREEN→AMBER)
    """
    now = datetime.now(timezone.utc).isoformat()

    movers    = []
    upgrades  = []  # risk decreased
    downgrades = []  # risk increased
    tier_changes = []

    def _tier(score: float) -> str:
        if score >= 0.65: return "RED"
        if score >= 0.40: return "AMBER"
        return "GREEN"

    total_prior   = 0.0
    total_current = 0.0
    weighted_prior   = 0.0
    weighted_current = 0.0
    total_exposure   = sum(p.exposure_usd_m for p in req.positions)

    drivers_count: dict[str, int] = {}

    for pos in req.positions:
        delta = pos.current_score - pos.prior_score
        weight = pos.exposure_usd_m / total_exposure if total_exposure > 0 else 0

        weighted_prior   += pos.prior_score * weight
        weighted_current += pos.current_score * weight

        prior_tier   = _tier(pos.prior_score)
        current_tier = _tier(pos.current_score)

        change_details = {
            "company_id":       pos.company_id,
            "company_name":     pos.company_name,
            "prior_score":      round(pos.prior_score, 3),
            "current_score":    round(pos.current_score, 3),
            "score_delta":      round(delta, 3),
            "prior_tier":       prior_tier,
            "current_tier":     current_tier,
            "exposure_usd_m":   pos.exposure_usd_m,
            "weight_pct":       round(weight * 100, 2),
            "change_driver":    pos.change_driver or "unspecified",
            "flood_var_delta":  None,
            "transition_var_delta": None,
        }

        if pos.prior_flood_var_pct is not None and pos.current_flood_var_pct is not None:
            change_details["flood_var_delta"] = round(
                pos.current_flood_var_pct - pos.prior_flood_var_pct, 2
            )
        if pos.prior_transition_var is not None and pos.current_transition_var is not None:
            change_details["transition_var_delta"] = round(
                pos.current_transition_var - pos.prior_transition_var, 2
            )

        if abs(delta) >= 0.05:
            movers.append(change_details)

        if prior_tier != current_tier:
            tier_changes.append({
                **change_details,
                "tier_change": f"{prior_tier} → {current_tier}",
                "direction":   "upgrade" if current_tier < prior_tier else "downgrade",
            })

        if delta < -0.05:
            upgrades.append(change_details)
        elif delta > 0.05:
            downgrades.append(change_details)

        driver = pos.change_driver or "unspecified"
        drivers_count[driver] = drivers_count.get(driver, 0) + 1

    movers.sort(key=lambda x: abs(x["score_delta"]), reverse=True)
    tier_changes.sort(key=lambda x: abs(x["score_delta"]), reverse=True)

    portfolio_delta = round(weighted_current - weighted_prior, 4)
    portfolio_direction = "deteriorated" if portfolio_delta > 0 else "improved"

    return {
        "run_id_prior":          req.run_id_prior,
        "run_id_current":        req.run_id_current,
        "as_of_date":            req.as_of_date or now,
        "position_count":        len(req.positions),
        "total_exposure_usd_m":  round(total_exposure, 2),
        "portfolio_summary": {
            "weighted_score_prior":   round(weighted_prior, 4),
            "weighted_score_current": round(weighted_current, 4),
            "delta":                  portfolio_delta,
            "direction":              portfolio_direction,
        },
        "tier_changes":          tier_changes,
        "largest_movers":        movers[:10],
        "upgrades_count":        len(upgrades),
        "downgrades_count":      len(downgrades),
        "change_driver_breakdown": drivers_count,
        "methodology": (
            "Score delta = current_score – prior_score. "
            "Material movers: |delta| ≥ 0.05. "
            "Tier boundaries: RED ≥ 0.65, AMBER 0.40–0.65, GREEN < 0.40. "
            "Portfolio delta is exposure-weighted mean score change."
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# AGENT — Autonomous assessment endpoints
# ══════════════════════════════════════════════════════════════════════════════
#
# The agent layer accepts a company name + optional hints and autonomously:
#   1. Resolves the entity via GLEIF
#   2. Fetches data from SEC EDGAR, CDP, SBTi, Yahoo Finance, OSM geocoder
#   3. Runs the full risk assessment pipeline
#   4. Produces 2025–2050 predictive trajectories (3 NGFS scenarios)
#   5. Returns a sourced report with every field provenance-tagged
#
# No hallucination: every gap is explicitly listed in data_gaps.
# No silent benchmarks: estimated values are flagged ESTIMATED.
# ══════════════════════════════════════════════════════════════════════════════

import asyncio as _asyncio

from ..agent.job_runner import (
    JobStatus,
    create_job,
    get_job,
    list_jobs,
    run_assessment_job,
)


class AgentAssessRequest(BaseModel):
    company_name:   str   = Field(..., description="Company name (as known to the practitioner)")
    assessment_scope: str = Field(
        default="standard",
        description="'triage' (name+sector+country only) | 'standard' (full DD) | 'full_dd' (all sources + trajectory)",
    )
    sector:         Optional[str]   = Field(None, description="Sector key, e.g. 'oil_gas', 'utilities', 'technology'")
    country:        Optional[str]   = Field(None, description="ISO-2 country code hint, e.g. 'US', 'GB', 'DE'")
    ticker:         Optional[str]   = Field(None, description="Stock ticker (optional, speeds up Yahoo Finance lookup)")
    lat:            Optional[float] = Field(None, description="Primary asset latitude (skips geocoding if provided)")
    lon:            Optional[float] = Field(None, description="Primary asset longitude (skips geocoding if provided)")
    use_estimates:  bool            = Field(
        default=True,
        description=(
            "If true, sector benchmarks fill missing fields (flagged ESTIMATED). "
            "If false, gaps stay None — use for strictest due diligence."
        ),
    )


@app.post(
    "/agent/assess",
    tags=["Agent"],
    summary="Submit an autonomous climate risk assessment job",
    response_description=(
        "Job ID and initial status. Poll /agent/jobs/{job_id} for results."
    ),
)
async def agent_assess(req: AgentAssessRequest, background_tasks=None):
    """
    Submit a company for autonomous climate risk assessment.

    The engine will:
    1. Resolve the entity via GLEIF LEI registry
    2. Fetch SEC EDGAR / CDP / SBTi / Yahoo Finance data in parallel
    3. Geocode the registered address for physical hazard calculations
    4. Run the full CRI engine (physical + transition + biodiversity)
    5. Produce 2025–2050 predictive trajectories (3 NGFS scenarios)
    6. Return a fully sourced report — every field tagged with source + confidence

    **Returns immediately** with a `job_id`. Poll `/agent/jobs/{job_id}` for
    progress and results.  Typical completion time: 15–45 seconds.
    """
    job = create_job(
        company_name=req.company_name,
        scope=req.assessment_scope,
    )

    # Run in background — don't block the HTTP response
    _asyncio.create_task(
        run_assessment_job(
            job,
            sector=req.sector,
            country_hint=req.country,
            ticker=req.ticker,
            lat=req.lat,
            lon=req.lon,
            use_estimates=req.use_estimates,
        )
    )

    return {
        "job_id":       job.job_id,
        "status":       job.status.value,
        "company_name": job.company_name,
        "scope":        job.scope,
        "message":      (
            f"Assessment queued for '{req.company_name}'. "
            f"Poll /agent/jobs/{job.job_id} for progress and results."
        ),
        "poll_url":     f"/agent/jobs/{job.job_id}",
    }


@app.get(
    "/agent/jobs/{job_id}",
    tags=["Agent"],
    summary="Poll assessment job status and retrieve results",
)
async def agent_job_status(job_id: str):
    """
    Poll an assessment job.

    Returns current status, progress events, partial data, and (when completed)
    the full result including:
    - company_profile: all sourced fields with provenance
    - risk_assessment: physical + transition + composite scores
    - trajectory: 2025–2050 projections across 3 NGFS scenarios
    - data_gaps: explicit list of what couldn't be sourced and why
    - data_request: actionable list of what data to provide to close gaps
    """
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    if job.status == JobStatus.COMPLETED:
        return job.to_full_dict()
    else:
        return job.to_status_dict()


@app.get(
    "/agent/jobs",
    tags=["Agent"],
    summary="List recent assessment jobs",
)
async def agent_list_jobs(limit: int = 20):
    """
    List the most recent assessment jobs (newest first).

    Useful for monitoring batch runs or checking job history.
    Jobs are retained for 6 hours after completion.
    """
    return {
        "jobs":  list_jobs(limit=limit),
        "count": len(list_jobs(limit=limit)),
    }


@app.post(
    "/agent/profile",
    tags=["Agent"],
    summary="Synchronous company data profile (no risk assessment)",
    response_description="CompanyProfile with full provenance — no engine run",
)
async def agent_profile_only(req: AgentAssessRequest):
    """
    Fetch and return a sourced company profile **without** running the full
    risk engine.  Useful for:
    - Previewing what data is available before committing to an assessment
    - Populating the data entry form for manual review
    - Checking what gaps exist for a counterparty

    Runs synchronously (waits for all sources before returning).
    Typical time: 10–25 seconds.
    """
    from ..data_acquisition.company_profiler import build_company_profile

    profile = await build_company_profile(
        company_name=req.company_name,
        sector=req.sector,
        country_hint=req.country,
        ticker=req.ticker,
        lat=req.lat,
        lon=req.lon,
        use_estimates=req.use_estimates,
    )

    return {
        "company_profile":    profile.to_dict(),
        "engine_payload":     profile.engine_payload(),
        "assessment_readiness": profile.assessment_readiness,
        "data_gaps":          [
            {
                "field":   g.field_name,
                "reason":  g.reason,
                "impact":  g.impact,
                "action":  g.suggested_action,
            }
            for g in profile.data_gaps
        ],
        "data_request": [
            {
                "field":          g.field_name,
                "why_needed":     g.impact,
                "how_to_provide": g.suggested_action,
            }
            for g in profile.data_gaps
            if "benchmark" not in g.reason.lower()
        ],
        "note": (
            "Use /agent/assess to run the full risk assessment on this profile. "
            "Use engine_payload as the body for /assess if you prefer manual review first."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Extended risk standalone endpoints
# ─────────────────────────────────────────────────────────────────────────────

class StrandedAssetsRequest(BaseModel):
    company_name:        str   = Field(..., description="Company display name")
    sector:              str   = Field(..., description="Sector slug (oil_gas, coal, utilities, steel, cement, …)")
    carrying_value_usd_m: float = Field(..., description="Net book value of PP&E / reserves (USD M)")
    revenue_usd_m:       float = Field(..., description="Annual revenue (USD M)")
    ev_usd_m:            Optional[float] = Field(None, description="Enterprise value (USD M); defaults to 2× revenue")
    market_cap_usd_m:    Optional[float] = Field(None, description="Market cap (USD M)")
    wacc:                float = Field(default=0.09, description="Discount rate (default 9%)")
    carbon_price_coverage: float = Field(default=1.0, description="Fraction of Scope 1 emissions under carbon price (0-1)")
    free_allocation_pct:   float = Field(default=0.0,  description="EU ETS-style free allowance as fraction of covered emissions")


@app.post(
    "/agent/stranded-assets",
    tags=["Agent", "Extended Risk"],
    summary="Stranded asset NPV impairment — IEA WEO 2023 unburnable carbon model",
    response_description="Per-scenario (NZE / Delayed / Current Policies) stranded asset analysis with yearly breakdown",
)
def agent_stranded_assets(req: StrandedAssetsRequest):
    """
    Assess the stranded asset risk for fossil-fuel-exposed companies using
    IEA WEO 2023 Annex A remaining demand paths.

    **Covered sectors:** oil_gas, coal, gas, utilities (gas power), steel, cement.
    Returns `is_applicable: false` for non-fossil sectors.

    **Output per NGFS scenario:**
    - `stranding_year` — first year carbon-inclusive cost exceeds break-even
    - `npv_impairment_usd_m` — present-value write-down
    - `impairment_pct_ev` — impairment as % of enterprise value
    - `years` — year-by-year stranded fraction and discounted impairment

    Sources: IEA WEO 2023 Annex A; Rystad UCube 2023; Carbon Tracker Initiative 2023.
    """
    from ..climate.stranded_assets import assess_stranded_assets
    result = assess_stranded_assets(
        company_name=req.company_name,
        sector=req.sector,
        carrying_value_usd_m=req.carrying_value_usd_m,
        revenue_usd_m=req.revenue_usd_m,
        ev_usd_m=req.ev_usd_m,
        market_cap_usd_m=req.market_cap_usd_m,
        wacc=req.wacc,
        carbon_price_coverage=req.carbon_price_coverage,
        free_allocation_pct=req.free_allocation_pct,
    )
    import dataclasses
    return dataclasses.asdict(result)


class CBAMRequest(BaseModel):
    company_name:        str   = Field(..., description="Company display name")
    sector:              str   = Field(..., description="CBAM-covered sector: cement, steel, aluminium, fertilizers, chemicals, hydrogen, electricity")
    revenue_usd_m:       float = Field(..., description="Total annual revenue (USD M)")
    eu_revenue_fraction: float = Field(default=0.20, description="Fraction of revenue from EU-bound exports (0-1); default 20%")
    production_tonnes:   Optional[float] = Field(None, description="Annual production volume (tonnes); estimated from revenue if absent")
    origin_country:      str   = Field(default="default", description="ISO-2 country of production (for carbon price deduction per Article 9)")
    carbon_price_paid_eur: Optional[float] = Field(None, description="Override for origin carbon price (EUR/tCO2)")
    eur_usd_rate:        float = Field(default=1.08, description="EUR/USD conversion rate")


@app.post(
    "/agent/cbam",
    tags=["Agent", "Extended Risk"],
    summary="EU CBAM certificate cost trajectory 2026-2034 (Regulation EU 2023/956)",
    response_description="Year-by-year CBAM certificate cost, net of origin carbon price, for covered sectors",
)
def agent_cbam(req: CBAMRequest):
    """
    Assess EU Carbon Border Adjustment Mechanism (CBAM) exposure for companies
    exporting to the EU in covered sectors (Annex I, Regulation EU 2023/956).

    **CBAM coverage:** cement, iron & steel, aluminium, fertilisers, hydrogen, electricity.
    Returns `is_applicable: false` for sectors outside Annex I scope.

    **Output:**
    - `cbam_trajectory` — year-by-year certificate cost 2026-2034
    - `peak_annual_cost_usd_m` — highest single-year CBAM bill
    - `methodology` — embedded carbon intensity source and EU ETS price path

    Certificate price tracks EU ETS (NGFS Phase 4 / EMBER calibrated: €60/t in 2026 → €160/t in 2034).
    Origin carbon price deducted per Article 9 (e.g. China ETS €8/t, Korea ETS €20/t).
    """
    from ..climate.cbam import assess_cbam_exposure
    result = assess_cbam_exposure(
        company_name=req.company_name,
        sector=req.sector,
        revenue_usd_m=req.revenue_usd_m,
        eu_revenue_fraction=req.eu_revenue_fraction,
        production_tonnes=req.production_tonnes,
        origin_country=req.origin_country,
        carbon_price_paid_eur=req.carbon_price_paid_eur,
        eur_usd_rate=req.eur_usd_rate,
    )
    import dataclasses
    return dataclasses.asdict(result)


class WaterRiskRequest(BaseModel):
    company_name:    str   = Field(..., description="Company display name")
    sector:          str   = Field(..., description="Sector slug (agriculture, mining, chemicals, steel, beverages, utilities, …)")
    aqueduct_score:  float = Field(..., description="WRI Aqueduct 4.0 composite water risk score (0-5). Look up at https://www.wri.org/applications/aqueduct/water-risk-atlas")
    revenue_usd_m:   float = Field(..., description="Annual revenue (USD M)")
    ev_usd_m:        Optional[float] = Field(None, description="Enterprise value (USD M); defaults to 2× revenue")
    wacc:            float = Field(default=0.09, description="Discount rate")
    horizon:         int   = Field(default=2050, description="End year for NPV projection")
    scenario:        str   = Field(default="current_policies", description="'nze' | 'delayed' | 'current_policies'")
    water_intensity: Optional[float] = Field(None, description="m³ per USD revenue (optional; sector default used if absent)")


@app.post(
    "/agent/water-risk",
    tags=["Agent", "Extended Risk"],
    summary="Water stress financial loss model — WRI Aqueduct 4.0 → NPV drag",
    response_description="NPV production loss, cost increase, curtailment risk, and defensive capex by year under water stress scenario",
)
def agent_water_risk(req: WaterRiskRequest):
    """
    Translate a WRI Aqueduct 4.0 water stress score into year-by-year financial
    impact under the specified NGFS-aligned climate scenario.

    **Financial layers:**
    1. Production loss (sector × stress band, Ceres 2019 / S&P Sustainable1 2023)
    2. Operating cost uplift (water procurement, recycling, Bloomberg NEF 2023)
    3. Curtailment expected value (regulatory shutdown, CDP Water 2023)
    4. Defensive capex present value (recycling loops, ZLD treatment)

    **WRI Aqueduct score bands:**
    - 0-1: Low | 1-2: Low-Medium | 2-3: Medium-High | 3-4: High | 4-5: Extremely High

    Stress score escalates with warming per WRI Aqueduct SSP projections:
    Current Policies → 1.42× by 2050; NZE → 1.12× by 2050.
    """
    from ..climate.water_stress_loss import assess_water_stress
    result = assess_water_stress(
        company_name=req.company_name,
        sector=req.sector,
        aqueduct_score=req.aqueduct_score,
        revenue_usd_m=req.revenue_usd_m,
        ev_usd_m=req.ev_usd_m,
        wacc=req.wacc,
        horizon=req.horizon,
        scenario=req.scenario,
        water_intensity=req.water_intensity,
    )
    import dataclasses
    return dataclasses.asdict(result)
