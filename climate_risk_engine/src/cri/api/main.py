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

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import requests

from fastapi import Depends, FastAPI, File, Form, HTTPException, Security, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field

import dataclasses

from .agents import router as agents_router, _run_monitoring_cycle, _load_watchlist

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
    InquiryRequest,
    InquiryResponse,
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


# ── Daily monitoring scheduler ────────────────────────────────────────────────
# Runs every day at 08:00 UTC — re-assesses watchlist companies and sends
# email alerts when risk scores shift by more than ALERT_THRESHOLD points.
def _start_scheduler() -> None:
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        import asyncio

        async def _daily_monitor() -> None:
            wl = _load_watchlist()
            if wl:
                await _run_monitoring_cycle(wl)

        sched = AsyncIOScheduler(timezone="UTC")
        sched.add_job(_daily_monitor, "cron", hour=8, minute=0, id="daily_monitor")
        sched.start()
        logging.getLogger("cri.api.scheduler").info("Daily monitoring scheduler started (08:00 UTC)")
    except ImportError:
        logging.getLogger("cri.api.scheduler").warning(
            "apscheduler not installed — daily monitoring disabled. "
            "Run: pip install apscheduler"
        )
    except Exception as exc:
        logging.getLogger("cri.api.scheduler").error("Scheduler failed to start: %s", exc)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    _start_scheduler()
    try:                                            # the resident agent works on its own from the moment the engine (the app) opens
        from ..resident import autopilot as _ap
        _ap.start()
    except Exception as exc:                        # noqa: BLE001
        logging.getLogger(__name__).warning("autopilot not started: %s", exc)
    # Preload in-memory datasets in the background so the first request has no lag
    try:
        from ..data.asset_register import warm_up_async
        warm_up_async()
        import threading
        from ..climate.tropical_cyclone import tracks as _tc_tracks
        threading.Thread(target=_tc_tracks, daemon=True).start()
    except Exception as exc:                        # noqa: BLE001
        logging.getLogger(__name__).warning("warm-up skipped: %s", exc)
    try:                                            # the case index and the local model are loaded now, so the first question in a demo is as fast as the tenth
        import threading as _th
        from ..resident import answer as _an, casebase as _cb

        def _warm_ai():
            try:
                if _cb.available():
                    _cb.stats()
                _an.warm()
            except Exception:                       # noqa: BLE001
                pass
        _th.Thread(target=_warm_ai, daemon=True).start()
    except Exception as exc:                        # noqa: BLE001
        logging.getLogger(__name__).warning("AI warm-up skipped: %s", exc)
    yield


# Initialize FastAPI app
app = FastAPI(
    title="Climate Risk Intelligence API",
    description="REST API for climate financial risk modelling",
    version="0.5.0",
    lifespan=_lifespan,
)

# Add CORS middleware (allow all origins for dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Agentic AI routes ─────────────────────────────────────────────────────────
app.include_router(agents_router)

# Build registries
SCENARIO_REGISTRY = {
    "nze_2050": NZE_2050,
    "delayed_transition": DELAYED_TRANSITION,
    "current_policies": CURRENT_POLICIES,
}

COMPANY_REGISTRY = get_all_companies()

logger = logging.getLogger("cri.api.inquiry")

# Engagement-inquiry notification config (contact form on the marketing site).
# RESEND_API_KEY must be set in Render's environment for real email delivery;
# every inquiry is logged either way so nothing is lost if it's unset.
RESEND_API_KEY = os.environ.get("RESEND_API_KEY")
INQUIRY_NOTIFY_EMAIL = os.environ.get("INQUIRY_NOTIFY_EMAIL", "shri@climrisk.io")
INQUIRY_FROM_EMAIL = os.environ.get("INQUIRY_FROM_EMAIL", "onboarding@resend.dev")

# ── API Key Authentication ────────────────────────────────────────────────────
# Set CRI_API_KEYS in Render env as a comma-separated list of valid keys.
# When unset (dev / self-hosted), auth is skipped so existing usage isn't broken.
_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
_VALID_KEYS: set[str] = {
    k.strip() for k in os.environ.get("CRI_API_KEYS", "").split(",") if k.strip()
}

def _require_api_key(api_key: Optional[str] = Security(_API_KEY_HEADER)) -> None:
    """FastAPI dependency — validates X-API-Key header on protected endpoints.

    Behaviour:
    - CRI_API_KEYS not set  → auth skipped (development / on-premise mode)
    - CRI_API_KEYS set, key missing or wrong → 403
    - CRI_API_KEYS set, key matches → allow
    """
    if not _VALID_KEYS:
        return   # auth not configured — open access (dev / on-premise)
    if not api_key or api_key not in _VALID_KEYS:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "Invalid or missing API key.",
                "hint": "Pass your key in the X-API-Key request header. "
                        "Contact team@climrisk.io to obtain a key.",
            },
        )

_AUTH = Depends(_require_api_key)

# ── Resident AI agent (in-engine, trained NLU; dock injected into every engine HTML page) ──
from ..resident.api import make_router as _resident_router, INJECT as _DOCK_TAG   # noqa: E402
_res_prot, _res_pub = _resident_router(_AUTH)
app.include_router(_res_pub)
app.include_router(_res_prot)

from .. import extensions as _ext   # noqa: E402  (private extensions load only where listed; absent in commercial builds)
for _m in _ext.modules():
    if hasattr(_m, "make_router"):
        for _r in _m.make_router(_AUTH):
            app.include_router(_r)

from ..story_api import make_router as _story_router   # noqa: E402
_st_prot, _st_pub = _story_router(_AUTH)
app.include_router(_st_pub)
app.include_router(_st_prot)


@app.middleware("http")
async def _inject_resident_dock(request, call_next):
    resp = await call_next(request)
    p = request.url.path
    if (request.method != "GET" or "text/html" not in (resp.headers.get("content-type") or "")
            or p.startswith(("/docs", "/redoc", "/openapi", "/resident/dashboard"))):
        return resp
    body = b"".join([chunk async for chunk in resp.body_iterator])
    if b"</body>" in body and _DOCK_TAG not in body:
        i = body.rfind(b"</body>")
        body = body[:i] + _DOCK_TAG + body[i:]
    headers = {k: v for k, v in resp.headers.items() if k.lower() != "content-length"}
    from fastapi.responses import Response as _R
    return _R(content=body, status_code=resp.status_code, headers=headers, media_type="text/html")

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
    return HealthResponse(status="ok", version="0.5.0")


# ── License tier metadata keyed by API key ────────────────────────────────────
# In production, replace with a Supabase table lookup.
# Add a row here for each client when you issue their license.
_LICENSE_TIERS: dict[str, dict] = {
    "CRI2026": {
        "tier": "professional",
        "company": "ClimRisk Demo",
        "features": [
            "assessments", "ratings", "tcfd", "issb", "csrd",
            "pdf_reports", "scenarios", "agent_assess", "portfolio"
        ],
        "max_assessments": 500,
        "seats": 5,
        "expires": "2027-12-31",
        "valid": True,
    },
}

@app.get("/license/validate")
def validate_license(
    api_key: Optional[str] = Security(_API_KEY_HEADER),
) -> dict:
    """Returns license tier and features for the provided API key.

    Called by ClimRisk Desktop on startup to determine what features to unlock.
    Public endpoint (no _AUTH dependency) so the desktop can validate before
    the user is "inside" the app.

    - No key supplied          → free / unauthenticated
    - Key matches _LICENSE_TIERS → return that tier
    - Key in _VALID_KEYS but not in tier map → analyst (default paid tier)
    - CRI_API_KEYS not set at all (open access mode) → professional (dev/on-premise)
    - Invalid key              → 403
    """
    if not api_key:
        return {
            "tier": "free",
            "features": ["assessments"],
            "max_assessments": 5,
            "seats": 1,
            "valid": False,
            "message": "Supply X-API-Key header to activate a license.",
        }

    # Open access mode (CRI_API_KEYS not configured — dev / self-hosted)
    if not _VALID_KEYS:
        meta = _LICENSE_TIERS.get(api_key)
        if meta:
            return meta
        # Unknown key in open mode → grant professional for on-premise installs
        return {
            "tier": "professional",
            "features": [
                "assessments", "ratings", "tcfd", "issb", "csrd",
                "pdf_reports", "scenarios", "agent_assess"
            ],
            "max_assessments": 500,
            "seats": 5,
            "valid": True,
            "message": "Open access mode — all Professional features enabled.",
        }

    # Keys are configured — must match
    if api_key not in _VALID_KEYS:
        raise HTTPException(
            status_code=403,
            detail={"error": "Invalid license key.", "valid": False,
                    "hint": "Contact hello@climrisk.io to obtain a valid key."},
        )

    # Return tier metadata for this key (default to Analyst if not in map)
    meta = _LICENSE_TIERS.get(api_key, {
        "tier": "analyst",
        "features": ["assessments", "ratings"],
        "max_assessments": 100,
        "seats": 1,
        "valid": True,
    })
    return meta


@app.post("/inquiry", response_model=InquiryResponse)
def submit_inquiry(request: InquiryRequest) -> InquiryResponse:
    """Receive an engagement inquiry from the marketing site's contact form
    and notify the ClimRisk team by email.

    The inquiry is always logged server-side (visible in Render's logs) so
    nothing is lost even when email delivery isn't configured or fails.
    """
    lines = [
        f"Name: {request.first_name} {request.last_name}",
        f"Work Email: {request.work_email}",
        f"Company: {request.company}",
        f"Primary Function: {request.primary_function}",
        f"Primary Objective: {request.primary_objective}",
        "",
        "Notes:",
        request.notes or "(none)",
    ]
    body_text = "\n".join(lines)

    logger.info("New demo request received:\n%s", body_text)

    if not RESEND_API_KEY:
        logger.warning("RESEND_API_KEY is not set — inquiry was logged but no email was sent.")
        return InquiryResponse(received=True, message="Inquiry received.")

    try:
        resp = requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {RESEND_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "from": f"ClimRisk Website <{INQUIRY_FROM_EMAIL}>",
                "to": [INQUIRY_NOTIFY_EMAIL],
                "reply_to": request.work_email,
                "subject": f"Demo Request — {request.company} ({request.primary_objective})",
                "text": body_text,
            },
            timeout=10,
        )
        if resp.status_code >= 400:
            logger.error("Resend API error %s: %s", resp.status_code, resp.text)
    except Exception as exc:  # noqa: BLE001 — never let email delivery break the form
        logger.error("Failed to send inquiry email: %s", exc)

    return InquiryResponse(received=True, message="Inquiry received.")


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
                        id=a.id, name=a.name, commodity=a.commodity.value, region=a.region,
                        lat=a.lat, lon=a.lon, equipment_type=a.equipment_type,
                        baseline_production=a.baseline_production, production_unit=a.production_unit,
                        carrying_value_usd_m=round(a.carrying_value, 1),
                    )
                    for a in company.assets
                ],
            )
        )
    return companies


@app.post("/runs", response_model=RunResponse, dependencies=[_AUTH])
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



# ──────────────────────────────────────────────────────────────────────────────
# DASHBOARD ASSESSMENT ENDPOINT
# Drives the dashboard at /platform/dashboard.
# Resolves company name → runs full rating → returns AssessmentData shape.
# ──────────────────────────────────────────────────────────────────────────────

class AssessRequest(BaseModel):
    company_name: str = Field(..., description="Company name (fuzzy-matched against registry)")
    sector_hint: str  = Field("technology", description="Sector hint for parametric build if company not in registry")
    jurisdiction: str = Field("US", description="ISO 2-letter country code for parametric risk scoring")
    revenue_usd_m: float = Field(5_000.0, description="Approximate annual revenue in USD millions")
    assessment_scope: str = Field("standard", description="standard | full")


@app.post("/agent/assess", dependencies=[_AUTH])
def agent_assess(request: AssessRequest) -> dict:
    """
    Synchronous company assessment for the ClimRisk dashboard.

    Resolution order:
    1. Exact/fuzzy match against the company registry (known companies)
    2. Parametric synthetic build using IPCC AR6 + NGFS Phase 4 embedded data
       (works for ANY company in ANY sector in ANY country — no geocoding API)

    Returns an AssessmentData payload with sector-calibrated physical risk
    (by climate zone), transition risk (by NGFS carbon pathway), and a
    2025–2050 CRI trajectory under all three NGFS scenarios.
    """
    from ..outcomes.ratings import RatingEngine, WeightProfile
    from ..outcomes.tiers import Tier

    # ── 1. Resolve company from registry (fuzzy match) ─────────────────────────
    query = request.company_name.lower().strip()
    company = None
    _use_registry = False
    _scenarios_to_use: dict = {}

    # Exact match
    if query in COMPANY_REGISTRY:
        company = COMPANY_REGISTRY[query]
        _use_registry = True
    # Substring match on the registered name — only for queries long enough to be unambiguous.
    # (A first-word match used to map e.g. "Rio Grande Foods" onto Rio Tinto; removed.)
    elif len(query) >= 4:
        for cid, c in COMPANY_REGISTRY.items():
            if query in c.name.lower() or c.name.lower() in query:
                company = c
                _use_registry = True
                break

    # ── 2. Parametric build for unknown companies ──────────────────────────────
    _data_source = "registry"
    _climate_zone = None
    _data_notes: list[str] = []

    if company is None:
        # Build a synthetic Company object from IPCC AR6 + NGFS parameters.
        # This produces sector- and jurisdiction-specific risk scores without
        # any external API calls or geocoding.
        try:
            from ..data.profile_builder import build_synthetic_company
            built = build_synthetic_company(
                name          = request.company_name,
                sector        = request.sector_hint,
                jurisdiction  = request.jurisdiction,
                revenue_usd_m = request.revenue_usd_m,
            )
            company = built.company
            _scenarios_to_use = built.scenarios   # augmented with zone hazard paths
            _data_source = "parametric_ipcc_ngfs"
            _climate_zone = built.climate_zone
            _data_notes = built.data_notes
            # Ground the synthetic company's Scope 1/2 in the emissions estimator trained on observed
            # (EPA GHGRP) data instead of the generic sector-intensity table.
            try:
                from ..ml.emissions_estimator import EmissionsEstimator
                _em0 = EmissionsEstimator().estimate(sector=request.sector_hint, revenue_usd_m=request.revenue_usd_m,
                                                     jurisdiction=request.jurisdiction or "US",
                                                     company_name=request.company_name, ticker=company.id)
                _a = company.assets[0]
                if _a.baseline_production > 0 and _em0.scope1_mt_co2e > 0:
                    _a.emissions.scope1_intensity = _em0.scope1_mt_co2e / _a.baseline_production
                    _a.emissions.scope2_intensity = _em0.scope2_mt_co2e / _a.baseline_production
                    _a.emissions.scope3_intensity = _em0.scope3_mt_co2e / _a.baseline_production
                    _data_notes.append("Scope 1/2/3: emissions estimator (EPA GHGRP-trained; median error ~3.5×)")
            except Exception as _ee:
                logging.getLogger("cri.api").warning("emissions override failed: %s", _ee)
        except Exception as _e:
            import logging
            logging.getLogger("cri.api").error(f"profile_builder failed: {_e}")
            # Never substitute another company's results — fail loudly instead.
            raise HTTPException(status_code=502, detail=f"Could not build a profile for '{request.company_name}': {_e}")

    # Use registry scenarios for known companies; augmented scenarios for synthetic
    if not _scenarios_to_use:
        _scenarios_to_use = {
            "nze":     SCENARIO_REGISTRY["nze_2050"],
            "delayed": SCENARIO_REGISTRY["delayed_transition"],
            "cp":      SCENARIO_REGISTRY["current_policies"],
        }

    # ── Run full rating ────────────────────────────────────────────────────────
    wp = WeightProfile.EQUAL
    nze_r = run_engine(company=company, scenario=_scenarios_to_use["nze"])
    dt_r  = run_engine(company=company, scenario=_scenarios_to_use["delayed"])
    cp_r  = run_engine(company=company, scenario=_scenarios_to_use["cp"])

    rating_engine = RatingEngine()
    rr = rating_engine.rate(
        company_name=company.name,
        sector=company.sector,
        nze_results=nze_r,
        dt_results=dt_r,
        cp_results=cp_r,
        data_quality=company.data_quality,
        weight_profile=wp,
    )

    # ── Emissions estimates ────────────────────────────────────────────────────
    s1 = s2 = s3 = None
    try:
        from ..ml.emissions_estimator import EmissionsEstimator
        # NB: before this fix the call omitted `jurisdiction`, raised TypeError,
        # and the bare except below hid it — emissions were never returned.
        em = EmissionsEstimator().estimate(
            sector=company.sector,
            revenue_usd_m=company.financials.revenue,
            jurisdiction=request.jurisdiction or "US",
            company_name=company.name,
            ticker=company.id,
        )
        s1 = {"value": round(em.scope1_mt_co2e, 3),
              "range_80pct": [em.scope1_lo, em.scope1_hi],
              "tier": "verified" if em.scope1_coverage == "complete" else
                      ("reported_lower_bound" if em.scope1_coverage == "lower_bound" else "estimated"),
              "source": em.scope1_source}
        s2 = {"value": round(em.scope2_mt_co2e, 3), "tier": "estimated", "source": "Grid factor × Scope 1 share (parametric)"}
        s3 = {"value": round(em.scope3_mt_co2e, 3), "tier": "estimated", "source": "Sector supply-chain multiplier (parametric)"}
    except Exception as exc:
        logging.getLogger(__name__).warning("emissions estimate failed: %s", exc)

    # ── ML trajectory ─────────────────────────────────────────────────────────
    traj_nze = traj_delayed = traj_cp = None
    try:
        from ..ml.risk_predictor import RiskPredictor
        traj = RiskPredictor().predict(
            base_cri_score=rr.composite_score,
            sector=company.sector,
            jurisdiction=getattr(company, "hq_region", "US"),
            revenue_usd_m=company.financials.revenue,
            scope1_mt_co2e=(s1["value"] if s1 else 1.0),
            eal_p50_usd_m=50.0,
            wacc_adjusted=company.financials.wacc_base,
        )
        traj_nze     = {"scores": traj.nze_scores,     "lo": traj.nze_lo,     "hi": traj.nze_hi}
        traj_delayed = {"scores": traj.delayed_scores, "lo": traj.delayed_lo, "hi": traj.delayed_hi}
        traj_cp      = {"scores": traj.cp_scores,      "lo": traj.cp_lo,      "hi": traj.cp_hi}
    except Exception:
        pass

    # ── Build AssessmentData shape ─────────────────────────────────────────────
    def _score(x): return round(x, 1) if x is not None else None

    def _ev(r): return round(float(r.enterprise_value), 1)

    ev_cp = _ev(cp_r)

    def _vs_cp(r):
        return round((_ev(r) / ev_cp - 1.0) * 100, 2) if ev_cp else None

    # Implied temperature rise is not reported here: the budget-allocation ITR module
    # (cri.climate.paris_alignment) is under recalibration — its sector budgets are too small, so it
    # rated even fast-decarbonising companies above 2 °C. Use POST /sbti/alignment for target checks.
    paris = None

    def _phys(r, year):
        y = next((y for y in r.years if y.year == year), None)
        return round(float(y.physical_loss_cost), 2) if y else None

    peak_year = None
    if traj_cp and traj_cp.get("scores"):
        sc = traj_cp["scores"]
        peak_year = 2025 + max(range(len(sc)), key=lambda k: sc[k])

    return {
        "company_profile": {
            "resolved_name": company.name,
            "sector": company.sector,
            "jurisdiction": getattr(company, "hq_region", "US"),
            "revenue_usd_m": {"value": company.financials.revenue,
                              "tier": "reported" if _data_source == "registry" else "user-supplied"},
            "scope1_mt_co2e": s1,
            "scope2_mt_co2e": s2,
            "scope3_mt_co2e": s3,
            "data_gaps": list(_data_notes),
        },
        "risk_assessment": {
            "scenarios": {
                "nze": {
                    "enterprise_value_usd_m": _ev(nze_r),
                    "npv_impact_pct": _vs_cp(nze_r),
                    "exposure_score":    _score(rr.physical.score),
                    "transition_score":  _score(rr.transition.score),
                    "financial_score":   _score(rr.financial.score),
                },
                "delayed": {"enterprise_value_usd_m": _ev(dt_r), "npv_impact_pct": _vs_cp(dt_r)},
                "cp": {"enterprise_value_usd_m": ev_cp, "npv_impact_pct": 0.0},
                "npv_impact_basis": "enterprise value vs the Current Policies run of the same model",
            },
            "rating": {
                "rating": str(rr.rating),
                "physical_pillar":    _score(rr.physical.score),
                "transition_pillar":  _score(rr.transition.score),
                "financial_pillar":   _score(rr.financial.score),
            },
        },
        "extended_risk": {
            "paris_alignment": paris,
            "paris_alignment_note": "ITR withheld pending recalibration; SBTi target check: POST /sbti/alignment",
            "physical_risk": {
                "annual_physical_loss_usd_m": {str(yr): {"nze": _phys(nze_r, yr), "delayed": _phys(dt_r, yr),
                                                         "cp": _phys(cp_r, yr)} for yr in (2030, 2040, 2050)},
                "basis": "Engine physical-loss cost per year (USD m) by scenario",
            },
        },
        "trajectory": {
            "nze":     traj_nze,
            "delayed": traj_delayed,
            "cp":      traj_cp,
            "peak_risk_year": peak_year,
        },
        "ml_intelligence": None,
        "ml_intelligence_note": "Disclosure scoring runs on a company document — use POST /agent/disclosure.",
        "model_status": ("Legacy CRI rating model: company-level screening from a registry or sector-average "
                         "parametric profile. For asset-level physical, policy and transition results use the "
                         "Risk Analyst (/analyst/company, /public/company-risk)."),
        "model_used": "cri-engine-registry" if _data_source == "registry" else "cri-engine-parametric",
        "company_id": company.id,
        "data_provenance": {
            "source": _data_source,
            "climate_zone": _climate_zone,
            "notes": _data_notes,
            "methodology": (
                "IPCC AR6 Ch12 physical hazard profiles × NGFS Phase 4 carbon-price pathways (sector/jurisdiction "
                "parametric build — not company-specific asset data)"
                if _data_source == "parametric_ipcc_ngfs"
                else "CRI company registry with reported asset-level data"
            ),
        },
    }


class ParametricAssessRequest(BaseModel):
    company_name: str  = Field(..., description="Company name")
    sector:       str  = Field(..., description="Sector (free text — normalised internally)")
    jurisdiction: str  = Field("US", description="ISO 2-letter country code")
    revenue_usd_m: float = Field(5_000.0, description="Annual revenue USD millions")


@app.post("/assess/parametric", dependencies=[_AUTH])
def assess_parametric(request: ParametricAssessRequest) -> dict:
    """
    Fast parametric climate risk assessment using embedded IPCC AR6 + NGFS data.

    Works for ANY company in ANY sector in ANY country without relying on
    the company registry, geocoding APIs, or external data sources.

    Returns sector-calibrated physical and transition risk scores for all
    three NGFS Phase 4 scenarios (NZE, Delayed Transition, Current Policies).

    Physical risk is derived from IPCC AR6 Chapter 12 climate zone hazard
    profiles.  Transition risk is derived from NGFS Phase 4 carbon pathways
    and IEA sector emission intensity averages.
    """
    from ..data.parameters import (
        country_to_climate_zone,
        normalise_sector,
        compute_composite_score,
        get_sector_transition_params,
        get_sector_physical_sensitivity,
    )
    from ..ml.risk_predictor import RiskPredictor

    canonical_sector = normalise_sector(request.sector)
    climate_zone     = country_to_climate_zone(request.jurisdiction)
    trans_p          = get_sector_transition_params(canonical_sector)

    # NGFS Phase 4 carbon prices at 2030 for each scenario (USD/tCO2e)
    _cp_2030 = {"nze": 130.0, "delayed": 45.0, "cp": 18.0}

    scores = {}
    for scen in ("nze", "delayed", "cp"):
        scores[scen] = compute_composite_score(
            sector          = canonical_sector,
            climate_zone    = climate_zone,
            scenario        = scen,
            carbon_price_usd = _cp_2030[scen],
            year            = 2030,
        )

    composite_now = scores["delayed"]["composite_score"]   # use delayed as baseline

    # 2025–2050 trajectory from ML predictor
    traj_nze = traj_delayed = traj_cp = None
    try:
        traj = RiskPredictor().predict(
            base_cri_score  = composite_now,
            sector          = canonical_sector,
            jurisdiction    = request.jurisdiction,
            revenue_usd_m   = request.revenue_usd_m,
            scope1_mt_co2e  = trans_p["carbon_intensity_t_per_m_rev"] * request.revenue_usd_m * 0.6 / 1_000_000,
            eal_p50_usd_m   = request.revenue_usd_m * 0.025,
            wacc_adjusted   = 0.09,
        )
        traj_nze     = {"scores": traj.nze_scores,     "lo": traj.nze_lo,     "hi": traj.nze_hi}
        traj_delayed = {"scores": traj.delayed_scores, "lo": traj.delayed_lo, "hi": traj.delayed_hi}
        traj_cp      = {"scores": traj.cp_scores,      "lo": traj.cp_lo,      "hi": traj.cp_hi}
    except Exception:
        pass

    # ── HF Enrichment 1: CarbonLens + ND-GAIN country scores ─────────────────
    hf_enrichment: dict = {}
    try:
        from ..connectors.hf_carbontlens import HFEnricher
        hf_enrichment = HFEnricher.enrich(
            country    = None,
            jurisdiction = request.jurisdiction,
        )
    except Exception:
        pass

    # ── HF Enrichment 2: Monte Carlo Climate VaR ─────────────────────────────
    climate_var_result: dict = {}
    try:
        from ..engine.climate_var import compute_climate_var
        physical_now    = scores["delayed"].get("physical_score",    composite_now * 0.70)
        transition_now  = scores["delayed"].get("transition_score",  composite_now * 0.30)
        asset_value_m   = request.revenue_usd_m * 2.5   # rough EV proxy

        var_res = compute_climate_var(
            physical_score         = physical_now,
            transition_score       = transition_now,
            portfolio_value_usd_m  = asset_value_m,
            sector                 = canonical_sector,
            ndgain_vulnerability   = hf_enrichment.get("vulnerability_score"),
            ndgain_readiness       = hf_enrichment.get("readiness_score"),
        )
        climate_var_result = {
            "haircut_pct":          var_res.haircut_pct,
            "expected_loss_usd_m":  var_res.expected_loss_usd_m,
            "var_95_usd_m":         var_res.var_95_usd_m,
            "cvar_95_usd_m":        var_res.cvar_95_usd_m,
            "composite_score":      var_res.composite_score,
            "risk_tier":            var_res.risk_tier,
            "sector_multiplier":    var_res.sector_multiplier,
            "ndgain_adjustment":    var_res.ndgain_adjustment,
            "hazard_contributions": var_res.hazard_contributions,
        }
    except Exception:
        pass

    return {
        "company": request.company_name,
        "sector":  canonical_sector,
        "jurisdiction": request.jurisdiction,
        "climate_zone": climate_zone,
        "revenue_usd_m": request.revenue_usd_m,
        "scores_by_scenario": scores,
        "composite_score_current": composite_now,
        "trajectory": {
            "nze":     traj_nze,
            "delayed": traj_delayed,
            "cp":      traj_cp,
        },
        "sector_params": {
            "carbon_intensity_t_per_m_rev": trans_p["carbon_intensity_t_per_m_rev"],
            "carbon_price_coverage": trans_p["carbon_price_coverage"],
            "policy_risk_score": trans_p["policy_risk_score"],
            "stranded_asset_risk": trans_p["stranded_asset_risk"],
        },
        # ── HuggingFace data enrichments ─────────────────────────────────────
        "hf_ndgain": hf_enrichment,          # ND-GAIN country vulnerability / readiness
        "hf_climate_var": climate_var_result, # Monte Carlo Climate VaR (climate-risk-quant)
        "methodology": {
            "physical_risk": "IPCC AR6 Ch12 climate zone hazard profiles (Table 12.1)",
            "transition_risk": "NGFS Phase 4 carbon pathways + IEA sector emission intensities",
            "financial_translation": "NGFS financial sector impact study (Annex 3, 2023)",
            "climate_var": "Monte Carlo Climate VaR — adapted from deena-lad/climate-risk-quant (MIT)",
            "country_risk": "ND-GAIN Country Index 2022 via CarbonLens / SubramaniMokkala HF Space",
            "model_version": "cri-parametric-v2.1-hf",
        },
    }


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


@app.post("/runs/scoped", response_model=ScopedRunResponse, dependencies=[_AUTH])
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


@app.post("/runs/file", response_model=FileRunResponse, dependencies=[_AUTH])
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


@app.post("/ratings", response_model=RatingResponse, dependencies=[_AUTH])
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


# ── HuggingFace NLP Disclosure Analyzer ──────────────────────────────────────

class DisclosureAnalysisRequest(BaseModel):
    text: str = Field(..., description="Climate disclosure text to analyze (e.g. from annual report)")
    ticker: Optional[str] = Field(None, description="Ticker symbol for CarbonLens cross-reference")
    jurisdiction: Optional[str] = Field(None, description="ISO-2 country code for ND-GAIN context")


@app.post("/analyze/disclosure", dependencies=[_AUTH])
def analyze_disclosure_text(request: DisclosureAnalysisRequest) -> dict:
    """
    Analyze a climate disclosure text using HuggingFace NLP (ClimateRiskSense AI).

    Detects climate risk events, named entities (companies/assets/locations),
    extracts transition and physical risk signals, and performs sentiment analysis.

    Cross-references with CarbonLens (ND-GAIN country scores, XGBoost credit risk)
    when a ticker or jurisdiction is provided.

    Data sources:
      - MANSOORahme67/climaterisksense-ai (HuggingFace, NLP)
      - SubramaniMokkala/climate-credit-risk-analyzer CarbonLens (ticker lookup)
      - ND-GAIN Extended country index (country risk context)
    """
    from ..connectors.hf_nlp_analyzer import get_analyzer
    from ..connectors.hf_carbontlens import HFEnricher

    # NLP analysis
    analyzer = get_analyzer()
    nlp_result = analyzer.analyze_text(request.text)

    # CarbonLens / ND-GAIN enrichment
    hf_enrichment: dict = {}
    try:
        hf_enrichment = HFEnricher.enrich(
            ticker       = request.ticker,
            jurisdiction = request.jurisdiction,
        )
    except Exception:
        pass

    return {
        "nlp_analysis":   nlp_result,
        "hf_enrichment":  hf_enrichment,
        "text_length":    len(request.text),
        "methodology": {
            "nlp": "MANSOORahme67/climaterisksense-ai (HuggingFace Spaces)",
            "country_risk": "ND-GAIN Country Index 2022 via CarbonLens",
        },
    }


# ── HuggingFace Climate Monitoring — live hazard score for a location ─────────

class LocationHazardRequest(BaseModel):
    lat:      float = Field(..., description="Latitude")
    lon:      float = Field(..., description="Longitude")
    location: str   = Field("", description="Human-readable location name (optional)")


@app.post("/analyze/location-hazard", dependencies=[_AUTH])
def location_hazard_score(request: LocationHazardRequest) -> dict:
    """
    Retrieve a live climate hazard score for a geographic coordinate using
    the towardsinnovationlab/Climate_Risk_Monitoring HuggingFace Space.

    Returns historical weather risk, satellite-derived anomalies, and a
    0-100 hazard score that can supplement the engine's IPCC AR6 zone profiles.

    Data source: towardsinnovationlab/Climate_Risk_Monitoring (HuggingFace, MCP Server)
    """
    from ..connectors.hf_climate_monitor import get_client
    client = get_client()
    report = client.get_risk_report(lat=request.lat, lon=request.lon, location=request.location)
    score  = client.hazard_score(lat=request.lat, lon=request.lon, location=request.location)
    return {
        "hazard_score":  score,
        "risk_report":   report,
        "methodology": {
            "source": "towardsinnovationlab/Climate_Risk_Monitoring (HuggingFace Spaces MCP)",
        },
    }


@app.post("/reports/tcfd", response_model=DisclosureResponse, dependencies=[_AUTH])
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


@app.post("/reports/issb", response_model=DisclosureResponse, dependencies=[_AUTH])
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


@app.post("/reports/csrd", response_model=DisclosureResponse, dependencies=[_AUTH])
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


class BRSRRequest(DisclosureRequest):
    usd_inr: Optional[float] = Field(None, gt=0, description="INR per USD used to convert engine revenue (e.g. RBI reference rate)")
    revenue_inr_crore: Optional[float] = Field(None, gt=0, description="Company-reported turnover, INR crore (preferred)")
    ppp_inr_per_intl_usd: Optional[float] = Field(None, gt=0, description="PPP conversion factor for the BRSR Core PPP-adjusted intensity")


@app.post("/reports/brsr", response_model=DisclosureResponse, dependencies=[_AUTH])
def generate_brsr_report(request: BRSRRequest) -> DisclosureResponse:
    """SEBI BRSR Core climate disclosure (Principle 6) with Scope 1-2 intensity per INR crore of turnover."""
    from ..outcomes.disclosure import generate_brsr
    if request.company_id.lower() not in COMPANY_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Company '{request.company_id}' not found.")
    company = COMPANY_REGISTRY[request.company_id.lower()]
    nze = run_engine(company=company, scenario=SCENARIO_REGISTRY["nze_2050"])
    dt  = run_engine(company=company, scenario=SCENARIO_REGISTRY["delayed_transition"])
    cp  = run_engine(company=company, scenario=SCENARIO_REGISTRY["current_policies"])
    report = generate_brsr(company, nze, dt, cp, reporting_year=request.reporting_year, usd_inr=request.usd_inr,
                           revenue_inr_crore=request.revenue_inr_crore,
                           ppp_inr_per_intl_usd=request.ppp_inr_per_intl_usd)
    return DisclosureResponse(**report.to_dict())


class PhysicalRiskAssetIn(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    name: Optional[str] = None
    sector: Optional[str] = Field(None, description="e.g. cement, iron-and-steel, electricity-generation, real estate, data centre")
    asset_type: Optional[str] = None
    commissioned_year: Optional[int] = Field(None, ge=1800, le=2100, description="Construction / commissioning year")


class PhysicalRiskReportRequest(BaseModel):
    assets: list[PhysicalRiskAssetIn] = Field(..., min_length=1, max_length=25)
    client: Optional[str] = None
    title: Optional[str] = None
    include_image: bool = True


def _physical_risk_report(request: PhysicalRiskReportRequest) -> dict:
    from ..physical_report import build_portfolio_report
    rep = build_portfolio_report([a.model_dump() for a in request.assets], title=request.title, client=request.client,
                                 include_image=request.include_image)
    if not rep["assets"]:
        raise HTTPException(status_code=422, detail="; ".join(rep["failed"]) or "no site could be assessed")
    return rep


@app.get("/reports/physical-risk/dashboard", tags=["Reports"])
def physical_risk_dashboard():
    from fastapi.responses import HTMLResponse
    return HTMLResponse((Path(__file__).parent / "physical_report_dashboard.html").read_text(encoding="utf-8"))


@app.post("/reports/physical-risk", dependencies=[_AUTH], tags=["Reports"])
def physical_risk_report(request: PhysicalRiskReportRequest) -> dict:
    """Physical Risk Assessment, grouped by sector: how each site sits, which hazard zones it is in, what its
    construction year means, what will go wrong and what to do. No financial figures."""
    return _jsonable(_physical_risk_report(request))


@app.post("/reports/physical-risk.html", dependencies=[_AUTH], tags=["Reports"])
def physical_risk_report_html(request: PhysicalRiskReportRequest):
    from fastapi.responses import HTMLResponse
    from ..physical_report import render_html
    return HTMLResponse(render_html(_physical_risk_report(request)))


@app.post("/reports/physical-risk.pdf", dependencies=[_AUTH], tags=["Reports"])
def physical_risk_report_pdf(request: PhysicalRiskReportRequest):
    from fastapi.responses import Response
    from ..physical_report import render_pdf
    return Response(render_pdf(_physical_risk_report(request)), media_type="application/pdf",
                    headers={"Content-Disposition": 'attachment; filename="physical_risk_assessment.pdf"'})


# ── Asset Twin: 3D model of a real site with a hazard-replay and protection agent ──────────────────────

class AssetTwinBuildRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    name: Optional[str] = "Asset"
    sector: Optional[str] = None
    asset_type: Optional[str] = None
    commissioned_year: Optional[int] = Field(None, ge=1800, le=2100)
    radius_m: float = Field(300.0, ge=120, le=500)
    edits: Optional[list[dict]] = Field(None, description="Corrections: patch a component by id (height_m, kind, equip_height_m, "
                                                          "design_wind_ms, floor_offset_m, name, removed) or add one (x, y, w, d, height_m, kind)")
    incidents: Optional[list[dict]] = Field(None, description="Known past incidents: {date, hazard, note, depth_m}")
    include_observed: bool = True


class AssetTwinEvaluateRequest(BaseModel):
    components: list[dict]
    site: dict
    hazard: dict = Field(..., description='{"flood": {"q": [...21], "wet_share": s} | {"depth_m": x}, "wind": {"v10_ms": v}, '
                                          '"what_if": {"flood_add_m": 0, "wind_scale": 1}}')
    measures: Optional[list[dict]] = None
    linear: Optional[list[dict]] = None


class AssetTwinAnalyseRequest(BaseModel):
    twin: dict


@app.get("/asset-twin/dashboard", tags=["Asset Twin"])
def asset_twin_dashboard():
    from fastapi.responses import HTMLResponse
    return HTMLResponse((Path(__file__).parent / "asset_twin_dashboard.html").read_text(encoding="utf-8"))


@app.post("/asset-twin/build", dependencies=[_AUTH], tags=["Asset Twin"])
def asset_twin_build(request: AssetTwinBuildRequest) -> dict:
    """Build the 3D twin: real OSM footprints (or a flagged illustrative layout), terrain, flood scenarios, cyclone winds,
    storm history, and the agent's findings (problem, cause, protection, verified residual)."""
    from .. import asset_twin as T
    try:
        tw = T.build_twin(request.lat, request.lon, request.name or "Asset", request.sector, request.asset_type,
                          request.commissioned_year, request.radius_m, request.edits, request.incidents, request.include_observed)
        tw["agent"] = T.analyse(tw)
    except Exception as exc:                                    # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Could not build the twin: {type(exc).__name__}: {str(exc)[:200]}")
    return _jsonable(tw)


@app.post("/asset-twin/evaluate", dependencies=[_AUTH], tags=["Asset Twin"])
def asset_twin_evaluate(request: AssetTwinEvaluateRequest) -> dict:
    """Re-evaluate every component for one flood scenario and / or wind, with protection measures applied."""
    from .. import asset_twin as T
    return _jsonable(T.evaluate(request.components, request.site, request.hazard, request.measures, request.linear))


class AssetTwinSimulateRequest(BaseModel):
    twin: dict
    scenario: str
    add_m: float = 0.0
    barrier_m: float = 0.0


@app.post("/asset-twin/simulate", dependencies=[_AUTH], tags=["Asset Twin"])
def asset_twin_simulate(request: AssetTwinSimulateRequest) -> dict:
    """Run the 2D shallow-water solver for one mapped flood scenario on the twin's terrain and structures, optionally with a
    flood wall (barrier_m) in the model or extra water level (add_m). Returns depth frames and per-structure depth, speed, arrival."""
    from .. import asset_twin as T
    try:
        sim = T.simulate_flood(request.twin, request.scenario, request.add_m, request.barrier_m)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown flood scenario")
    except Exception as exc:                                    # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Solver could not run: {type(exc).__name__}: {str(exc)[:160]}")
    blk = sim.pop("block", None)
    sim["block_present"] = blk is not None
    return _jsonable(sim)


class ClimateScenarioRequest(BaseModel):
    lat: float
    lon: float
    ssp: str = "ssp245"
    year: int = 2070


class AssetTwinClimateRequest(BaseModel):
    twin: dict
    ssp: str = "ssp245"
    year: int = 2070
    rp: int = 100
    kind: Optional[str] = None            # auto | river | coastal
    band: str = "median"                  # low | median | high
    run_solver: bool = True


@app.post("/climate-scenario/compose", dependencies=[_AUTH], tags=["Climate Scenario"])
def climate_scenario_compose(request: ClimateScenarioRequest) -> dict:
    """One emissions pathway and year -> the physical inputs (rain ratio, river depth factor, sea-level rise, heat shift, cyclone wind scale) with a stated confidence for each."""
    from .. import climate_scenario as CS
    try:
        return _jsonable(CS.compose(request.lat, request.lon, request.ssp, request.year))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:                                    # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Scenario data unavailable: {type(exc).__name__}: {str(exc)[:160]}")


@app.post("/asset-twin/climate", dependencies=[_AUTH], tags=["Asset Twin"])
def asset_twin_climate(request: AssetTwinClimateRequest) -> dict:
    """Compose a climate scenario at the twin's location, re-run the 2D solver with the climate-shifted water level, and compare every structure now and then."""
    from .. import climate_scenario as CS
    try:
        a = request.twin["asset"]
        comp = CS.compose(a["lat"], a["lon"], request.ssp, request.year)
        res = CS.apply_to_twin(request.twin, comp, request.rp, request.kind, request.band, request.run_solver)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:                                    # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Scenario could not run: {type(exc).__name__}: {str(exc)[:160]}")
    return _jsonable(res)


@app.post("/asset-twin/analyse", dependencies=[_AUTH], tags=["Asset Twin"])
def asset_twin_analyse(request: AssetTwinAnalyseRequest) -> dict:
    """Re-run the agent on an edited twin (after the user corrected heights, equipment levels or design winds)."""
    from .. import asset_twin as T
    ag = T.analyse(request.twin)
    ag["sims"] = request.twin.get("sims")
    return _jsonable(ag)


@app.post("/reports/physical", response_model=PhysicalHazardReportResponse, dependencies=[_AUTH])
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


@app.get(
    "/reports/client/{company_id}",
    summary="Download a 4-page PDF climate risk report",
    tags=["Client Reports"],
    dependencies=[_AUTH],
    response_model=None,   # streams a PDF; a string return annotation here broke /openapi.json
)
def download_client_report(
    company_id: str,
    reporting_year: int = 2025,
    weight_profile: str = "equal",
):
    """
    Generate and stream a 4-page branded PDF climate risk assessment for a company.

    The PDF includes:
    - Page 1: Executive summary — CRI rating, pillar labels, key findings
    - Page 2: Scenario trajectory table (NZE / Delayed / Current Policies, 2025–2050)
    - Page 3: Emissions profile (Scope 1/2/3) and disclosure quality analysis
    - Page 4: Methodology and data sources

    Returns the PDF as an attachment (Content-Disposition: attachment).
    Requires: reportlab>=4.1 (included in requirements.txt).
    """
    from fastapi.responses import Response
    from ..outcomes.ratings import RatingEngine, WeightProfile
    from ..outcomes.tiers import Tier

    cid = company_id.lower()
    if cid not in COMPANY_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Company '{company_id}' not found. Use GET /companies to list available IDs.",
        )

    company = COMPANY_REGISTRY[cid]

    # Run all three scenarios for rating + trajectory
    try:
        wp = WeightProfile(weight_profile.lower())
    except ValueError:
        wp = WeightProfile.EQUAL

    nze_results = run_engine(company=company, scenario=SCENARIO_REGISTRY["nze_2050"])
    dt_results  = run_engine(company=company, scenario=SCENARIO_REGISTRY["delayed_transition"])
    cp_results  = run_engine(company=company, scenario=SCENARIO_REGISTRY["current_policies"])

    rating_engine = RatingEngine()
    rating_result = rating_engine.rate(
        company_name=company.name,
        sector=company.sector,
        nze_results=nze_results,
        dt_results=dt_results,
        cp_results=cp_results,
        data_quality=company.data_quality,
        weight_profile=wp,
    )

    # ML trajectory (best-effort — falls back gracefully)
    nze_scores = delayed_scores = cp_scores = None
    try:
        from ..ml.risk_predictor import RiskPredictor
        predictor = RiskPredictor()
        traj = predictor.predict(
            base_cri_score=rating_result.composite_score,
            sector=company.sector,
            jurisdiction=getattr(company, "hq_region", "US"),
            revenue_usd_m=company.financials.revenue,
            scope1_mt_co2e=company.assets[0].emissions.scope1_intensity * company.financials.revenue / 1_000_000
            if company.assets else 1.0,
            eal_p50_usd_m=50.0,
            wacc_adjusted=company.financials.wacc_base,
        )
        nze_scores     = traj.nze_scores
        delayed_scores = traj.delayed_scores
        cp_scores      = traj.cp_scores
    except Exception:
        pass

    # Emissions estimate (best-effort)
    scope1_mt = scope2_mt = scope3_mt = None
    try:
        from ..ml.emissions_estimator import EmissionsEstimator
        est = EmissionsEstimator()
        em = est.estimate(
            sector=company.sector,
            revenue_usd_m=company.financials.revenue,
            jurisdiction=getattr(company, "jurisdiction", None) or "US",
            company_name=company.name,
            ticker=company.id,
        )
        scope1_mt = em.scope1_mt_co2e
        scope2_mt = em.scope2_mt_co2e
        scope3_mt = em.scope3_mt_co2e
    except Exception:
        pass

    # Generate PDF
    try:
        from ..outcomes.client_report import generate_client_pdf
        pdf_bytes = generate_client_pdf(
            company_name=company.name,
            sector=company.sector,
            rating=str(rating_result.rating),
            rating_label=rating_result.rating_label,
            summary=rating_result.summary,
            physical_label=rating_result.physical.label,
            transition_label=rating_result.transition.label,
            financial_label=rating_result.financial.label,
            nze_scores=nze_scores,
            delayed_scores=delayed_scores,
            cp_scores=cp_scores,
            scope1_mt=scope1_mt,
            scope2_mt=scope2_mt,
            scope3_mt=scope3_mt,
            reporting_year=reporting_year,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in company.name)
    filename = f"CRI_Report_{safe_name}_{reporting_year}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
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
            "source": "WRI Aqueduct 4.0 dataset (Y2023M07D05, CC BY 4.0), shipped with the engine",
            "url": "https://www.wri.org/data/aqueduct-global-maps-40-data",
            "coverage": "global — 68,506 basin × admin-1 units at 0.05°; 13 indicators + industry-weighted overall risk; "
                        "future water stress 2030/2050/2080 (SSP1-2.6, SSP3-7.0, SSP5-8.5)",
            "hazards": ["water_stress", "water_depletion", "variability", "groundwater_decline", "riverine_flood",
                        "coastal_flood", "drought"],
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
    dependencies=[_AUTH],
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


# ── LLM Agent endpoints ───────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message:   str         = Field(..., description="Natural-language question or instruction")
    context:   dict | None = Field(None, description="Optional dashboard context (company, sector, etc.)")
    max_rounds: int        = Field(3, ge=1, le=5, description="Max tool-call rounds (1–5)")


class DecisionBriefRequest(BaseModel):
    company_name:  str   = Field(..., description="Company name")
    sector:        str   = Field(..., description="Sector")
    jurisdiction:  str   = Field("US", description="ISO 2-letter country code")
    revenue_usd_m: float = Field(5000.0, description="Annual revenue USD millions")


@app.post("/agent/chat", dependencies=[_AUTH], tags=["LLM Agent"])
def agent_chat(request: ChatRequest) -> dict:
    """
    Natural-language climate risk Q&A powered by Groq LLaMA-3.3-70b.

    The LLM decides which engine tools to call (assess_company, analyze_disclosure,
    get_location_hazard, compare_companies, etc.), executes them against the live
    engine, then synthesises a concise narrative answer.

    Falls back to Anthropic Claude Haiku, then a template, if Groq is unavailable.

    **Example queries:**
    - "What is the climate risk for Shell?"
    - "Compare Toyota and Ford's climate exposure"
    - "Analyse this sustainability report text: [paste text]"
    - "What is the hazard risk at 28.6°N, 77.2°E?"
    - "Explain the Delayed Transition scenario"
    """
    from ..ai.llm_agent import run_agent

    result = run_agent(
        user_message=request.message,
        context=request.context,
        max_rounds=request.max_rounds,
    )

    return {
        "narrative":        result.narrative,
        "tool_calls_made":  result.tool_calls_made,
        "raw_tool_results": result.raw_tool_results,
        "model_used":       result.model_used,
        "tokens_used":      result.tokens_used,
        "rounds":           result.rounds,
        "fallback_used":    result.fallback_used,
        "error":            result.error,
    }


@app.post("/agent/decision-brief", dependencies=[_AUTH], tags=["LLM Agent"])
def agent_decision_brief(request: DecisionBriefRequest) -> dict:
    """
    Generate a board-ready climate risk decision brief for a single company.

    Runs assess_company + Monte Carlo VaR internally, then uses the LLM to
    produce a structured narrative with: risk signal, composite score, top
    drivers, VaR, and three specific recommended actions.
    """
    from ..ai.llm_agent import get_decision_brief

    result = get_decision_brief(
        company_name=request.company_name,
        sector=request.sector,
        jurisdiction=request.jurisdiction,
        revenue_usd_m=request.revenue_usd_m,
    )

    return {
        "company":          request.company_name,
        "sector":           request.sector,
        "jurisdiction":     request.jurisdiction,
        "decision_brief":   result.narrative,
        "tool_calls_made":  result.tool_calls_made,
        "raw_engine_data":  result.raw_tool_results,
        "model_used":       result.model_used,
        "tokens_used":      result.tokens_used,
        "fallback_used":    result.fallback_used,
        "error":            result.error,
    }


@app.get("/agent/tools", tags=["LLM Agent"])
def list_agent_tools() -> dict:
    """List all tools available to the LLM agent with their descriptions."""
    from ..ai.tools import TOOL_REGISTRY
    return {
        "tools": [
            {
                "name":        name,
                "description": entry["definition"]["function"]["description"],
                "parameters":  list(
                    entry["definition"]["function"]["parameters"]
                    .get("properties", {}).keys()
                ),
            }
            for name, entry in TOOL_REGISTRY.items()
        ],
        "total": len(TOOL_REGISTRY),
    }


# ──────────────────────────────────────────────────────────────────────────────
# ASSET INTELLIGENCE — coordinate-precise physical risk
# ──────────────────────────────────────────────────────────────────────────────

class AssetAssessRequest(BaseModel):
    lat:          float  = Field(..., ge=-90,  le=90,   description="Latitude in decimal degrees")
    lon:          float  = Field(..., ge=-180, le=180,  description="Longitude in decimal degrees")
    asset_name:   str    = Field("Asset",                description="Name or description of the asset")
    asset_type:   str    = Field("industrial",           description="Asset type (industrial, warehouse, port, data_centre, agriculture, hospital, ...)")
    include_visuals: bool = Field(True,                  description="Return DEM grid, Sentinel-2 texture and overlay layers for the 3D dashboard (~400 KB)")
    radius_km:    float  = Field(2.5, ge=1.0, le=5.0,    description="Half-width of the analysis window in km")


@app.post("/asset/assess", dependencies=[_AUTH], tags=["Asset Intelligence"])
def assess_asset_endpoint(request: AssetAssessRequest) -> dict:
    """
    Evidence-based physical climate risk for a real asset location.

    Fuses three observation layers over a ~5 km × 5 km window:
    - **Terrain** — AWS terrain tiles (~10 m), pysheds D8 flow routing, slope, TPI
    - **Sentinel-2 L2A pixels** (AWS Open Data) — NDVI, MNDWI, and a multi-date
      water-frequency history that reveals observed flooding. Uses the Prithvi-EO-2.0
      flood model on a Hugging Face endpoint when CLIMRISK_PRITHVI_ENDPOINT is set.
    - **ESA WorldCover 2021** (10 m) — built-up, flammable vegetation, wetlands

    Returns findings (flood, debris flow, landslide, wildfire, urban heat) with
    severity, measured evidence + source, and remediation actions. With
    include_visuals, also returns the DEM, true-colour texture and overlay layers
    for the 3D dashboard. Typical latency 20–40 s (cached for 6 h per location).
    """
    from ..analysis.asset_intelligence import assess_asset
    return assess_asset(
        lat=request.lat,
        lon=request.lon,
        asset_name=request.asset_name,
        asset_type=request.asset_type,
        radius_km=request.radius_km,
        include_visuals=request.include_visuals,
    )


@app.get("/asset/dashboard", tags=["Asset Intelligence"])
def asset_dashboard():
    """
    Serve the Asset Intelligence 3D dashboard HTML.

    Opens an interactive single-page application with:
    - 3D terrain (Three.js) built from the 10 m AWS DEM, draped with the actual
      Sentinel-2 true-colour scene for the site
    - Overlay layers: water history, NDVI, water index, land cover, slope, flow paths
    - Evidence-backed findings and remediation actions
    - AI Deep Dive (Groq LLaMA-3.3-70b narrative via /agent/chat)

    No authentication required to load the dashboard itself.
    API key is entered in the dashboard UI and used for /asset/assess calls.
    """
    from fastapi.responses import HTMLResponse
    dashboard_path = Path(__file__).parent / "asset_dashboard.html"
    return HTMLResponse(content=dashboard_path.read_text(encoding="utf-8"))


# ──────────────────────────────────────────────────────────────────────────────
# COMPANY EXPOSURE — where a company's physical assets are, pre-screened
# ──────────────────────────────────────────────────────────────────────────────

class CompanyExposureRequest(BaseModel):
    keys: list[str] = Field(..., min_length=1, max_length=50,
                            description="Owner keys from /company/search (select all subsidiaries that apply)")


@app.get("/company/search", dependencies=[_AUTH], tags=["Company Exposure"])
def company_search(q: str, limit: int = 12) -> dict:
    """
    Find owners in the asset register (Climate TRACE global heavy industry + EPA GHGRP US).
    In-memory lookup, typically < 10 ms.
    """
    from ..data.asset_register import search, stats
    return {"query": q, "results": search(q, limit=min(max(limit, 1), 50)), "register": stats()}


@app.post("/company/exposure", dependencies=[_AUTH], tags=["Company Exposure"])
def company_exposure_endpoint(request: CompanyExposureRequest) -> dict:
    """
    All assets for the selected owners with precomputed hazard screening
    (cyclone: validated IBTrACS model; flood: low-lying elevation signal) and a
    portfolio summary. No network or modelling at request time.
    """
    from ..data.asset_register import company_exposure
    return company_exposure(request.keys)


@app.get("/asset/world-land", tags=["Asset Intelligence"])
def world_land() -> dict:
    """Natural Earth 1:110m land outline (public domain) for the offline portfolio map."""
    import json as _json
    return {"rings": _json.loads((Path(__file__).parent / "world_land_110m.json").read_text())}


# ──────────────────────────────────────────────────────────────────────────────
# RISK ANALYST — physical → policy → transition → financial distribution
# ──────────────────────────────────────────────────────────────────────────────

class AnalystAssetRequest(BaseModel):
    lat:          float = Field(..., ge=-90, le=90)
    lon:          float = Field(..., ge=-180, le=180)
    name:         Optional[str] = None
    sector:       Optional[str] = Field(None, description="Register sector, e.g. electricity-generation, cement, iron-and-steel")
    asset_type:   Optional[str] = Field(None, description="Register asset type, e.g. coal, gas, BF/BOF, integrated dry")
    capacity:     Optional[float] = Field(None, description="Capacity in register units (MW, t/yr, bbl/d)")
    emissions_t:  Optional[float] = Field(None, ge=0, description="Annual scope-1 emissions, tCO2e")
    value_usd:    Optional[float] = Field(None, gt=0, description="Replacement / book value, USD (overrides the estimate)")
    register_index: Optional[int] = Field(None, description="Asset index from /company/exposure")
    discount_rate: float = Field(0.07, ge=0.0, le=0.2)
    narrative_llm: bool = Field(False, description="Also write an LLM narrative (Groq) from the computed numbers")
    site:         Optional[dict] = Field(None, description="Optional /asset/assess result to fold satellite findings in")
    financials:   Optional[dict] = Field(None, description="Earnings layer (US$/yr, same perimeter as the emissions): "
                  "revenue_usd, ebitda_usd required; da_usd, interest_usd, net_debt_usd, book_value_usd, tax_rate, "
                  "electricity_mwh, fuel (coal|gas|oil) + fuel_gj, commodity (coal|gas|oil|none), price_realisation, "
                  "operating_leverage, sustaining_capex_usd, covenant_max_leverage, covenant_min_icr, lgd")
    commissioned_year: Optional[int] = Field(None, ge=1850, le=2100, description="Year the plant started operating — sets remaining life and undepreciated value")
    retirement_year:   Optional[int] = Field(None, ge=2000, le=2100, description="Legal phase-out or planned closure year")
    export_share_eu:   Optional[float] = Field(None, ge=0, le=1, description="Share of production (emissions) exported into the EU — drives the CBAM border charge for in-scope products")
    abatement:    bool = Field(True, description="Include the operator's abatement response (sector levers); false = cost of inaction only")
    levers:       Optional[list[dict]] = Field(None, max_length=8, description="Own abatement levers replacing the library: "
                  "[{name, share (0-1 or [low,mode,high]), mac (US$/t or [low,mode,high]), lead?, avail?, life?, opex_share?, source?}]")


class AnalystCompanyRequest(BaseModel):
    keys:          list[str] = Field(..., min_length=1, max_length=50)
    top_n:         int = Field(30, ge=1, le=80, description="Assets (by emissions) that get live flood-depth reads")
    discount_rate: float = Field(0.07, ge=0.0, le=0.2)


def _jsonable(x):
    import numpy as _np
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items() if not str(k).startswith("_")}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, _np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (_np.floating,)):
        return None if not _np.isfinite(x) else float(x)
    if isinstance(x, (_np.integer,)):
        return int(x)
    if isinstance(x, float) and x != x:
        return None
    return x


@app.post("/analyst/asset", dependencies=[_AUTH], tags=["Risk Analyst"])
def analyst_asset(request: AnalystAssetRequest) -> dict:
    """
    Four-stage Risk Analyst for one asset: physical hazards (cyclone, river and coastal
    flood with defences) → carbon-pricing policies in force → NGFS Phase 5 transition
    paths → Monte Carlo loss distribution to 2050, with findings and actions.
    Unknown sector/emissions are filled from the nearest register asset (≤ 1 km).
    Typical latency: < 1 s cached, 3–30 s on first visit (flood depths are range-read).
    """
    from ..finance.analyst import analyse_asset
    try:
        r = analyse_asset(request.lat, request.lon, name=request.name, sector=request.sector,
                          atype=request.asset_type, capacity=request.capacity, emissions_t=request.emissions_t,
                          value_usd=request.value_usd, register_index=request.register_index, site=request.site,
                          discount=request.discount_rate, narrative_llm=request.narrative_llm,
                          financials=request.financials, abatement_on=request.abatement, levers=request.levers,
                          commissioned_year=request.commissioned_year, retirement_year=request.retirement_year,
                          export_share_eu=request.export_share_eu)
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc).strip("'\""))
    return _jsonable(r)


@app.post("/analyst/company", dependencies=[_AUTH], tags=["Risk Analyst"])
def analyst_company(request: AnalystCompanyRequest) -> dict:
    """Run every register asset of the selected owners through the analyst and aggregate the distribution."""
    from ..finance.analyst import analyse_company
    return _jsonable(analyse_company(request.keys, top_n=request.top_n, discount=request.discount_rate))


# ══════════════════════════════════════════════════════════════════════════════
# EXTENDED API — ported from the earlier engine build so every endpoint the website
# and the desktop app reference is served by this (local) engine. Each route below was
# re-tested after the port (tests/test_extended_api.py).
# ══════════════════════════════════════════════════════════════════════════════

from datetime import datetime, timezone
from .schemas import (  # noqa: E402  — schemas used by the ported endpoints
    AssetAssessmentResult, ArticleRequest, ArticleResponse, BatchAssessmentRequest, BatchAssessmentResponse,
    CreateCompanyRequest, DecisionBrief, DecisionRequest, FinancialImpact, NarrativeRequest, NarrativeResponse,
    PortfolioCreditRequest, PortfolioCreditResponse, PortfolioPositionIn, PortfolioRiskRequest,
    PortfolioRiskResponse, PortfolioStreamRequest, RecommendedAction, RiskDriver, RiskSignal,
)

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
            commodity=Commodity.MANUFACTURING,   # generic proxy (Commodity.OTHER does not exist)
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
                    asset_rcn = company.financials.revenue * 0.05   # revenue is stored in USD m (was an undefined name)
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
    brief: DecisionBrief = get_decision_brief(dec_request)   # was an undefined coroutine name
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
    brief: DecisionBrief = get_decision_brief(dec_request)   # was an undefined coroutine name
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
    positions:      list[PortfolioPositionIn]
    benchmark_name: str = Field("Benchmark", description="Label only, e.g. 'MSCI World (factsheet 2026-08)'")
    benchmark_sector_weights: dict[str, float] = Field(
        ..., description="Benchmark sector weights from the index provider's factsheet, keyed by the same sector "
                         "names as the positions (weights are normalised to 100%)")


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


@app.post("/portfolio/benchmark", tags=["Portfolio — Benchmark & Counterparty"])
def portfolio_benchmark_endpoint(req: BenchmarkRequest):
    """
    Active climate risk of a portfolio against a benchmark, both measured with the same ClimRisk model.

    The portfolio model is sector × region based (company scores do not enter it; for asset-level risk use
    /analyst/company). The benchmark is represented by its sector weights, supplied from the index factsheet,
    at global region exposure. Active risk is split into
      * sector allocation — portfolio sector weights vs benchmark sector weights (both at global exposure);
      * regional exposure — the positions' own regions vs global exposure, at portfolio sector weights.
    No third-party climate VaR figures are used.
    """
    from ..climate.portfolio import aggregate_portfolio_risk, PortfolioPosition
    if not req.positions:
        raise HTTPException(status_code=422, detail="At least one position required")
    total = sum(p.exposure_usd_m for p in req.positions)
    if total <= 0:
        raise HTTPException(status_code=422, detail="Total exposure must be > 0")
    bw_raw = {k: float(v) for k, v in req.benchmark_sector_weights.items() if v and v > 0}
    bsum = sum(bw_raw.values())
    if bsum <= 0:
        raise HTTPException(status_code=422, detail="benchmark_sector_weights must contain positive weights")
    bw = {k: v / bsum for k, v in bw_raw.items()}

    def run(ps: list) -> dict:
        r = aggregate_portfolio_risk(ps, run_stress=False)
        t = r.total_exposure_usd_m or 1.0
        return {"eal_current_policies_pct": 100 * r.portfolio_eal_cp_usd_m / t,
                "var99_current_policies_pct": 100 * r.portfolio_var99_cp_usd_m / t,
                "cvar99_current_policies_pct": 100 * r.portfolio_cvar99_cp_usd_m / t,
                "eal_net_zero_pct": 100 * r.portfolio_eal_nze_usd_m / t,
                "var99_net_zero_pct": 100 * r.portfolio_var99_nze_usd_m / t}

    def pos(i, name, exp, sector, region="Global", ph=50.0, tr=50.0):
        return PortfolioPosition(company_id=str(i), company_name=name, exposure_usd_m=exp, sector=sector,
                                 region=region, physical_score=ph, transition_score=tr)

    port = [PortfolioPosition(company_id=p.company_id, company_name=p.company_name, exposure_usd_m=p.exposure_usd_m,
                              sector=p.sector, region=p.region, physical_score=p.physical_score,
                              transition_score=p.transition_score, policy_score=p.policy_score,
                              technology_score=p.technology_score, market_score=p.market_score,
                              litigation_score=p.litigation_score) for p in req.positions]
    pw: dict[str, float] = {}
    for p in req.positions:
        pw[p.sector] = pw.get(p.sector, 0.0) + p.exposure_usd_m / total
    neutral_port = [pos(k, f"neutral {k}", w * total, k) for k, w in pw.items()]
    bench = [pos(k, f"benchmark {k}", w * total, k) for k, w in bw.items()]

    P, N, B = run(port), run(neutral_port), run(bench)
    keys = list(P)
    r3 = lambda d: {k: round(v, 3) for k, v in d.items()}
    sectors = sorted(set(pw) | set(bw))
    return {
        "benchmark": req.benchmark_name,
        "total_exposure_usd_m": round(total, 2),
        "portfolio": r3(P), "benchmark_metrics": r3(B),
        "active": r3({k: P[k] - B[k] for k in keys}),
        "attribution": {"sector_allocation": r3({k: N[k] - B[k] for k in keys}),
                        "regional_exposure": r3({k: P[k] - N[k] for k in keys})},
        "active_climate_var99_usd_m": round((P["var99_current_policies_pct"] - B["var99_current_policies_pct"]) / 100 * total, 2),
        "sector_tilt_table": [{"sector": k, "portfolio_weight_pct": round(100 * pw.get(k, 0.0), 2),
                               "benchmark_weight_pct": round(100 * bw.get(k, 0.0), 2),
                               "active_weight_pct": round(100 * (pw.get(k, 0.0) - bw.get(k, 0.0)), 2)} for k in sectors],
        "methodology": ("Portfolio and benchmark are both run through the ClimRisk portfolio model "
                        "(sector/region hazard exposure, Gaussian-copula Monte Carlo, current-policies and net-zero "
                        "pathways) — a sector × region model; company scores do not change it. Benchmark = supplied "
                        "sector weights at global exposure. Active = sector allocation + regional exposure. Metrics are "
                        "% of exposure. No third-party climate VaR values are used."),
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

    from ..climate.portfolio import aggregate_portfolio_risk, PortfolioPosition
    ps = [PortfolioPosition(company_id=f"{i}", company_name=p.counterparty_name, exposure_usd_m=p.exposure_usd_m,
                            sector=p.sector_key, region=p.region) for i, p in enumerate(req.positions)]
    res = aggregate_portfolio_risk(ps, run_stress=False)
    by_pos = {r.company_id: r for r in res.positions}
    for i, p in enumerate(req.positions):
        key = (p.sector_key if req.group_by == "sector" else p.region if req.group_by == "region" else p.counterparty_id)
        r = by_pos.get(f"{i}")
        g = groups[key]
        g["eal"] = g.get("eal", 0.0) + (r.eal_cp_usd_m if r else 0.0)
        g["cvar99"] = g.get("cvar99", 0.0) + (r.mc_cvar99_usd_m if r else 0.0)
    rows = []
    hhi = 0.0
    for key, grp in groups.items():
        wt = grp["exposure"] / total_exp
        hhi += wt ** 2
        rows.append({
            "group_key":         key,
            "label":             grp["label"],
            "exposure_usd_m":    round(grp["exposure"], 2),
            "weight_pct":        round(wt * 100, 2),
            "expected_annual_loss_usd_m": round(grp.get("eal", 0.0), 3),
            "expected_annual_loss_pct":   round(100 * grp.get("eal", 0.0) / grp["exposure"], 3) if grp["exposure"] else 0.0,
            "cvar99_contribution_usd_m":  round(grp.get("cvar99", 0.0), 3),
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
        "portfolio_cvar99_usd_m": round(res.portfolio_cvar99_cp_usd_m, 3),
        "methodology": (
            "Herfindahl-Hirschman Index (HHI) for concentration. Expected annual loss and CVaR99 contributions "
            "(current-policies pathway) from the ClimRisk portfolio model (sector × region, Gaussian-copula Monte Carlo). "
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
    import numpy as _np
    from cri.finance import transition as _tr

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
            "slr_2050_m":        slr.slr_2050_median_m,
            "slr_2100_m":        slr.slr_2100_median_m,
            "inundation_prob_2050": slr.p_inundation_2050,
            "inundation_prob_2100": slr.p_inundation_2100,
            "risk_tier":         slr.risk_tier,
            "expected_loss_2050_usd_m": slr.expected_loss_2050_usd_m,
        }

    # Transition: NGFS Phase 5 carbon-price paths (median of GCAM / MESSAGEix / REMIND, world
    # region, US$2024) applied to the stated Scope 1 emissions, held flat (no abatement).
    # (The earlier build called a function that never existed; this is the real calculation.)
    _drv = _tr.drivers(None, req.sector_key, "", None)
    _names = {"net_zero_2050": "Net Zero 2050", "delayed_transition": "Delayed transition",
              "current_policies": "Current Policies"}
    _yrs = _np.array(_drv["years"]); _disc = 1.0 / 1.07 ** (_yrs - 2025)
    transition_scenarios = {}
    for scenario, ngfs in _names.items():
        P = _np.median(_np.array(list(_drv["price"][ngfs].values())), axis=0)
        cost = req.scope1_emissions_mt_co2e * 1e6 * P / 1e6            # USD m per year
        pv = float((cost * _disc).sum())
        transition_scenarios[scenario] = {
            "carbon_price_2030_usd_t": round(float(P[_yrs.tolist().index(2030)]), 1),
            "carbon_price_2050_usd_t": round(float(P[-1]), 1),
            "carbon_cost_2030_usd_m":  round(float(cost[_yrs.tolist().index(2030)]), 1),
            "carbon_cost_2050_usd_m":  round(float(cost[-1]), 1),
            "carbon_cost_pv_2026_2050_usd_m": round(pv, 1),
            "carbon_cost_pv_pct_ev":   round(100 * pv / max(req.ev_usd_m, 1e-9), 1),
            "basis": "NGFS Phase 5 world carbon price (median of 3 models), Scope 1 held flat, gross of free allocation and pass-through, 7% real discount",
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

class SFDRExportRequest(SFDRRequest):
    """Same inputs as /regulatory/sfdr-pai, plus the file format."""
    file_format: str = Field("xlsx", description="'xlsx' or 'csv'")


@app.post("/regulatory/sfdr-pai/export", tags=["Regulatory"])
async def sfdr_pai_export(req: SFDRExportRequest):
    """
    Download the SFDR PAI pack (all 18 mandatory indicators, SFDR RTS Annex I Table 1)
    as an Excel workbook or CSV — the same numbers /regulatory/sfdr-pai returns.
    """
    import csv
    import io
    from fastapi.responses import StreamingResponse
    from ..regulatory.sfdr_pai import compute_sfdr_pai
    pai = compute_sfdr_pai(
        company_id=req.company_id, company_name=req.company_name, sector_key=req.sector_key,
        revenue_eur_m=req.revenue_eur_m, ev_eur_m=req.ev_eur_m, outstanding_loan_eur_m=req.outstanding_loan_eur_m,
        scope1_tco2e=req.scope1_tco2e, scope2_tco2e=req.scope2_tco2e, scope3_tco2e=req.scope3_tco2e,
        has_carbon_reduction_target=req.has_carbon_reduction_target, board_female_pct=req.board_female_pct,
        gender_pay_gap_pct=req.gender_pay_gap_pct, ungc_compliant=req.ungc_compliant,
        reference_period=req.reference_period,
    )
    header = ["PAI #", "Indicator", "Value", "Unit", "Flag", "Data quality (1-5)", "Needs additional data", "Note"]
    rows = [[i.pai_id, i.name, i.value, i.unit, i.flag, i.data_quality, i.requires_additional_data, i.note]
            for i in pai.indicators]
    stem = f"SFDR_PAI_{req.company_name.replace(' ', '_')}_{req.reference_period}"
    if req.file_format.lower() == "csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow([f"SFDR RTS Annex I Table 1 — {req.company_name} — reference period {req.reference_period}"])
        w.writerow(header)
        w.writerows(rows)
        w.writerow([])
        w.writerow(["Data limitations"])
        for d in pai.data_limitations:
            w.writerow([d])
        w.writerow(["Methodology", pai.methodology_note])
        return StreamingResponse(io.BytesIO(buf.getvalue().encode("utf-8")), media_type="text/csv",
                                 headers={"Content-Disposition": f'attachment; filename="{stem}.csv"'})
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    wb = Workbook(); ws = wb.active; ws.title = "PAI indicators"
    ws["A1"] = "SFDR RTS Annex I — Principal Adverse Impact indicators"; ws["A1"].font = Font(bold=True, size=12)
    ws["A2"] = (f"Company: {req.company_name} | Sector: {req.sector_key} | Reference period: {req.reference_period} | "
                f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d')} by the ClimRisk engine")
    ws["A2"].font = Font(italic=True, size=9)
    for j, h in enumerate(header, start=1):
        cell = ws.cell(row=4, column=j, value=h)
        cell.font = Font(bold=True, color="FFFFFF"); cell.fill = PatternFill("solid", fgColor="0B1220")
    for r_i, row in enumerate(rows, start=5):
        for j, v in enumerate(row, start=1):
            ws.cell(row=r_i, column=j, value=v).alignment = Alignment(wrap_text=j in (2, 8), vertical="top")
    for col, wdt in zip("ABCDEFGH", (7, 48, 14, 18, 8, 12, 12, 60)):
        ws.column_dimensions[col].width = wdt
    ws2 = wb.create_sheet("Limitations & method")
    ws2["A1"] = "Data limitations"; ws2["A1"].font = Font(bold=True)
    for k, d in enumerate(pai.data_limitations, start=2):
        ws2.cell(row=k, column=1, value=d)
    k = len(pai.data_limitations) + 3
    ws2.cell(row=k, column=1, value="Methodology").font = Font(bold=True)
    ws2.cell(row=k + 1, column=1, value=pai.methodology_note)
    ws2.column_dimensions["A"].width = 120
    buf = io.BytesIO(); wb.save(buf); buf.seek(0)
    return StreamingResponse(buf, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                             headers={"Content-Disposition": f'attachment; filename="{stem}.xlsx"'})


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
    aqueduct_score:  Optional[float] = Field(None, ge=0, le=5, description="WRI Aqueduct 4.0 water risk score (0-5); omit and give lat/lon to look it up")
    lat:             Optional[float] = Field(None, ge=-90, le=90, description="Site latitude — looks up the Aqueduct 4.0 sector-weighted overall water risk")
    lon:             Optional[float] = Field(None, ge=-180, le=180)
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
    score, looked_up = req.aqueduct_score, None
    if score is None:
        if req.lat is None or req.lon is None:
            raise HTTPException(status_code=422, detail="Give aqueduct_score, or lat and lon to look it up")
        from ..connectors.wri_aqueduct import point as _aq_point
        looked_up = _aq_point(req.lat, req.lon, req.sector)
        ow = looked_up.get("overall_water_risk") or {}
        score = ow.get("sector_score") if ow.get("sector_score") is not None else ow.get("score")
        if score is None:
            raise HTTPException(status_code=422, detail="No Aqueduct 4.0 overall water risk score at this location")
    result = assess_water_stress(
        company_name=req.company_name,
        sector=req.sector,
        aqueduct_score=score,
        revenue_usd_m=req.revenue_usd_m,
        ev_usd_m=req.ev_usd_m,
        wacc=req.wacc,
        horizon=req.horizon,
        scenario=req.scenario,
        water_intensity=req.water_intensity,
    )
    import dataclasses
    out = dataclasses.asdict(result)
    if looked_up is not None:
        out["aqueduct_lookup"] = {"score_used": score, "weighting": looked_up["overall_water_risk"].get("sector_weighting")
                                  or "Default weighting", "admin1": looked_up.get("admin1"), "iso3": looked_up.get("iso3"),
                                  "source": looked_up.get("source")}
    return out


@app.get("/physical/water-risk", tags=["Physical Risk"])
def physical_water_risk(lat: float, lon: float, sector: Optional[str] = None) -> dict:
    """WRI Aqueduct 4.0 at a coordinate: 13 baseline indicators, overall and industry-weighted water risk,
    and future water stress / depletion / variability for 2030, 2050, 2080 under three scenarios."""
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise HTTPException(status_code=422, detail="lat/lon out of range")
    from ..connectors.wri_aqueduct import point as _aq_point
    return _aq_point(lat, lon, sector)



# ──────────────────────────────────────────────────────────────────────────────
# LIFE CYCLE ASSESSMENT — ISO 14067 product carbon footprints and study reports (cri.lca)
# ──────────────────────────────────────────────────────────────────────────────

class LCAConstituentIn(BaseModel):
    name: str
    t_per_t_cement: float = Field(..., ge=0, le=1)
    kgco2e_per_t: Optional[float] = Field(None, description="Upstream factor; omit to declare the flow as excluded")
    source: str = ""
    dq: int = Field(3, ge=1, le=5)


class LCATransportIn(BaseModel):
    mode: str = Field(..., pattern="^(truck|rail|ship)$")
    t: float = Field(..., ge=0, description="tonnes transported per declared unit")
    km: float = Field(..., ge=0)
    dq: int = Field(3, ge=1, le=5)
    label: str = ""


class LCACementRequest(BaseModel):
    name: str = "Cement plant"
    country: Optional[str] = Field(None, description="ISO3; or give lat/lon")
    lat: Optional[float] = None
    lon: Optional[float] = None
    clinker_ratio: float = Field(..., gt=0, le=1, description="t clinker per t cement")
    kiln_thermal_gj_per_t_clinker: float = Field(..., gt=0, le=10)
    fuel_mix: dict[str, float] = Field(..., description="Share of kiln thermal energy by fuel key (sums to 1); see GET /lca/factors")
    kwh_per_t_clinker: float = Field(..., ge=0, description="kiln + raw mill electricity")
    kwh_per_t_cement_grinding: float = Field(..., ge=0)
    clinker_cao: Optional[float] = Field(None, ge=0, le=0.8)
    clinker_mgo: Optional[float] = Field(None, ge=0, le=0.1)
    ckd_t_per_t_clinker: float = Field(0.0, ge=0, le=0.3)
    ckd_calcination: float = Field(1.0, ge=0, le=1)
    raw_meal_toc: Optional[float] = Field(None, ge=0, le=0.02)
    raw_meal_per_t_clinker: Optional[float] = Field(None, ge=1.2, le=2.0)
    electricity_g_per_kwh: Optional[float] = Field(None, ge=0, description="Supplier-specific factor (PPA); overrides grid")
    grinding_fuels_gj_per_t_cement: dict[str, float] = {}
    constituents: list[LCAConstituentIn] = []
    transport: list[LCATransportIn] = []
    capture_rate: float = Field(0.0, ge=0, le=1)
    dq: dict[str, int] = Field(default_factory=dict, description="data quality 1–5 for composition, fuels, electricity")
    plan_target_year: Optional[int] = Field(None, ge=2026, le=2050)
    plan_clinker_ratio: Optional[float] = Field(None, gt=0, le=1)
    plan_thermal_gj_per_t_clinker: Optional[float] = Field(None, gt=0, le=10)
    plan_biomass_share: Optional[float] = Field(None, ge=0, le=1)
    plan_capture_rate: Optional[float] = Field(None, ge=0, le=1)
    plan_capture_start_year: Optional[int] = Field(None, ge=2026, le=2050)
    plan_electricity_g_per_kwh: Optional[float] = Field(None, ge=0)
    cbam_year: int = Field(2026, ge=2026, le=2040)


@app.get("/lca/factors", tags=["Life Cycle Assessment"])
def lca_factors() -> dict:
    """The emission-factor database behind the LCA module, with every source."""
    from ..lca.engine import factors
    f = factors()
    return {"version": f["version"], "gwp": f["gwp"], "fuels": f["fuels"], "freight": f["freight"],
            "process_defaults": f["process"],
            "grid": {"source": f["grid"]["source"], "world_g_per_kwh": f["grid"]["world_g_per_kwh"],
                     "countries": {k: v["g_per_kwh"] for k, v in f["grid"]["countries"].items()}},
            "grid_pathways_source": f["grid_index"]["source"]}


@app.post("/lca/cement", dependencies=[_AUTH], tags=["Life Cycle Assessment"])
def lca_cement(request: LCACementRequest) -> dict:
    """Cradle-to-gate carbon footprint of cement & clinker (ISO 14067 methodology), 2025–2050 trajectory and CBAM view."""
    from ..lca.cement import CementPlant, Constituent, assess
    from ..lca.engine import TransportLeg
    d = request.model_dump()
    d["constituents"] = [Constituent(**c) for c in d["constituents"]]
    d["transport"] = [TransportLeg(**t) for t in d["transport"]]
    cy = d.pop("cbam_year")
    try:
        return _jsonable(assess(CementPlant(**d), cbam_year=cy))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


class LCASteelRequest(BaseModel):
    route: str = Field(..., pattern="^(bf_bof|eaf|dri_eaf)$")
    kwh_per_t: float = Field(..., ge=0)
    country: Optional[str] = None
    g_per_kwh: Optional[float] = Field(None, ge=0)
    extra_fuels: dict[str, float] = {}


class LCAAluminiumRequest(BaseModel):
    kwh_per_t: float = Field(..., ge=0)
    country: Optional[str] = None
    g_per_kwh: Optional[float] = Field(None, ge=0)
    alumina_t: Optional[float] = Field(None, ge=0)
    alumina_kgco2e_per_t: Optional[float] = Field(None, ge=0)
    pfc_kgco2e_per_t: Optional[float] = Field(None, ge=0)


class LCAAmmoniaRequest(BaseModel):
    kwh_per_t: float = Field(..., ge=0)
    country: Optional[str] = None
    g_per_kwh: Optional[float] = Field(None, ge=0)
    gas_gj_per_t: Optional[float] = Field(None, gt=0)
    co2_recovered_for_urea_kg_per_t: float = Field(0.0, ge=0)


class LCACustomRequest(BaseModel):
    declared_unit: str
    country: Optional[str] = None
    fuels: list[dict] = []
    electricity: list[dict] = []
    process: list[dict] = []
    materials: list[dict] = []
    transport: list[dict] = []
    capture_rate: float = Field(0.0, ge=0, le=1)
    exclusions: list[str] = []


def _lca_call(fn, **kw):
    try:
        return _jsonable(fn(**kw))
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/lca/steel", dependencies=[_AUTH], tags=["Life Cycle Assessment"])
def lca_steel(request: LCASteelRequest) -> dict:
    from ..lca.products import steel
    return _lca_call(steel, **request.model_dump())


@app.post("/lca/aluminium", dependencies=[_AUTH], tags=["Life Cycle Assessment"])
def lca_aluminium(request: LCAAluminiumRequest) -> dict:
    from ..lca.products import aluminium
    return _lca_call(aluminium, **request.model_dump())


@app.post("/lca/ammonia", dependencies=[_AUTH], tags=["Life Cycle Assessment"])
def lca_ammonia(request: LCAAmmoniaRequest) -> dict:
    from ..lca.products import ammonia
    return _lca_call(ammonia, **request.model_dump())


@app.post("/lca/custom", dependencies=[_AUTH], tags=["Life Cycle Assessment"])
def lca_custom(request: LCACustomRequest) -> dict:
    from ..lca.products import custom
    return _lca_call(custom, **request.model_dump())


class LCAStudyRequest(BaseModel):
    product: str = Field(..., pattern="^(cement|steel|aluminium|ammonia|custom)$")
    params: dict = Field(..., description="Same body as POST /lca/<product>")
    study: dict = Field(default_factory=dict, description="ISO 14067 study options: boundary, reference_year, distribution, "
                        "use, end_of_life, allocation, cutoff, luc_kgco2e, biogenic_uptake_share, commissioner, practitioner…")
    n: int = Field(5000, ge=500, le=20000)


def _lca_study(request: LCAStudyRequest):
    from pydantic import ValidationError
    from ..lca import cfp, products
    from ..lca.cement import CementPlant, Constituent, inventory as cement_inventory
    from ..lca.engine import TransportLeg
    try:
        p, prod = request.params, request.product
        if prod == "cement":
            d = LCACementRequest(**p).model_dump()
            d["constituents"] = [Constituent(**c) for c in d["constituents"]]
            d["transport"] = [TransportLeg(**t) for t in d["transport"]]
            d.pop("cbam_year")
            plant = CementPlant(**d)
            inv, country = cement_inventory(plant), plant.resolved_country()
        elif prod == "steel":
            m = LCASteelRequest(**p)
            inv, _ = products.steel_inventory(**m.model_dump())
            country = m.country
        elif prod == "aluminium":
            m = LCAAluminiumRequest(**p)
            inv, _ = products.aluminium_inventory(**m.model_dump())
            country = m.country
        elif prod == "ammonia":
            m = LCAAmmoniaRequest(**p)
            inv, _ = products.ammonia_inventory(**m.model_dump())
            country = m.country
        else:
            m = LCACustomRequest(**p)
            inv, _ = products.custom_inventory(**m.model_dump())
            country = m.country
        result = cfp.run_study(inv, request.study, prod, country, n=request.n)
        return result
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors(include_url=False, include_context=False))
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc).strip("'\""))


@app.get("/lca/eol-options", tags=["Life Cycle Assessment"])
def lca_eol_options() -> dict:
    """End-of-life materials and treatments with a published DESNZ factor, for the use/end-of-life stages."""
    from ..lca.cfp import eol_options
    return eol_options()


@app.post("/lca/study", dependencies=[_AUTH], tags=["Life Cycle Assessment"])
def lca_study(request: LCAStudyRequest) -> dict:
    """ISO 14067 carbon-footprint study: boundary A1–A3 / A1–A4 / A1–C, distribution, use stage, end of life, allocation,
    cut-off, GHG-type split, uncertainty, sensitivity, hotspots, data quality (§6.3.5)."""
    return _jsonable(_lca_study(request))


@app.post("/lca/study/report", dependencies=[_AUTH], tags=["Life Cycle Assessment"])
def lca_study_report(request: LCAStudyRequest) -> dict:
    """The study plus its ISO 14067 clause 9 report (sections + Markdown)."""
    from ..lca.cfp import report
    res = _lca_study(request)
    return _jsonable({"study": res, "report": report(res, request.study)})


@app.post("/lca/study/report.xlsx", dependencies=[_AUTH], tags=["Life Cycle Assessment"])
def lca_study_report_xlsx(request: LCAStudyRequest):
    from fastapi.responses import Response
    from ..lca.cfp import report, report_xlsx
    res = _lca_study(request)
    return Response(report_xlsx(res, report(res, request.study)),
                    media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": f'attachment; filename="cfp_study_{request.product}.xlsx"'})


@app.get("/lca/dashboard", tags=["Life Cycle Assessment"])
def lca_dashboard():
    """Product carbon footprint (LCA) workbench — served by the local engine."""
    from fastapi.responses import HTMLResponse
    return HTMLResponse((Path(__file__).parent / "lca_dashboard.html").read_text(encoding="utf-8"))



# ──────────────────────────────────────────────────────────────────────────────
# SBTi target alignment & deviation alerts
# ──────────────────────────────────────────────────────────────────────────────

class SBTiAlignmentRequest(BaseModel):
    company_name: str = ""
    scope: str = Field("scope12", pattern="^(scope12|scope3)$")
    base_year: int = Field(..., ge=2000, le=2035)
    base_emissions_t: float = Field(..., gt=0, description="Base-year emissions in the target boundary, tCO2e")
    target_year: int = Field(..., ge=2020, le=2060)
    target_reduction_pct: float = Field(..., gt=0, le=100)
    reported: dict[int, float] = Field(default_factory=dict, description="Reported emissions by year, tCO2e")
    net_zero_year: Optional[int] = Field(None, ge=2030, le=2070)
    net_zero_reduction_pct: Optional[float] = Field(None, gt=0, le=100)
    submission_year: Optional[int] = Field(None, ge=2015, le=2040)
    tolerance_pct: float = Field(5.0, ge=0, le=50)


@app.post("/sbti/alignment", tags=["Transition Risk"])
def sbti_alignment(req: SBTiAlignmentRequest) -> dict:
    """Check a target against the SBTi absolute-contraction minimum and raise deviation alerts from reported emissions."""
    from ..climate.sbti_alignment import check
    try:
        out = check(req.base_year, req.base_emissions_t, req.target_year, req.target_reduction_pct, req.scope,
                    req.reported, req.net_zero_year, req.net_zero_reduction_pct, req.submission_year, req.tolerance_pct)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    out["company_name"] = req.company_name
    return out



# ──────────────────────────────────────────────────────────────────────────────
# Consolidated portfolio export — every asset through the Risk Analyst, one CSV
# ──────────────────────────────────────────────────────────────────────────────

class PortfolioExportAsset(BaseModel):
    name: str
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    sector: Optional[str] = None
    type: Optional[str] = None
    capacity: Optional[float] = None
    emissions_t: Optional[float] = None
    value_usd: Optional[float] = None
    company: Optional[str] = None


class PortfolioExportRequest(BaseModel):
    assets: list[PortfolioExportAsset] = Field(..., min_length=1, max_length=500)
    draws: int = Field(2000, ge=200, le=10000)


@app.post("/portfolio/export.csv", dependencies=[_AUTH], tags=["Portfolio"])
def portfolio_export_csv(req: PortfolioExportRequest):
    """Run every asset through the Risk Analyst and return one consolidated CSV (one row per asset)."""
    from fastapi.responses import Response
    from ..portfolio_export import analyse_rows, to_csv
    rows = [dict(a.model_dump(), row=i + 1, error=None) for i, a in enumerate(req.assets)]
    csv_text = to_csv(analyse_rows(rows, n=req.draws))
    return Response(content=csv_text, media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="climrisk_portfolio.csv"'})


# ──────────────────────────────────────────────────────────────────────────────
# CARBON ACCOUNTING — GHG inventory (GHG Protocol / ISO 14064-1) and assurance pack
# ──────────────────────────────────────────────────────────────────────────────

def _carbon_calc(payload: dict):
    from ..carbon.inventory import InventoryError, calculate, from_dict
    try:
        inv = from_dict(payload)
        return inv, calculate(inv)
    except (InventoryError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc).strip("'\""))


@app.get("/carbon/factors", tags=["Carbon Accounting"])
def carbon_factors(q: str = "", scope: Optional[str] = None, limit: int = 50) -> dict:
    """Search the emission-factor library (UK DESNZ 2026 activity factors)."""
    from ..carbon.factors import lib, search
    L = lib()
    return {"results": search(q, scope, min(max(limit, 1), 500)),
            "library": {"version": L["version"], "desnz": L["desnz"]["source"],
                        "electricity": ["UK DESNZ 2026", L["electricity"]["us_egrid"]["source"],
                                        L["electricity"]["us_egrid"]["residual_mix"]["source"],
                                        L["electricity"]["europe_aib"]["source"], L["electricity"]["ember_derived"]["source"],
                                        L["electricity"]["td_losses"]["source"]],
                        "spend": L["spend"]["source"]}}


@app.get("/carbon/grid-factor", tags=["Carbon Accounting"])
def carbon_grid_factor(country: str, us_state: Optional[str] = None, egrid_subregion: Optional[str] = None) -> dict:
    """Location-based and residual-mix grid factors for a country / US state / eGRID subregion."""
    from ..carbon import factors as F
    try:
        loc = F.electricity_location(country, us_state, egrid_subregion)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip("'\""))
    return {"location": loc, "residual_mix": F.electricity_residual(country, egrid_subregion or loc.get("subregion")),
            "td_losses": F.td_loss_share(country, egrid_subregion or loc.get("subregion"))}


@app.post("/carbon/inventory", dependencies=[_AUTH], tags=["Carbon Accounting"])
def carbon_inventory(payload: dict) -> dict:
    """Calculate a corporate GHG inventory from activity data (Scope 1, Scope 2 location + market, Scope 3 by category)."""
    _, res = _carbon_calc(payload)
    return res


@app.post("/carbon/inventory/report", dependencies=[_AUTH], tags=["Carbon Accounting"])
def carbon_inventory_report(payload: dict) -> dict:
    """ISO 14064-1:2018 clause 9 GHG report content for the inventory."""
    from ..carbon.assurance import report_sections
    inv, res = _carbon_calc(payload)
    return {"report": report_sections(res, inv), "totals_t": res["totals_t"], "fingerprint": res["fingerprint"]}


@app.post("/carbon/inventory/assurance.xlsx", dependencies=[_AUTH], tags=["Carbon Accounting"])
def carbon_inventory_assurance(payload: dict):
    """Assurance pack: audit trail from every input to every reported number, factors, completeness and checks."""
    from fastapi.responses import Response
    from ..carbon.assurance import assurance_workbook
    inv, res = _carbon_calc(payload)
    name = f"GHG_inventory_{inv.organisation.replace(' ', '_')[:40]}_{inv.reporting_year}.xlsx"
    return Response(content=assurance_workbook(res, inv),
                    media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/carbon/template.xlsx", tags=["Carbon Accounting"])
def carbon_template():
    """Blank intake workbook (entities, activities, electricity, spend, supplier data, exclusions)."""
    from fastapi.responses import Response
    from ..carbon.assurance import template_workbook
    return Response(content=template_workbook(),
                    media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": 'attachment; filename="ClimRisk_GHG_inventory_template.xlsx"'})


@app.post("/carbon/inventory/upload", dependencies=[_AUTH], tags=["Carbon Accounting"])
async def carbon_inventory_upload(file: UploadFile = File(...), organisation: str = Form(...),
                                  reporting_year: int = Form(...), consolidation: str = Form("operational_control"),
                                  gwp: str = Form("AR6"), revenue_musd: Optional[float] = Form(None),
                                  employees: Optional[float] = Form(None)) -> dict:
    """Parse a filled intake template and return {payload, result} (the payload can be edited and re-posted)."""
    from ..carbon.assurance import parse_workbook
    try:
        payload = parse_workbook(await file.read(), organisation, reporting_year, consolidation, gwp, revenue_musd, employees)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"could not read the workbook: {exc}")
    _, res = _carbon_calc(payload)
    return {"payload": payload, "result": res}


class BaseYearRequest(BaseModel):
    base_year_total_t: float = Field(..., gt=0)
    changes: list[dict] = Field(default_factory=list, description="[{type, description, base_year_effect_t}]")
    threshold: float = Field(0.05, gt=0, lt=1)


@app.post("/carbon/base-year", tags=["Carbon Accounting"])
def carbon_base_year(req: BaseYearRequest) -> dict:
    """GHG Protocol base-year recalculation test for structural and methodology changes."""
    from ..carbon.inventory import base_year_recalculation
    return base_year_recalculation(req.base_year_total_t, req.changes, req.threshold)


@app.get("/carbon/dashboard", tags=["Carbon Accounting"])
def carbon_dashboard():
    """Carbon accounting workbench — served by the local engine."""
    from fastapi.responses import HTMLResponse
    return HTMLResponse((Path(__file__).parent / "carbon_dashboard.html").read_text(encoding="utf-8"))


# ──────────────────────────────────────────────────────────────────────────────
# PUBLIC live company check (website widget) — Risk Analyst on the company's mapped assets
# ──────────────────────────────────────────────────────────────────────────────

_PUBLIC_CACHE: dict = {}


@app.get("/public/company-risk", tags=["Risk Analyst"])
def public_company_risk(q: str) -> dict:
    """Find a company in the global asset register and run its largest assets through the Risk Analyst.
    Returns only what the model actually computed; companies without mapped assets get found=false."""
    import time as _t
    from ..data.asset_register import search
    from ..finance.analyst import analyse_company
    q = (q or "").strip()
    if len(q) < 3:
        raise HTTPException(status_code=422, detail="Type at least 3 characters")
    hits = search(q, limit=10)
    if not hits:
        return {"found": False, "query": q,
                "note": "No mapped physical assets for this name. The register covers heavy-industry and power "
                        "facilities (Climate TRACE, EPA GHGRP): power, oil & gas, cement, steel, aluminium, "
                        "chemicals, mining, refining and similar."}
    pick = next((h for h in hits if h["match"] == "group"), None) or next((h for h in hits if h["match"] == "exact"), hits[0])
    keys = [pick["key"]]
    ck = tuple(keys)
    if ck in _PUBLIC_CACHE and _t.time() - _PUBLIC_CACHE[ck][0] < 86400:
        return _PUBLIC_CACHE[ck][1]
    c = analyse_company(keys, top_n=10, n=1500)
    if not c.get("found"):
        return {"found": False, "query": q, "note": "Company matched but no assets could be analysed."}
    scen = {v["label"]: {"mean": v["total_npv"]["mean"], "p95": v["total_npv"]["p95"],
                         "pct_of_value_mean": (v["total_npv"].get("pct_of_value") or {}).get("mean")}
            for v in c["scenarios"].values()}
    assets = c["assets"]
    val = sum(a["value_mode"] or 0 for a in assets)
    aal = sum((a["physical_aal_pct"] or 0) / 100 * (a["value_mode"] or 0) for a in assets)
    priced = [a for a in assets if (a["carbon_price_now"] or 0) > 0]
    out = {"found": True, "query": q, "matched": pick.get("name") or pick["key"], "match_type": pick["match"],
           "companies": c["companies"], "n_assets": c["n_assets"], "flood_assessed_assets": c["flood_assessed_assets"],
           "portfolio_value_usd": val, "value_note": "Asset values are replacement-cost estimates from capacity (screening grade).",
           "physical_aal_usd": aal, "physical_aal_pct": (aal / val * 100) if val else None,
           "assets_under_carbon_price": len(priced),
           "scenarios_pv_loss_2026_2050_usd": scen,
           "top_assets": [{k: a[k] for k in ("name", "country", "sector", "physical_aal_pct", "cyclone", "river", "coast",
                                             "carbon_price_now", "instrument", "npv_delayed_mean")} for a in assets[:6]],
           "method": c["method"], "elapsed_s": c["elapsed_s"],
           "sources": ["Climate TRACE / EPA GHGRP asset register", "NOAA IBTrACS", "JRC & Deltares flood maps",
                       "FLOPROS", "World Bank Carbon Pricing Dashboard", "NGFS Phase 5"]}
    out = _jsonable(out)
    _PUBLIC_CACHE[ck] = (_t.time(), out)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# TRANSITION PLAN BUILDER (ESRS E1-1 / TPT / IFRS S2)
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/transition-plan", dependencies=[_AUTH], tags=["Transition Plan"])
def transition_plan(payload: dict) -> dict:
    """Targets → levers → capex/opex → year-by-year emissions path, MACC, SBTi check and ESRS E1-1 fields.
    Pass `inventory` (a /carbon/inventory payload) instead of base_emissions_t to seed the base year from it."""
    from ..transition_plan import PlanError, build, from_dict
    d = dict(payload)
    if "inventory" in d and not d.get("base_emissions_t"):
        _, inv_res = _carbon_calc(d.pop("inventory"))
        T = inv_res["totals_t"]
        d["base_emissions_t"] = {"scope1": T["scope1"], "scope2": T["scope2_market"], "scope3": T["scope3"]}
        d.setdefault("base_year", inv_res["reporting_year"])
        d.setdefault("organisation", inv_res["organisation"])
    try:
        return _jsonable(build(from_dict(d)))
    except (PlanError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc).strip("'\""))


@app.get("/transition-plan/dashboard", tags=["Transition Plan"])
def transition_plan_dashboard():
    from fastapi.responses import HTMLResponse
    return HTMLResponse((Path(__file__).parent / "transition_plan_dashboard.html").read_text(encoding="utf-8"))


# ──────────────────────────────────────────────────────────────────────────────
# INSURER VIEW — policy terms -> AAL, exceedance curves, PML, accumulation, indicative technical premium, climate change
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/insurance/analyse", dependencies=[_AUTH], tags=["Insurance"])
def insurance_analyse(payload: dict) -> dict:
    """sites: [{name, lat, lon, tiv_usd, deductible_pct|deductible_usd, limit_pct|limit_usd, wind_/flood_ deductible_usd and limit_usd, covers, sector}]
    options: n_years, seed, corr_scale, reinsurance {quota_share_pct, xl:[{attach, limit, reinstatements}]}, climate {ssp, year, band},
    cost_of_capital, expense_ratio, uncertainty_load, capital_quantile."""
    from .. import insurance as ins
    p = dict(payload)
    sites = p.pop("sites", None)
    if not isinstance(sites, list) or not sites:
        raise HTTPException(status_code=422, detail="sites: give a non-empty list")
    allowed = {"n_years", "seed", "corr_scale", "cost_of_capital", "expense_ratio", "uncertainty_load", "capital_quantile"}
    kw = {k: p[k] for k in allowed if p.get(k) is not None}
    if p.get("reinsurance"):
        kw["reinsurance_program"] = p["reinsurance"]
    if p.get("climate"):
        c = p["climate"]
        if c.get("ssp") not in ("ssp126", "ssp245", "ssp370", "ssp585") or c.get("band", "median") not in ("low", "median", "high"):
            raise HTTPException(status_code=422, detail="climate: ssp must be ssp126, ssp245, ssp370 or ssp585 and band low, median or high")
        kw["climate"] = {"ssp": c["ssp"], "year": int(c.get("year", 2070)), "band": c.get("band", "median")}
    if "n_years" in kw:
        kw["n_years"] = int(max(500, min(int(kw["n_years"]), 20000)))
    try:
        return _jsonable(ins.run(sites, **kw))
    except ins.InsuranceError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/insurance/dashboard", tags=["Insurance"])
def insurance_dashboard():
    from fastapi.responses import HTMLResponse
    return HTMLResponse((Path(__file__).parent / "insurance_dashboard.html").read_text(encoding="utf-8"))


# ──────────────────────────────────────────────────────────────────────────────
# REAL ESTATE — operational carbon, stranding vs a supplied pathway (e.g. CRREM), regulatory checks
# ──────────────────────────────────────────────────────────────────────────────

class RealEstateRequest(BaseModel):
    buildings: list[dict] = Field(..., min_length=1, max_length=2000)
    pathways: dict = Field(default_factory=dict,
                           description='{"NLD|Office": {"source": "CRREM V2.04 1.5°C (licensed)", "kgco2e_m2": {"2025": 40, ...}}}; key "*" applies to all')
    scenario: str = "Current Policies"
    carbon_price_usd_t: float = Field(0.0, ge=0)
    annual_improvement_pct: float = Field(0.0, ge=0, le=20)


@app.post("/real-estate/assess", dependencies=[_AUTH], tags=["Real Estate"])
def real_estate_assess(req: RealEstateRequest) -> dict:
    """EUI and operational carbon per building, 2050 trajectory with grid decarbonisation and retrofits,
    stranding year against the supplied pathway, and EPC/MEPS regulatory checks."""
    from ..real_estate import REError, assess_portfolio, from_dict
    try:
        bs = [from_dict(b) for b in req.buildings]
        return _jsonable(assess_portfolio(bs, req.pathways, req.scenario, req.carbon_price_usd_t, req.annual_improvement_pct))
    except (REError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc).strip("'\""))


@app.get("/real-estate/dashboard", tags=["Real Estate"])
def real_estate_dashboard():
    from fastapi.responses import HTMLResponse
    return HTMLResponse((Path(__file__).parent / "real_estate_dashboard.html").read_text(encoding="utf-8"))


# ──────────────────────────────────────────────────────────────────────────────
# CSRD DOUBLE MATERIALITY (IRO register) and CARBON MARKETS (EU ETS position, credit screen)
# ──────────────────────────────────────────────────────────────────────────────

class MaterialityRequest(BaseModel):
    iros: list[dict] = Field(..., min_length=1)
    threshold: float = Field(3.0, ge=1, le=5)


@app.post("/materiality/assess", dependencies=[_AUTH], tags=["CSRD"])
def materiality_assess(req: MaterialityRequest) -> dict:
    """Score an IRO register for impact and financial materiality (ESRS 1 ch.3) and list material topics."""
    from ..materiality import MaterialityError, assess, from_dict
    try:
        return assess([from_dict(i) for i in req.iros], req.threshold)
    except (MaterialityError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/materiality/suggest-e1", tags=["CSRD"])
def materiality_suggest(payload: dict) -> dict:
    """Starter climate (E1) IROs pre-filled from engine results (inventory totals, physical AAL, carbon price)."""
    from ..materiality import suggest_e1
    return {"iros": suggest_e1(payload.get("inventory_totals_t"), payload.get("physical_aal_pct"),
                               payload.get("carbon_price_usd_t"))}


class ETSRequest(BaseModel):
    installations: list[dict] = Field(..., min_length=1)
    compliance_year: int = Field(..., ge=2013, le=2035)
    eua_price: Optional[float] = Field(None, gt=0, description="EUA price per tonne (EUR); default: World Bank latest (USD)")
    projection_to: int = Field(2034, ge=2025, le=2040)
    emissions_change_pct_per_year: float = 0.0
    allocation_decline_pct_per_year: float = Field(0.0, ge=0, le=50)


@app.post("/carbon-markets/ets-position", dependencies=[_AUTH], tags=["Carbon Markets"])
def carbon_markets_ets(req: ETSRequest) -> dict:
    """EU ETS compliance balance per installation and projected shortfall as CBAM phases out free allocation."""
    from ..carbon_markets import MarketError, ets_position
    try:
        return ets_position(req.installations, req.compliance_year, req.eua_price, "EUR", req.projection_to,
                            req.emissions_change_pct_per_year, req.allocation_decline_pct_per_year)
    except MarketError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


class CreditsRequest(BaseModel):
    lots: list[dict] = Field(..., min_length=1)
    current_year: int = Field(2026, ge=2015, le=2050)
    max_vintage_age: int = Field(5, ge=1, le=30)


@app.post("/carbon-markets/credits", dependencies=[_AUTH], tags=["Carbon Markets"])
def carbon_markets_credits(req: CreditsRequest) -> dict:
    """Carbon-credit quality and usage screen (ICVCM CCP label, removals, vintage, SBTi and EU claim rules)."""
    from ..carbon_markets import MarketError, screen_credits
    try:
        return screen_credits(req.lots, req.current_year, req.max_vintage_age)
    except (MarketError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/materiality/dashboard", tags=["CSRD"])
def materiality_dashboard():
    from fastapi.responses import HTMLResponse
    return HTMLResponse((Path(__file__).parent / "materiality_dashboard.html").read_text(encoding="utf-8"))


@app.get("/carbon-markets/dashboard", tags=["Carbon Markets"])
def carbon_markets_dashboard():
    from fastapi.responses import HTMLResponse
    return HTMLResponse((Path(__file__).parent / "carbon_markets_dashboard.html").read_text(encoding="utf-8"))


# ──────────────────────────────────────────────────────────────────────────────
# RISK ANALYST on a client's own site list (New Assessment with sites / upload)
# ──────────────────────────────────────────────────────────────────────────────

class AnalystSitesRequest(BaseModel):
    label: str = "Portfolio"
    sites: list[dict] = Field(..., min_length=1, max_length=500)
    top_n: int = Field(30, ge=1, le=200)


@app.post("/analyst/sites", dependencies=[_AUTH], tags=["Risk Analyst"])
def analyst_sites(request: AnalystSitesRequest) -> dict:
    """Run a client's own sites through the Risk Analyst and aggregate the loss distribution."""
    from ..finance.analyst import analyse_sites
    try:
        return _jsonable(analyse_sites(request.sites, request.label, top_n=request.top_n))
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc).strip("'\""))


@app.post("/analyst/sites/upload", dependencies=[_AUTH], tags=["Risk Analyst"])
async def analyst_sites_upload(file: UploadFile = File(...), label: str = Form("Portfolio"), top_n: int = Form(30)) -> dict:
    """Upload an asset list (.xlsx/.csv: name, lat, lon [, sector, type, capacity, emissions_t, value_usd]) and analyse it."""
    import tempfile
    from ..portfolio_export import read_portfolio
    from ..finance.analyst import analyse_sites
    suffix = Path(file.filename or "x.csv").suffix or ".csv"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(await file.read())
        tmp.flush()
        try:
            rows = read_portfolio(tmp.name)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"could not read the asset list: {exc}")
    bad = [r for r in rows if r.get("error")]
    good = [r for r in rows if not r.get("error")]
    if not good:
        raise HTTPException(status_code=422, detail="no rows with valid name, lat and lon")
    from ..finance.analyst import _FIN_COLS
    _FIN_COLS = _FIN_COLS + ("commissioned_year", "retirement_year", "export_share_eu")
    try:
        res = analyse_sites([{k: r.get(k) for k in ("name", "lat", "lon", "sector", "type", "capacity", "emissions_t",
                                                   "value_usd", "company") + _FIN_COLS} for r in good], label, top_n=top_n)
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc).strip("'\""))
    res["skipped_rows"] = [{"row": r["row"], "name": r["name"], "error": r["error"]} for r in bad]
    return _jsonable(res)
