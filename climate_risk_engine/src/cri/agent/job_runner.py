"""
job_runner.py — Async job runner for the ClimRisk assessment agent.

Lifecycle
---------
  1. Client calls POST /agent/assess → receives {job_id, status: "queued"}
  2. Engine runs the full pipeline in background:
       entity resolution → data acquisition → risk assessment → trajectory
  3. Client polls GET /agent/jobs/{job_id} for status and partial results
  4. When status = "completed", the full AssessmentResult is in the response

In-memory job store (no external dependencies required).
For production: replace _JOB_STORE with Redis or Supabase.

Job states
----------
queued      → accepted, not started yet
running     → data acquisition in progress
completed   → full result available
failed      → pipeline error (check error field)
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── In-memory job store ────────────────────────────────────────────────────────
_JOB_STORE: dict[str, "AssessmentJob"] = {}
_JOB_TTL_SECONDS = 3600 * 6   # keep completed jobs 6 hours


class JobStatus(str, Enum):
    QUEUED    = "queued"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"


@dataclass
class ProgressEvent:
    stage:    str    # e.g. "entity_resolution", "data_acquisition", "risk_assessment"
    message:  str
    pct:      int    # 0-100
    ts:       float  = field(default_factory=time.time)


@dataclass
class AssessmentJob:
    job_id:       str
    company_name: str
    scope:        str        # "triage" | "standard" | "full_dd"
    status:       JobStatus  = JobStatus.QUEUED
    created_at:   float      = field(default_factory=time.time)
    updated_at:   float      = field(default_factory=time.time)
    progress:     list[ProgressEvent] = field(default_factory=list)
    partial_data: dict[str, Any]      = field(default_factory=dict)
    result:       Optional[dict]      = None
    error:        Optional[str]       = None

    def add_progress(self, stage: str, message: str, pct: int) -> None:
        evt = ProgressEvent(stage=stage, message=message, pct=pct)
        self.progress.append(evt)
        self.updated_at = time.time()
        logger.info("[%s] %s (%d%%) — %s", self.job_id[:8], stage, pct, message)

    def to_status_dict(self) -> dict:
        """Lightweight status response (no full result)."""
        return {
            "job_id":       self.job_id,
            "company_name": self.company_name,
            "scope":        self.scope,
            "status":       self.status.value,
            "created_at":   self.created_at,
            "updated_at":   self.updated_at,
            "progress":     [
                {"stage": e.stage, "message": e.message, "pct": e.pct, "ts": e.ts}
                for e in self.progress
            ],
            "partial_data": self.partial_data,
            "error":        self.error,
        }

    def to_full_dict(self) -> dict:
        """Full response including completed result."""
        d = self.to_status_dict()
        d["result"] = self.result
        return d


def create_job(company_name: str, scope: str) -> AssessmentJob:
    """Create a new job and register it in the store."""
    job = AssessmentJob(
        job_id=str(uuid.uuid4()),
        company_name=company_name,
        scope=scope,
    )
    _JOB_STORE[job.job_id] = job
    _evict_old_jobs()
    return job


def get_job(job_id: str) -> Optional[AssessmentJob]:
    """Retrieve a job by ID."""
    return _JOB_STORE.get(job_id)


def list_jobs(limit: int = 20) -> list[dict]:
    """Return most recent jobs (newest first)."""
    jobs = sorted(_JOB_STORE.values(), key=lambda j: j.created_at, reverse=True)
    return [j.to_status_dict() for j in jobs[:limit]]


def _evict_old_jobs() -> None:
    """Remove jobs older than TTL from the in-memory store."""
    cutoff = time.time() - _JOB_TTL_SECONDS
    stale  = [jid for jid, j in _JOB_STORE.items() if j.created_at < cutoff]
    for jid in stale:
        del _JOB_STORE[jid]
    if stale:
        logger.debug("Evicted %d stale jobs", len(stale))


async def run_assessment_job(
    job: AssessmentJob,
    sector:       Optional[str]  = None,
    country_hint: Optional[str]  = None,
    ticker:       Optional[str]  = None,
    lat:          Optional[float] = None,
    lon:          Optional[float] = None,
    use_estimates: bool           = True,
) -> None:
    """
    Full pipeline for a single company assessment.

    Runs in a background asyncio task.  Updates job.status, job.progress,
    job.partial_data, and job.result as it goes.

    Stages
    ------
    10%  Entity resolution (GLEIF)
    30%  Data acquisition (EDGAR, CDP, SBTi, Yahoo, geocoding) — parallel
    55%  Risk assessment (physical + transition using engine modules)
    80%  Predictive trajectory (2025–2050)
    100% Report assembly
    """
    from ..data_acquisition.company_profiler import build_company_profile

    job.status = JobStatus.RUNNING
    job.updated_at = time.time()

    try:
        # ── Stage 1: Entity resolution ─────────────────────────────────────
        job.add_progress("entity_resolution", f"Resolving '{job.company_name}' via GLEIF LEI registry", 5)

        # ── Stage 2: Parallel data acquisition ────────────────────────────
        job.add_progress("data_acquisition", "Fetching from SEC EDGAR, CDP, SBTi, Yahoo Finance…", 10)

        profile = await build_company_profile(
            company_name=job.company_name,
            sector=sector,
            country_hint=country_hint,
            ticker=ticker,
            lat=lat,
            lon=lon,
            use_estimates=use_estimates,
        )

        job.add_progress("data_acquisition", "Data acquisition complete", 35)
        job.partial_data["profile"] = profile.to_dict()
        job.partial_data["data_gaps_count"] = len(profile.data_gaps)
        job.partial_data["assessment_readiness"] = profile.assessment_readiness

        # ── Stage 3: Risk assessment ───────────────────────────────────────
        job.add_progress("risk_assessment", "Running physical + transition risk assessment", 40)
        risk_result = await _run_risk_assessment(job, profile)
        job.partial_data["risk_summary"] = risk_result.get("summary", {})

        # ── Stage 3.5: Extended risk analyses ────────────────────────────────
        job.add_progress(
            "extended_analysis",
            "Running extended risk modules (stranded assets, CBAM, water stress, litigation, CSRD materiality)",
            55,
        )
        extended = await _run_extended_analyses(job, profile, risk_result)

        # ── Stage 4: Predictive trajectory ────────────────────────────────
        job.add_progress("trajectory", "Computing 2025–2050 predictive trajectory", 70)
        trajectory = await _run_trajectory(job, profile, risk_result)

        # ── Stage 5: Assemble full result ─────────────────────────────────
        job.add_progress("assembly", "Assembling sourced assessment report", 90)

        job.result = {
            "company_profile":   profile.to_dict(),
            "risk_assessment":   risk_result,
            "extended_risk":     extended,
            "trajectory":        trajectory,
            "data_gaps":         [
                {
                    "field":   g.field_name,
                    "reason":  g.reason,
                    "impact":  g.impact,
                    "action":  g.suggested_action,
                }
                for g in profile.data_gaps
            ],
            "data_request":      _build_data_request(profile),
            "assessment_scope":  job.scope,
        }

        job.status = JobStatus.COMPLETED
        job.add_progress("done", "Assessment complete", 100)

    except Exception as exc:
        logger.exception("Assessment job %s failed: %s", job.job_id, exc)
        job.status = JobStatus.FAILED
        job.error  = str(exc)
        job.updated_at = time.time()


async def _run_extended_analyses(job: AssessmentJob, profile, risk_result: dict) -> dict:
    """
    Run the extended climate risk modules in parallel:
      - Stranded assets (IEA WEO unburnable carbon NPV impairment)
      - EU CBAM exposure (2026-2034 certificate cost trajectory)
      - Water stress financial loss (WRI Aqueduct → NPV drag)
      - Litigation risk (Grantham 5-factor composite score)
      - Impact materiality (CSRD ESRS E1 double materiality — impact side)

    All modules are non-blocking and return gracefully on missing data.
    Results are attached to the main assessment result dict under 'extended_risk'.
    """
    from ..climate.stranded_assets import assess_stranded_assets
    from ..climate.cbam import assess_cbam_exposure
    from ..climate.water_stress_loss import assess_water_stress
    from ..climate.litigation_risk import assess_litigation_risk
    from ..climate.impact_materiality import assess_impact_materiality

    company_name = profile.resolved_name or profile.input_name
    sector       = profile.sector or "default"
    revenue      = profile.revenue_usd_m.value if profile.revenue_usd_m else 1000.0
    jurisdiction = profile.jurisdiction or "default"

    # Pull EV from engine output if available
    ev_usd_m: Optional[float] = None
    try:
        ev_usd_m = risk_result.get("scenarios", {}).get("nze", {}).get("enterprise_value_usd_m")
    except Exception:
        pass
    ev_usd_m = ev_usd_m or revenue * 2.0

    # Emissions data
    scope1_mt  = profile.scope1_mt_co2e.value  if profile.scope1_mt_co2e  else None
    scope2_mt  = profile.scope2_mt_co2e.value  if profile.scope2_mt_co2e  else None
    scope3_mt  = profile.scope3_mt_co2e.value  if profile.scope3_mt_co2e  else None

    # Carrying value proxy: 2× revenue for oil/gas/mining/utilities, 1× for others
    _CAPEX_HEAVY = {"oil_gas", "coal", "utilities", "mining", "steel", "cement"}
    carry_multiplier = 2.0 if sector in _CAPEX_HEAVY else 1.0
    carrying_value   = revenue * carry_multiplier

    # SBTi and CDP for litigation / materiality
    sbti_status = getattr(profile, "sbti_status", None) or "unknown"
    cdp_score   = getattr(profile, "cdp_score", None)   or "unknown"
    net_zero_yr = getattr(profile, "net_zero_year", None)
    has_net_zero = net_zero_yr is not None and str(net_zero_yr).isdigit() and int(net_zero_yr) <= 2050
    has_sbti_val = "validated" in str(sbti_status).lower()

    # Water stress score: fall back to sector-based default if no geocoded score
    # Real implementations should use the WRI Aqueduct API with profile.lat/lon
    aqueduct_score: float = getattr(profile, "aqueduct_score", None) or 0.0
    if aqueduct_score == 0.0:
        # Sector-based conservative default for missing Aqueduct score
        _SECTOR_DEFAULT_SCORE = {
            "agriculture": 2.5, "mining": 2.0, "chemicals": 1.8,
            "steel": 1.5, "beverages": 2.2, "utilities": 1.5,
            "semiconductors": 2.0, "cement": 1.2, "oil_gas": 1.0,
        }
        aqueduct_score = _SECTOR_DEFAULT_SCORE.get(sector, 0.8)

    # Run all 5 extended modules in a thread pool (all are synchronous functions)
    loop = asyncio.get_event_loop()

    async def _safe_run(fn, *args, **kwargs):
        try:
            return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))
        except Exception as exc:
            logger.warning("Extended analysis '%s' failed: %s", fn.__name__, exc)
            return None

    (
        stranded_r,
        cbam_r,
        water_r,
        lit_r,
        mat_r,
    ) = await asyncio.gather(
        _safe_run(
            assess_stranded_assets,
            company_name, sector, carrying_value, revenue,
            ev_usd_m=ev_usd_m,
            wacc=0.09,
        ),
        _safe_run(
            assess_cbam_exposure,
            company_name, sector, revenue,
            origin_country=jurisdiction,
        ),
        _safe_run(
            assess_water_stress,
            company_name, sector, aqueduct_score, revenue,
            ev_usd_m=ev_usd_m,
            wacc=0.09,
            scenario="current_policies",
        ),
        _safe_run(
            assess_litigation_risk,
            company_name, sector,
            hq_country=jurisdiction,
            scope1_intensity=(scope1_mt * 1e6 / revenue) if (scope1_mt and revenue > 0) else None,
            sbti_status=sbti_status,
            cdp_score=cdp_score,
            revenue_usd_m=revenue,
            has_eu_operations=jurisdiction in {"DE", "FR", "NL", "GB", "SE", "DK", "NO", "BE", "ES"},
        ),
        _safe_run(
            assess_impact_materiality,
            company_name, sector,
            scope1_mt_co2e=scope1_mt,
            scope2_mt_co2e=scope2_mt,
            scope3_mt_co2e=scope3_mt,
            revenue_usd_m=revenue,
            has_net_zero_target=has_net_zero,
            has_sbti_validated=has_sbti_val,
            has_eu_operations=jurisdiction in {"DE", "FR", "NL", "GB", "SE", "DK", "NO", "BE", "ES"},
        ),
    )

    def _to_dict(obj):
        if obj is None:
            return None
        if hasattr(obj, "__dataclass_fields__"):
            import dataclasses
            return dataclasses.asdict(obj)
        return obj

    return {
        "stranded_assets":      _to_dict(stranded_r),
        "cbam":                 _to_dict(cbam_r),
        "water_stress":         _to_dict(water_r),
        "litigation_risk":      _to_dict(lit_r),
        "impact_materiality":   _to_dict(mat_r),
    }


async def _run_risk_assessment(job: AssessmentJob, profile) -> dict:
    """
    Call the full CRI engine (orchestrator.run_full) with the profiled data.

    Converts CompanyProfile → Company schema via profile_adapter, then runs
    all three NGFS scenarios and extracts the full pillar scores, per-hazard
    physical loss breakdown, and composite CRI rating.

    Falls back to shallow calculation only if the engine itself raises an
    unrecoverable error.
    """
    try:
        # ── Convert profile → engine Company schema ────────────────────────
        from ..data_acquisition.profile_adapter import profile_to_company
        from ..engine.orchestrator import run_full as _run_full

        company = await asyncio.get_event_loop().run_in_executor(
            None, profile_to_company, profile
        )

        # ── Run all three NGFS scenarios via the full engine ───────────────
        full_result = await asyncio.get_event_loop().run_in_executor(
            None, _run_full, company
        )

        # ── Extract per-scenario physical + transition breakdown ──────────
        def _extract_scenario(run_res) -> dict:
            phys_by_hazard = {}
            total_gross_var = 0.0
            total_net_var   = 0.0
            for yr in run_res.years:
                for h, v in yr.physical_loss_by_hazard.items():
                    phys_by_hazard[h] = phys_by_hazard.get(h, 0.0) + v
                total_gross_var += yr.gross_var
                total_net_var   += yr.net_var

            return {
                "enterprise_value_usd_m":   round(run_res.enterprise_value, 1),
                "npv_impact_pct":            run_res.npv_impact_pct,
                "ebitda_compression_2030":   run_res.ebitda_compression_2030_pct,
                "ebitda_compression_2040":   run_res.ebitda_compression_2040_pct,
                "gross_var_npv_usd_m":       round(total_gross_var, 1),
                "net_var_npv_usd_m":         round(total_net_var, 1),
                "portfolio_impairment_pct":  run_res.portfolio_impairment_pct,
                "peak_annual_gross_var":     run_res.peak_annual_gross_var,
                "dominant_hazard":           run_res.dominant_hazard,
                "physical_loss_by_hazard":   {k: round(v, 2) for k, v in phys_by_hazard.items()},
                "exposure_score":            getattr(run_res, "exposure_score", None),
                "transition_score":          getattr(run_res, "transition_score", None),
                "financial_score":           getattr(run_res, "financial_score", None),
                "adaptive_score":            getattr(run_res, "adaptive_score", None),
            }

        scenarios_out = {
            "nze":     _extract_scenario(full_result.nze),
            "delayed": _extract_scenario(full_result.delayed),
            "cp":      _extract_scenario(full_result.cp),
        }

        # ── Rating ────────────────────────────────────────────────────────
        rating = full_result.rating
        rating_out = {
            "rating":              getattr(rating, "rating", None),
            "physical_pillar":     getattr(rating, "physical_pillar", None),
            "transition_pillar":   getattr(rating, "transition_pillar", None),
            "financial_pillar":    getattr(rating, "financial_pillar", None),
            "adaptive_pillar":     getattr(rating, "adaptive_pillar", None),
            "sector_rank_pct":     getattr(rating, "sector_rank_pct", None),
        }

        # ── Aggregate physical loss across all hazards (worst-case = CP scenario) ──
        cp_physical = scenarios_out["cp"].get("physical_loss_by_hazard", {})
        dominant_hazard = scenarios_out["cp"].get("dominant_hazard", "")
        flood_var_pct = 0.0
        ev_usd_m = scenarios_out["nze"].get("enterprise_value_usd_m") or 1.0
        if ev_usd_m > 0:
            net_var = scenarios_out["cp"].get("net_var_npv_usd_m", 0.0) or 0.0
            flood_var_pct = (net_var / ev_usd_m * 100) if ev_usd_m > 0 else 0.0

        # ── Transition risk (NZE scenario — highest carbon price) ─────────
        scope1 = profile.scope1_mt_co2e.value if profile.scope1_mt_co2e else None
        revenue = profile.revenue_usd_m.value if profile.revenue_usd_m else None
        nze_ebitda_compression = scenarios_out["nze"].get("ebitda_compression_2030") or 0.0
        transition_var_pct = abs(nze_ebitda_compression) * 100 if nze_ebitda_compression else 0.0

        # ── Composite (use engine pillar scores if available) ─────────────
        rating_letter = rating_out.get("rating")
        _RATING_SCORE = {
            "A": 0.10, "A+": 0.05, "A-": 0.15,
            "B": 0.30, "B+": 0.25, "B-": 0.35,
            "C": 0.55, "C+": 0.50, "C-": 0.60,
            "D": 0.80, "D+": 0.75, "D-": 0.85, "E": 0.95,
        }
        composite = _RATING_SCORE.get(rating_letter, 0.40)
        triage = "GREEN" if composite < 0.35 else ("AMBER" if composite < 0.60 else "RED")

        summary = {
            "company_name":         profile.resolved_name or profile.input_name,
            "jurisdiction":         profile.jurisdiction,
            "sector":               profile.sector,
            "assessment_readiness": profile.assessment_readiness,
            "engine_version":       "orchestrator.run_full",
            "physical": {
                "flood_var_pct":         round(flood_var_pct, 3),
                "dominant_hazard":       dominant_hazard,
                "physical_loss_by_hazard": cp_physical,
                "net_var_usd_m":         scenarios_out["cp"].get("net_var_npv_usd_m"),
                "gross_var_usd_m":       scenarios_out["cp"].get("gross_var_npv_usd_m"),
            },
            "transition": {
                "scope1_mt_co2e":             scope1,
                "transition_var_pct_revenue": round(transition_var_pct, 3),
                "ebitda_compression_2030_nze": scenarios_out["nze"].get("ebitda_compression_2030"),
                "ebitda_compression_2030_delayed": scenarios_out["delayed"].get("ebitda_compression_2030"),
            },
            "climate_targets": {
                "sbti_status":          profile.sbti_status,
                "temperature_ambition": profile.temperature_ambition,
                "net_zero_year":        profile.net_zero_year,
                "cdp_score":            profile.cdp_score,
            },
            "scenarios":    scenarios_out,
            "rating":       rating_out,
            "composite": {
                "score":       composite,
                "triage":      triage,
                "methodology": "Full CRI orchestrator.run_full — 3 NGFS scenarios, pillar-weighted composite rating",
            },
        }
        return summary

    except Exception as exc:
        logger.warning(
            "Full engine run failed for %s: %s — falling back to shallow assessment",
            job.company_name, exc,
        )
        return await _run_risk_assessment_shallow(job, profile)


async def _run_risk_assessment_shallow(job: AssessmentJob, profile) -> dict:
    """
    Shallow fallback when the full engine cannot be called (e.g., missing
    optional dependencies like GIS stack).  Uses only scalar metrics.
    """
    lat     = profile.lat.value if profile.lat else None
    lon     = profile.lon.value if profile.lon else None
    revenue = profile.revenue_usd_m.value if profile.revenue_usd_m else None
    scope1  = profile.scope1_mt_co2e.value if profile.scope1_mt_co2e else None

    summary: dict = {
        "company_name":         profile.resolved_name or profile.input_name,
        "jurisdiction":         profile.jurisdiction,
        "sector":               profile.sector,
        "assessment_readiness": profile.assessment_readiness,
        "engine_version":       "shallow_fallback",
    }

    # Physical (flood only via legacy assessor)
    if lat is not None and lon is not None:
        try:
            from ..chronic.flood import FloodRiskAssessor
            flood = FloodRiskAssessor()
            flood_result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: flood.assess(lat, lon)
            )
            summary["physical"] = {
                "flood_var_pct":   getattr(flood_result, "var_pct", None),
                "flood_score":     getattr(flood_result, "score", None),
                "dominant_hazard": "flood",
                "note": "Shallow fallback — only flood VaR computed",
            }
        except Exception as e:
            summary["physical"] = {"note": f"Flood assessment unavailable: {e}"}
    else:
        summary["physical"] = {"note": "No coordinates — physical risk cannot be computed"}

    # Transition
    if scope1 is not None and revenue is not None:
        cp2030 = 130.0
        cost_m = scope1 * cp2030 / 1_000_000
        tv_pct = (cost_m / revenue * 100) if revenue > 0 else 0.0
        summary["transition"] = {
            "scope1_mt_co2e":             scope1,
            "carbon_price_usd":           cp2030,
            "carbon_cost_usd_m":          round(cost_m, 2),
            "transition_var_pct_revenue": round(tv_pct, 3),
        }
    else:
        summary["transition"] = {
            "note": "Cannot compute transition risk — missing scope1 or revenue"
        }

    summary["climate_targets"] = {
        "sbti_status":          profile.sbti_status,
        "temperature_ambition": profile.temperature_ambition,
        "net_zero_year":        profile.net_zero_year,
        "cdp_score":            profile.cdp_score,
    }

    phys_var  = (summary.get("physical") or {}).get("flood_var_pct", 0) or 0
    trans_var = (summary.get("transition") or {}).get("transition_var_pct_revenue", 0) or 0
    composite = round(0.5 * min(1.0, phys_var / 20.0) + 0.5 * min(1.0, trans_var / 33.0), 3)
    triage    = "GREEN" if composite < 0.35 else ("AMBER" if composite < 0.60 else "RED")
    summary["composite"] = {"score": composite, "triage": triage, "methodology": "shallow"}
    return summary


async def _run_trajectory(job: AssessmentJob, profile, risk_result: dict) -> dict:
    """
    Generate 2025–2050 predictive trajectory.
    Delegates to the trajectory module.
    """
    try:
        from ..data_acquisition._trajectory import build_trajectory
        return await build_trajectory(profile, risk_result)
    except Exception as exc:
        logger.warning("Trajectory module unavailable: %s", exc)
        return _simple_trajectory(profile, risk_result)


def _simple_trajectory(profile, risk_result: dict) -> dict:
    """
    Fallback: compute a simple linear/logistic trajectory without the
    full trajectory module.  Covers SSP2-4.5 (mid-case).
    """
    composite_now = (risk_result.get("composite") or {}).get("score", 0.4)
    scope1_now    = profile.scope1_mt_co2e.value if profile.scope1_mt_co2e else 1.0

    # NGFS NZ2050 carbon price ramp (USD/tCO2e)
    _CARBON_PRICE = {
        2025: 60,  2030: 130, 2035: 220, 2040: 350, 2045: 500, 2050: 700
    }

    revenue = profile.revenue_usd_m.value if profile.revenue_usd_m else 5000.0

    years = list(range(2025, 2051))
    trajectory = []
    for year in years:
        # Simple linear Scope 1 reduction (assume 4%/year aligned with SBTi)
        yrs_from_base = year - 2025
        scope1_y = scope1_now * (0.96 ** yrs_from_base)

        # Carbon cost
        cp = _CARBON_PRICE.get(year) or (
            _CARBON_PRICE[2050] + (year - 2050) * 20  # extrapolate post-2050
        )
        carbon_cost  = scope1_y * cp / 1_000_000
        trans_var    = min(1.0, (carbon_cost / revenue * 100) / 33.0) if revenue > 0 else 0

        # Physical risk grows ~1.5% per year under SSP2-4.5
        phys_base = (risk_result.get("physical") or {}).get("flood_var_pct", 5.0) or 5.0
        phys_y    = phys_base * (1 + 0.015 * yrs_from_base)
        phys_score = min(1.0, phys_y / 20.0)

        composite_y = round(0.5 * phys_score + 0.5 * trans_var, 3)

        trajectory.append({
            "year":             year,
            "scope1_mt_co2e":   round(scope1_y, 3),
            "carbon_price_usd": cp,
            "carbon_cost_usd_m": round(carbon_cost, 2),
            "flood_var_pct":    round(phys_y, 3),
            "transition_var_pct": round(min(99.9, (carbon_cost / revenue * 100) if revenue > 0 else 0), 3),
            "composite_score":  composite_y,
            "triage":           "GREEN" if composite_y < 0.35 else ("AMBER" if composite_y < 0.60 else "RED"),
        })

    # Inflection points
    inflections = []
    for i in range(1, len(trajectory)):
        prev = trajectory[i-1]
        curr = trajectory[i]
        if prev["triage"] != curr["triage"]:
            inflections.append({
                "year":    curr["year"],
                "event":   f"Triage changes {prev['triage']} → {curr['triage']}",
                "score":   curr["composite_score"],
                "driver":  (
                    "rising carbon price" if curr["composite_score"] > prev["composite_score"]
                    else "emissions reduction"
                ),
            })

    return {
        "scenario":     "SSP2-4.5 / NGFS Net Zero 2050 (baseline)",
        "methodology":  (
            "Linear Scope 1 reduction 4%/yr (SBTi 1.5°C near-term pathway). "
            "Physical risk +1.5%/yr flood VaR amplification (IPCC AR6 SSP2-4.5). "
            "Carbon price: NGFS NZ2050 schedule. "
            "Composite: 0.5×physical + 0.5×transition, each capped at 1.0."
        ),
        "years":        trajectory,
        "inflections":  inflections,
    }


def _build_data_request(profile) -> list[dict]:
    """
    Build an explicit list of data requests to close assessment gaps.
    Shown to the practitioner when profile is PARTIAL or TRIAGE_ONLY.
    """
    requests = []
    for gap in profile.data_gaps:
        # Skip gaps that are low-impact (EV estimated, etc.)
        if "benchmark" in gap.reason.lower():
            continue
        requests.append({
            "field":       gap.field_name,
            "why_needed":  gap.impact,
            "how_to_provide": gap.suggested_action,
        })
    return requests
