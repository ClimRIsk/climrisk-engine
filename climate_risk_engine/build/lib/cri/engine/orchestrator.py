"""Top-level entry points for the CRI engine.

Three functions are exposed:

run(company, scenario, ...)
    The original single-scenario path. Returns a full RunResults with
    valuation. Backward-compatible — existing code unchanged.

run_full(company, ...)
    Convenience function that runs all three NGFS scenarios, computes the
    composite CRI rating, and returns a FullRunResult. Pillar scores
    (exposure_score, transition_score, financial_score, adaptive_score) are
    written back into each RunResults object so callers never get None there.
    This is the recommended entry point for the web platform and API layer.

run_scoped(company, scope, ...)
    The new modular path. The caller specifies which pillars they need
    via a ReportScope, and the engine runs only the required computation.
    Returns a ScopedResult containing only the requested pillar reports.

    Examples
    --------
    # Morelli: physical risk only
    result = run_scoped(company, ReportScope.PHYSICAL)
    print(result.physical.physical_score)
    print(result.physical.narrative)

    # ING: full financial risk with valuation
    result = run_scoped(company, ReportScope.FINANCIAL)
    print(result.valuation_results["nze"].enterprise_value)

    # Full CRI package (existing behaviour, new interface)
    result = run_scoped(company, ReportScope.FULL_CRI)
    print(result.rating_result.rating)

Scope → computation map
-----------------------
PHYSICAL            : simulate × 3 scenarios → PhysicalRiskReport
TRANSITION          : simulate × 3 scenarios → TransitionRiskReport
PHYSICAL_TRANSITION : simulate × 3 scenarios → both reports
FINANCIAL           : simulate + DCF × 3 scenarios → valuation dict
FULL_CRI            : all of the above + composite CRI rating
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field

from .. import scenarios as _scen
from ..data.schemas import Company, RunResults, Scenario, YearResult
from ..engine.scope import ReportScope, ScopedResult
from ..financial.dcf import value
from ..financial.metrics import climate_adjusted_wacc, compute_year
from ..operations.company import simulate
from ..outcomes.physical_report import build_physical_report
from ..outcomes.transition_report import build_transition_report


# ---------------------------------------------------------------------------
# FullRunResult — returned by run_full()
# ---------------------------------------------------------------------------

@dataclass
class FullRunResult:
    """Results of a full three-scenario CRI run with composite rating.

    Attributes
    ----------
    company_id  : Company identifier.
    results     : Dict with keys "nze", "delayed", "cp" → RunResults.
                  Each RunResults has its four pillar score fields populated.
    rating      : Full RatingResult (letter, pillar scores, sector rank, etc.)
    """
    company_id: str
    results: dict  # "nze" | "delayed" | "cp" → RunResults
    rating: object  # RatingResult (avoid circular import at class-definition time)

    @property
    def cp(self) -> RunResults:
        return self.results["cp"]

    @property
    def nze(self) -> RunResults:
        return self.results["nze"]

    @property
    def delayed(self) -> RunResults:
        return self.results["delayed"]


def _input_hash(scenario: Scenario, company: Company) -> str:
    payload = json.dumps(
        {
            "scenario": scenario.model_dump(mode="json"),
            "company": company.model_dump(mode="json"),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Original single-scenario run — backward-compatible
# ---------------------------------------------------------------------------

def run(
    company: Company,
    scenario: Scenario,
    baseline_npv: float | None = None,
    model_version: str = "0.3.0",
) -> RunResults:
    """Run the full pipeline: scenario → operational → financial → valuation.

    This is the original interface. It remains fully supported.
    For modular scope-based runs, use run_scoped().
    """
    ops = simulate(company, scenario)
    # Thread prev_revenue so working-capital change is computed for each year
    year_results: list[YearResult] = []
    prev_rev: float | None = None
    for o in ops:
        yr = compute_year(company, o, prev_revenue=prev_rev)
        year_results.append(yr)
        prev_rev = o.revenue

    wacc = climate_adjusted_wacc(company, scenario)
    dcf = value(company, year_results, wacc)

    npv_impact_pct = None
    if baseline_npv is not None and baseline_npv != 0:
        npv_impact_pct = dcf.enterprise_value / baseline_npv - 1.0

    def _by_year(y: int) -> YearResult | None:
        for yr in year_results:
            if yr.year == y:
                return yr
        return None

    baseline_ebitda = company.financials.ebitda

    def _compression(year: int) -> float | None:
        yr = _by_year(year)
        if yr is None or baseline_ebitda == 0:
            return None
        return yr.ebitda / baseline_ebitda - 1.0

    return RunResults(
        run_id=str(uuid.uuid4())[:8],
        scenario_id=scenario.id,
        company_id=company.id,
        model_version=model_version,
        years=year_results,
        npv_fcf=dcf.npv_fcf,
        terminal_value=dcf.terminal_value,
        enterprise_value=dcf.enterprise_value,
        equity_value=dcf.equity_value,
        implied_share_price=dcf.implied_share_price,
        wacc_used=dcf.wacc_used,
        baseline_npv=baseline_npv,
        npv_impact_pct=npv_impact_pct,
        ebitda_compression_2030_pct=_compression(2030),
        ebitda_compression_2040_pct=_compression(2040),
        input_hash=_input_hash(scenario, company),
        scenario_version=scenario.version,
    )


# ---------------------------------------------------------------------------
# run_full — recommended entry point for platform + API
# ---------------------------------------------------------------------------

def run_full(
    company: Company,
    model_version: str = "0.3.0",
) -> FullRunResult:
    """Run all three NGFS scenarios and compute the composite CRI rating.

    This is the recommended entry point for the web platform and API layer.
    It runs NZE 2050, Delayed Transition, and Current Policies in sequence,
    then calls RatingEngine to compute physical / transition / financial /
    adaptive pillar scores and writes them back into each RunResults object.
    Callers are guaranteed that RunResults.exposure_score etc. are never None.

    Parameters
    ----------
    company      : Fully populated Company (assets, financials, emissions).
    model_version: Engine version string embedded in output provenance.

    Returns
    -------
    FullRunResult with .results dict (keys "nze", "delayed", "cp") and
    .rating (RatingResult). Access individual RunResults via .nze / .cp etc.

    Example
    -------
    >>> full = run_full(company)
    >>> print(full.rating.rating, full.rating.composite_score)
    >>> print(full.cp.exposure_score, full.cp.transition_score)
    """
    from ..outcomes.ratings import RatingEngine

    # Step 1: run CP first — its enterprise_value becomes baseline_npv for the
    # other two scenarios so npv_impact_pct is correctly signed.
    r_cp  = run(company, _scen.CURRENT_POLICIES, model_version=model_version)
    baseline_ev = r_cp.enterprise_value if r_cp.enterprise_value > 0 else None

    r_nze = run(company, _scen.NZE_2050,
                baseline_npv=baseline_ev, model_version=model_version)
    r_dt  = run(company, _scen.DELAYED_TRANSITION,
                baseline_npv=baseline_ev, model_version=model_version)

    # Step 2: compute composite rating across all three scenarios
    engine = RatingEngine()
    rating = engine.rate(
        company_name=company.name,
        sector=company.sector,
        nze_results=r_nze,
        dt_results=r_dt,
        cp_results=r_cp,
        data_quality=company.data_quality,
    )

    # Step 3: back-populate pillar scores into every RunResults so downstream
    # consumers never see None in exposure_score / transition_score / etc.
    # adaptive_score: measures headroom — how far the company is from Critical (100).
    # Higher adaptive_score = more buffer before reaching a Critical (E) rating.
    adaptive = max(0.0, 100.0 - rating.composite_score)

    for r in (r_nze, r_dt, r_cp):
        r.exposure_score   = round(rating.physical.score, 2)
        r.transition_score = round(rating.transition.score, 2)
        r.financial_score  = round(rating.financial.score, 2)
        r.adaptive_score   = round(adaptive, 2)

    # Step 4: gather physical hazard provenance at full resolution (_depth=0).
    # We use the NZE 2050 scenario as the reference (SSP1-2.6 = most conservative
    # warming trajectory) so provenance reflects the best available data quality.
    # The same provenance block is attached to all three RunResults — provenance
    # is per-asset, not per-scenario.
    try:
        from ..operations.company import get_physical_provenance
        provenance = get_physical_provenance(
            company, _scen.NZE_2050, reference_year=2026
        )
        for r in (r_nze, r_dt, r_cp):
            r.physical_hazard_detail = provenance
    except Exception:
        # Provenance is supplementary — never let it break a run
        pass

    return FullRunResult(
        company_id=company.id,
        results={"nze": r_nze, "delayed": r_dt, "cp": r_cp},
        rating=rating,
    )


# ---------------------------------------------------------------------------
# New scoped run — modular, pillar-selective
# ---------------------------------------------------------------------------

def run_scoped(
    company: Company,
    scope: ReportScope = ReportScope.FULL_CRI,
    model_version: str = "0.3.0",
) -> ScopedResult:
    """Run only the engine pillars required by the requested scope.

    The three NGFS scenarios (NZE 2050, Delayed Transition, Current Policies)
    are always simulated together so results are cross-scenario comparable.
    What changes is *which outputs* are computed and returned.

    Parameters
    ----------
    company : Company profile (assets, financials, emissions).
    scope   : Which pillar(s) to compute. Defaults to FULL_CRI.
              See ReportScope docstring for options.

    Returns
    -------
    ScopedResult with .physical / .transition / .valuation_results populated
    according to the scope. Unpopulated fields are None.
    """
    run_id = str(uuid.uuid4())[:8]

    nze_scenario = _scen.NZE_2050
    dly_scenario = _scen.DELAYED_TRANSITION
    cp_scenario  = _scen.CURRENT_POLICIES

    shared_hash = _input_hash(nze_scenario, company)
    scenario_version = nze_scenario.version

    # -------------------------------------------------------------------
    # Step 1: Operational simulation (always needed — feeds all pillars)
    # -------------------------------------------------------------------
    ops_nze = simulate(company, nze_scenario)
    ops_dly = simulate(company, dly_scenario)
    ops_cp  = simulate(company, cp_scenario)

    result = ScopedResult(scope=scope, run_id=run_id)

    # -------------------------------------------------------------------
    # Step 2: Physical Risk pillar
    # -------------------------------------------------------------------
    if scope.needs_physical:
        result.physical = build_physical_report(
            company=company,
            ops_nze=ops_nze,
            ops_delayed=ops_dly,
            ops_cp=ops_cp,
            run_id=run_id,
            model_version=model_version,
            input_hash=shared_hash,
            scenario_version=scenario_version,
        )

    # -------------------------------------------------------------------
    # Step 3: Transition Risk pillar
    # -------------------------------------------------------------------
    if scope.needs_transition:
        result.transition = build_transition_report(
            company=company,
            ops_nze=ops_nze,
            ops_delayed=ops_dly,
            ops_cp=ops_cp,
            run_id=run_id,
            model_version=model_version,
            input_hash=shared_hash,
            scenario_version=scenario_version,
        )

    # -------------------------------------------------------------------
    # Step 4: Financial valuation (DCF across all scenarios)
    # -------------------------------------------------------------------
    if scope.needs_valuation:
        valuation = {}
        for label, scenario, ops in [
            ("nze",     nze_scenario, ops_nze),
            ("delayed", dly_scenario, ops_dly),
            ("cp",      cp_scenario,  ops_cp),
        ]:
            # Thread prev_revenue for working capital change
            yrs: list[YearResult] = []
            prev_rev2: float | None = None
            for o in ops:
                yr = compute_year(company, o, prev_revenue=prev_rev2)
                yrs.append(yr)
                prev_rev2 = o.revenue
            wacc = climate_adjusted_wacc(company, scenario)
            dcf = value(company, yrs, wacc)
            valuation[label] = run(company, scenario, model_version=model_version)
        result.valuation_results = valuation

    # -------------------------------------------------------------------
    # Step 5: Composite CRI rating (FULL_CRI only)
    # -------------------------------------------------------------------
    if scope.needs_composite_rating and result.valuation_results:
        try:
            from ..outcomes.ratings import rate
            nze_r = result.valuation_results["nze"]
            dly_r = result.valuation_results["delayed"]
            cp_r  = result.valuation_results["cp"]
            # baseline_npv = CP enterprise value (no-action baseline)
            baseline = cp_r.enterprise_value
            nze_r2 = run(company, nze_scenario, baseline_npv=baseline)
            dly_r2 = run(company, dly_scenario, baseline_npv=baseline)
            result.rating_result = rate(nze_r2, dly_r2, cp_r)
        except Exception:
            pass   # rating is additive — a failure here doesn't break the run

    return result
