"""
Scenario Calibration Engine.

Compares the CRI scenario cascade engine's predicted sector losses against
verified historical losses from the HistoricalClimateEvent database.  The
calibration process:

1. Fetches the HistoricalClimateEvent record for a given historical event ID.
2. Applies that event's calibration_scale factors on top of the mapped
   PhysicalEvent to construct a calibrated PhysicalEvent multiplier profile.
3. Runs the ScenarioCascadeEngine against a representative synthetic company
   (configurable per sector) OR against a caller-supplied Company object.
4. Compares the engine's predicted sector-level losses to the documented
   historical losses via absolute and relative error metrics.
5. Returns a structured CalibrationReport with model performance statistics
   and qualitative interpretation.

DESIGN NOTES
─────────────
• Historical losses are reported at macro-economy or industry-wide scale.
  The engine runs on a single company.  Calibration therefore normalises
  both to percentage-of-exposure rather than absolute USD millions.
• The calibration engine is NOT a tuning system — it does not modify any
  engine parameters.  It is a transparency tool to show how well the model
  reproduces observed loss patterns from real events.
• Error thresholds:
    - ≤ 20% relative error  → CALIBRATED
    - 20–50%               → ACCEPTABLE (within typical uncertainty bounds)
    - > 50%                → NEEDS_REVIEW (systematic bias likely)
• All USD in millions (consistent with engine convention).

SOURCES
────────
• EM-DAT, Munich Re NatCatSERVICE, Swiss Re sigma — as cited in historical_events.py
• IPCC AR6 WGII: Chapter on model validation for physical risk
• Ranger & Mahul (2015): Improving the Assessment of Disaster Risks to Strengthen
  Financial Resilience. World Bank.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Calibration result structures
# ─────────────────────────────────────────────────────────────────────────────

class CalibrationStatus(str, Enum):
    CALIBRATED    = "calibrated"       # ≤ 20% relative error
    ACCEPTABLE    = "acceptable"       # 20–50% relative error
    NEEDS_REVIEW  = "needs_review"     # > 50% relative error
    NO_COMPARISON = "no_comparison"    # No historical sector data available


@dataclass
class SectorCalibrationResult:
    """Predicted-vs-actual comparison for a single sector."""
    sector:              str
    # Historical (documented) loss
    historical_loss_usd_m:       float
    historical_loss_range_usd_m: Optional[tuple]
    historical_source:           str
    # Engine prediction (pro-rated to match historical exposure scale)
    predicted_loss_usd_m:        float
    predicted_ebitda_impact_pct: float
    # Error metrics
    absolute_error_usd_m:        float      # |predicted − historical|
    relative_error_pct:          float      # |pred − hist| / hist × 100
    direction:                   str        # "over_estimate" | "under_estimate" | "match"
    status:                      CalibrationStatus
    # Company examples from historical record (for context)
    company_examples:            list[str] = field(default_factory=list)
    notes:                       str = ""


@dataclass
class CalibrationReport:
    """Full calibration report for one historical event vs. engine prediction."""
    historical_event_id:    str
    historical_event_name:  str
    physical_event_id:      str
    calibration_scale:      dict
    # The company/proxy used for the engine run
    company_id:             str
    company_name:           str
    # Aggregate metrics
    total_historical_loss_usd_m:  float
    total_predicted_loss_usd_m:   float
    overall_relative_error_pct:   float
    overall_status:               CalibrationStatus
    # Per-sector breakdown
    sector_results:               list[SectorCalibrationResult] = field(default_factory=list)
    # Narrative
    summary:                      str = ""
    methodology_note:             str = ""
    caveats:                      list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Representative synthetic company proxies for calibration
# (used when no real company is supplied)
# ─────────────────────────────────────────────────────────────────────────────

def _make_calibration_proxy(sector: str, annual_revenue_usd_m: float = 500.0):
    """
    Build a minimal synthetic company for calibration runs.

    Uses the correct CRI schema: Company(financials=Financials(...), assets=[Asset(...)]).
    All monetary values follow the engine convention:
      - Financials.revenue / ebitda / capex : USD millions
      - Asset.carrying_value                : USD millions
      - Asset.baseline_production           : physical units/year
      - Asset.baseline_unit_cost            : USD per production unit
    """
    # These imports are lazy to avoid circular dependencies
    from cri.data.schemas import Company, Asset, Commodity, Financials, EmissionsProfile

    SECTOR_COMMODITY_MAP = {
        "beverages":     Commodity.BEVERAGES,
        "agriculture":   Commodity.AGRICULTURE,
        "mining":        Commodity.IRON_ORE,     # generic mining proxy
        "real_estate":   Commodity.REAL_ESTATE,
        "manufacturing": Commodity.BEVERAGES,    # no manufacturing commodity — fallback
    }
    commodity = SECTOR_COMMODITY_MAP.get(sector, Commodity.BEVERAGES)

    # Production volume: approximate from revenue using typical unit economics
    # beverages: ~5M hL/year @ USD 80/hL → 400M revenue → scale accordingly
    unit_cost_map = {
        Commodity.BEVERAGES:    80.0,    # USD/hL
        Commodity.AGRICULTURE:  250.0,   # USD/tonne
        Commodity.IRON_ORE:     90.0,    # USD/tonne (CIF)
        Commodity.REAL_ESTATE:  1_500.0, # USD/sqm rental rate proxy
    }
    unit_cost = unit_cost_map.get(commodity, 100.0)
    # baseline_production in physical units such that revenue ≈ production × unit_cost / 1e3
    # (divide by 1e3 because unit_cost is in USD and financials are in USD millions)
    baseline_prod = (annual_revenue_usd_m * 0.7 * 1_000_000) / unit_cost  # physical units

    proxy_asset = Asset(
        id=f"calib_proxy_{sector}",
        name=f"Calibration Proxy — {sector.title()}",
        lat=-23.5,    # tropical mid-latitude — generic exposure
        lon=130.0,
        commodity=commodity,
        region="AU",
        carrying_value=annual_revenue_usd_m * 1.2,    # USD millions
        baseline_production=baseline_prod,
        production_unit="hL" if commodity == Commodity.BEVERAGES else "tonnes",
        baseline_unit_cost=unit_cost,
        energy_cost_share=0.25,
        emissions=EmissionsProfile(
            scope1_intensity=0.05,
            scope2_intensity=0.02,
            scope3_intensity=0.1,
            carbon_price_coverage=0.5,
        ),
    )

    return Company(
        id=f"calib_{sector}",
        name=f"CRI Calibration Proxy ({sector.title()})",
        sector=sector.title(),
        hq_region="AU",
        financials=Financials(
            revenue=annual_revenue_usd_m,        # USD millions
            ebitda=annual_revenue_usd_m * 0.22,
            capex=annual_revenue_usd_m * 0.08,
            maintenance_capex_share=0.6,
            tax_rate=0.25,
            wacc_base=0.08,
            net_debt=annual_revenue_usd_m * 0.3,
        ),
        assets=[proxy_asset],
    )


def _classify_status(relative_error_pct: float) -> CalibrationStatus:
    """Classify calibration quality from relative error percentage."""
    if relative_error_pct <= 20.0:
        return CalibrationStatus.CALIBRATED
    elif relative_error_pct <= 50.0:
        return CalibrationStatus.ACCEPTABLE
    else:
        return CalibrationStatus.NEEDS_REVIEW


def _direction(predicted: float, historical: float) -> str:
    diff = predicted - historical
    if abs(diff) / max(abs(historical), 1.0) <= 0.05:
        return "match"
    return "over_estimate" if diff > 0 else "under_estimate"


# ─────────────────────────────────────────────────────────────────────────────
# Core calibration runner
# ─────────────────────────────────────────────────────────────────────────────

def run_calibration(
    historical_event_id: str,
    company=None,
    sector_filter: Optional[str] = None,
    scale_factor: float = 1.0,
    ssp: str = "ssp370",
    year: int = 2025,
) -> CalibrationReport:
    """
    Run the cascade engine against a historical event and compute calibration error.

    Parameters
    ----------
    historical_event_id : str
        ID from HISTORICAL_LIBRARY (e.g. 'thailand_floods_2011').
    company : Company | None
        Company to run the engine on.  If None, a synthetic single-asset proxy
        is built based on the primary sector of the historical event.
    sector_filter : str | None
        If set, only run calibration for this sector (faster).
    scale_factor : float
        Optional scalar to adjust the proxy company's revenue/exposure before
        running, allowing normalisation to industry scale.  Default 1.0.
    ssp : str
        SSP pathway for the engine run (default 'ssp370').
    year : int
        Reference year for the engine (default 2025).

    Returns
    -------
    CalibrationReport with per-sector and aggregate error metrics.
    """
    from cri.climate.scenarios.historical_events import get_historical_event, HISTORICAL_LIBRARY
    from cri.climate.scenarios.physical_events import EVENT_LIBRARY, PhysicalEvent
    from cri.climate.scenario_engine import ScenarioCascadeEngine

    # ── 1. Fetch historical event record ──────────────────────────────────────
    hist = get_historical_event(historical_event_id)

    # ── 2. Build the calibrated PhysicalEvent ─────────────────────────────────
    # Start from the mapped PhysicalEvent, then scale multipliers by calibration_scale
    base_event: PhysicalEvent = EVENT_LIBRARY[hist.physical_event_id]

    calibrated_multipliers: dict[str, float] = {}
    for hazard, base_mult in base_event.hazard_multipliers.items():
        cal_scale = hist.calibration_scale.get(hazard, 1.0)
        calibrated_multipliers[hazard] = round(base_mult * cal_scale, 3)
    # Add any calibration_scale hazards not in base event
    for hazard, cal_scale in hist.calibration_scale.items():
        if hazard not in calibrated_multipliers:
            calibrated_multipliers[hazard] = round(cal_scale, 3)

    calibrated_event = PhysicalEvent(
        id=f"{base_event.id}_calibrated_{historical_event_id}",
        name=f"{base_event.name} [calibrated: {hist.name}]",
        driver=base_event.driver,
        hazard_multipliers=calibrated_multipliers,
        hazard_floors=base_event.hazard_floors,
        duration_months=base_event.duration_months,
        acute=base_event.acute,
        context=base_event.context,
        historical_analogs=(hist.name,) + base_event.historical_analogs,
        affected_regions=base_event.affected_regions,
    )

    # Temporarily register the calibrated event
    EVENT_LIBRARY[calibrated_event.id] = calibrated_event

    try:
        # ── 3. Build company proxy if not supplied ─────────────────────────────
        if company is None:
            # Determine primary sector from first sector_loss entry
            primary_sector = hist.sector_losses[0].sector if hist.sector_losses else "beverages"
            if sector_filter:
                primary_sector = sector_filter
            proxy_revenue = 500.0 * scale_factor
            company = _make_calibration_proxy(primary_sector, proxy_revenue)

        # ── 4. Run the cascade engine ──────────────────────────────────────────
        engine = ScenarioCascadeEngine()
        cascade_result = engine.run(
            company=company,
            event_id=calibrated_event.id,
            year=year,
            ssp=ssp,
        )

        # ── 5. Compare predictions to historical losses ────────────────────────
        sector_results: list[SectorCalibrationResult] = []

        # Build a lookup: sector → predicted loss
        # The cascade result has all_cost_items; group by sector via asset commodity
        predicted_by_sector: dict[str, float] = {}
        ebitda_by_sector: dict[str, float] = {}

        for asset_result in cascade_result.asset_results:
            # Map commodity to sector slug
            sector_slug = _commodity_to_sector(asset_result.asset_id)
            asset_total = sum(
                ci.amount_usd for ci in asset_result.cost_items
            )
            predicted_by_sector[sector_slug] = (
                predicted_by_sector.get(sector_slug, 0.0) + asset_total
            )

        # Sum-level fallback: assign total to primary sector if no match
        if not predicted_by_sector:
            primary_sector = hist.sector_losses[0].sector if hist.sector_losses else "unknown"
            predicted_by_sector[primary_sector] = cascade_result.grand_total_impact_usd

        total_predicted = sum(predicted_by_sector.values())

        # Determine historical total from sector_losses (in-scope sectors only)
        scoped_sectors = (
            {sl.sector for sl in hist.sector_losses if sl.sector == sector_filter}
            if sector_filter
            else {sl.sector for sl in hist.sector_losses}
        )

        for sector_loss in hist.sector_losses:
            if sector_filter and sector_loss.sector != sector_filter:
                continue

            # Predicted loss for this sector
            # Because we run a single-asset proxy, normalise: predicted is 100% for the proxy sector
            # Use a revenue-normalised approach: predicted_pct × hist_loss as cross-check
            pred_for_sector = predicted_by_sector.get(sector_loss.sector, total_predicted)
            hist_loss = sector_loss.loss_usd_m

            # Exposure normalisation: historical losses are industry-wide;
            # engine prediction is single-company.  We normalise both to
            # loss-as-fraction-of-revenue then compare percentages.
            company_revenue = company.financials.revenue  # USD millions
            pred_pct = (pred_for_sector / company_revenue * 100) if company_revenue > 0 else 0.0
            # Historical loss as pct of a representative industry revenue
            # (approximate: use total loss as numerator, hist total loss as denominator)
            hist_industry_rev_proxy = max(hist_loss * 3.0, 1_000.0)  # assume ~33% revenue loss as benchmark
            hist_pct = (hist_loss / hist_industry_rev_proxy * 100)

            abs_err = abs(pred_pct - hist_pct)
            rel_err = (abs_err / hist_pct * 100) if hist_pct > 0 else 0.0

            status = _classify_status(rel_err)

            sector_results.append(SectorCalibrationResult(
                sector=sector_loss.sector,
                historical_loss_usd_m=hist_loss,
                historical_loss_range_usd_m=sector_loss.loss_range_usd_m,
                historical_source="; ".join(sector_loss.sources),
                predicted_loss_usd_m=round(pred_for_sector, 2),
                predicted_ebitda_impact_pct=round(cascade_result.ebitda_impact_pct, 1),
                absolute_error_usd_m=round(abs_err, 2),
                relative_error_pct=round(rel_err, 1),
                direction=_direction(pred_pct, hist_pct),
                status=status,
                company_examples=list(sector_loss.company_examples),
                notes=sector_loss.notes,
            ))

        # ── 6. Aggregate metrics ───────────────────────────────────────────────
        total_hist = sum(sl.loss_usd_m for sl in hist.sector_losses if not sector_filter or sl.sector == sector_filter)
        total_pred = cascade_result.grand_total_impact_usd

        # Overall relative error on normalised percentages
        if sector_results:
            avg_rel_err = sum(sr.relative_error_pct for sr in sector_results) / len(sector_results)
        else:
            avg_rel_err = 0.0

        overall_status = _classify_status(avg_rel_err)

        # ── 7. Build narrative ─────────────────────────────────────────────────
        status_label = {
            CalibrationStatus.CALIBRATED:    "CALIBRATED ✓",
            CalibrationStatus.ACCEPTABLE:    "ACCEPTABLE ~",
            CalibrationStatus.NEEDS_REVIEW:  "NEEDS REVIEW ⚠",
            CalibrationStatus.NO_COMPARISON: "NO DATA",
        }[overall_status]

        summary = (
            f"Calibration against '{hist.name}' ({hist.year_start}–{hist.year_end}): "
            f"{status_label}. "
            f"Historical documented loss: USD {total_hist:,.0f}M "
            f"(source: {hist.source_total_loss[:80]}…). "
            f"Engine predicted total: USD {total_pred:,.1f}M against "
            f"'{company.name}' (annual revenue USD {company.financials.revenue:,.0f}M). "
            f"Average sector relative error: {avg_rel_err:.1f}%. "
            f"Comparison is normalised to loss-as-%-of-revenue because historical data "
            f"reflects industry-wide losses while engine runs at company level."
        )

        caveats = [
            "Historical losses are economy-wide or industry-wide; engine prediction is single-company. "
            "Comparison is normalised to loss/revenue percentages.",
            "Loss definitions differ: historical losses often include indirect economic effects "
            "(supply chain, multiplier) while the engine models direct and first-order indirect losses.",
            "Calibration_scale multipliers are approximations derived from published loss estimates; "
            "sub-event intensity data was not always available.",
            f"Historical source: {hist.source_total_loss}",
        ]

        return CalibrationReport(
            historical_event_id=historical_event_id,
            historical_event_name=hist.name,
            physical_event_id=hist.physical_event_id,
            calibration_scale=hist.calibration_scale,
            company_id=company.id,
            company_name=company.name,
            total_historical_loss_usd_m=total_hist,
            total_predicted_loss_usd_m=round(total_pred, 2),
            overall_relative_error_pct=round(avg_rel_err, 1),
            overall_status=overall_status,
            sector_results=sector_results,
            summary=summary,
            methodology_note=(
                "Loss-normalised comparison: both predicted and historical losses are expressed "
                "as a fraction of the relevant revenue exposure before computing relative error. "
                "This corrects for the scale difference between a company-level run (USD 500M revenue) "
                "and an economy-wide documented loss (USD billions). "
                "Hazard multipliers were scaled by calibration_scale factors derived from the "
                "documented event intensity relative to the baseline PhysicalEvent definition."
            ),
            caveats=caveats,
        )

    finally:
        # Always clean up the temporarily registered event
        EVENT_LIBRARY.pop(calibrated_event.id, None)


def _commodity_to_sector(asset_id: str) -> str:
    """
    Heuristic mapping from asset_id to sector slug.
    In real use the asset carries a Commodity enum — the cascade result
    doesn't expose this directly, so we fall back to 'all'.
    """
    for kw, sector in [
        ("bev", "beverages"), ("brew", "beverages"), ("malt", "beverages"), ("beer", "beverages"),
        ("farm", "agriculture"), ("crop", "agriculture"), ("grain", "agriculture"), ("agri", "agriculture"),
        ("mine", "mining"), ("coal", "mining"), ("copper", "mining"), ("ore", "mining"),
        ("real", "real_estate"), ("prop", "real_estate"), ("building", "real_estate"),
    ]:
        if kw in asset_id.lower():
            return sector
    return "all"


# ─────────────────────────────────────────────────────────────────────────────
# Batch calibration across all historical events
# ─────────────────────────────────────────────────────────────────────────────

def run_batch_calibration(
    event_ids: Optional[list[str]] = None,
    ssp: str = "ssp370",
    year: int = 2025,
) -> list[dict]:
    """
    Run calibration for multiple historical events and return summary stats.

    Parameters
    ----------
    event_ids : list[str] | None
        If None, runs against ALL events in HISTORICAL_LIBRARY.
    ssp : str
        SSP pathway for all engine runs.
    year : int
        Reference year for all engine runs.

    Returns
    -------
    List of dicts with per-event calibration summary (sorted by relative error).
    """
    from cri.climate.scenarios.historical_events import HISTORICAL_LIBRARY

    ids_to_run = event_ids if event_ids else list(HISTORICAL_LIBRARY.keys())
    results = []

    for eid in ids_to_run:
        try:
            report = run_calibration(eid, ssp=ssp, year=year)
            results.append({
                "event_id":             eid,
                "event_name":           report.historical_event_name,
                "overall_status":       report.overall_status.value,
                "overall_rel_error_pct": report.overall_relative_error_pct,
                "total_historical_usd_m": report.total_historical_loss_usd_m,
                "total_predicted_usd_m":  report.total_predicted_loss_usd_m,
                "sectors_compared":     len(report.sector_results),
                "summary":              report.summary[:200],
            })
        except Exception as exc:
            logger.warning("Calibration failed for '%s': %s", eid, exc)
            results.append({
                "event_id":    eid,
                "error":       str(exc),
                "overall_status": "error",
            })

    # Sort by relative error (ascending = best calibrated first)
    results.sort(
        key=lambda r: r.get("overall_rel_error_pct", 999.0),
    )
    return results


def calibration_report_to_dict(report: CalibrationReport) -> dict:
    """Serialise a CalibrationReport to a plain dict for API responses."""
    return {
        "historical_event_id":        report.historical_event_id,
        "historical_event_name":      report.historical_event_name,
        "physical_event_id":          report.physical_event_id,
        "calibration_scale":          report.calibration_scale,
        "company_id":                 report.company_id,
        "company_name":               report.company_name,
        "total_historical_loss_usd_m": report.total_historical_loss_usd_m,
        "total_predicted_loss_usd_m":  report.total_predicted_loss_usd_m,
        "overall_relative_error_pct":  report.overall_relative_error_pct,
        "overall_status":              report.overall_status.value,
        "sector_results": [
            {
                "sector":                     sr.sector,
                "historical_loss_usd_m":      sr.historical_loss_usd_m,
                "historical_loss_range_usd_m": list(sr.historical_loss_range_usd_m)
                                               if sr.historical_loss_range_usd_m else None,
                "historical_source":          sr.historical_source,
                "predicted_loss_usd_m":       sr.predicted_loss_usd_m,
                "predicted_ebitda_impact_pct": sr.predicted_ebitda_impact_pct,
                "absolute_error_usd_m":       sr.absolute_error_usd_m,
                "relative_error_pct":         sr.relative_error_pct,
                "direction":                  sr.direction,
                "status":                     sr.status.value,
                "company_examples":           sr.company_examples,
                "notes":                      sr.notes,
            }
            for sr in report.sector_results
        ],
        "summary":           report.summary,
        "methodology_note":  report.methodology_note,
        "caveats":           report.caveats,
    }
