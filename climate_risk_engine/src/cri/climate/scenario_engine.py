"""
Physical Climate Scenario Cascade Engine.

Takes a Company, a PhysicalEvent from the event library, and a reference year,
and produces a ScenarioCascadeResult containing:

  • Per-asset itemised cost breakdowns (CostItem lists)
  • Company-level financial aggregates
  • Physical-financial index metrics (EBITDA haircut, credit spread proxy)
  • A structured narrative for investor/credit reporting

ARCHITECTURE
────────────
1. For each company asset, retrieve the baseline AssetHazardProfile by calling
   PhysicalHazardEngine.assess() at _depth=0 (full resolution).

2. Apply PhysicalEvent.hazard_multipliers to each HazardScore.severity_index
   (with ceiling at 5.0) and enforce any hazard_floors.

3. Pass the modified profile to the sector-specific damage chain function
   (get_sector_chain(commodity)) which returns a list[CostItem].

4. Aggregate costs to company level, compute financial metrics, generate
   narrative, return ScenarioCascadeResult.

THREADING
─────────
Asset assessment calls (step 1) can be parallelised — each asset is
independent.  The engine uses concurrent.futures.ThreadPoolExecutor when
more than 3 assets are present.

USAGE
─────
    engine = ScenarioCascadeEngine()
    result = engine.run(company, "el_nino_super_drought", year=2026)
    print(f"Total impact: USD {result.grand_total_impact_usd:,.0f}")
    print(f"EBITDA compression: {result.ebitda_impact_pct*100:.1f}%")
    for ar in result.asset_results:
        print(f"\\n{ar.asset_name}:")
        for item in ar.cost_items:
            print(f"  {item.category.value:25s} {item.description[:50]:<50s} "
                  f"USD {item.amount_usd:>12,.0f}")
"""

from __future__ import annotations

import concurrent.futures
import math
from dataclasses import replace
from typing import Optional

from ..data.schemas import (
    AssetCascadeResult,
    CostCategory,
    CostItem,
    ScenarioCascadeResult,
    Company,
    Scenario,
)
from .hazard_layers import PhysicalHazardEngine, AssetHazardProfile, HazardScore
from .scenarios.physical_events import PhysicalEvent, get_event
from .scenarios.sector_chains import get_sector_chain
from ..climate.ssp_scenarios import ngfs_to_ssp

# Module-level hazard engine singleton (mirrors operations/company.py pattern)
_HAZARD_ENGINE = PhysicalHazardEngine()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_event_to_profile(
    profile: AssetHazardProfile,
    event: PhysicalEvent,
) -> AssetHazardProfile:
    """Apply event hazard multipliers and floors to a baseline hazard profile.

    Returns a new AssetHazardProfile (the original is not modified) with
    severity_index values scaled by the event's multipliers, floored by
    hazard_floors, and capped at 5.0.
    """
    new_hazards: dict[str, HazardScore] = {}
    for hname, h in profile.hazards.items():
        if not h.applicable:
            new_hazards[hname] = h
            continue

        mult = event.hazard_multipliers.get(hname, 1.0)
        floor = event.hazard_floors.get(hname, 0.0)
        new_sev = min(5.0, max(floor, h.severity_index * mult))

        # Rebuild HazardScore with updated severity (preserve all fields)
        new_score = HazardScore(
            hazard=h.hazard,
            annual_probability=min(1.0, h.annual_probability * mult),
            severity_index=round(new_sev, 3),
            production_loss_pct=min(1.0, h.production_loss_pct * mult),
            trend_2030=min(5.0, max(floor, h.trend_2030 * mult)),
            trend_2050=min(5.0, max(floor, h.trend_2050 * mult)),
            applicable=h.applicable,
            data_source=h.data_source,
            notes=h.notes + f" [event×{mult:.2f}]",
        )
        new_hazards[hname] = new_score

    # Rebuild profile preserving all metadata
    return AssetHazardProfile(
        asset_id=profile.asset_id,
        asset_name=profile.asset_name,
        region=profile.region,
        lat=profile.lat,
        lon=profile.lon,
        elevation_m=profile.elevation_m,
        is_coastal=profile.is_coastal,
        coastal_factor=profile.coastal_factor,
        lulc_type=profile.lulc_type,
        ssp=profile.ssp,
        hazards=new_hazards,
        physical_risk_score=profile.physical_risk_score,
        annual_loss_pct=profile.annual_loss_pct,
        peak_loss_2050_pct=profile.peak_loss_2050_pct,
        top_hazards=profile.top_hazards,
        critical_year=profile.critical_year,
        spatial_resolution=profile.spatial_resolution,
        downscaling_method=profile.downscaling_method,
        live_baseline=profile.live_baseline,
        live_projection=profile.live_projection,
    )


def _activated_hazards(event: PhysicalEvent, profile: AssetHazardProfile) -> list[str]:
    """List hazards materially elevated (multiplier > 1.2) with severity > 1.5."""
    activated = []
    for hname, h in profile.hazards.items():
        if not h.applicable:
            continue
        mult = event.hazard_multipliers.get(hname, 1.0)
        floor = event.hazard_floors.get(hname, 0.0)
        effective_sev = min(5.0, max(floor, h.severity_index * mult))
        if effective_sev >= 1.5 and (mult > 1.2 or floor > 1.5):
            activated.append(hname)
    return sorted(
        activated,
        key=lambda h: profile.hazards[h].severity_index if h in profile.hazards else 0.0,
        reverse=True,
    )


def _aggregate_by_category(cost_items: list[CostItem]) -> dict[str, float]:
    """Sum costs by CostCategory, return dict[category_value, usd]."""
    agg: dict[str, float] = {}
    for item in cost_items:
        agg[item.category.value] = agg.get(item.category.value, 0.0) + item.amount_usd
    return agg


def _recovery_months(cost_items: list[CostItem]) -> int:
    """Estimate recovery duration from the severity of physical damage items."""
    damage = sum(i.amount_usd for i in cost_items
                 if i.category == CostCategory.PHYSICAL_DAMAGE)
    if damage == 0:
        return 1
    if damage < 500_000:
        return 2
    if damage < 5_000_000:
        return 6
    if damage < 20_000_000:
        return 12
    return 24


def _implied_credit_spread(ebitda_impact_pct: float) -> float:
    """Approximate credit spread widening (bps) from EBITDA compression.

    Calibrated to Moody's credit rating transition data:
    ~10% EBITDA stress ≈ 20 bps; ~30% ≈ 80 bps; ~50%+ ≈ 200+ bps.
    """
    if ebitda_impact_pct <= 0:
        return 0.0
    # Piecewise linear interpolation
    if ebitda_impact_pct <= 0.10:
        return ebitda_impact_pct * 200       # 0–20 bps
    if ebitda_impact_pct <= 0.30:
        return 20 + (ebitda_impact_pct - 0.10) * 300   # 20–80 bps
    if ebitda_impact_pct <= 0.50:
        return 80 + (ebitda_impact_pct - 0.30) * 600   # 80–200 bps
    return min(500.0, 200 + (ebitda_impact_pct - 0.50) * 1200)  # 200–500 bps


def _narrative(
    company: Company,
    event: PhysicalEvent,
    result: "ScenarioCascadeResult",
) -> str:
    """Generate a structured scenario narrative for credit/investor reporting."""
    impact_m = result.grand_total_impact_usd  # already in USD millions
    ebitda_pct = result.ebitda_impact_pct * 100
    top_assets = sorted(result.asset_results, key=lambda a: a.total_impact_usd, reverse=True)
    top_names = ", ".join(a.asset_name for a in top_assets[:3])

    top_cost_cat = max(
        (CostCategory.PRODUCTION_LOSS, CostCategory.PHYSICAL_DAMAGE,
         CostCategory.SUPPLY_CHAIN, CostCategory.EMERGENCY_RESPONSE),
        key=lambda c: sum(i.amount_usd for i in result.all_cost_items if i.category == c),
        default=CostCategory.PRODUCTION_LOSS,
    )

    analogs_str = "; ".join(event.historical_analogs[:2]) if event.historical_analogs else ""
    analog_note = f" Historical precedents: {analogs_str}." if analogs_str else ""

    lines = [
        f"SCENARIO: {event.name}",
        f"",
        f"Under a {event.name.lower()} scenario (duration: {event.duration_months} months, "
        f"driver: {event.driver.value.replace('_', ' ').title()}), {company.name} faces "
        f"an estimated total financial impact of USD {impact_m:.1f}M, representing "
        f"an EBITDA compression of {ebitda_pct:.1f}%.",
        f"",
        f"Most-exposed assets: {top_names}. "
        f"The dominant cost driver is {top_cost_cat.value.replace('_', ' ')} "
        f"(USD {sum(i.amount_usd for i in result.all_cost_items if i.category == top_cost_cat):.1f}M).",
        f"",
        f"Physical damage costs of USD {result.total_physical_damage_usd:.1f}M reflect "
        f"direct asset impairment. Production and revenue losses of "
        f"USD {result.total_production_loss_usd:.1f}M arise from operational curtailment "
        f"and distribution disruption. Recovery capex of "
        f"USD {result.total_recovery_capex_usd:.1f}M is required to restore full capacity.",
        f"",
        f"The implied credit spread widening is estimated at "
        f"{result.implied_credit_spread_bps:.0f} bps (Moody's transition calibration). "
        f"Management response capability and insurance coverage will materially "
        f"influence the realised financial impact.",
    ]
    if analog_note:
        lines.append(f"")
        lines.append(f"Calibration note:{analog_note}")

    return "\n".join(lines)


def _key_vulnerabilities(
    company: Company,
    event: PhysicalEvent,
    asset_results: list[AssetCascadeResult],
) -> list[str]:
    """Identify 3–5 key vulnerability statements for the cascade result."""
    vulns = []
    total_damage = sum(a.total_physical_damage_usd for a in asset_results)
    total_prod = sum(a.total_production_loss_usd for a in asset_results)
    total_supply = sum(a.total_supply_chain_usd for a in asset_results)

    # Water-related
    water_hazards = {"water_stress", "drought"}
    water_assets = [a for a in asset_results
                    if any(h in a.activated_hazards for h in water_hazards)]
    if water_assets:
        vulns.append(
            f"{len(water_assets)} asset(s) face water stress / drought activation — "
            f"water-intensive operations at risk of allocation cuts."
        )

    # Flood
    flood_hazards = {"flood_riverine", "flood_coastal", "saltwater_intrusion"}
    flood_assets = [a for a in asset_results
                    if any(h in a.activated_hazards for h in flood_hazards)]
    if flood_assets and total_damage > 0:
        vulns.append(
            f"Flood exposure in {len(flood_assets)} asset(s): "
            f"USD {total_damage/1e6:.1f}M physical damage risk."
        )

    # Production loss dominance
    if total_prod > total_damage * 1.5:
        vulns.append(
            "Production loss exceeds physical damage — supply chain and "
            "operational disruption are the primary financial risk pathways."
        )

    # Cyclone
    cyclone_assets = [a for a in asset_results if "cyclone" in a.activated_hazards]
    if cyclone_assets:
        vulns.append(
            f"Cyclone exposure at {len(cyclone_assets)} asset(s) — "
            f"structural damage and logistics disruption risk."
        )

    # Highly exposed single asset concentration
    if asset_results:
        top = max(asset_results, key=lambda a: a.total_impact_usd)
        total_all = sum(a.total_impact_usd for a in asset_results)
        if total_all > 0 and top.total_impact_usd / total_all > 0.50:
            vulns.append(
                f"Asset concentration risk: {top.asset_name} accounts for "
                f"{top.total_impact_usd/total_all*100:.0f}% of total scenario impact."
            )

    return vulns[:5]


# ---------------------------------------------------------------------------
# Per-asset cascade worker
# ---------------------------------------------------------------------------

def _process_asset(args: tuple) -> Optional[AssetCascadeResult]:
    """Worker function for parallel asset processing.

    Returns AssetCascadeResult or None on failure.
    """
    asset, company, event, ssp_id, year = args
    try:
        # Step 1: baseline hazard profile at full resolution
        baseline_profile = _HAZARD_ENGINE.assess(
            asset.id,
            asset.name,
            asset.region,
            year,
            ssp=ssp_id,
            lat=asset.lat,
            lon=asset.lon,
            equipment_type=asset.equipment_type,
            _depth=0,
        )

        # Step 2: apply event multipliers
        event_profile = _apply_event_to_profile(baseline_profile, event)
        activated = _activated_hazards(event, baseline_profile)

        # Capture event-modified hazard severity map
        event_severity: dict[str, float] = {
            hname: round(h.severity_index, 2)
            for hname, h in event_profile.hazards.items()
            if h.applicable
        }

        # Step 3: run sector damage chain
        chain_fn = get_sector_chain(asset.commodity.value)
        cost_items = chain_fn(event, event_profile, asset, company)

        # Filter out zero-value items (keep non-zero)
        cost_items = [c for c in cost_items if c.amount_usd > 0]

        # Step 4: aggregate by category
        agg = _aggregate_by_category(cost_items)

        # Production loss fraction estimate
        total_usd = sum(i.amount_usd for i in cost_items)
        rev_proxy = max(
            asset.carrying_value * 0.6,
            company.financials.revenue / max(len(company.assets), 1)
        )
        prod_loss_pct = min(1.0, (
            agg.get(CostCategory.PRODUCTION_LOSS.value, 0) / max(rev_proxy, 1)
        ))

        narrative_asset = (
            f"{asset.name} ({asset.region}): {len(activated)} hazard(s) activated "
            f"({', '.join(activated[:3])}). "
            f"Estimated impact USD {total_usd/1e6:.1f}M "
            f"({prod_loss_pct*100:.0f}% production loss)."
        ) if activated else f"{asset.name}: no material hazard activation under this event."

        return AssetCascadeResult(
            asset_id=asset.id,
            asset_name=asset.name,
            region=asset.region,
            commodity=asset.commodity.value,
            event_hazard_severity=event_severity,
            activated_hazards=activated,
            cost_items=cost_items,
            total_physical_damage_usd=agg.get(CostCategory.PHYSICAL_DAMAGE.value, 0),
            total_production_loss_usd=agg.get(CostCategory.PRODUCTION_LOSS.value, 0),
            total_supply_chain_usd=agg.get(CostCategory.SUPPLY_CHAIN.value, 0),
            total_emergency_response_usd=agg.get(CostCategory.EMERGENCY_RESPONSE.value, 0),
            total_recovery_capex_usd=agg.get(CostCategory.RECOVERY_CAPEX.value, 0),
            total_consequential_usd=agg.get(CostCategory.CONSEQUENTIAL.value, 0),
            total_impact_usd=total_usd,
            production_loss_pct=round(prod_loss_pct, 4),
            recovery_months=_recovery_months(cost_items),
            narrative=narrative_asset,
        )

    except Exception as exc:
        # Never let one bad asset break the full run
        return AssetCascadeResult(
            asset_id=asset.id,
            asset_name=asset.name,
            region=asset.region,
            commodity=asset.commodity.value,
            narrative=f"Error during cascade computation: {exc}",
        )


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class ScenarioCascadeEngine:
    """Compute the full financial cascade of a physical climate event on a company.

    Usage
    -----
    engine = ScenarioCascadeEngine()
    result = engine.run(company, "el_nino_super_drought", year=2026)
    """

    def __init__(self, max_workers: int = 8) -> None:
        self._max_workers = max_workers

    def run(
        self,
        company: Company,
        event_id: str,
        year: int = 2026,
        ssp: str = "ssp370",
    ) -> ScenarioCascadeResult:
        """Run the full physical cascade for a company under a named event.

        Parameters
        ----------
        company  : Fully populated Company (assets, financials, emissions).
        event_id : Key from EVENT_LIBRARY (e.g. "el_nino_super_drought").
        year     : Reference year for hazard baseline (default 2026).
        ssp      : SSP scenario for baseline hazard retrieval (default ssp370 =
                   Current Policies — gives the highest baseline to multiply from).

        Returns
        -------
        ScenarioCascadeResult with itemised costs, aggregates, and narrative.
        """
        event = get_event(event_id)

        # Parallelise asset processing for companies with many assets
        args_list = [(asset, company, event, ssp, year) for asset in company.assets]

        if len(args_list) > 3:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(self._max_workers, len(args_list))
            ) as ex:
                asset_results = list(ex.map(_process_asset, args_list))
        else:
            asset_results = [_process_asset(a) for a in args_list]

        # Filter None results (shouldn't happen, but defensive)
        asset_results = [ar for ar in asset_results if ar is not None]

        # ── Company-level aggregates ──────────────────────────────────────────
        total_phys   = sum(a.total_physical_damage_usd for a in asset_results)
        total_prod   = sum(a.total_production_loss_usd for a in asset_results)
        total_supply = sum(a.total_supply_chain_usd for a in asset_results)
        total_emerg  = sum(a.total_emergency_response_usd for a in asset_results)
        total_capex  = sum(a.total_recovery_capex_usd for a in asset_results)
        total_consq  = sum(a.total_consequential_usd for a in asset_results)
        grand_total  = sum(a.total_impact_usd for a in asset_results)

        # ── Financial index metrics ───────────────────────────────────────────
        baseline_ebitda = max(company.financials.ebitda, 1.0)
        baseline_rev    = max(company.financials.revenue, 1.0)
        baseline_capex  = max(company.financials.capex, 1.0)

        ebitda_impact_pct = min(1.0, (total_prod + total_emerg + total_supply) / baseline_ebitda)
        revenue_impact_pct = min(1.0, total_prod / baseline_rev)
        capex_burden_pct   = min(5.0, total_capex / baseline_capex)
        credit_spread_bps  = _implied_credit_spread(ebitda_impact_pct)
        # Company recovery = worst single-asset recovery (limiting constraint)
        company_recovery_months = max((a.recovery_months for a in asset_results), default=1)

        # ── Flatten all cost items for cross-asset analysis ───────────────────
        all_items: list[CostItem] = []
        for ar in asset_results:
            all_items.extend(ar.cost_items)

        # ── Build preliminary result for narrative generation ─────────────────
        result = ScenarioCascadeResult(
            company_id=company.id,
            company_name=company.name,
            event_id=event.id,
            event_name=event.name,
            event_driver=event.driver.value,
            event_duration_months=event.duration_months,
            reference_year=year,
            asset_results=asset_results,
            total_physical_damage_usd=round(total_phys),
            total_production_loss_usd=round(total_prod),
            total_supply_chain_usd=round(total_supply),
            total_emergency_response_usd=round(total_emerg),
            total_recovery_capex_usd=round(total_capex),
            total_consequential_usd=round(total_consq),
            grand_total_impact_usd=round(grand_total),
            ebitda_impact_pct=round(ebitda_impact_pct, 4),
            revenue_impact_pct=round(revenue_impact_pct, 4),
            capex_burden_pct=round(capex_burden_pct, 4),
            implied_credit_spread_bps=round(credit_spread_bps, 1),
            recovery_months=company_recovery_months,
            scenario_narrative=_narrative(company, event, result := ScenarioCascadeResult(
                company_id=company.id,
                company_name=company.name,
                event_id=event.id,
                event_name=event.name,
                event_driver=event.driver.value,
                event_duration_months=event.duration_months,
                reference_year=year,
                asset_results=asset_results,
                total_physical_damage_usd=round(total_phys),
                total_production_loss_usd=round(total_prod),
                total_supply_chain_usd=round(total_supply),
                total_emergency_response_usd=round(total_emerg),
                total_recovery_capex_usd=round(total_capex),
                total_consequential_usd=round(total_consq),
                grand_total_impact_usd=round(grand_total),
                ebitda_impact_pct=round(ebitda_impact_pct, 4),
                revenue_impact_pct=round(revenue_impact_pct, 4),
                capex_burden_pct=round(capex_burden_pct, 4),
                implied_credit_spread_bps=round(credit_spread_bps, 1),
                recovery_months=company_recovery_months,
                historical_analogs=list(event.historical_analogs),
                all_cost_items=all_items,
            )),
            key_vulnerabilities=_key_vulnerabilities(company, event, asset_results),
            historical_analogs=list(event.historical_analogs),
            all_cost_items=all_items,
        )

        # Patch narrative into result (we already built it against the preliminary)
        result.scenario_narrative = _narrative(company, event, result)

        return result

    def run_multi(
        self,
        company: Company,
        event_ids: list[str],
        year: int = 2026,
        ssp: str = "ssp370",
    ) -> dict[str, ScenarioCascadeResult]:
        """Run multiple events and return a dict of {event_id: result}.

        Useful for stress-testing across the full event library or a subset.
        """
        return {eid: self.run(company, eid, year=year, ssp=ssp) for eid in event_ids}

    def worst_case(
        self,
        company: Company,
        event_ids: Optional[list[str]] = None,
        year: int = 2026,
    ) -> ScenarioCascadeResult:
        """Run all (or specified) events and return the worst-case result by EBITDA impact."""
        from .scenarios.physical_events import EVENT_LIBRARY
        ids = event_ids or list(EVENT_LIBRARY.keys())
        results = self.run_multi(company, ids, year=year)
        return max(results.values(), key=lambda r: r.ebitda_impact_pct)
