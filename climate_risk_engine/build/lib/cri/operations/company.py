"""Translate scenario drivers into company-level operational trajectories.

This is where the shape of the V2 model lives — but corrected: emissions
and capex are allocated per-asset/segment (no triple counting), Scope 3
is tracked separately, and margin is *derived* from revenue and cost
structure rather than assumed constant.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..climate.physical import adaptation_capex, expected_loss_fraction
from ..climate.transition import TransitionDrivers, demand_shift, price_at, resolve
from ..climate.hazard_layers import PhysicalHazardEngine
from ..climate.ssp_scenarios import ngfs_to_ssp
from ..data.schemas import Asset, Company, Scenario, ScenarioFamily, SegmentBaseline
from ..financial.abatement import (
    coverage_after_abatement,
    transition_capex as compute_transition_capex,
)

# Module-level singleton — avoid re-instantiating on every compute_year call
_HAZARD_ENGINE = PhysicalHazardEngine()

# Map ScenarioFamily → SSP id for PhysicalHazardEngine
_FAMILY_TO_SSP: dict[ScenarioFamily, str] = {
    ScenarioFamily.NZE_2050: "ssp126",
    ScenarioFamily.BELOW_2C_ORDERLY: "ssp245",
    ScenarioFamily.DELAYED_TRANSITION: "ssp245",
    ScenarioFamily.CURRENT_POLICIES: "ssp370",
    ScenarioFamily.CUSTOM: "ssp245",
}


@dataclass
class OperationalYear:
    """One year of operational output for a whole company."""

    year: int
    revenue: float
    opex: float
    carbon_cost: float
    physical_loss_cost: float
    emissions_scope1: float
    emissions_scope2: float
    emissions_scope3: float
    adaptation_capex: float
    transition_capex: float
    stranded_writedown: float
    revenue_by_commodity: dict[str, float]
    stranded_assets: list[str]
    # Per-hazard dollar cost breakdown (USD) — populated by PhysicalHazardEngine
    physical_loss_by_hazard: dict[str, float] = None   # type: ignore[assignment]

    def __post_init__(self):
        if self.physical_loss_by_hazard is None:
            self.physical_loss_by_hazard = {}


# ---------------------------------------------------------------------------
# Per-asset computation
# ---------------------------------------------------------------------------


def _volume_for_asset(
    asset: Asset,
    drivers: TransitionDrivers,
    year: int,
    scenario: Scenario,
) -> float:
    """Expected produced volume after demand-side shift and physical loss."""
    shift = demand_shift(drivers, asset.commodity)
    # Fully inelastic default (elasticity=1.0) — volume moves 1-to-1 with demand
    elasticity = drivers.commodity_elasticity.get(asset.commodity, 1.0)
    volume_after_demand = asset.baseline_production * (1.0 + elasticity * shift)

    loss = expected_loss_fraction(scenario, asset, year)
    return max(0.0, volume_after_demand * (1.0 - loss))


def _asset_contribution(
    asset: Asset,
    drivers: TransitionDrivers,
    year: int,
    scenario: Scenario,
) -> dict:
    price = price_at(drivers, asset.commodity) or 0.0
    volume = _volume_for_asset(asset, drivers, year, scenario)
    revenue = price * volume

    # Opex with a simple energy-cost index. Rising carbon price pushes up
    # energy cost; scenario commodity-price path absorbs some of it.
    energy_inflator = 1.0 + 0.5 * (drivers.carbon_price / 100.0)  # placeholder curve
    unit_cost = (
        asset.baseline_unit_cost
        * (1 - asset.energy_cost_share + asset.energy_cost_share * energy_inflator)
    )
    opex = volume * unit_cost

    # Emissions — only Scope 1 + 2 directly priced, Scope 3 tracked but not costed
    s1 = volume * asset.emissions.scope1_intensity
    s2 = volume * asset.emissions.scope2_intensity
    s3 = volume * asset.emissions.scope3_intensity

    # Abatement offset: transition_capex purchases real emission reductions.
    # Once infrastructure is in place, the company pays carbon cost only on
    # REMAINING (unabated) emissions. coverage_after_abatement(year) = 1 at
    # baseline, declining to 0.05 by 2050 under NZE 2050. Under CP it stays
    # near 1.0 (only ~5% abated by 2030). This avoids double-charging:
    # companies should not pay both full transition capex AND full carbon tax
    # on emissions that the capex is specifically deployed to eliminate.
    remaining_emissions_frac = coverage_after_abatement(
        scenario.family, year, custom_targets=scenario.abatement_targets
    )

    priced_emissions = (s1 + s2) * remaining_emissions_frac * \
        asset.emissions.carbon_price_coverage * (
        1 - asset.emissions.free_allocation
    )
    carbon_cost = priced_emissions * drivers.carbon_price

    # Physical loss cost — computed with per-hazard breakdown via PhysicalHazardEngine
    # This ensures the financial loss is attributed to individual hazards (heat, flood,
    # drought, cyclone, etc.) so disclosure reports show meaningful differentiation.
    undisrupted_volume = (
        asset.baseline_production
        * (1.0 + drivers.commodity_elasticity.get(asset.commodity, 1.0)
           * demand_shift(drivers, asset.commodity))
    )
    margin_per_unit = max(0.0, price - unit_cost)

    # Physical loss cost uses BASELINE (2026) commodity prices, not scenario-year prices.
    # Rationale: physical risk is a hazard-probability effect; the dollar cost should
    # reflect stable baseline economics so that scenario ordering (CP loss > NZE loss)
    # is governed by disruption probability alone — not by scenario commodity price
    # pathways (which belong to the transition/revenue channel and are already captured
    # in the revenue and EBITDA lines).
    _baseline_drivers = resolve(scenario, 2026)
    baseline_price = price_at(_baseline_drivers, asset.commodity) or price
    baseline_margin = max(0.0, baseline_price - unit_cost)

    ssp_id = _FAMILY_TO_SSP.get(scenario.family, "ssp245")
    try:
        hazard_profile = _HAZARD_ENGINE.assess(
            asset.id, asset.name, asset.region, year,
            ssp=ssp_id,
            lat=asset.lat,
            lon=asset.lon,
            equipment_type=asset.equipment_type,
            _depth=1,   # _depth=1 skips recursive critical_year/peak sub-calls
        )
        # Per-hazard dollar cost = foregone_volume_from_hazard × margin
        physical_loss_by_hazard: dict[str, float] = {}
        total_survival = 1.0
        for hname, h in hazard_profile.hazards.items():
            if h.applicable and h.production_loss_pct > 0:
                hazard_loss_cost = undisrupted_volume * h.production_loss_pct * baseline_margin
                physical_loss_by_hazard[hname] = round(hazard_loss_cost, 2)
                total_survival *= (1.0 - h.production_loss_pct)
        total_loss_frac = 1.0 - total_survival
        physical_loss_cost = undisrupted_volume * total_loss_frac * baseline_margin
    except Exception:
        # Fall back to scenario-based loss if hazard engine unavailable
        loss = expected_loss_fraction(scenario, asset, year)
        physical_loss_cost = undisrupted_volume * loss * baseline_margin
        physical_loss_by_hazard = {"combined": round(physical_loss_cost, 2)}

    # ── Stranded asset detection ──────────────────────────────────────────────
    # An asset is stranded when commodity price < full unit cost of production.
    # Full unit cost = operating unit_cost + carbon compliance cost per unit.
    # When stranded, a one-off book-value writedown is triggered in that year.
    # Writedown proxy = baseline_revenue × (1 / assumed 20y asset life × 15y remaining).
    # This is a v0.3 approximation; Phase 2 will use actual remaining book value.
    scope1_per_unit = asset.emissions.scope1_intensity  # tCO2 / unit
    carbon_cost_per_unit = (
        scope1_per_unit
        * asset.emissions.carbon_price_coverage
        * (1 - asset.emissions.free_allocation)
        * drivers.carbon_price
    )
    full_unit_cost = unit_cost + carbon_cost_per_unit
    is_stranded = price > 0 and price < full_unit_cost
    # Writedown: only trigger once (when price first drops below breakeven).
    # Here we compute the potential writedown; orchestrator sums across assets.
    stranded_writedown = 0.0
    if is_stranded:
        baseline_revenue = asset.baseline_production * asset.baseline_unit_cost
        remaining_value_frac = 0.75  # assume 15y remaining of a 20y asset life
        stranded_writedown = baseline_revenue * remaining_value_frac

    return {
        "commodity": asset.commodity.value,
        "revenue": revenue,
        "opex": opex,
        "carbon_cost": carbon_cost,
        "physical_loss_cost": physical_loss_cost,
        "physical_loss_by_hazard": physical_loss_by_hazard,
        "scope1": s1,
        "scope2": s2,
        "scope3": s3,
        "adaptation_capex": adaptation_capex(scenario, asset, year),
        "stranded_writedown": stranded_writedown,
        "is_stranded": is_stranded,
        "asset_name": asset.name,
    }


# ---------------------------------------------------------------------------
# Per-segment fallback (when no asset-level data is available)
# ---------------------------------------------------------------------------


def _segment_contribution(
    segment: SegmentBaseline,
    drivers: TransitionDrivers,
    year: int,
) -> dict:
    shift = demand_shift(drivers, segment.commodity)
    elasticity = drivers.commodity_elasticity.get(segment.commodity, 1.0)
    volume = segment.volume_baseline * (1.0 + elasticity * shift)

    price = price_at(drivers, segment.commodity) or 0.0
    revenue = price * volume

    # Opex derived from baseline EBITDA margin, then inflated by carbon exposure
    baseline_opex = segment.revenue_baseline * (1.0 - segment.ebitda_margin_baseline)
    energy_inflator = 1.0 + 0.3 * (drivers.carbon_price / 100.0)
    opex = baseline_opex * (volume / max(segment.volume_baseline, 1e-9)) * energy_inflator

    s1 = volume * segment.emissions.scope1_intensity
    s2 = volume * segment.emissions.scope2_intensity
    s3 = volume * segment.emissions.scope3_intensity

    priced_emissions = (s1 + s2) * segment.emissions.carbon_price_coverage * (
        1 - segment.emissions.free_allocation
    )
    carbon_cost = priced_emissions * drivers.carbon_price

    return {
        "commodity": segment.commodity.value,
        "revenue": revenue,
        "opex": opex,
        "carbon_cost": carbon_cost,
        "physical_loss_cost": 0.0,     # segment fallback has no geography
        "scope1": s1,
        "scope2": s2,
        "scope3": s3,
        "adaptation_capex": 0.0,
        "stranded_writedown": 0.0,     # segments have no breakeven detection
        "is_stranded": False,
        "asset_name": segment.commodity.value,
    }


# ---------------------------------------------------------------------------
# Company simulation
# ---------------------------------------------------------------------------


def simulate_year(
    company: Company,
    scenario: Scenario,
    year: int,
) -> OperationalYear:
    """Aggregate operational outputs for one year for one company."""
    drivers = resolve(scenario, year, region=company.hq_region)

    contributions: list[dict] = []

    if company.assets:
        contributions.extend(
            _asset_contribution(a, drivers, year, scenario) for a in company.assets
        )
    # Segments can be used alongside assets (for bits of the business we
    # haven't modelled asset-by-asset yet)
    contributions.extend(
        _segment_contribution(s, drivers, year) for s in company.segments
    )

    agg_revenue = sum(c["revenue"] for c in contributions)
    agg_opex = sum(c["opex"] for c in contributions)
    agg_carbon_cost = sum(c["carbon_cost"] for c in contributions)
    agg_physical = sum(c["physical_loss_cost"] for c in contributions)
    agg_adapt_capex = sum(c["adaptation_capex"] for c in contributions)
    agg_stranded_writedown = sum(c.get("stranded_writedown", 0.0) for c in contributions)
    stranded_asset_names = [
        c["asset_name"] for c in contributions if c.get("is_stranded", False)
    ]

    # Transition capex — computed from abatement module using company-level
    # Scope 1+2 baseline emissions and the scenario's carbon price path.
    total_s1s2 = sum(c["scope1"] + c["scope2"] for c in contributions)
    agg_transition_capex = compute_transition_capex(
        scenario.family, year, total_s1s2, drivers.carbon_price,
        custom_targets=scenario.abatement_targets
    )

    revenue_by_commodity: dict[str, float] = {}
    for c in contributions:
        revenue_by_commodity[c["commodity"]] = (
            revenue_by_commodity.get(c["commodity"], 0.0) + c["revenue"]
        )

    # Aggregate per-hazard loss costs across all assets
    agg_hazard_losses: dict[str, float] = {}
    for c in contributions:
        for hazard, cost in c.get("physical_loss_by_hazard", {}).items():
            agg_hazard_losses[hazard] = agg_hazard_losses.get(hazard, 0.0) + cost

    return OperationalYear(
        year=year,
        revenue=agg_revenue,
        opex=agg_opex,
        carbon_cost=agg_carbon_cost,
        physical_loss_cost=agg_physical,
        physical_loss_by_hazard=agg_hazard_losses,
        emissions_scope1=sum(c["scope1"] for c in contributions),
        emissions_scope2=sum(c["scope2"] for c in contributions),
        emissions_scope3=sum(c["scope3"] for c in contributions),
        adaptation_capex=agg_adapt_capex,
        # Transition capex: computed from abatement module (scenario-driven MACC).
        # Asset-level MACC curves with sector calibration are on the roadmap for v0.4.
        transition_capex=agg_transition_capex,
        # Stranded assets: assets where commodity price < full unit cost of production.
        # Writedown = remaining book value proxy (75% of baseline revenue, v0.3 approx).
        stranded_writedown=agg_stranded_writedown,
        stranded_assets=stranded_asset_names,
        revenue_by_commodity=revenue_by_commodity,
    )


def simulate(company: Company, scenario: Scenario) -> list[OperationalYear]:
    start, end = scenario.horizon
    return [simulate_year(company, scenario, y) for y in range(start, end + 1)]


# ---------------------------------------------------------------------------
# Physical hazard provenance
# ---------------------------------------------------------------------------

def get_physical_provenance(
    company: Company,
    scenario: Scenario,
    reference_year: int = 2026,
) -> list:
    """Build per-asset physical hazard data provenance records.

    Runs the hazard engine at *full resolution* (_depth=0) for each asset at
    ``reference_year`` and captures every provenance field that the normal
    simulation loop discards.

    Returns
    -------
    list[AssetPhysicalProvenance]
        One record per asset.  Assets that raise an unexpected exception are
        silently skipped — provenance is best-effort, never blocking.

    Data quality classification
    ---------------------------
    LIVE              live_projection["live"] == True
    REGIONAL_BASELINE lat/lon provided, but live APIs unavailable
    GLOBAL_FALLBACK   no coordinates — region-code lookup only
    """
    from ..data.schemas import AssetPhysicalProvenance, HazardProvenance, PhysicalDataQuality

    ssp_id = _FAMILY_TO_SSP.get(scenario.family, "ssp245")
    provenance_list: list = []

    for asset in company.assets:
        try:
            profile = _HAZARD_ENGINE.assess(
                asset.id,
                asset.name,
                asset.region,
                reference_year,
                ssp=ssp_id,
                lat=asset.lat,
                lon=asset.lon,
                equipment_type=asset.equipment_type,
                _depth=0,   # full resolution — triggers live API calls
            )

            # Determine data quality tier
            live_ok = bool(
                profile.live_projection and profile.live_projection.get("live")
            )
            if live_ok:
                quality = PhysicalDataQuality.LIVE
            elif asset.lat is not None and asset.lon is not None:
                quality = PhysicalDataQuality.REGIONAL_BASELINE
            else:
                quality = PhysicalDataQuality.GLOBAL_FALLBACK

            # Build per-hazard provenance records
            hazard_prov: dict = {}
            for hname, h in profile.hazards.items():
                if not h.applicable:
                    continue
                # Only heat_stress and flood_riverine can use live API data
                is_live_hazard = live_ok and hname in ("heat_stress", "flood_riverine")
                observed = bool(
                    getattr(profile, "observed_alerts", {}).get(hname, False)
                )
                hazard_prov[hname] = HazardProvenance(
                    hazard=hname,
                    data_source=h.data_source,
                    notes=h.notes,
                    is_live=is_live_hazard,
                    observed_active=observed,
                )

            # Satellite observation summary (populated when connectors return data)
            satellite_obs = None
            observed_alerts = getattr(profile, "observed_alerts", {})
            observation_sources = getattr(profile, "observation_sources", [])
            if observed_alerts or observation_sources:
                satellite_obs = {
                    "active_alerts": observed_alerts,
                    "sources": observation_sources,
                }

            provenance_list.append(
                AssetPhysicalProvenance(
                    asset_id=asset.id,
                    asset_name=asset.name,
                    region=asset.region,
                    lat=asset.lat,
                    lon=asset.lon,
                    elevation_m=profile.elevation_m,
                    data_quality=quality,
                    downscaling_method=profile.downscaling_method,
                    spatial_resolution=profile.spatial_resolution,
                    hazard_provenance=hazard_prov,
                    live_baseline=profile.live_baseline,
                    live_projection=profile.live_projection,
                    satellite_observations=satellite_obs,
                )
            )

        except Exception:
            # Provenance is supplementary — never let it break a run
            continue

    return provenance_list
