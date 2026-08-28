"""Canonical data contracts for the CRI engine.

Every Scenario, Company, Asset, Run, and Results object in the system is
modelled here. The API, engine, and data loaders all import from this
module — there is one source of truth.

These models are deliberately strict (Pydantic v2). If a field is missing
or has the wrong type, we fail loud at the edge rather than silently in
the middle of a DCF.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ScenarioFamily(str, Enum):
    """Canonical scenario families we support out of the box."""

    NZE_2050 = "nze_2050"
    BELOW_2C_ORDERLY = "below_2c_orderly"
    DELAYED_TRANSITION = "delayed_transition"
    CURRENT_POLICIES = "current_policies"
    CUSTOM = "custom"


class HazardType(str, Enum):
    """Physical climate hazards we model."""

    HEAT_STRESS = "heat_stress"
    WATER_STRESS = "water_stress"
    FLOOD = "flood"
    DROUGHT = "drought"
    CYCLONE = "cyclone"
    SEA_LEVEL = "sea_level"
    WILDFIRE = "wildfire"


class Commodity(str, Enum):
    """Commodity segments we support.

    Two groups:
    - Extractive / energy: iron_ore → electricity (original set)
    - Non-extractive sectors: beverages → agriculture (v0.4 expansion)

    For non-extractive assets the 'price' in scenario curves represents a
    revenue-per-unit proxy (e.g. USD/hl for beverages, USD/tonne for food)
    so that the asset simulation (volume × price = revenue) still works
    correctly without requiring a separate revenue-based model path.
    """

    # ── Extractive and energy commodities ────────────────────────────────────
    IRON_ORE = "iron_ore"
    COPPER = "copper"
    ALUMINIUM = "aluminium"
    COAL_THERMAL = "coal_thermal"
    COAL_METALLURGICAL = "coal_metallurgical"
    CRUDE_OIL = "crude_oil"
    NATURAL_GAS = "natural_gas"
    REFINED_PRODUCTS = "refined_products"
    CEMENT = "cement"
    ELECTRICITY = "electricity"

    # ── Non-extractive sector proxies (v0.4) ─────────────────────────────────
    BEVERAGES = "beverages"                    # beer, spirits, soft drinks — USD/hl
    FOOD = "food"                              # food manufacturing/processing — USD/tonne
    CHEMICALS = "chemicals"                    # specialty + commodity chemicals — USD/tonne
    MANUFACTURING = "manufacturing"            # general industrial — USD/unit
    RETAIL = "retail"                          # consumer retail — USD/basket equiv.
    FINANCIAL_SERVICES = "financial_services"  # banks, insurers — USD/loan notional
    REAL_ESTATE = "real_estate"                # commercial + residential — USD/sqm/year
    AGRICULTURE = "agriculture"                # crop + livestock — USD/tonne crop equiv.


class EmissionsScope(str, Enum):
    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"


class PhysicalDataQuality(str, Enum):
    """Data quality tier for physical hazard assessment per asset.

    LIVE              — NASA POWER observed baseline + Open-Meteo CMIP6 live
                        projection both succeeded; delta-downscaling applied.
    REGIONAL_BASELINE — lat/lon provided but live API unavailable; regional
                        WRI/GIS lookup with elevation correction.
    GLOBAL_FALLBACK   — No coordinates; global region-code lookup only.
    """
    LIVE = "live"
    REGIONAL_BASELINE = "regional_baseline"
    GLOBAL_FALLBACK = "global_fallback"


# ---------------------------------------------------------------------------
# Physical hazard provenance models
# ---------------------------------------------------------------------------


class HazardProvenance(BaseModel):
    """Data lineage record for a single hazard at a single asset.

    Tells the reader exactly where the score came from and whether any
    live observational data (satellite, API) was incorporated.
    """

    hazard: str
    data_source: str                # e.g. "WRI Aqueduct 3.0 embedded", "NASA POWER 2.3"
    notes: str = ""
    is_live: bool = False           # True when score used live API data (not embedded table)
    observed_active: bool = False   # True when satellite/GDACS confirms active event now


class AssetPhysicalProvenance(BaseModel):
    """Full data-lineage record for physical hazard assessment on one asset.

    Attached to RunResults.physical_hazard_detail so any consumer can audit
    exactly what data underpins each hazard score — which layer fired, whether
    live climate APIs responded, and whether real-time satellite observations
    were incorporated.
    """

    asset_id: str
    asset_name: str
    region: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    elevation_m: float = 0.0

    # Overall data quality classification
    data_quality: PhysicalDataQuality = PhysicalDataQuality.GLOBAL_FALLBACK

    # Downscaling method applied (e.g. "delta_cmip6_nasa_power", "embedded_tables_v1")
    downscaling_method: str = ""
    # Effective spatial resolution (e.g. "0.25deg NASA POWER", "25km WRI Aqueduct")
    spatial_resolution: str = ""

    # Per-hazard lineage records (only for applicable hazards)
    hazard_provenance: Dict[str, HazardProvenance] = Field(default_factory=dict)

    # Raw live-API payloads (None if APIs were unavailable)
    live_baseline: Optional[Dict] = None     # NASA POWER climatological normals
    live_projection: Optional[Dict] = None   # Open-Meteo CMIP6 delta

    # Real-time satellite observation summary (None if not attempted)
    satellite_observations: Optional[Dict] = None


# ---------------------------------------------------------------------------
# Scenario layer
# ---------------------------------------------------------------------------


class CarbonPricePath(BaseModel):
    """Carbon price in USD / tCO2e by year, potentially per-region."""

    region: str = "global"
    # year -> USD/tCO2e
    path: Dict[int, float]


class CommodityCurve(BaseModel):
    """Demand index (and/or price) path for one commodity under one scenario."""

    commodity: Commodity
    # year -> index, where base year == 100
    demand_index: Dict[int, float]
    # year -> USD per physical unit (e.g., USD/tonne)
    price_path: Dict[int, float]
    # Elasticity of volume to demand-index changes. Default 1.0 means
    # demand index fully translates to volume (before physical disruption).
    price_elasticity: float = 1.0


class HazardPath(BaseModel):
    """Expected hazard severity/frequency path per region."""

    hazard: HazardType
    region: str
    # year -> severity index (0..1) or expected annual production loss (fraction)
    path: Dict[int, float]


class Scenario(BaseModel):
    """A named, versioned climate scenario."""

    id: str                                    # e.g. "nze_2050_v1"
    name: str                                  # human-readable
    family: ScenarioFamily
    horizon: tuple[int, int]                   # (start_year, end_year) inclusive
    description: str = ""
    version: str = "0.1.0"

    carbon_prices: List[CarbonPricePath] = Field(default_factory=list)
    commodity_curves: List[CommodityCurve] = Field(default_factory=list)
    hazards: List[HazardPath] = Field(default_factory=list)

    # Additional risk premium (bps on WACC) applied under this scenario
    risk_premium_bps: int = 0

    # ── Custom abatement path ──────────────────────────────────────────────────
    # When set, overrides the hard-coded _SCENARIO_TARGETS lookup in abatement.py.
    # Dict maps milestone years to required cumulative abatement fraction (0–1).
    # e.g. {2030: 0.30, 2040: 0.60, 2050: 0.90}
    # Only used when family == ScenarioFamily.CUSTOM (ignored for named scenarios).
    abatement_targets: Optional[Dict[int, float]] = None

    @field_validator("horizon")
    @classmethod
    def _valid_horizon(cls, v: tuple[int, int]) -> tuple[int, int]:
        start, end = v
        if end <= start:
            raise ValueError("horizon end must be after start")
        return v

    # --- convenience lookups ------------------------------------------------

    def carbon_price(self, year: int, region: str = "global") -> float:
        """Carbon price in USD/tCO2e for (region, year), falling back to global."""
        for cp in self.carbon_prices:
            if cp.region == region and year in cp.path:
                return cp.path[year]
        for cp in self.carbon_prices:
            if cp.region == "global" and year in cp.path:
                return cp.path[year]
        return 0.0

    def commodity_curve(self, commodity: Commodity) -> Optional[CommodityCurve]:
        for c in self.commodity_curves:
            if c.commodity == commodity:
                return c
        return None

    def hazard(self, hazard: HazardType, region: str) -> Optional[HazardPath]:
        for h in self.hazards:
            if h.hazard == hazard and h.region == region:
                return h
        return None


# ---------------------------------------------------------------------------
# Company layer
# ---------------------------------------------------------------------------


class EmissionsProfile(BaseModel):
    """Emissions intensity at the company or asset level."""

    # tCO2e per physical unit of production (e.g., per tonne of ore)
    scope1_intensity: float = 0.0
    scope2_intensity: float = 0.0
    scope3_intensity: float = 0.0

    # Share of emissions actually subject to a carbon price in a given region.
    # Starts simple: a single global fraction. Can be refined to by-region later.
    carbon_price_coverage: float = 1.0

    # Free allowance fraction (EU ETS style)
    free_allocation: float = 0.0


class Asset(BaseModel):
    """A production unit (mine, refinery, mill, power plant)."""

    id: str
    name: str
    commodity: Commodity
    region: str                                     # e.g., "AU-WA", "CL-02", "global"
    baseline_production: float                      # units / year, at start year
    # For mines: often tonnes; for refineries: barrels; etc. Stored free-text.
    production_unit: str = "tonnes"
    emissions: EmissionsProfile = EmissionsProfile()
    carrying_value: float = 0.0                     # USD, used for stranded-asset writedowns
    remaining_life_years: Optional[int] = None
    # Unit operating cost at baseline year (USD per production unit)
    baseline_unit_cost: float = 0.0
    # Fraction of unit cost attributable to energy (proxy for carbon exposure)
    energy_cost_share: float = 0.3

    # ── GIS coordinates (decimal degrees, WGS-84) ────────────────────────────
    # When provided, PhysicalHazardEngine uses real spatial data (elevation,
    # coastal distance, terrain) instead of region-code lookup tables.
    # Source: operator-provided or geocoded from asset name/region centroid.
    lat: Optional[float] = None     # latitude  (positive = North)
    lon: Optional[float] = None     # longitude (positive = East)

    # Equipment type — drives hazard sensitivity multipliers
    # e.g. "open_pit_mine", "wind_farm", "lng_terminal", "pipeline", "solar_farm"
    equipment_type: Optional[str] = None


class SegmentBaseline(BaseModel):
    """Per-commodity baseline used when we don't yet have asset-level data."""

    commodity: Commodity
    revenue_baseline: float                          # USD
    volume_baseline: float                           # physical units
    ebitda_margin_baseline: float                    # 0..1
    emissions: EmissionsProfile = EmissionsProfile()


class Financials(BaseModel):
    """Company-wide baseline financials (most-recent fiscal year)."""

    revenue: float
    ebitda: float
    capex: float
    maintenance_capex_share: float = 0.6             # share of capex that is maintenance
    tax_rate: float = 0.25
    wacc_base: float = 0.08
    net_debt: float = 0.0
    shares_outstanding: float = 1.0
    market_cap: Optional[float] = None


class Company(BaseModel):
    """A company we can run the engine on."""

    id: str                                          # ticker or slug
    name: str
    sector: str                                      # free-text for now
    hq_region: str = "global"
    financials: Financials
    segments: List[SegmentBaseline] = Field(default_factory=list)
    assets: List[Asset] = Field(default_factory=list)
    # Multi-pillar exposure weights (from V1 methodology, kept as reporting lens)
    exposure_weight: float = 0.3
    transition_weight: float = 0.3
    # Data-quality flag influences a confidence multiplier in reporting
    data_quality: str = "medium"                    # "low" | "medium" | "high"


# ---------------------------------------------------------------------------
# Run + Results layer
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    """A single engine invocation."""

    scenario_id: str
    company_id: str
    # Optional per-run overrides (e.g., user slides carbon price)
    overrides: Dict[str, float] = Field(default_factory=dict)
    model_version: str = "0.3.0"


class YearResult(BaseModel):
    """Single-year output row."""

    year: int
    revenue: float
    opex: float
    carbon_cost: float
    physical_loss_cost: float
    ebitda: float
    da: float
    ebit: float
    nopat: float
    transition_capex: float
    adaptation_capex: float
    maintenance_capex: float
    working_capital_change: float
    fcf: float
    # Breakdowns for transparency
    revenue_by_commodity: Dict[str, float] = Field(default_factory=dict)
    emissions_by_scope: Dict[str, float] = Field(default_factory=dict)
    # Per-hazard dollar cost breakdown (USD) — populated by PhysicalHazardEngine
    # Keys: heat_stress, flood_riverine, flood_coastal, sea_level_rise,
    #       saltwater_intrusion, landslide, wildfire, cyclone, drought, water_stress
    physical_loss_by_hazard: Dict[str, float] = Field(default_factory=dict)
    # Stranded asset fields — populated by operations/company.py breakeven test
    stranded_writedown: float = 0.0
    stranded_assets: List[str] = Field(default_factory=list)


class RunResults(BaseModel):
    """Full results of one engine invocation."""

    run_id: str
    scenario_id: str
    company_id: str
    model_version: str

    # Per-year trajectory
    years: List[YearResult]

    # Valuation
    npv_fcf: float
    terminal_value: float
    enterprise_value: float
    equity_value: float
    implied_share_price: float
    wacc_used: float

    # Headline comparisons vs. baseline scenario (populated if provided)
    baseline_npv: Optional[float] = None
    npv_impact_pct: Optional[float] = None
    ebitda_compression_2030_pct: Optional[float] = None
    ebitda_compression_2040_pct: Optional[float] = None

    # Pillar scores (V1 lens retained for reporting)
    exposure_score: Optional[float] = None
    transition_score: Optional[float] = None
    financial_score: Optional[float] = None
    adaptive_score: Optional[float] = None

    # Provenance
    input_hash: str = ""
    scenario_version: str = ""

    # Physical hazard data lineage — one record per asset, attached by run_full().
    # None when the engine is invoked via run() directly (not run_full()).
    physical_hazard_detail: Optional[List[AssetPhysicalProvenance]] = None


# ---------------------------------------------------------------------------
# Scenario cascade result models
# ---------------------------------------------------------------------------


class CostCategory(str, Enum):
    """Top-level cost category for itemised cascade output."""
    PHYSICAL_DAMAGE = "physical_damage"          # Structural / equipment damage
    INVENTORY_LOSS = "inventory_loss"            # Raw material / finished goods
    PRODUCTION_LOSS = "production_loss"          # Lost output / margin
    SUPPLY_CHAIN = "supply_chain"                # Input shortages, logistics
    EMERGENCY_RESPONSE = "emergency_response"    # Crisis management, temporary fixes
    ENERGY_UTILITY = "energy_utility"            # Extra power, water, fuel costs
    LABOUR = "labour"                            # Overtime, evacuation, safety
    RECOVERY_CAPEX = "recovery_capex"            # Repair and rebuild investment
    INSURANCE = "insurance"                      # Deductibles, premium increase
    CONSEQUENTIAL = "consequential"              # Contract penalties, lost customers
    FINANCIAL = "financial"                      # Covenant breach, refinancing cost


class CostItem(BaseModel):
    """Single itemised financial cost line from a scenario cascade.

    Every number here must have a traceable assumption — ``source_assumption``
    records the methodology so analysts can audit and challenge individual lines.
    """
    category: CostCategory
    description: str                            # Human-readable line item name
    amount_usd: float                           # Estimated cost in USD
    duration_note: str = ""                     # e.g. "30 days × AUD 85k/day"
    confidence: str = "medium"                  # "low" | "medium" | "high"
    source_assumption: str = ""                 # Methodology / data source


class AssetCascadeResult(BaseModel):
    """Full financial impact cascade for one asset under one physical event."""

    asset_id: str
    asset_name: str
    region: str
    commodity: str

    # Hazard severity after event multipliers have been applied (0–5 scale)
    event_hazard_severity: Dict[str, float] = Field(default_factory=dict)
    # Hazards that are materially elevated by this event
    activated_hazards: List[str] = Field(default_factory=list)

    # Itemised cost breakdown
    cost_items: List[CostItem] = Field(default_factory=list)

    # Aggregated by category
    total_physical_damage_usd: float = 0.0
    total_production_loss_usd: float = 0.0
    total_supply_chain_usd: float = 0.0
    total_emergency_response_usd: float = 0.0
    total_recovery_capex_usd: float = 0.0
    total_consequential_usd: float = 0.0
    total_impact_usd: float = 0.0

    production_loss_pct: float = 0.0           # Fraction of annual output lost
    recovery_months: int = 0                    # Estimated months to full recovery

    # Asset-level narrative (1-2 sentences)
    narrative: str = ""


class ScenarioCascadeResult(BaseModel):
    """Company-wide financial cascade from a single physical climate event.

    This is the primary output of ScenarioCascadeEngine.run().  It contains
    per-asset breakdowns, company-level aggregates, and a structured narrative
    that can be used directly in investor reporting or credit committee briefs.
    """

    company_id: str
    company_name: str
    event_id: str
    event_name: str
    event_driver: str
    event_duration_months: int
    reference_year: int

    # Per-asset results
    asset_results: List[AssetCascadeResult] = Field(default_factory=list)

    # ── Company-level aggregates ──────────────────────────────────────────────
    total_physical_damage_usd: float = 0.0
    total_production_loss_usd: float = 0.0
    total_supply_chain_usd: float = 0.0
    total_emergency_response_usd: float = 0.0
    total_recovery_capex_usd: float = 0.0
    total_consequential_usd: float = 0.0
    grand_total_impact_usd: float = 0.0

    # ── Financial index metrics ───────────────────────────────────────────────
    # All expressed as fractions (0.15 = 15%) relative to company baseline
    ebitda_impact_pct: float = 0.0             # EBITDA compression
    revenue_impact_pct: float = 0.0            # Revenue reduction
    capex_burden_pct: float = 0.0              # Recovery capex / baseline capex
    # Credit proxy: excess cost as spread over baseline WACC (bps)
    implied_credit_spread_bps: float = 0.0
    # Estimated months to restore full company-wide operational capacity
    recovery_months: int = 0

    # ── Scenario narrative ────────────────────────────────────────────────────
    scenario_narrative: str = ""
    key_vulnerabilities: List[str] = Field(default_factory=list)
    # Historical analogs used for calibration
    historical_analogs: List[str] = Field(default_factory=list)

    # ── Cost breakdown summary (all items, all assets flattened) ─────────────
    all_cost_items: List[CostItem] = Field(default_factory=list)
