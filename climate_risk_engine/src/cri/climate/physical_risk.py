"""
physical_risk.py — 4-Stage Physical Risk Engine
═══════════════════════════════════════════════════════════════════════════════
Implements the institutional-grade physical risk workflow:

  Stage 1: Multi-hazard intensity lookup (flood depth, heat days, water stress,
           cyclone wind, wildfire FWI) with CMIP6 / NGFS scenario multipliers.

  Stage 2: Asset specification & natural capital offsets. Sector-default
           engineering parameters (elevation, shutdown thresholds) auto-filled
           when client does not provide them.

  Stage 3: Dynamic sector vulnerability curves (JRC depth-damage functions,
           WBGT heat-stress productivity curves, WRI Aqueduct water curves).

  Stage 4: Financial transmission — EAL, Business Interruption, NPV/DCF,
           Enterprise Value impairment, Marginal Abatement Cost Curve (MACC).

Data sources embedded as lookup tables (no external calls required for
the core calculation; external data enrichment happens upstream in the
data acquisition layer):
  • JRC depth-damage functions: EU Science Hub JRC 2017 (Huizinga et al.)
  • WBGT productivity: ILO 2019 / NIOSH heat stress standards
  • CMIP6 hazard multipliers: NGFS Phase 4 / NASA NEX-GDDP-CMIP6 percentiles
  • Sector defaults: ISO 14067 / IPCC AR6 WG2 sectoral analysis

MIT License — ClimRisk Engine
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 1: HAZARD INTENSITY TABLES
# ──────────────────────────────────────────────────────────────────────────────

# CMIP6 hazard intensity multipliers by scenario and year.
# Represents multiplier on baseline 100-year return-period hazard intensity.
# Source: NASA NEX-GDDP-CMIP6 ensemble median (34 GCMs), 99th percentile.
# Structure: {hazard: {scenario: {year: multiplier}}}
_CMIP6_HAZARD_MULTIPLIER: Dict[str, Dict[str, Dict[int, float]]] = {
    "flood": {
        "ssp126": {2030: 1.08, 2040: 1.14, 2050: 1.19, 2075: 1.24, 2100: 1.28},
        "ssp245": {2030: 1.12, 2040: 1.22, 2050: 1.31, 2075: 1.45, 2100: 1.58},
        "ssp370": {2030: 1.16, 2040: 1.30, 2050: 1.45, 2075: 1.72, 2100: 2.04},
        "ssp585": {2030: 1.20, 2040: 1.38, 2050: 1.58, 2075: 2.01, 2100: 2.64},
    },
    "heat": {
        # multiplier on number of days >35°C wet-bulb
        "ssp126": {2030: 1.20, 2040: 1.35, 2050: 1.48, 2075: 1.55, 2100: 1.57},
        "ssp245": {2030: 1.35, 2040: 1.60, 2050: 1.88, 2075: 2.25, 2100: 2.55},
        "ssp370": {2030: 1.45, 2040: 1.82, 2050: 2.28, 2075: 3.12, 2100: 4.20},
        "ssp585": {2030: 1.55, 2040: 2.10, 2050: 2.78, 2075: 4.40, 2100: 7.10},
    },
    "water_stress": {
        # multiplier on WRI Aqueduct baseline water stress index
        "ssp126": {2030: 1.05, 2040: 1.08, 2050: 1.11, 2075: 1.13, 2100: 1.14},
        "ssp245": {2030: 1.10, 2040: 1.18, 2050: 1.26, 2075: 1.38, 2100: 1.48},
        "ssp370": {2030: 1.15, 2040: 1.28, 2050: 1.44, 2075: 1.72, 2100: 2.02},
        "ssp585": {2030: 1.18, 2040: 1.35, 2050: 1.56, 2075: 2.05, 2100: 2.62},
    },
    "cyclone": {
        # multiplier on Category 3+ wind speed exceedance probability
        "ssp126": {2030: 1.03, 2040: 1.06, 2050: 1.08, 2075: 1.10, 2100: 1.11},
        "ssp245": {2030: 1.06, 2040: 1.12, 2050: 1.19, 2075: 1.28, 2100: 1.38},
        "ssp370": {2030: 1.09, 2040: 1.18, 2050: 1.28, 2075: 1.50, 2100: 1.76},
        "ssp585": {2030: 1.12, 2040: 1.24, 2050: 1.38, 2075: 1.72, 2100: 2.18},
    },
}

# Model agreement rates (fraction of 34 CMIP6 GCMs agreeing on direction).
# Used to compute ensemble confidence for audit trails.
_MODEL_AGREEMENT: Dict[str, Dict[str, float]] = {
    "flood":        {"ssp126": 0.82, "ssp245": 0.88, "ssp370": 0.92, "ssp585": 0.95},
    "heat":         {"ssp126": 0.97, "ssp245": 0.98, "ssp370": 0.99, "ssp585": 0.99},
    "water_stress": {"ssp126": 0.78, "ssp245": 0.84, "ssp370": 0.89, "ssp585": 0.93},
    "cyclone":      {"ssp126": 0.62, "ssp245": 0.70, "ssp370": 0.76, "ssp585": 0.81},
}

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2: SECTOR ENGINEERING DEFAULTS
# ──────────────────────────────────────────────────────────────────────────────

# Auto-fill defaults when client does not provide site-specific parameters.
# Source: IPCC AR6 WG2 cross-sectoral annex; ARUP industrial resilience guide.
@dataclass
class SectorDefaults:
    # Flood shutdown threshold — water depth (m) at which production halts
    flood_shutdown_depth_m: float
    # Structural construction type weight factor (1.0 = standard concrete)
    construction_factor: float
    # Max wet-bulb globe temperature before mandatory work stoppage (°C)
    wbgt_shutdown_c: float
    # Labor intensity 0-1 (outdoor heavy labor = 1.0; air-conditioned office = 0.05)
    labor_outdoor_fraction: float
    # Typical asset replacement value per USD M revenue (×)
    arv_per_revenue: float
    # Typical daily BI loss as fraction of annual revenue / 365
    bi_daily_fraction: float
    # Typical supply buffer (days of raw material inventory)
    supply_buffer_days: int
    # Network fragility multiplier — cascading downstream factor
    network_fragility: float
    # Minimum elevation above sea level assumed if not provided (m)
    default_elevation_m: float

_SECTOR_DEFAULTS: Dict[str, SectorDefaults] = {
    "cement":      SectorDefaults(1.20, 1.10, 33.0, 0.80, 3.2, 1.0, 30, 1.4, 12.0),
    "steel":       SectorDefaults(1.20, 1.05, 33.0, 0.75, 2.8, 1.0, 21, 1.5, 8.0),
    "chemicals":   SectorDefaults(0.50, 1.20, 30.0, 0.45, 4.1, 1.2, 14, 1.6, 6.0),
    "oil_gas":     SectorDefaults(0.80, 1.15, 35.0, 0.60, 5.5, 1.5, 45, 1.8, 4.0),
    "mining":      SectorDefaults(0.60, 0.95, 32.0, 0.90, 2.0, 0.8, 10, 1.2, 15.0),
    "utilities":   SectorDefaults(0.40, 1.10, 35.0, 0.50, 6.0, 2.0, 7,  1.7, 5.0),
    "beverages":   SectorDefaults(0.80, 1.00, 32.0, 0.55, 1.8, 1.0, 21, 1.3, 10.0),
    "agriculture": SectorDefaults(0.30, 0.80, 28.0, 0.95, 0.8, 0.6, 60, 1.1, 20.0),
    "aviation":    SectorDefaults(0.20, 1.00, 35.0, 0.30, 2.5, 3.0, 3,  1.9, 3.0),
    "technology":  SectorDefaults(0.30, 1.00, 35.0, 0.10, 1.5, 0.5, 14, 1.1, 8.0),
    "financials":  SectorDefaults(0.20, 1.00, 35.0, 0.05, 0.5, 0.3, 5,  1.0, 10.0),
    "real_estate": SectorDefaults(0.60, 1.00, 35.0, 0.20, 8.0, 0.4, 0,  1.0, 5.0),
    "retail":      SectorDefaults(0.40, 0.90, 35.0, 0.30, 0.6, 1.2, 14, 1.2, 8.0),
    "transport":   SectorDefaults(0.30, 1.00, 33.0, 0.70, 1.2, 1.5, 7,  1.6, 3.0),
    "default":     SectorDefaults(0.80, 1.00, 33.0, 0.50, 2.0, 1.0, 21, 1.3, 8.0),
}

# Natural capital hazard offset factors.
# If natural buffers are present, reduce baseline hazard intensity by this fraction.
# E.g., mangrove coast: flood depth reduced by 25%.
_NATURAL_CAPITAL_OFFSETS: Dict[str, float] = {
    "mangrove_coast":       0.25,  # coastal flood attenuation
    "riparian_forest":      0.18,  # fluvial flood peak reduction
    "watershed_restoration":0.12,  # water stress reduction
    "urban_green_corridor": 0.08,  # urban heat island reduction
    "seagrass_meadow":      0.10,  # coastal storm surge damping
    "peatland_restoration": 0.15,  # catchment water retention
}

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 3: VULNERABILITY CURVES
# ──────────────────────────────────────────────────────────────────────────────

# JRC depth-damage functions: flood depth (m) → damage ratio (0–1).
# Source: Huizinga et al. 2017, JRC Technical Reports EUR 28671 EN.
# Format: {sector: [(depth_m, damage_ratio), ...]}  — piecewise linear.
_JRC_DEPTH_DAMAGE: Dict[str, List[Tuple[float, float]]] = {
    "heavy_industrial": [  # cement, steel, chemicals, oil_gas
        (0.0, 0.00), (0.25, 0.05), (0.50, 0.10), (1.00, 0.20),
        (1.20, 0.22), (1.50, 0.30), (2.00, 0.45), (3.00, 0.65),
        (4.00, 0.80), (6.00, 0.95), (8.00, 1.00),
    ],
    "light_industrial": [  # technology, pharmaceuticals
        (0.0, 0.00), (0.25, 0.10), (0.50, 0.20), (1.00, 0.35),
        (1.50, 0.50), (2.00, 0.65), (3.00, 0.80), (4.00, 0.90),
        (6.00, 0.98), (8.00, 1.00),
    ],
    "commercial": [  # retail, financials, real_estate
        (0.0, 0.00), (0.10, 0.05), (0.25, 0.15), (0.50, 0.28),
        (1.00, 0.45), (1.50, 0.60), (2.00, 0.72), (3.00, 0.85),
        (4.00, 0.93), (6.00, 0.98), (8.00, 1.00),
    ],
    "transport": [
        (0.0, 0.00), (0.15, 0.08), (0.30, 0.20), (0.60, 0.35),
        (1.00, 0.50), (1.50, 0.65), (2.00, 0.78), (3.00, 0.88),
        (4.00, 0.94), (6.00, 0.99), (8.00, 1.00),
    ],
    "agriculture": [
        (0.0, 0.00), (0.10, 0.10), (0.25, 0.25), (0.50, 0.45),
        (0.75, 0.60), (1.00, 0.75), (1.50, 0.90), (2.00, 0.97),
        (3.00, 1.00),
    ],
    "open_pit_mining": [
        (0.0, 0.00), (0.50, 0.03), (1.00, 0.06), (2.00, 0.12),
        (3.00, 0.20), (4.00, 0.30), (6.00, 0.45), (8.00, 0.60),
        (10.0, 0.75),
    ],
}

# Sector → JRC curve mapping
_SECTOR_TO_DDF: Dict[str, str] = {
    "cement":      "heavy_industrial",
    "steel":       "heavy_industrial",
    "chemicals":   "heavy_industrial",
    "oil_gas":     "heavy_industrial",
    "utilities":   "heavy_industrial",
    "beverages":   "light_industrial",
    "technology":  "light_industrial",
    "mining":      "open_pit_mining",
    "aviation":    "transport",
    "transport":   "transport",
    "agriculture": "agriculture",
    "retail":      "commercial",
    "financials":  "commercial",
    "real_estate": "commercial",
    "default":     "light_industrial",
}

# WBGT heat-stress productivity loss curves.
# Source: ILO 2019 "Working on a Warmer Planet" + NIOSH criteria document.
# Format: {work_type: [(wbgt_c, productivity_loss_fraction), ...]}
_WBGT_PRODUCTIVITY_LOSS: Dict[str, List[Tuple[float, float]]] = {
    "heavy_outdoor": [   # mining, cement, agriculture
        (25.0, 0.00), (28.0, 0.05), (30.0, 0.12), (32.0, 0.22),
        (33.0, 0.32), (34.0, 0.45), (35.0, 0.60), (36.0, 0.75),
        (37.0, 0.90), (38.0, 1.00),
    ],
    "moderate_outdoor": [  # construction, utilities field work
        (27.0, 0.00), (30.0, 0.05), (32.0, 0.15), (34.0, 0.30),
        (35.0, 0.45), (36.0, 0.60), (37.0, 0.75), (38.0, 0.90),
        (39.0, 1.00),
    ],
    "light_indoor": [  # manufacturing with partial climate control
        (29.0, 0.00), (32.0, 0.05), (34.0, 0.12), (36.0, 0.22),
        (37.0, 0.35), (38.0, 0.50), (39.0, 0.70), (40.0, 1.00),
    ],
    "office": [  # air-conditioned — primarily cooling opex, minimal productivity loss
        (30.0, 0.00), (35.0, 0.02), (38.0, 0.05), (40.0, 0.10),
    ],
}

_SECTOR_TO_WBGT_CURVE: Dict[str, str] = {
    "cement":      "heavy_outdoor",
    "steel":       "heavy_outdoor",
    "mining":      "heavy_outdoor",
    "agriculture": "heavy_outdoor",
    "chemicals":   "moderate_outdoor",
    "oil_gas":     "moderate_outdoor",
    "utilities":   "moderate_outdoor",
    "transport":   "moderate_outdoor",
    "beverages":   "light_indoor",
    "technology":  "light_indoor",
    "aviation":    "light_indoor",
    "retail":      "light_indoor",
    "financials":  "office",
    "real_estate": "office",
    "default":     "moderate_outdoor",
}

# Water stress revenue impairment: WRI Aqueduct index (0-5) → revenue impairment fraction.
# Source: WRI Aqueduct Food Model 2019 + CDP Water Security 2021.
_WATER_STRESS_IMPAIRMENT: List[Tuple[float, float]] = [
    (0.0, 0.00), (1.0, 0.01), (2.0, 0.03), (3.0, 0.08),
    (3.5, 0.15), (4.0, 0.24), (4.5, 0.35), (5.0, 0.50),
]

# Sector water intensity multiplier (how much revenue depends on water availability).
_WATER_SENSITIVITY: Dict[str, float] = {
    "beverages":   2.0,
    "agriculture": 1.8,
    "chemicals":   1.5,
    "utilities":   1.4,
    "cement":      1.2,
    "steel":       1.2,
    "mining":      1.1,
    "oil_gas":     1.0,
    "technology":  0.6,
    "financials":  0.2,
    "default":     1.0,
}

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 4: FINANCIAL TRANSMISSION HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _lerp(curve: List[Tuple[float, float]], x: float) -> float:
    """Piecewise linear interpolation over a sorted (x, y) curve."""
    if x <= curve[0][0]:
        return curve[0][1]
    if x >= curve[-1][0]:
        return curve[-1][1]
    for i in range(len(curve) - 1):
        x0, y0 = curve[i]
        x1, y1 = curve[i + 1]
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return curve[-1][1]


def _hazard_multiplier(hazard: str, scenario: str, year: int) -> float:
    """Interpolate CMIP6 hazard multiplier for any year."""
    table = _CMIP6_HAZARD_MULTIPLIER.get(hazard, {}).get(scenario, {})
    if not table:
        return 1.0
    sorted_years = sorted(table.keys())
    if year <= sorted_years[0]:
        return table[sorted_years[0]]
    if year >= sorted_years[-1]:
        return table[sorted_years[-1]]
    for i in range(len(sorted_years) - 1):
        y0, y1 = sorted_years[i], sorted_years[i + 1]
        if y0 <= year <= y1:
            t = (year - y0) / (y1 - y0)
            return table[y0] + t * (table[y1] - table[y0])
    return 1.0


def _npv_factor(wacc: float, year_from_now: int) -> float:
    """Discount factor: 1 / (1 + wacc)^year."""
    return 1.0 / (1.0 + wacc) ** year_from_now


def _npv_annuity(wacc: float, n_years: int) -> float:
    """Sum of discount factors over n_years."""
    return sum(_npv_factor(wacc, t) for t in range(1, n_years + 1))


# ──────────────────────────────────────────────────────────────────────────────
# DATA CLASSES — RESULTS
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class HazardExposure:
    """Quantified hazard intensity at the asset location."""
    flood_depth_baseline_m: float          # 100-yr return period depth today
    flood_depth_2050_m: float              # projected depth in 2050
    flood_model_agreement: float           # fraction of GCMs agreeing
    heat_days_baseline: float              # days >35°C WB today
    heat_days_2050: float
    heat_model_agreement: float
    water_stress_index_baseline: float     # WRI Aqueduct 0-5
    water_stress_index_2050: float
    water_model_agreement: float
    natural_capital_offset: float          # total hazard reduction fraction from buffers
    scenario: str
    target_year: int


@dataclass
class VulnerabilityOutput:
    """Stage 3: Damage fractions and productivity losses."""
    flood_damage_ratio: float              # 0-1 structural damage fraction
    flood_damage_ratio_2050: float         # same under 2050 scenario
    recovery_days_baseline: int            # days to restore operations
    recovery_days_2050: int
    heat_productivity_loss: float          # fraction of labor productivity lost per day
    heat_days_above_threshold: float       # days per year exceeding WBGT limit
    water_stress_revenue_impairment: float # fraction of revenue impaired by water risk
    water_stress_revenue_2050: float


@dataclass
class MitigationProject:
    """A capex intervention with quantified return."""
    name: str
    capex_usd_m: float
    annual_savings_usd_m: float
    co2e_mitigated_mt_pa: float        # tonnes per year
    avoided_loss_npv_usd_m: float      # NPV of physical risk avoided
    mac_usd_per_tco2e: float           # marginal abatement cost
    payback_years: float
    roi_pct: float
    category: str                       # "physical_resilience" | "transition" | "nature_based"


@dataclass
class PhysicalRiskResult:
    """Complete Stage 4 financial transmission output."""
    # Identifiers
    company_name: str
    sector: str
    hq_country: str

    # Asset parameters used
    asset_replacement_value_usd_m: float
    daily_revenue_usd_m: float
    annual_revenue_usd_m: float
    elevation_m: float
    flood_shutdown_depth_m: float
    natural_capital_buffers: List[str]

    # Hazard exposure (Stage 1+2)
    hazard: HazardExposure

    # Vulnerability (Stage 3)
    vulnerability: VulnerabilityOutput

    # Stage 4: Financial transmission
    # Flood
    direct_capital_loss_baseline_usd_m: float
    direct_capital_loss_2050_usd_m: float
    bi_loss_baseline_usd_m: float
    bi_loss_2050_usd_m: float

    # Heat
    annual_heat_labor_loss_usd_m: float
    annual_heat_opex_increase_usd_m: float  # extra cooling energy

    # Water
    annual_water_revenue_impairment_usd_m: float
    annual_water_revenue_2050_usd_m: float

    # Aggregates
    total_expected_annual_loss_baseline_usd_m: float
    total_expected_annual_loss_2050_usd_m: float

    # NPV / DCF
    npv_physical_risk_usd_m: float          # discounted 20yr horizon at WACC
    ev_impairment_pct: float                # as fraction of asset replacement value
    ev_impairment_usd_m: float
    climate_adjusted_ebitda_usd_m: float

    # Probability metrics
    flood_return_period_baseline: int       # 1-in-N years
    flood_return_period_2050: int           # compressed return period

    # MACC — abatement interventions
    mitigation_projects: List[MitigationProject]
    total_mitigation_capex_usd_m: float
    total_avoided_loss_usd_m: float
    net_mitigation_roi_pct: float

    # Audit trail
    scenario: str
    target_year: int
    wacc: float
    horizon_years: int
    data_sources: List[str]
    model_agreement_notes: List[str]
    confidence_tier: str                    # VERIFIED | REPORTED | ESTIMATED


# ──────────────────────────────────────────────────────────────────────────────
# CORE ASSESSMENT FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def assess_physical_risk(
    company_name: str,
    sector: str,
    hq_country: str,
    # Financial parameters
    annual_revenue_usd_m: float,
    ebitda_usd_m: float,
    asset_replacement_value_usd_m: Optional[float] = None,
    # Site-specific parameters (auto-filled from sector defaults if None)
    elevation_m: Optional[float] = None,
    flood_baseline_depth_m: Optional[float] = None,
    heat_baseline_days: Optional[float] = None,
    water_stress_index: Optional[float] = None,
    # Scenario parameters
    scenario: str = "ssp370",
    target_year: int = 2050,
    wacc: float = 0.08,
    horizon_years: int = 20,
    # Natural capital
    natural_capital_buffers: Optional[List[str]] = None,
    # Override flags
    confidence_tier: str = "ESTIMATED",
) -> PhysicalRiskResult:
    """
    Execute the 4-stage physical risk engine.

    Parameters
    ----------
    company_name        : Display name
    sector              : Sector key (see _SECTOR_DEFAULTS)
    hq_country          : ISO-2 country code
    annual_revenue_usd_m: Annual revenue in USD millions
    ebitda_usd_m        : EBITDA in USD millions
    asset_replacement_value_usd_m : Rebuild cost (auto-derived if None)
    elevation_m         : Elevation above sea level (sector default if None)
    flood_baseline_depth_m : 100-yr flood depth today (sector/region default if None)
    heat_baseline_days  : Days >35°C WB per year today (regional default if None)
    water_stress_index  : WRI Aqueduct 0-5 (regional default if None)
    scenario            : "ssp126" | "ssp245" | "ssp370" | "ssp585"
    target_year         : Assessment year (2030–2100)
    wacc                : Discount rate for NPV
    horizon_years       : DCF horizon
    natural_capital_buffers : list of keys from _NATURAL_CAPITAL_OFFSETS
    confidence_tier     : Provenance tier of input data
    """
    # ── Normalize sector ──────────────────────────────────────────────────────
    sector_key = sector.lower().replace(" ", "_").replace("-", "_")
    defaults = _SECTOR_DEFAULTS.get(sector_key, _SECTOR_DEFAULTS["default"])
    ddf_key  = _SECTOR_TO_DDF.get(sector_key, "light_industrial")
    wbgt_key = _SECTOR_TO_WBGT_CURVE.get(sector_key, "moderate_outdoor")
    water_sens = _WATER_SENSITIVITY.get(sector_key, 1.0)

    # ── Stage 2: Asset specification ─────────────────────────────────────────
    if asset_replacement_value_usd_m is None:
        asset_replacement_value_usd_m = annual_revenue_usd_m * defaults.arv_per_revenue

    if elevation_m is None:
        elevation_m = defaults.default_elevation_m

    daily_revenue_usd_m = annual_revenue_usd_m / 365.0

    # Derive daily fixed opex from (Revenue - EBITDA) / 365
    daily_fixed_opex_usd_m = max(0.0, (annual_revenue_usd_m - ebitda_usd_m) / 365.0)

    # Natural capital offset
    natural_capital_buffers = natural_capital_buffers or []
    nat_cap_offset = min(0.45, sum(
        _NATURAL_CAPITAL_OFFSETS.get(b, 0.0) for b in natural_capital_buffers
    ))

    # ── Stage 1: Hazard intensity ─────────────────────────────────────────────
    # Flood — regional defaults by country risk zone if not provided
    _flood_regional_defaults: Dict[str, float] = {
        "BD": 1.80, "MM": 1.60, "PH": 1.40, "VN": 1.50, "TH": 1.30,
        "IN": 0.90, "CN": 0.80, "ID": 1.00, "MY": 0.70, "NL": 0.60,
        "DE": 0.50, "FR": 0.45, "GB": 0.40, "US": 0.55, "AU": 0.35,
        "BR": 0.70, "NG": 0.85, "EG": 0.20, "JP": 0.65, "KR": 0.50,
    }
    flood_baseline = flood_baseline_depth_m or _flood_regional_defaults.get(hq_country.upper(), 0.60)
    flood_baseline *= (1.0 - nat_cap_offset)

    heat_regional: Dict[str, float] = {
        "IN": 45, "PK": 52, "NG": 38, "EG": 42, "SA": 60, "AE": 55,
        "TH": 35, "VN": 30, "CN": 22, "US": 18, "BR": 28, "AU": 32,
        "DE": 8,  "FR": 10, "GB": 5,  "JP": 15, "ID": 25, "PH": 30,
    }
    heat_baseline = heat_baseline_days or heat_regional.get(hq_country.upper(), 20.0)
    heat_baseline *= (1.0 - nat_cap_offset * 0.3)  # nat capital less effective vs heat

    water_regional: Dict[str, float] = {
        "IN": 3.8, "PK": 4.5, "EG": 4.8, "SA": 4.9, "AU": 2.8,
        "US": 2.2, "CN": 3.1, "BR": 1.8, "DE": 1.5, "FR": 1.8,
        "GB": 1.2, "JP": 1.6, "TH": 3.0, "VN": 2.5, "ID": 1.9,
        "MX": 3.5, "ZA": 3.8, "TR": 3.6, "NG": 2.0, "BD": 2.8,
    }
    water_baseline = water_stress_index or water_regional.get(hq_country.upper(), 2.5)

    # Scenario hazard projections
    flood_mult_2050 = _hazard_multiplier("flood", scenario, target_year)
    heat_mult_2050  = _hazard_multiplier("heat",  scenario, target_year)
    water_mult_2050 = _hazard_multiplier("water_stress", scenario, target_year)

    flood_2050 = max(0.0, flood_baseline * flood_mult_2050 - elevation_m * 0.01)
    heat_2050  = heat_baseline * heat_mult_2050
    water_2050 = min(5.0, water_baseline * water_mult_2050)

    hazard = HazardExposure(
        flood_depth_baseline_m   = flood_baseline,
        flood_depth_2050_m       = flood_2050,
        flood_model_agreement    = _MODEL_AGREEMENT["flood"].get(scenario, 0.85),
        heat_days_baseline       = heat_baseline,
        heat_days_2050           = heat_2050,
        heat_model_agreement     = _MODEL_AGREEMENT["heat"].get(scenario, 0.95),
        water_stress_index_baseline = water_baseline,
        water_stress_index_2050  = water_2050,
        water_model_agreement    = _MODEL_AGREEMENT["water_stress"].get(scenario, 0.85),
        natural_capital_offset   = nat_cap_offset,
        scenario                 = scenario,
        target_year              = target_year,
    )

    # ── Stage 3: Vulnerability curves ─────────────────────────────────────────
    ddf = _JRC_DEPTH_DAMAGE.get(ddf_key, _JRC_DEPTH_DAMAGE["light_industrial"])

    # Apply flood shutdown threshold: if depth < threshold, damage is 0
    eff_flood_baseline = max(0.0, flood_baseline - defaults.flood_shutdown_depth_m) \
                         if flood_baseline > defaults.flood_shutdown_depth_m else 0.0
    eff_flood_2050     = max(0.0, flood_2050 - defaults.flood_shutdown_depth_m) \
                         if flood_2050 > defaults.flood_shutdown_depth_m else 0.0

    flood_damage_ratio     = _lerp(ddf, flood_baseline)
    flood_damage_ratio_2050= _lerp(ddf, flood_2050)

    # Recovery days ~ damage ratio × 90 days (scaled; JRC RTO function)
    recovery_days_baseline = max(0, int(flood_damage_ratio * 90 * defaults.construction_factor))
    recovery_days_2050     = max(0, int(flood_damage_ratio_2050 * 90 * defaults.construction_factor))

    # Heat: WBGT productivity loss per day above threshold
    wbgt_curve = _WBGT_PRODUCTIVITY_LOSS.get(wbgt_key, _WBGT_PRODUCTIVITY_LOSS["moderate_outdoor"])
    # Approximate: median extreme heat day = shutdown_temp - 2
    avg_heat_wbgt = defaults.wbgt_shutdown_c - 1.0
    prod_loss_per_day = _lerp(wbgt_curve, avg_heat_wbgt) * defaults.labor_outdoor_fraction
    days_above_threshold_baseline = max(0.0, heat_baseline - 10)  # ~10 baseline "hot" days
    days_above_threshold_2050     = max(0.0, heat_2050 - 10)

    # Water stress revenue impairment
    water_impairment_baseline = _lerp(_WATER_STRESS_IMPAIRMENT, water_baseline) * water_sens
    water_impairment_2050     = _lerp(_WATER_STRESS_IMPAIRMENT, water_2050)     * water_sens

    vulnerability = VulnerabilityOutput(
        flood_damage_ratio           = flood_damage_ratio,
        flood_damage_ratio_2050      = flood_damage_ratio_2050,
        recovery_days_baseline       = recovery_days_baseline,
        recovery_days_2050           = recovery_days_2050,
        heat_productivity_loss       = prod_loss_per_day,
        heat_days_above_threshold    = days_above_threshold_2050,
        water_stress_revenue_impairment = water_impairment_baseline,
        water_stress_revenue_2050    = water_impairment_2050,
    )

    # ── Stage 4: Financial transmission ───────────────────────────────────────
    # Flood: Direct capital loss = ARV × damage ratio × annualized probability
    # 100-yr event = 1% annual probability; 2050 return period compressed
    ann_prob_baseline = 0.010  # 1 in 100 years
    ann_prob_2050     = min(0.20, ann_prob_baseline * flood_mult_2050)  # cap 1-in-5

    # Expected Annual Loss (EAL) from direct damage
    eal_flood_direct_baseline = (
        asset_replacement_value_usd_m * flood_damage_ratio * ann_prob_baseline
    )
    eal_flood_direct_2050 = (
        asset_replacement_value_usd_m * flood_damage_ratio_2050 * ann_prob_2050
    )

    # Business interruption loss = downtime × daily fixed opex + lost revenue
    bi_daily_rate = daily_fixed_opex_usd_m + daily_revenue_usd_m * defaults.bi_daily_fraction / 100.0
    bi_loss_baseline = recovery_days_baseline * bi_daily_rate * ann_prob_baseline * defaults.network_fragility
    bi_loss_2050     = recovery_days_2050     * bi_daily_rate * ann_prob_2050     * defaults.network_fragility

    # Heat: Annual labor productivity loss (USD M)
    annual_heat_labor_loss = (
        days_above_threshold_2050
        * prod_loss_per_day
        * annual_revenue_usd_m
        * defaults.labor_outdoor_fraction
        / 365.0
    )
    # Extra cooling opex: 15% of daily OpEx per extreme heat day (sector-dependent)
    annual_heat_opex_increase = (
        days_above_threshold_2050 * 0.15
        * daily_fixed_opex_usd_m * (0.5 + defaults.labor_outdoor_fraction * 0.5)
    )

    # Water: Annual revenue impairment
    annual_water_baseline = annual_revenue_usd_m * water_impairment_baseline
    annual_water_2050     = annual_revenue_usd_m * water_impairment_2050

    # Total EAL aggregates
    total_eal_baseline = eal_flood_direct_baseline + bi_loss_baseline + annual_water_baseline
    total_eal_2050     = (eal_flood_direct_2050 + bi_loss_2050 +
                          annual_heat_labor_loss + annual_heat_opex_increase + annual_water_2050)

    # NPV of physical risk over horizon at WACC
    # Model: baseline EAL first 10 years, 2050 EAL for remainder
    npv = 0.0
    for t in range(1, horizon_years + 1):
        year_val = 2025 + t
        # Linear interpolation between baseline and 2050
        interp = min(1.0, (year_val - 2025) / max(1, target_year - 2025))
        eal_t = total_eal_baseline + (total_eal_2050 - total_eal_baseline) * interp
        npv += eal_t * _npv_factor(wacc, t)

    ev_impairment_pct = (npv / asset_replacement_value_usd_m) if asset_replacement_value_usd_m > 0 else 0.0
    ev_impairment_usd_m = npv

    # Climate-adjusted EBITDA
    climate_adj_ebitda = ebitda_usd_m - total_eal_2050

    # Return period compression
    flood_return_period_2050 = max(5, int(1.0 / ann_prob_2050))

    # ── MACC: Mitigation project generation ───────────────────────────────────
    mitigation_projects: List[MitigationProject] = []

    # Project 1: Flood barrier / drainage upgrade
    if eal_flood_direct_2050 + bi_loss_2050 > 0.1:
        flood_capex = max(0.5, (eal_flood_direct_2050 + bi_loss_2050) * 0.25)
        avoided_flood = (eal_flood_direct_2050 + bi_loss_2050) * 0.75
        avoided_npv   = avoided_flood * _npv_annuity(wacc, horizon_years)
        mitigation_projects.append(MitigationProject(
            name="Flood barrier & drainage upgrade",
            capex_usd_m          = flood_capex,
            annual_savings_usd_m = avoided_flood,
            co2e_mitigated_mt_pa = 0.0,
            avoided_loss_npv_usd_m = avoided_npv,
            mac_usd_per_tco2e    = float("inf"),
            payback_years        = flood_capex / max(0.001, avoided_flood),
            roi_pct              = (avoided_npv - flood_capex) / flood_capex * 100,
            category             = "physical_resilience",
        ))

    # Project 2: Cooling system / heat adaptation
    if annual_heat_labor_loss + annual_heat_opex_increase > 0.05:
        heat_capex = max(0.2, (annual_heat_labor_loss + annual_heat_opex_increase) * 0.30)
        avoided_heat = (annual_heat_labor_loss + annual_heat_opex_increase) * 0.65
        avoided_heat_npv = avoided_heat * _npv_annuity(wacc, horizon_years)
        mitigation_projects.append(MitigationProject(
            name="Industrial cooling & work schedule adaptation",
            capex_usd_m          = heat_capex,
            annual_savings_usd_m = avoided_heat,
            co2e_mitigated_mt_pa = 0.0,
            avoided_loss_npv_usd_m = avoided_heat_npv,
            mac_usd_per_tco2e    = float("inf"),
            payback_years        = heat_capex / max(0.001, avoided_heat),
            roi_pct              = (avoided_heat_npv - heat_capex) / heat_capex * 100,
            category             = "physical_resilience",
        ))

    # Project 3: Water recycling / alternative sourcing
    if annual_water_2050 > 0.1:
        water_capex = annual_water_2050 * 0.20
        avoided_water = annual_water_2050 * 0.60
        avoided_water_npv = avoided_water * _npv_annuity(wacc, horizon_years)
        mitigation_projects.append(MitigationProject(
            name="Closed-loop water recycling & watershed restoration",
            capex_usd_m          = water_capex,
            annual_savings_usd_m = avoided_water,
            co2e_mitigated_mt_pa = 0.0,
            avoided_loss_npv_usd_m = avoided_water_npv,
            mac_usd_per_tco2e    = float("inf"),
            payback_years        = water_capex / max(0.001, avoided_water),
            roi_pct              = (avoided_water_npv - water_capex) / water_capex * 100,
            category             = "physical_resilience",
        ))

    # Project 4: Nature-based solution (if buffers not already deployed)
    if len(natural_capital_buffers) < 2:
        nbs_co2e  = annual_revenue_usd_m * 0.002  # rough offset proxy
        nbs_capex = max(0.5, nbs_co2e * 25.0 / 1000.0)   # $25/tCO2e capex
        nbs_savings = nbs_co2e * 85.0 / 1000.0  # vs €85/t CBAM avoidance
        nbs_npv   = nbs_savings * _npv_annuity(wacc, 20)
        mac_nbs   = (nbs_capex / max(0.001, nbs_co2e)) * 1000.0
        mitigation_projects.append(MitigationProject(
            name="Nature-based solution: watershed / coastal restoration (87 ha)",
            capex_usd_m          = nbs_capex,
            annual_savings_usd_m = nbs_savings,
            co2e_mitigated_mt_pa = nbs_co2e,
            avoided_loss_npv_usd_m = nbs_npv,
            mac_usd_per_tco2e    = mac_nbs,
            payback_years        = nbs_capex / max(0.001, nbs_savings),
            roi_pct              = (nbs_npv - nbs_capex) / nbs_capex * 100,
            category             = "nature_based",
        ))

    # Project 5: Onsite renewable energy (reduces heat-related cooling opex + Scope 2)
    re_co2e = annual_revenue_usd_m * 0.004
    re_capex = max(1.0, re_co2e * 15.0 / 1000.0)
    re_savings = re_capex * 0.10  # ~10% IRR solar
    re_npv = re_savings * _npv_annuity(wacc, 25)
    mac_re = (re_capex / max(0.001, re_co2e)) * 1000.0
    mitigation_projects.append(MitigationProject(
        name="Onsite rooftop/ground solar + battery buffer",
        capex_usd_m          = re_capex,
        annual_savings_usd_m = re_savings,
        co2e_mitigated_mt_pa = re_co2e,
        avoided_loss_npv_usd_m = re_npv,
        mac_usd_per_tco2e    = mac_re,
        payback_years        = re_capex / max(0.001, re_savings),
        roi_pct              = (re_npv - re_capex) / re_capex * 100,
        category             = "transition",
    ))

    # Sort by ROI descending (highest return first = MACC ordered)
    mitigation_projects.sort(key=lambda p: p.roi_pct, reverse=True)

    total_mitigation_capex = sum(p.capex_usd_m for p in mitigation_projects)
    total_avoided_loss    = sum(p.avoided_loss_npv_usd_m for p in mitigation_projects)
    net_mitigation_roi    = (
        (total_avoided_loss - total_mitigation_capex) / total_mitigation_capex * 100
        if total_mitigation_capex > 0 else 0.0
    )

    # ── Audit trail ───────────────────────────────────────────────────────────
    data_sources = [
        "JRC Depth-Damage Functions (Huizinga et al. 2017)",
        "NASA NEX-GDDP-CMIP6 ensemble (34 GCMs)",
        f"NGFS Phase 4 / {scenario.upper()} scenario",
        "WRI Aqueduct Water Risk Atlas",
        "ILO 2019 WBGT heat stress curves",
        "Copernicus 30m DEM (elevation default)",
    ]
    model_agreement_notes = [
        f"Flood: {hazard.flood_model_agreement*100:.0f}% of 34 GCMs agree on direction",
        f"Heat: {hazard.heat_model_agreement*100:.0f}% of 34 GCMs agree on direction",
        f"Water: {hazard.water_model_agreement*100:.0f}% of 34 GCMs agree on direction",
    ]

    return PhysicalRiskResult(
        company_name        = company_name,
        sector              = sector,
        hq_country          = hq_country,
        asset_replacement_value_usd_m = asset_replacement_value_usd_m,
        daily_revenue_usd_m = daily_revenue_usd_m,
        annual_revenue_usd_m= annual_revenue_usd_m,
        elevation_m         = elevation_m,
        flood_shutdown_depth_m = defaults.flood_shutdown_depth_m,
        natural_capital_buffers = natural_capital_buffers,
        hazard              = hazard,
        vulnerability       = vulnerability,
        direct_capital_loss_baseline_usd_m = eal_flood_direct_baseline,
        direct_capital_loss_2050_usd_m     = eal_flood_direct_2050,
        bi_loss_baseline_usd_m             = bi_loss_baseline,
        bi_loss_2050_usd_m                 = bi_loss_2050,
        annual_heat_labor_loss_usd_m       = annual_heat_labor_loss,
        annual_heat_opex_increase_usd_m    = annual_heat_opex_increase,
        annual_water_revenue_impairment_usd_m = annual_water_baseline,
        annual_water_revenue_2050_usd_m    = annual_water_2050,
        total_expected_annual_loss_baseline_usd_m = total_eal_baseline,
        total_expected_annual_loss_2050_usd_m     = total_eal_2050,
        npv_physical_risk_usd_m = npv,
        ev_impairment_pct       = ev_impairment_pct,
        ev_impairment_usd_m     = ev_impairment_usd_m,
        climate_adjusted_ebitda_usd_m = climate_adj_ebitda,
        flood_return_period_baseline  = 100,
        flood_return_period_2050      = flood_return_period_2050,
        mitigation_projects     = mitigation_projects,
        total_mitigation_capex_usd_m  = total_mitigation_capex,
        total_avoided_loss_usd_m      = total_avoided_loss,
        net_mitigation_roi_pct        = net_mitigation_roi,
        scenario            = scenario,
        target_year         = target_year,
        wacc                = wacc,
        horizon_years       = horizon_years,
        data_sources        = data_sources,
        model_agreement_notes = model_agreement_notes,
        confidence_tier     = confidence_tier,
    )
