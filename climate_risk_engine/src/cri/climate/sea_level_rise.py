"""
Sea Level Rise (SLR) Physical Risk Module.

Implements IPCC AR6 WG1 (2021) sea level rise projections under SSP1-2.6,
SSP2-4.5, and SSP5-8.5, with coastal inundation lookup by asset elevation and
distance to coast.

Scope
-----
- Global mean SLR trajectories at 2030, 2050, 2075, 2100 (median + likely range)
- Regional adjustment via IPCC AR6 Chapter 9 Table 9.9 multipliers
- Elevation-based inundation probability for coastal assets
- Transition risk flag: policy scenarios where rapid decarbonisation accelerates
  coastal adaptation investment requirements
- Integration point for physical_risk_financial.py year loop

Methodology
-----------
Global mean SLR (metres above 2015 baseline), IPCC AR6 WGI Chapter 9:

  Scenario    2030    2050    2075    2100   (median)
  SSP1-2.6    0.09    0.18    0.28    0.38
  SSP2-4.5    0.09    0.20    0.35    0.52
  SSP5-8.5    0.10    0.23    0.46    0.77

Low-confidence High End (H++): +0.5 m on top of SSP5-8.5 by 2100 for ice-sheet
instability scenarios (not used in baseline; returned as tail scenario).

Inundation rule (simplified bathtub model — conservative lower bound):
  P(inundation by year T) ≈ logistic(asset_elevation – projected_SLR – tide_range/2)
  where logistic is centred at 0 with scale 0.3 m.

This is intentionally conservative: it ignores protective infrastructure
(sea walls, managed retreat) which site-specific data is needed to model.

Coastal proximity score
-----------------------
Required to flag whether SLR is relevant for a given asset.
Estimated from lat/lon distance to the nearest coastline using a simplified
lookup table of major coastal regions. For production use, a full global
coastline database (GSHHG) should be queried.

References
----------
IPCC AR6 WGI Chapter 9, Fox-Kemper et al. (2021).
  https://www.ipcc.ch/report/ar6/wg1/chapter/chapter-9/
IPCC AR6 WGI Technical Summary Box TS.4.
Oppenheimer et al. (2019) IPCC SROCC Chapter 4.
NOAA Tides and Currents — coastal DEM data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ── SLR projections (metres above 2015 baseline, IPCC AR6 median) ─────────────

_SLR_MEDIAN: dict[str, dict[int, float]] = {
    "SSP1-2.6": {2030: 0.09, 2050: 0.18, 2075: 0.28, 2100: 0.38},
    "SSP2-4.5": {2030: 0.09, 2050: 0.20, 2075: 0.35, 2100: 0.52},
    "SSP5-8.5": {2030: 0.10, 2050: 0.23, 2075: 0.46, 2100: 0.77},
}

# Likely range (17th–83rd percentile) as ± offset from median
_SLR_RANGE: dict[str, dict[int, float]] = {
    "SSP1-2.6": {2030: 0.04, 2050: 0.07, 2075: 0.10, 2100: 0.13},
    "SSP2-4.5": {2030: 0.04, 2050: 0.07, 2075: 0.11, 2100: 0.16},
    "SSP5-8.5": {2030: 0.04, 2050: 0.08, 2075: 0.14, 2100: 0.23},
}

# H++ tail scenario (ice-sheet instability) — add-on to SSP5-8.5 at 2100
_H_PLUS_PLUS_2100 = 0.50

# Tidal range (m) by broad coastal region — used in inundation probability
_TIDAL_RANGE_M: dict[str, float] = {
    "microtidal":   0.3,   # Mediterranean, Baltic, parts of Gulf of Mexico
    "mesotidal":    2.0,   # most Atlantic coasts, North Sea
    "macrotidal":   5.0,   # Bay of Fundy, Severn estuary, NW Australia
    "default":      1.5,
}

# Regional SLR multipliers (relative to global mean, IPCC AR6 Table 9.9)
# Values > 1.0 = faster than global mean (e.g. subsiding deltas)
# Values < 1.0 = slower (e.g. glacial isostatic adjustment uplift areas)
_REGIONAL_MULTIPLIERS: dict[str, float] = {
    # North America
    "US_East_Coast":        1.25,
    "US_Gulf_Coast":        1.50,   # subsidence (Mississippi delta)
    "US_West_Coast":        0.80,
    "Canada_Atlantic":      0.90,
    # Europe
    "North_Sea":            1.10,
    "Baltic":               0.70,   # GIA uplift
    "Mediterranean":        1.05,
    # Asia
    "South_Asia":           1.20,
    "Southeast_Asia":       1.40,   # delta subsidence (Mekong, Ganges-Brahmaputra)
    "China_Coast":          1.30,   # Yellow/Yangtze delta subsidence
    "Japan":                1.05,
    # Other
    "Small_Island_States":  1.00,
    "Australia":            0.95,
    "default":              1.00,
}


def _interp_slr(scenario: str, year: float) -> tuple[float, float]:
    """
    Linear interpolation of SLR median and likely-range half-width for a year.

    Returns (median_m, range_half_m).
    """
    checkpoints = sorted(_SLR_MEDIAN[scenario].keys())
    if year <= checkpoints[0]:
        return _SLR_MEDIAN[scenario][checkpoints[0]], _SLR_RANGE[scenario][checkpoints[0]]
    if year >= checkpoints[-1]:
        return _SLR_MEDIAN[scenario][checkpoints[-1]], _SLR_RANGE[scenario][checkpoints[-1]]

    for i in range(len(checkpoints) - 1):
        y0, y1 = checkpoints[i], checkpoints[i + 1]
        if y0 <= year <= y1:
            t = (year - y0) / (y1 - y0)
            med = _SLR_MEDIAN[scenario][y0] + t * (_SLR_MEDIAN[scenario][y1] - _SLR_MEDIAN[scenario][y0])
            rng = _SLR_RANGE[scenario][y0] + t * (_SLR_RANGE[scenario][y1] - _SLR_RANGE[scenario][y0])
            return med, rng

    return 0.0, 0.0


def _inundation_prob(elevation_m: float, slr_m: float, tide_range_m: float) -> float:
    """
    Logistic probability of inundation.

    P = 1 / (1 + exp((elevation - slr - tide_range/2) / scale))

    Scale = 0.30 m (bathtub model with storm surge uncertainty).
    """
    effective_water_level = slr_m + tide_range_m / 2.0
    x = elevation_m - effective_water_level
    scale = 0.30
    return 1.0 / (1.0 + math.exp(x / scale))


def _coastal_region_from_coords(lat: float, lon: float) -> str:
    """
    Coarse assignment of coastal region from lat/lon for regional SLR multiplier.
    Bounding-box heuristic — replace with GSHHG database call for production.
    """
    # US Gulf Coast
    if 25 <= lat <= 32 and -98 <= lon <= -80:
        return "US_Gulf_Coast"
    # US East Coast
    if 24 <= lat <= 48 and -80 <= lon <= -65:
        return "US_East_Coast"
    # US West Coast
    if 32 <= lat <= 50 and -130 <= lon <= -117:
        return "US_West_Coast"
    # North Sea
    if 50 <= lat <= 58 and -4 <= lon <= 9:
        return "North_Sea"
    # Baltic
    if 54 <= lat <= 66 and 10 <= lon <= 30:
        return "Baltic"
    # Mediterranean
    if 30 <= lat <= 47 and -5 <= lon <= 37:
        return "Mediterranean"
    # South Asia
    if 5 <= lat <= 24 and 61 <= lon <= 92:
        return "South_Asia"
    # Southeast Asia
    if -10 <= lat <= 22 and 95 <= lon <= 140:
        return "Southeast_Asia"
    # China Coast
    if 18 <= lat <= 40 and 108 <= lon <= 125:
        return "China_Coast"
    # Japan
    if 24 <= lat <= 46 and 122 <= lon <= 148:
        return "Japan"
    # Australia
    if -45 <= lat <= -10 and 113 <= lon <= 155:
        return "Australia"
    # Canada Atlantic
    if 43 <= lat <= 62 and -68 <= lon <= -52:
        return "Canada_Atlantic"
    return "default"


@dataclass
class SLRAssetResult:
    """SLR exposure result for one coastal asset."""
    asset_id: str
    asset_name: str
    # Inputs
    asset_elevation_m: float
    distance_to_coast_km: Optional[float]
    coastal_region: str
    tidal_regime: str
    # SLR projections at key horizons
    slr_2030_median_m: float
    slr_2050_median_m: float
    slr_2100_median_m: float
    slr_2100_h_plus_plus_m: float
    scenario: str
    regional_multiplier: float
    # Inundation probabilities
    p_inundation_2030: float
    p_inundation_2050: float
    p_inundation_2100: float
    # Financial impact
    ev_usd_m: float
    asset_at_risk_usd_m: float            # ev × p_inundation_2050
    expected_loss_2050_usd_m: float       # direct damage + relocation cost estimate
    expected_loss_2100_usd_m: float
    # Risk classification
    risk_tier: str                        # "critical" | "high" | "medium" | "low" | "negligible"
    disclosure_flag: bool                 # TCFD / SFDR coastal asset disclosure recommended
    notes: list[str] = field(default_factory=list)


def compute_slr_risk(
    asset_id: str,
    asset_name: str,
    asset_lat: float,
    asset_lon: float,
    asset_elevation_m: float,
    ev_usd_m: float,
    scenario: str = "SSP2-4.5",
    distance_to_coast_km: Optional[float] = None,
    tidal_regime: str = "default",
    custom_regional_multiplier: Optional[float] = None,
) -> SLRAssetResult:
    """
    Compute SLR physical risk for a coastal asset.

    Parameters
    ----------
    asset_id, asset_name : str
    asset_lat, asset_lon : float
        Asset coordinates (WGS84).
    asset_elevation_m : float
        Asset elevation above Mean Higher High Water (MHHW) in metres.
        Use 0 for at sea level. Negative = below current high-tide line.
    ev_usd_m : float
        Enterprise / property value in USD millions.
    scenario : str
        IPCC AR6 scenario: "SSP1-2.6", "SSP2-4.5", or "SSP5-8.5".
    distance_to_coast_km : float, optional
        Distance from asset to nearest coastline. If None, the module
        assumes coastal exposure is possible (conservative).
    tidal_regime : str
        "microtidal" | "mesotidal" | "macrotidal" | "default"
    custom_regional_multiplier : float, optional
        Override the automatic regional multiplier.

    Returns
    -------
    SLRAssetResult
    """
    if scenario not in _SLR_MEDIAN:
        raise ValueError(f"Unknown scenario '{scenario}'. Choose from: {list(_SLR_MEDIAN)}")

    # Not coastal — short-circuit with negligible result
    if distance_to_coast_km is not None and distance_to_coast_km > 50:
        return _negligible_result(
            asset_id, asset_name, asset_elevation_m, distance_to_coast_km,
            ev_usd_m, scenario,
            note=f"Asset {distance_to_coast_km:.0f} km from coast — SLR exposure negligible",
        )

    coastal_region = _coastal_region_from_coords(asset_lat, asset_lon)
    reg_mult = custom_regional_multiplier or _REGIONAL_MULTIPLIERS.get(coastal_region, 1.0)
    tide_m = _TIDAL_RANGE_M.get(tidal_regime, _TIDAL_RANGE_M["default"])

    # Project SLR at key horizons (apply regional multiplier)
    med_2030, rng_2030 = _interp_slr(scenario, 2030)
    med_2050, rng_2050 = _interp_slr(scenario, 2050)
    med_2100, rng_2100 = _interp_slr(scenario, 2100)

    slr_2030 = med_2030 * reg_mult
    slr_2050 = med_2050 * reg_mult
    slr_2100 = med_2100 * reg_mult
    slr_hpp  = (med_2100 + _H_PLUS_PLUS_2100) * reg_mult

    # Inundation probabilities
    p30 = _inundation_prob(asset_elevation_m, slr_2030, tide_m)
    p50 = _inundation_prob(asset_elevation_m, slr_2050, tide_m)
    p00 = _inundation_prob(asset_elevation_m, slr_2100, tide_m)

    # Financial impact
    rcv = ev_usd_m * 0.40           # replacement cost proxy
    at_risk = ev_usd_m * p50        # EV exposure at median 2050 SLR
    # Direct damage + adaptation / relocation cost (estimated as 1.5× direct)
    el_2050 = rcv * p50 * 1.5
    el_2100 = rcv * p00 * 1.5

    # Risk tier
    if p50 >= 0.5:
        tier = "critical"
    elif p50 >= 0.25:
        tier = "high"
    elif p50 >= 0.10:
        tier = "medium"
    elif p50 >= 0.02:
        tier = "low"
    else:
        tier = "negligible"

    notes = []
    if reg_mult > 1.1:
        notes.append(f"Regional subsidence: SLR multiplier {reg_mult:.2f}× (vs global mean)")
    if tide_m >= 4.0:
        notes.append(f"Macrotidal regime: {tide_m:.1f} m tidal range amplifies inundation risk")
    if asset_elevation_m < 1.0:
        notes.append("Asset elevation < 1 m above MHHW — already vulnerable to storm surge")
    if slr_hpp > 1.0:
        notes.append(f"H++ tail (ice-sheet instability): {slr_hpp:.2f} m regional SLR by 2100")

    return SLRAssetResult(
        asset_id=asset_id,
        asset_name=asset_name,
        asset_elevation_m=asset_elevation_m,
        distance_to_coast_km=distance_to_coast_km,
        coastal_region=coastal_region,
        tidal_regime=tidal_regime,
        slr_2030_median_m=round(slr_2030, 3),
        slr_2050_median_m=round(slr_2050, 3),
        slr_2100_median_m=round(slr_2100, 3),
        slr_2100_h_plus_plus_m=round(slr_hpp, 3),
        scenario=scenario,
        regional_multiplier=round(reg_mult, 3),
        p_inundation_2030=round(p30, 4),
        p_inundation_2050=round(p50, 4),
        p_inundation_2100=round(p00, 4),
        ev_usd_m=ev_usd_m,
        asset_at_risk_usd_m=round(at_risk, 2),
        expected_loss_2050_usd_m=round(el_2050, 2),
        expected_loss_2100_usd_m=round(el_2100, 2),
        risk_tier=tier,
        disclosure_flag=(tier in ("critical", "high", "medium")),
        notes=notes,
    )


def slr_to_dict(result: SLRAssetResult) -> dict:
    """Serialise SLR result to JSON-compatible dict."""
    return {
        "asset_id":          result.asset_id,
        "asset_name":        result.asset_name,
        "scenario":          result.scenario,
        "coastal_region":    result.coastal_region,
        "regional_multiplier": result.regional_multiplier,
        "asset_elevation_m": result.asset_elevation_m,
        "distance_to_coast_km": result.distance_to_coast_km,
        "slr_projections_m": {
            "2030_median":    result.slr_2030_median_m,
            "2050_median":    result.slr_2050_median_m,
            "2100_median":    result.slr_2100_median_m,
            "2100_h_plus_plus": result.slr_2100_h_plus_plus_m,
        },
        "inundation_probability": {
            "2030": result.p_inundation_2030,
            "2050": result.p_inundation_2050,
            "2100": result.p_inundation_2100,
        },
        "financial_impact_usd_m": {
            "asset_at_risk_2050":    result.asset_at_risk_usd_m,
            "expected_loss_2050":    result.expected_loss_2050_usd_m,
            "expected_loss_2100":    result.expected_loss_2100_usd_m,
        },
        "risk_tier":        result.risk_tier,
        "disclosure_flag":  result.disclosure_flag,
        "notes":            result.notes,
        "methodology": (
            "IPCC AR6 WGI Ch.9 SLR projections × regional multiplier. "
            "Inundation probability via logistic bathtub model (scale=0.3 m). "
            "Financial impact = RCV × P(inundation) × 1.5 (includes adaptation cost). "
            "H++ tail: +0.5 m ice-sheet instability add-on (SSP5-8.5 base)."
        ),
    }


def slr_portfolio_summary(results: list[SLRAssetResult]) -> dict:
    """Aggregate SLR results across a portfolio."""
    at_risk_total   = sum(r.asset_at_risk_usd_m for r in results)
    el_2050_total   = sum(r.expected_loss_2050_usd_m for r in results)
    el_2100_total   = sum(r.expected_loss_2100_usd_m for r in results)
    disclosure_assets = [r.asset_id for r in results if r.disclosure_flag]
    tier_counts: dict[str, int] = {}
    for r in results:
        tier_counts[r.risk_tier] = tier_counts.get(r.risk_tier, 0) + 1

    return {
        "total_assets_assessed": len(results),
        "assets_requiring_disclosure": len(disclosure_assets),
        "disclosure_asset_ids": disclosure_assets,
        "tier_distribution": tier_counts,
        "portfolio_financial_impact_usd_m": {
            "asset_at_risk_2050":  round(at_risk_total, 2),
            "expected_loss_2050":  round(el_2050_total, 2),
            "expected_loss_2100":  round(el_2100_total, 2),
        },
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _negligible_result(
    asset_id: str, asset_name: str, elevation_m: float,
    distance_km: Optional[float], ev_usd_m: float, scenario: str, note: str,
) -> SLRAssetResult:
    return SLRAssetResult(
        asset_id=asset_id, asset_name=asset_name,
        asset_elevation_m=elevation_m, distance_to_coast_km=distance_km,
        coastal_region="inland", tidal_regime="default",
        slr_2030_median_m=0.0, slr_2050_median_m=0.0,
        slr_2100_median_m=0.0, slr_2100_h_plus_plus_m=0.0,
        scenario=scenario, regional_multiplier=1.0,
        p_inundation_2030=0.0, p_inundation_2050=0.0, p_inundation_2100=0.0,
        ev_usd_m=ev_usd_m, asset_at_risk_usd_m=0.0,
        expected_loss_2050_usd_m=0.0, expected_loss_2100_usd_m=0.0,
        risk_tier="negligible", disclosure_flag=False, notes=[note],
    )
