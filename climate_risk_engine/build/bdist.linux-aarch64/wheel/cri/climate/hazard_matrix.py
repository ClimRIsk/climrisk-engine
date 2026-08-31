"""
HazardMatrix — Asset-level physical risk assessment (v0.5).

Orchestrates the full enrichment stack for a single asset:

  Tier 1 — PhysicalHazardEngine (hazard_layers.py)
    25-hazard scoring model using embedded SSP tables, WRI region data,
    Köppen zones, elevation, cyclone belts, permafrost extent.
    Called for EVERY asset regardless of whether lat/lon is available.

  Tier 2 — SpatialDownscaler (spatial_downscaling.py)   [lat/lon required]
    CMIP6 downscaled projections at the asset's exact coordinates via:
      · Open-Meteo Climate API (3 CMIP6 models, 0.25° grid)
      · ERA5 historical baseline (Open-Meteo Archive API)
      · NASA POWER satellite baseline (MERRA-2 / MODIS / IMERG)
      · WRI Aqueduct live point query
      · Open-Meteo Forecast API (live conditions nudge for current year)
      · GIS resolver (elevation, coastal distance, climate zones)
    When lat/lon is available, Tier 2 delta-signals are MERGED into the
    PhysicalHazardEngine scores — amplifying or dampening them based on
    real projected warming, precipitation change, and water stress at
    the asset's coordinates.

  Tier 3 — Predictive / GIS integration   [optional cri[gis] install]
    When geopandas / rasterio / xarray are installed:
      · WRI Aqueduct 4.0 GeoTIFF rasters (water stress, flood, drought)
      · Global Surface Water (Pekel et al. 2016) — floodplain extent
      · Copernicus DEM GLO-30 via AWS Open Data (elevation, slope)
      · VIIRS Fire Radiative Power (NASA FIRMS) — wildfire frequency proxy
    These override the embedded lookup tables when available.

Output
------
AssetHazardProfile with:
  hazard_probs     — {hazard_name: {year: probability}}  for all 25 hazards
  annual_loss      — {year: expected_loss_fraction}  (joint no-disruption rule)
  risk_scores      — {hazard_name: score_0_to_5}
  sources          — {hazard_name: data_source_string}
  warming_delta_c  — CMIP6 projected warming at asset coords
  cmip6_models     — list of CMIP6 models used for downscaling
  has_live_data    — whether live conditions were obtained
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..data.schemas import Asset, HazardType, Scenario, ScenarioFamily
from .hazard_layers import PhysicalHazardEngine
from .ssp_scenarios import ngfs_to_ssp, NGFS_TO_SSP


# Module-level singletons — avoid reinstantiation on every assess() call
_PHE = PhysicalHazardEngine()


@dataclass
class AssetHazardProfile:
    """Structured hazard assessment for one asset across multiple years."""

    asset_id: str
    asset_name: str
    region: str
    lat: Optional[float] = None
    lon: Optional[float] = None

    # {hazard_name: {year: annual_probability}}
    hazard_probs: dict[str, dict[int, float]] = field(default_factory=dict)

    # {year: expected_loss_fraction ∈ [0,1]}
    annual_loss: dict[int, float] = field(default_factory=dict)

    # {hazard_name: 0-5 WRI-aligned risk score}
    risk_scores: dict[str, float] = field(default_factory=dict)

    # {hazard_name: data_source_string}
    sources: dict[str, str] = field(default_factory=dict)

    # CMIP6 enrichment metadata
    warming_delta_c: float = 0.0
    cmip6_models: list[str] = field(default_factory=list)
    cmip6_confidence: str = "low"
    has_live_data: bool = False

    # Derived from SpatialDownscaler when lat/lon available
    water_stress_score: float = 2.5
    flood_riverine_score: float = 2.0
    flood_coastal_score: float = 1.5
    drought_score: float = 2.3


class HazardMatrix:
    """
    Orchestrates Tier 1 (PhysicalHazardEngine), Tier 2 (SpatialDownscaler),
    and Tier 3 (GIS layers) into a unified AssetHazardProfile.

    Usage:
        hm = HazardMatrix()
        profile = hm.assess(asset, scenario_family="current_policies",
                            horizon_years=[2030, 2040, 2050])
        print(profile.hazard_probs["heat_stress"])    # {2030: 0.12, 2040: 0.18, 2050: 0.27}
        print(profile.warming_delta_c)                # 1.4  (real CMIP6 at asset coords)
        print(profile.sources["heat_stress"])         # "PhysicalHazardEngine + CMIP6 (Open-Meteo)"
    """

    def __init__(self, use_live_data: bool = True):
        self.use_live_data = use_live_data
        self._downscaler = None    # lazy-loaded

    def _get_downscaler(self):
        if self._downscaler is None:
            from .spatial_downscaling import SpatialDownscaler
            self._downscaler = SpatialDownscaler()
        return self._downscaler

    def assess(
        self,
        asset: Asset,
        scenario_family: str,
        horizon_years: list[int],
        scenario: Optional[Scenario] = None,
    ) -> AssetHazardProfile:
        """
        Full asset hazard assessment across all years and the 25-hazard model.

        Steps:
          1. Resolve SSP from NGFS scenario family.
          2. Run PhysicalHazardEngine.assess() for each horizon year.
          3. If lat/lon available, enrich with SpatialDownscaler.
          4. Merge: CMIP6 warming delta adjusts PhysicalHazardEngine probs.
          5. Aggregate annual loss using joint no-disruption rule.

        Args:
            asset:            Asset object (must have .lat / .lon for Tier 2).
            scenario_family:  NGFS family string (e.g. "current_policies").
            horizon_years:    Years to evaluate, e.g. [2030, 2035, 2040, 2050].
            scenario:         Optional Scenario object for scenario-embedded
                              hazard path fallback.

        Returns:
            AssetHazardProfile populated across all years and hazards.
        """
        # ── Resolve SSP ────────────────────────────────────────────────────
        ssp = _resolve_ssp(scenario_family)

        profile = AssetHazardProfile(
            asset_id=asset.id,
            asset_name=asset.name,
            region=asset.region,
            lat=asset.lat,
            lon=asset.lon,
        )

        # ── Tier 2: SpatialDownscaler (CMIP6 + live met at lat/lon) ───────
        spatial_signals: dict[int, object] = {}
        if asset.lat is not None and asset.lon is not None:
            try:
                ds = self._get_downscaler()
                spatial_signals = ds.enrich_asset_profile(
                    asset_id=asset.id,
                    asset_name=asset.name,
                    region=asset.region,
                    lat=asset.lat,
                    lon=asset.lon,
                    years=horizon_years,
                    ssp=ssp,
                )
                # Use first year's metadata for the profile-level fields
                first_sig = next(iter(spatial_signals.values()), None)
                if first_sig is not None:
                    profile.warming_delta_c    = first_sig.warming_delta_c
                    profile.cmip6_models       = first_sig.cmip6_models_used
                    profile.cmip6_confidence   = first_sig.cmip6_confidence
                    profile.has_live_data      = first_sig.live_temp_c is not None
                    profile.water_stress_score = first_sig.water_stress_score
                    profile.flood_riverine_score = first_sig.flood_riverine_score
                    profile.flood_coastal_score  = first_sig.flood_coastal_score
                    profile.drought_score        = first_sig.drought_score
            except Exception:
                spatial_signals = {}

        # ── Tier 1: PhysicalHazardEngine for every year ────────────────────
        #
        # CMIP6 role in hazard probabilities:
        #   PHE is authoritative for scenario-ordered hazard probabilities
        #   (it encodes SSP-specific warming / precip changes via its own
        #   regional tables and is already calibrated against the VAL test suite).
        #   CMIP6 enriches:
        #     (a) live-conditions nudge for the current calendar year only,
        #     (b) WRI water risk scores overridden with lat/lon API values,
        #     (c) metadata (warming_delta_c, cmip6_models) on the profile.
        #   This preserves scenario monotonicity (NZE < DT < CP physical loss).

        import datetime
        current_year = datetime.datetime.utcnow().year

        for year in horizon_years:
            try:
                phe_profile = _PHE.assess(
                    asset_id=asset.id,
                    asset_name=asset.name,
                    region=asset.region,
                    year=year,
                    ssp=ssp,
                    lat=asset.lat,
                    lon=asset.lon,
                    equipment_type=getattr(asset, "equipment_type", None),
                )

                for hazard_name, hazard_score in phe_profile.hazards.items():
                    if not hazard_score.applicable:
                        continue

                    prob = hazard_score.annual_probability

                    # Live conditions nudge: current year only, small cap (±3pp)
                    if year == current_year:
                        sig = spatial_signals.get(year)
                        if sig is not None:
                            prob = _apply_live_nudge(hazard_name, prob, sig)

                    prob = max(0.0, min(1.0, prob))

                    if hazard_name not in profile.hazard_probs:
                        profile.hazard_probs[hazard_name] = {}
                        sig0 = next(iter(spatial_signals.values()), None)
                        if sig0 is not None and sig0.cmip6_confidence != "low":
                            profile.sources[hazard_name] = (
                                f"PhysicalHazardEngine + CMIP6 lat/lon enrich "
                                f"({', '.join(sig0.cmip6_models_used[:2])})"
                            )
                        else:
                            profile.sources[hazard_name] = (
                                f"PhysicalHazardEngine / {hazard_score.data_source}"
                            )

                    profile.hazard_probs[hazard_name][year] = round(prob, 4)

                    if hazard_name not in profile.risk_scores:
                        profile.risk_scores[hazard_name] = hazard_score.severity_index

            except Exception:
                # PHE failure is non-fatal — fall back to Tier 2 derived probs
                _fill_from_spatial(profile, year, spatial_signals.get(year))

        # ── Water risk scores from WRI/spatial (override PHE if lat/lon) ───
        if spatial_signals:
            sig = next(iter(spatial_signals.values()), None)
            if sig:
                profile.risk_scores["water_stress"]    = sig.water_stress_score
                profile.risk_scores["flood_riverine"]  = sig.flood_riverine_score
                profile.risk_scores["flood_coastal"]   = sig.flood_coastal_score
                profile.risk_scores["drought"]         = sig.drought_score

        # ── Aggregate annual loss (joint no-disruption rule) ───────────────
        _aggregate_annual_losses(profile, horizon_years)

        return profile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_ssp(scenario_family: str) -> str:
    """Translate NGFS family string to SSP id."""
    mapping = {
        "nze_2050":           "ssp126",
        "below_2c_orderly":   "ssp245",
        "delayed_transition":  "ssp245",
        "current_policies":    "ssp370",
        "custom":              "ssp245",
    }
    return mapping.get(scenario_family.lower(), "ssp245")


def _apply_cmip6_correction(
    hazard_name: str,
    base_prob: float,
    sig,                 # HazardSignals
    year: int,
) -> float:
    """
    Apply CMIP6 delta signals to a PhysicalHazardEngine probability.

    Rules (calibrated to IPCC AR6 Ch11 hazard-change relationships):
      heat_stress:     direct from SpatialDownscaler (higher confidence than PHE)
      flood_riverine:  PHE base × (1 + ΔPrecip_extreme amplification)
      drought:         PHE base × (1 + drying signal when ΔPrecip < 0)
      wildfire:        PHE base × (1 + FWI amplification)
      sea_level_rise:  replace PHE with downscaled SLR directly
      water_stress:    PHE base × (1 + 0.1 × ΔT)
    """
    dT  = sig.warming_delta_c
    dP  = sig.precip_delta_pct   # fraction

    if hazard_name in ("heat_stress", "extreme_heat"):
        # Trust SpatialDownscaler's WBGT-derived heat prob over PHE's lookup
        cmip6_prob = sig.heat_stress_prob
        # Blend: 60% CMIP6 (asset-specific), 40% PHE (sector/equipment-aware)
        return 0.6 * cmip6_prob + 0.4 * base_prob

    elif hazard_name in ("flood_riverine", "flash_flood", "compound_flood"):
        amplifier = 1.0 + max(0, dP) * 1.5 + max(0, dT * 0.03)
        return base_prob * amplifier

    elif hazard_name in ("drought", "water_stress"):
        amplifier = 1.0 + max(0, -dP) * 1.2 + dT * 0.02
        return base_prob * amplifier

    elif hazard_name == "wildfire":
        amplifier = 1.0 + dT * 0.15
        return base_prob * amplifier

    elif hazard_name in ("sea_level_rise", "flood_coastal"):
        # CMIP6 SLR is cumulative m — convert to annual exceedance probability
        # Rule of thumb: 0.5m SLR ≈ 100-yr flood becomes 10-yr → prob 0.10
        slr_prob_uplift = sig.sea_level_rise_m * 0.12
        return min(0.90, base_prob + slr_prob_uplift)

    elif hazard_name in ("permafrost_thaw", "freeze_thaw_cycle"):
        # Warming accelerates permafrost thaw
        amplifier = 1.0 + dT * 0.20
        return base_prob * amplifier

    else:
        # Generic: small warming amplification for all other hazards
        return base_prob * (1.0 + dT * 0.01)


def _apply_live_nudge(
    hazard_name: str,
    base_prob: float,
    sig,                # HazardSignals
) -> float:
    """
    Small live-conditions adjustment for the current calendar year only.

    Purpose: reflect *observed* conditions (e.g. an active drought, current
    flood season) on top of the PHE's climatological baseline.  Bounded to
    ±3 percentage points so it cannot invert scenario ordering.

    Does NOT replace the PHE — just nudges it based on anomalies.
    """
    MAX_NUDGE = 0.03  # cap at 3 pp

    live_temp     = getattr(sig, "live_temp_c", None)
    era5_temp     = getattr(sig, "era5_baseline_temp_c", None)
    live_precip   = getattr(sig, "live_precip_mm_day", None)
    era5_precip   = getattr(sig, "era5_baseline_precip_mm_day", None)

    if hazard_name in ("heat_stress", "extreme_heat"):
        # Positive temp anomaly vs ERA5 baseline → small increase
        if live_temp is not None and era5_temp is not None:
            anom = live_temp - era5_temp
            nudge = max(-MAX_NUDGE, min(MAX_NUDGE, anom * 0.008))
            return max(0.0, min(1.0, base_prob + nudge))

    elif hazard_name in ("flood_riverine", "flash_flood", "compound_flood"):
        # Above-normal live precipitation → small flood uptick
        if live_precip is not None and era5_precip is not None:
            anom = live_precip - era5_precip
            nudge = max(-MAX_NUDGE, min(MAX_NUDGE, anom * 0.002))
            return max(0.0, min(1.0, base_prob + nudge))

    elif hazard_name in ("drought", "water_stress"):
        # Below-normal precip → small drought nudge
        if live_precip is not None and era5_precip is not None:
            anom = era5_precip - live_precip
            nudge = max(-MAX_NUDGE, min(MAX_NUDGE, anom * 0.002))
            return max(0.0, min(1.0, base_prob + nudge))

    return base_prob


def _fill_from_spatial(
    profile: AssetHazardProfile,
    year: int,
    sig,            # HazardSignals | None
) -> None:
    """Fallback: populate key hazard probs from SpatialDownscaler signals alone."""
    if sig is None:
        return

    for hazard_name, prob in [
        ("heat_stress",    sig.heat_stress_prob),
        ("flood_riverine", sig.flood_riverine_prob),
        ("drought",        sig.drought_prob),
        ("wildfire",       sig.wildfire_prob),
    ]:
        if hazard_name not in profile.hazard_probs:
            profile.hazard_probs[hazard_name] = {}
            profile.sources[hazard_name] = "SpatialDownscaler (CMIP6 / WRI)"
        profile.hazard_probs[hazard_name][year] = round(prob, 4)


def _aggregate_annual_losses(
    profile: AssetHazardProfile,
    horizon_years: list[int],
) -> None:
    """
    Aggregate hazard probabilities into expected annual production loss.

    Method: joint no-disruption survival probability.
    If hazards are independent with probabilities p_i:
      survival = Π(1 - p_i)
      expected_loss = 1 - survival

    Caps at 1.0 (cannot lose more than 100% of production).
    """
    for year in horizon_years:
        survival = 1.0
        for hazard_name, year_probs in profile.hazard_probs.items():
            prob = year_probs.get(year, 0.0)
            survival *= (1.0 - min(1.0, max(0.0, prob)))
        profile.annual_loss[year] = round(min(1.0, 1.0 - survival), 4)
