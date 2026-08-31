"""
Open-Meteo Climate Projection Connector.

Queries the Open-Meteo Climate Change API for CMIP6-based future projections
at asset coordinates.

API: https://climate-api.open-meteo.com/v1/climate
Model: MRI_AGCM3_2_S (MRI AGCM3.2-S, JMA/MRI Japan)
       Primary: SSP5-8.5 (high emissions)
       Fallback: EC_Earth3P_HR (European high-res CMIP6)
Resolution: ~0.25° (~25km native, consistent with NASA NEX-GDDP-CMIP6)

SSP scaling rationale:
  Open-Meteo provides primarily SSP5-8.5 (worst-case) projections. We compute
  the warming/precip delta vs the NASA POWER observed baseline, then scale to
  other SSP scenarios using IPCC AR6 WG1 Table 4.5 GMST ratios at 2050:

    SSP5-8.5 (baseline): 2.00°C GMST → scale factor 1.000
    SSP3-7.0:            1.70°C GMST → scale factor 0.850
    SSP2-4.5:            1.35°C GMST → scale factor 0.675
    SSP1-2.6:            0.85°C GMST → scale factor 0.425

  At 2030 the ratios converge (scenarios haven't diverged strongly yet);
  we apply a year-weighted blend of 1.0 at 2026 → full ratio at 2050.

Variables:
  temperature_2m_max  — daily maximum temperature (°C)
  temperature_2m_mean — daily mean temperature (°C)
  precipitation_sum   — daily total precipitation (mm)

Source: Open-Meteo (2024). Climate Change API. https://open-meteo.com/en/docs/climate-api
        MRI-AGCM3.2-S: Mizuta et al. (2017), J. Meteor. Soc. Japan.
"""

from __future__ import annotations

import json
import math
import statistics
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_CACHE_DIR = Path(__file__).parent.parent / ".cache" / "open_meteo_climate"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Primary model: MRI_AGCM3_2_S — confirmed working in API tests
# Fallback: CMCC_CM2_VHR4
_PRIMARY_MODEL = "MRI_AGCM3_2_S"
_FALLBACK_MODEL = "CMCC_CM2_VHR4"

_BASE_URL = (
    "https://climate-api.open-meteo.com/v1/climate"
    "?latitude={lat}&longitude={lon}"
    "&start_date={year}-01-01&end_date={year}-12-31"
    "&models={model}"
    "&daily=temperature_2m_max,temperature_2m_mean,precipitation_sum"
)

# SSP scale factors vs SSP5-8.5 at 2050 (IPCC AR6 WG1 Table 4.5)
_SSP_SCALE_AT_2050: dict[str, float] = {
    "ssp585": 1.000,
    "ssp370": 0.850,
    "ssp245": 0.675,
    "ssp126": 0.425,
}


@dataclass
class ClimateProjection:
    """Annual statistics from CMIP6 projection at a coordinate for one year."""
    lat: float
    lon: float
    year: int
    model: str
    t2m_max_mean_c: float       # Annual mean of daily max temperature (°C)
    t2m_mean_c: float           # Annual mean temperature (°C)
    precip_mm_day: float        # Annual mean daily precipitation (mm/day)
    elevation_m: float          # Elevation returned by Open-Meteo (m)
    source: str = "Open-Meteo CMIP6 Climate Change API"


def _cache_path(lat: float, lon: float, year: int, model: str) -> Path:
    """Cache key: round to 0.25° grid cell."""
    lat_r = round(round(lat / 0.25) * 0.25, 2)
    lon_r = round(round(lon / 0.25) * 0.25, 2)
    return _CACHE_DIR / f"{lat_r}_{lon_r}_{year}_{model}.json"


def _fetch_raw(lat: float, lon: float, year: int, model: str) -> dict:
    """Fetch from Open-Meteo Climate API or load from cache."""
    path = _cache_path(lat, lon, year, model)
    if path.exists():
        with open(path) as f:
            return json.load(f)

    url = _BASE_URL.format(lat=round(lat, 4), lon=round(lon, 4),
                           year=year, model=model)
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        with open(path, "w") as f:
            json.dump(data, f)
        return data
    except Exception as exc:
        raise ConnectionError(f"Open-Meteo Climate API unavailable: {exc}") from exc


def _parse_annual_stats(raw: dict, model: str) -> tuple[float, float, float]:
    """
    Extract annual mean stats from daily Open-Meteo response.
    Returns (t2m_max_mean, t2m_mean, precip_mm_day) — all None-filtered.

    Key resolution: Open-Meteo single-model requests return plain keys
    (e.g. 'temperature_2m_max'). Multi-model requests return model-suffixed
    keys (e.g. 'temperature_2m_max_MRI_AGCM3_2_S'). We try the plain key
    first (more reliable for single-model queries), then fall back to the
    model-suffixed variant for multi-model response compatibility.
    """
    daily = raw.get("daily", {})

    def _resolve_key(base: str) -> str:
        """Return whichever key is actually present in the response."""
        if base in daily:
            return base
        suffixed = f"{base}_{model}"
        if suffixed in daily:
            return suffixed
        # Last resort: return base (will produce None via .get())
        return base

    key_tmax   = _resolve_key("temperature_2m_max")
    key_tmean  = _resolve_key("temperature_2m_mean")
    key_precip = _resolve_key("precipitation_sum")

    def mean_notnull(arr):
        vals = [v for v in (arr or []) if v is not None]
        return statistics.mean(vals) if vals else None

    return (
        mean_notnull(daily.get(key_tmax)),
        mean_notnull(daily.get(key_tmean)),
        mean_notnull(daily.get(key_precip)),
    )


def get_projection(lat: float, lon: float, year: int) -> Optional[ClimateProjection]:
    """
    Fetch CMIP6 climate projection for (lat, lon) at a given year.

    Tries primary model first; falls back to secondary if primary returns nulls.
    Returns ClimateProjection on success, None on API failure.

    Note: projections represent SSP5-8.5 (high emissions). Use
    scale_to_ssp() to adjust to other scenarios.
    """
    for model in (_PRIMARY_MODEL, _FALLBACK_MODEL):
        try:
            raw = _fetch_raw(lat, lon, year, model)
            tmax, tmean, precip = _parse_annual_stats(raw, model)
            if tmax is None and tmean is None:
                continue   # model returned nulls — try next
            return ClimateProjection(
                lat=lat,
                lon=lon,
                year=year,
                model=model,
                t2m_max_mean_c=tmax or tmean,   # use mean if max unavailable
                t2m_mean_c=tmean or tmax,
                # _parse_annual_stats returns annual mean of daily precipitation sums
                # (mm/day). No further division needed — value is already mm/day.
                precip_mm_day=precip,
                elevation_m=raw.get("elevation", 0.0),
            )
        except Exception:
            continue
    return None


def compute_warming_delta(
    cmip6_historical: ClimateProjection,
    cmip6_future: ClimateProjection,
    ssp_id: str,
    year: int,
    observed_precip_mm_day: Optional[float] = None,
) -> dict:
    """
    Compute actual warming delta at asset coordinates for a given SSP and year.

    Uses a model-consistent delta-downscaling approach:
      warming_delta = CMIP6_future_T2M_MAX − CMIP6_historical_T2M_MAX

    Both historical and future come from the same CMIP6 model, eliminating
    the model bias that arises when comparing CMIP6 output directly to
    observations (NASA POWER T2M_MAX is typically 10–20°C higher in tropics
    due to different sampling / reanalysis vs GCM grid differences).

    The raw delta is then scaled to the target SSP using IPCC AR6 WG1
    Table 4.5 GMST ratios (same as before).

    Returns:
        warming_c: warming delta (°C) relative to historical run, at this SSP/year
        precip_delta_pct: precipitation change (%) from CMIP6 model, SSP-scaled
        t2m_max_projected_obs_c: observed baseline + warming_c (for reporting)
        ssp_scale_applied: SSP scaling ratio applied
        cmip6_model: model used
        source: description for audit trail
    """
    # SSP scaling: interpolate between 1.0 (2026) and full ratio (2050)
    ssp_ratio_2050 = _SSP_SCALE_AT_2050.get(ssp_id, 1.0)
    year_weight = min(1.0, max(0.0, (year - 2026) / (2050 - 2026)))
    ssp_ratio = 1.0 - year_weight * (1.0 - ssp_ratio_2050)

    # Model-consistent delta: future minus historical (same CMIP6 model)
    future_tmax = cmip6_future.t2m_max_mean_c or cmip6_future.t2m_mean_c or 0.0
    hist_tmax   = cmip6_historical.t2m_max_mean_c or cmip6_historical.t2m_mean_c or future_tmax
    raw_warming = future_tmax - hist_tmax
    adjusted_warming = raw_warming * ssp_ratio

    # Precipitation delta (model-consistent %)
    # Guard: skip percentage if CMIP6 historical precip is near-zero (arid regions
    # like Atacama, Sahara). A near-zero denominator produces astronomically large
    # percentages that would inflate flood risk for hyperarid sites.
    # Threshold: 0.10 mm/day (≈ 36 mm/year). Below this, precip is climatologically
    # negligible and percentage change is not a useful signal.
    _MIN_PRECIP_FOR_PCT = 0.10   # mm/day
    precip_delta_pct = None
    if (
        cmip6_future.precip_mm_day is not None
        and cmip6_historical.precip_mm_day is not None
        and cmip6_historical.precip_mm_day >= _MIN_PRECIP_FOR_PCT
    ):
        raw_precip_change = (cmip6_future.precip_mm_day - cmip6_historical.precip_mm_day) / cmip6_historical.precip_mm_day
        # Also cap at ±200% to prevent model artefacts from propagating
        raw_precip_change = max(-0.90, min(2.00, raw_precip_change))
        precip_delta_pct = round(raw_precip_change * ssp_ratio * 100, 2)

    return {
        "warming_c": round(max(0.0, adjusted_warming), 3),
        "precip_delta_pct": precip_delta_pct,
        "cmip6_historical_tmax_c": round(hist_tmax, 2),
        "cmip6_future_tmax_c": round(future_tmax, 2),
        "ssp_scale_applied": round(ssp_ratio, 3),
        "cmip6_model": cmip6_future.model,
        "source": (
            f"Open-Meteo CMIP6 {cmip6_future.model} model-consistent delta "
            f"({hist_tmax:.2f}→{future_tmax:.2f}°C, raw +{raw_warming:.3f}°C), "
            f"SSP5-8.5→{ssp_id} scale {ssp_ratio:.3f} at year {year} "
            f"(IPCC AR6 WG1 Table 4.5)"
        ),
    }
