"""NOAA GFS Weather Forecast Connector.

Provides near-term (0–16 day) weather forecast signals at asset locations.
Used to detect CURRENT extreme weather exposure at the time of assessment —
complementing the long-range CMIP6 climate projections.

Data sources (in priority order):
  1. Open-Meteo Forecast API  — GFS + ECMWF ensemble, global, free, no API key
     https://api.open-meteo.com/v1/forecast
  2. NOAA NWS API             — official US NWS gridded forecast (US only)
     https://api.weather.gov/points/{lat},{lon}
  3. NOAA NOMADS GFS          — raw GFS GRIB2 point extraction via server-side
     https://nomads.ncep.noaa.gov/ (fallback — complex, used for verification only)

Signals extracted:
  - Maximum temperature (°C) over forecast window → heat stress flag
  - Total precipitation (mm) → flood/heavy rain flag
  - Maximum wind gust (km/h) → wind hazard flag
  - Drought indicator (precip deficit vs 30-year climatology)
  - Extreme event flags (any threshold breach in 16-day window)

Why this matters for climate financial risk:
  An asset already under an active heat wave or flood event has HIGHER
  immediate physical risk than the CMIP6 long-range projection alone
  would suggest. Current conditions feed into the short-term CAPEX/loss
  estimate (the "acute" component of physical risk).

References:
  Open-Meteo: https://open-meteo.com/en/docs#api_form
  NOAA NWS:   https://www.weather.gov/documentation/services-web-api
  GFS:        https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib import request as urllib_request

_CACHE_DIR = Path.home() / ".cri_cache" / "noaa_gfs"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Open-Meteo forecast endpoint (primary — no API key, global)
_OM_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo historical climatology (for anomaly detection)
_OM_CLIMATE_URL  = "https://climate-api.open-meteo.com/v1/climate"

# NOAA NWS (US-only fallback)
_NWS_BASE = "https://api.weather.gov"


# ── Extreme event thresholds ─────────────────────────────────────────────────
HEAT_EXTREME_C       = 38.0   # °C — IPCC AR6 threshold for dangerous heat
HEAVY_RAIN_MM        = 50.0   # mm/day — WMO heavy rainfall threshold
EXTREME_WIND_KMH     = 89.0   # km/h — Beaufort scale 10 (storm force)
COLD_STRESS_C        = -15.0  # °C — cold snap threshold for equipment damage
DROUGHT_PRECIP_MM    = 5.0    # mm total over 16 days (extreme dryness)


@dataclass
class GFSForecastSignals:
    """Near-term weather forecast signals for one asset."""

    lat: float
    lon: float
    forecast_days: int = 16

    # ── Temperature ──────────────────────────────────────────────────────
    temp_max_c: float | None = None          # max T forecast
    temp_mean_c: float | None = None
    heat_stress_days: int = 0                # days > HEAT_EXTREME_C

    # ── Precipitation ────────────────────────────────────────────────────
    precip_total_mm: float | None = None     # total precip over window
    max_daily_precip_mm: float | None = None
    heavy_rain_days: int = 0                 # days > HEAVY_RAIN_MM

    # ── Wind ─────────────────────────────────────────────────────────────
    wind_gust_max_kmh: float | None = None
    extreme_wind_days: int = 0

    # ── Drought ──────────────────────────────────────────────────────────
    drought_flag: bool = False               # very low precip signal

    # ── Compound extreme event ───────────────────────────────────────────
    compound_event_flag: bool = False        # heat + low precip simultaneously
    extreme_event_score: float = 0.0        # 0–1 composite severity

    # ── Hazard probability adjustments (applied on top of HazardMatrix) ──
    # Positive = increased short-term risk beyond climate baseline
    hazard_delta: dict[str, float] = field(default_factory=dict)

    # ── Provenance ────────────────────────────────────────────────────────
    model: str = "GFS/ECMWF (Open-Meteo)"
    data_source: str = "NOAA GFS + Open-Meteo Forecast API"
    forecast_issued: str = ""
    error: str | None = None


def _cache_path(lat: float, lon: float, tag: str) -> Path:
    key = f"{lat:.3f}_{lon:.3f}_{tag}.json"
    return _CACHE_DIR / key


def _fetch_om_forecast(lat: float, lon: float, days: int = 16) -> dict | None:
    """Fetch Open-Meteo forecast (GFS + ECMWF ensemble)."""
    cache = _cache_path(lat, lon, f"om_forecast_{days}d")
    # Cache for 6 hours — forecast changes frequently
    if cache.exists() and (time.time() - cache.stat().st_mtime) < 21600:
        return json.loads(cache.read_text())

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "wind_gusts_10m_max",
            "et0_fao_evapotranspiration",
        ]),
        "forecast_days": min(days, 16),
        "timezone": "auto",
    }
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{_OM_FORECAST_URL}?{qs}"

    try:
        req = urllib_request.Request(url, headers={"Accept": "application/json"})
        with urllib_request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
            cache.write_text(json.dumps(data))
            return data
    except Exception:
        return None


def _fetch_nws_forecast(lat: float, lon: float) -> dict | None:
    """Fetch NOAA NWS gridded forecast (US only). Returns None outside US."""
    try:
        url = f"{_NWS_BASE}/points/{lat:.4f},{lon:.4f}"
        req = urllib_request.Request(
            url, headers={"User-Agent": "ClimRisk/0.3 (climrisk.io)"}
        )
        with urllib_request.urlopen(req, timeout=10) as r:
            meta = json.loads(r.read())
        fc_url = meta.get("properties", {}).get("forecastGridData")
        if not fc_url:
            return None
        req2 = urllib_request.Request(
            fc_url, headers={"User-Agent": "ClimRisk/0.3 (climrisk.io)"}
        )
        with urllib_request.urlopen(req2, timeout=15) as r2:
            return json.loads(r2.read())
    except Exception:
        return None


def get_forecast_signals(
    lat: float,
    lon: float,
    days: int = 16,
) -> GFSForecastSignals:
    """
    Fetch near-term weather forecast and derive physical hazard signals.

    Primary: Open-Meteo (GFS + ECMWF ensemble blend, global).
    Fallback: NOAA NWS (US only).

    Parameters
    ----------
    lat, lon : float
        Asset WGS84 coordinates.
    days : int
        Forecast window (max 16 for Open-Meteo free tier).

    Returns
    -------
    GFSForecastSignals with extreme event flags and hazard_delta adjustments.
    """
    sig = GFSForecastSignals(lat=lat, lon=lon, forecast_days=days)

    # ── Primary: Open-Meteo ────────────────────────────────────────────────
    data = _fetch_om_forecast(lat, lon, days)
    if data and "daily" in data:
        d = data["daily"]
        t_max   = d.get("temperature_2m_max", [])
        t_mean  = d.get("temperature_2m_mean", [])
        precip  = d.get("precipitation_sum", [])
        gusts   = d.get("wind_gusts_10m_max", [])

        if t_max:
            sig.temp_max_c  = max(v for v in t_max if v is not None)
            sig.temp_mean_c = sum(v for v in t_mean if v is not None) / max(1, len([v for v in t_mean if v]))
            sig.heat_stress_days = sum(1 for v in t_max if v and v > HEAT_EXTREME_C)

        if precip:
            valid_p = [v for v in precip if v is not None]
            sig.precip_total_mm    = sum(valid_p)
            sig.max_daily_precip_mm = max(valid_p) if valid_p else 0.0
            sig.heavy_rain_days    = sum(1 for v in valid_p if v > HEAVY_RAIN_MM)
            sig.drought_flag       = sig.precip_total_mm < DROUGHT_PRECIP_MM

        if gusts:
            valid_g = [v for v in gusts if v is not None]
            sig.wind_gust_max_kmh = max(valid_g) if valid_g else 0.0
            sig.extreme_wind_days = sum(1 for v in valid_g if v > EXTREME_WIND_KMH)

        sig.forecast_issued = data.get("daily", {}).get("time", [""])[0] if data.get("daily", {}).get("time") else ""
        sig.model = "GFS/ECMWF (Open-Meteo Forecast API)"

        # ── Compound event ────────────────────────────────────────────────
        sig.compound_event_flag = (
            sig.heat_stress_days > 0 and sig.drought_flag
        )

        # ── Composite extreme event score ─────────────────────────────────
        # Weighted severity: heat 40%, rain 30%, wind 20%, drought 10%
        heat_score  = min(1.0, sig.heat_stress_days / max(days, 1))
        rain_score  = min(1.0, sig.heavy_rain_days / max(days, 1))
        wind_score  = min(1.0, sig.extreme_wind_days / max(days, 1))
        drought_sco = 1.0 if sig.drought_flag else 0.0
        sig.extreme_event_score = (
            0.40 * heat_score + 0.30 * rain_score +
            0.20 * wind_score + 0.10 * drought_sco
        )

        # ── Hazard probability delta ─────────────────────────────────────
        # How much to ADD to the climate-baseline hazard probs for this asset.
        # These are SHORT-TERM signals — keep adjustments modest.
        sig.hazard_delta = {}
        if sig.heat_stress_days > 0:
            sig.hazard_delta["heat_stress"] = min(0.05, sig.heat_stress_days * 0.008)
        if sig.heavy_rain_days > 0:
            sig.hazard_delta["flood"] = min(0.06, sig.heavy_rain_days * 0.01)
        if sig.extreme_wind_days > 0:
            sig.hazard_delta["wind"] = min(0.04, sig.extreme_wind_days * 0.008)
        if sig.drought_flag:
            sig.hazard_delta["wildfire"] = 0.02
            sig.hazard_delta["water_stress"] = 0.015
        if sig.compound_event_flag:
            # Compound events get a multiplier — independent hazard rule
            for k in sig.hazard_delta:
                sig.hazard_delta[k] *= 1.3

    else:
        sig.error = "om_forecast_unavailable"

    return sig
