"""
Open-Meteo forecast connector — short-to-medium horizon weather/climate data.

Horizons
--------
  0–16 days  : open-meteo.com/v1/forecast   (deterministic NWP; ECMWF IFS + GFS blend)
  17–90 days : seasonal-api.open-meteo.com  (SEAS5 ensemble; 5-model mean)

No API key required.  Responses are cached per (lat, lon, date) for 3 hours so
repeated calls within a forecast run don't re-hit the network.

Usage
-----
    from cri.connectors.open_meteo_forecast import get_forecast, get_seasonal

    days  = get_forecast(lat=25.77, lon=-80.19)   # Miami — 16 ForecastDay objects
    month = get_seasonal(lat=25.77, lon=-80.19)   # 90-day ensemble mean per day
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(__file__).parent.parent.parent.parent / "data" / "cache"
_CACHE_TTL_SEC = 3 * 3600  # 3 hours — forecasts update 4×/day


def _cache_path(label: str, lat: float, lon: float) -> Path:
    lat_r = round(lat, 2)
    lon_r = round(lon, 2)
    today = date.today().isoformat()
    return _CACHE_DIR / f"forecast_{label}_{lat_r}_{lon_r}_{today}.json"


def _cache_valid(path: Path) -> bool:
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < _CACHE_TTL_SEC


def _read_cache(path: Path) -> Optional[dict]:
    if _cache_valid(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _write_cache(path: Path, data: dict) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

# WMO weather interpretation codes → human-readable label + severity tier
# Severity: 0=clear, 1=minor, 2=moderate, 3=significant, 4=severe, 5=extreme
_WMO_DESCRIPTIONS: dict[int, tuple[str, int]] = {
    0:  ("Clear sky",                   0),
    1:  ("Mainly clear",                0),
    2:  ("Partly cloudy",               0),
    3:  ("Overcast",                    1),
    45: ("Fog",                         1),
    48: ("Freezing fog",                2),
    51: ("Light drizzle",               1),
    53: ("Moderate drizzle",            1),
    55: ("Dense drizzle",               2),
    61: ("Slight rain",                 1),
    63: ("Moderate rain",               2),
    65: ("Heavy rain",                  3),
    71: ("Slight snowfall",             2),
    73: ("Moderate snowfall",           3),
    75: ("Heavy snowfall",              4),
    77: ("Snow grains",                 2),
    80: ("Slight rain showers",         1),
    81: ("Moderate rain showers",       2),
    82: ("Violent rain showers",        4),
    85: ("Slight snow showers",         2),
    86: ("Heavy snow showers",          4),
    95: ("Thunderstorm",                3),
    96: ("Thunderstorm w/ slight hail", 4),
    99: ("Thunderstorm w/ heavy hail",  5),
}


@dataclass
class ForecastDay:
    """One day of NWP deterministic forecast for a single point."""
    date: str                        # ISO-8601 "YYYY-MM-DD"
    tmax_c: Optional[float]          # °C
    tmin_c: Optional[float]          # °C
    precip_mm: Optional[float]       # mm total
    wind_max_kmh: Optional[float]    # km/h sustained max
    wind_gust_kmh: Optional[float]   # km/h gusts
    wmo_code: Optional[int]          # WMO weather code
    # Derived
    weather_label: str = field(init=False)
    weather_severity: int = field(init=False)  # 0–5

    def __post_init__(self) -> None:
        desc, sev = _WMO_DESCRIPTIONS.get(self.wmo_code or 0, ("Unknown", 1))
        self.weather_label = desc
        self.weather_severity = sev

    @property
    def days_from_today(self) -> int:
        d = date.fromisoformat(self.date)
        return (d - date.today()).days

    @property
    def is_severe(self) -> bool:
        """True if any variable crosses a significant-impact threshold."""
        return (
            self.weather_severity >= 3
            or (self.wind_max_kmh or 0) >= 60
            or (self.precip_mm or 0) >= 50
            or (self.tmax_c or 0) >= 38
        )


@dataclass
class SeasonalDay:
    """One day of 3-month ensemble seasonal outlook."""
    date: str
    tmax_c_mean: Optional[float]     # ensemble mean °C
    tmax_c_p90: Optional[float]      # 90th percentile °C
    precip_mm_mean: Optional[float]  # ensemble mean mm
    precip_mm_p10: Optional[float]   # dry 10th percentile mm


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def _fetch_url(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "ClimRisk-CRI/1.0"})
    with urllib.request.urlopen(req, timeout=12) as r:
        return json.loads(r.read())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_forecast(
    lat: float,
    lon: float,
    days: int = 16,
) -> list[ForecastDay]:
    """
    Fetch 0–16 day deterministic NWP forecast for (lat, lon).

    Returns a list of ForecastDay objects (up to ``days`` items).
    Returns an empty list if the API is unreachable.
    """
    cpath = _cache_path("det", lat, lon)
    raw = _read_cache(cpath)

    if raw is None:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&daily=temperature_2m_max,temperature_2m_min,"
            "precipitation_sum,windspeed_10m_max,windgusts_10m_max,weathercode"
            f"&forecast_days={days}"
            "&timezone=auto"
        )
        try:
            raw = _fetch_url(url)
            _write_cache(cpath, raw)
        except Exception:
            return []

    daily = raw.get("daily", {})
    dates = daily.get("time", [])
    result = []
    for i, d in enumerate(dates):
        def _v(key: str) -> Optional[float]:
            vals = daily.get(key, [])
            v = vals[i] if i < len(vals) else None
            return float(v) if v is not None else None

        result.append(ForecastDay(
            date=d,
            tmax_c=_v("temperature_2m_max"),
            tmin_c=_v("temperature_2m_min"),
            precip_mm=_v("precipitation_sum"),
            wind_max_kmh=_v("windspeed_10m_max"),
            wind_gust_kmh=_v("windgusts_10m_max"),
            wmo_code=int(_v("weathercode") or 0),
        ))
    return result


def get_seasonal(
    lat: float,
    lon: float,
    months: int = 3,
) -> list[SeasonalDay]:
    """
    Fetch 17–90 day (up to 3 month) seasonal ensemble outlook for (lat, lon).

    Uses SEAS5 / multi-model ensemble mean + percentiles.
    Returns an empty list if the API is unreachable.
    """
    cpath = _cache_path("seas", lat, lon)
    raw = _read_cache(cpath)

    if raw is None:
        url = (
            "https://seasonal-api.open-meteo.com/v1/seasonal"
            f"?latitude={lat}&longitude={lon}"
            "&daily=temperature_2m_max,precipitation_sum"
            f"&forecast_months={months}"
            "&timezone=auto"
        )
        try:
            raw = _fetch_url(url)
            _write_cache(cpath, raw)
        except Exception:
            return []

    daily = raw.get("daily", {})
    dates = daily.get("time", [])

    # Seasonal API returns ensemble members as separate columns:
    # temperature_2m_max_member01..member06, precipitation_sum_member01..member06
    def _member_vals(prefix: str, idx: int) -> list[float]:
        out = []
        for m in range(1, 7):
            key = f"{prefix}_member{m:02d}"
            vals = daily.get(key, [])
            v = vals[idx] if idx < len(vals) else None
            if v is not None:
                out.append(float(v))
        return out

    result = []
    for i, d in enumerate(dates):
        t_members = _member_vals("temperature_2m_max", i)
        p_members = _member_vals("precipitation_sum", i)

        t_mean = (sum(t_members) / len(t_members)) if t_members else None
        t_p90  = sorted(t_members)[int(len(t_members) * 0.9)] if t_members else None
        p_mean = (sum(p_members) / len(p_members)) if p_members else None
        p_p10  = sorted(p_members)[int(len(p_members) * 0.1)] if p_members else None

        result.append(SeasonalDay(
            date=d,
            tmax_c_mean=t_mean,
            tmax_c_p90=t_p90,
            precip_mm_mean=p_mean,
            precip_mm_p10=p_p10,
        ))
    return result
