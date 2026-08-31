"""
NASA POWER Climate Baseline Connector.

Queries NASA POWER (Prediction Of Worldwide Energy Resources) for observed
climatological normals at asset coordinates.

Dataset: MERRA-2 / NASA POWER 2.3.0
Coverage: 2001-2020 monthly and annual climatologies
Resolution: 0.5° × 0.625° (regridded from MERRA-2 native 0.5° × 0.625°)
API: https://power.larc.nasa.gov/api/temporal/climatology/point

Variables used:
  T2M         — Temperature at 2m, mean (°C)
  T2M_MAX     — Temperature at 2m, maximum (°C)
  PRECTOTCORR — Precipitation, corrected (mm/day)
  RH2M        — Relative humidity at 2m (%)

Caching: responses cached by (lat, lon) rounded to 0.25° to avoid duplicate
API calls for assets in the same ~25km grid cell.

Source: Stackhouse, P.W. Jr. et al. (2024). POWER: A NASA Earth Science
Data Archival and Distribution System. NASA/TM-2024-20240001.
"""

from __future__ import annotations

import json
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_CACHE_DIR = Path(__file__).parent.parent / ".cache" / "nasa_power"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_BASE_URL = (
    "https://power.larc.nasa.gov/api/temporal/climatology/point"
    "?parameters=T2M,T2M_MAX,PRECTOTCORR,RH2M"
    "&community=RE&longitude={lon}&latitude={lat}&format=JSON"
)


@dataclass
class NASAPowerBaseline:
    """Observed climatological normals at a coordinate (2001-2020 mean)."""
    lat: float
    lon: float
    t2m_mean_c: float        # Annual mean temperature (°C)
    t2m_max_c: float         # Annual mean of daily maxima (°C)
    precip_mm_day: float     # Annual mean precipitation (mm/day)
    rh2m_pct: float          # Annual mean relative humidity (%)
    monthly_t2m_max: dict    # {JAN..DEC: float} for seasonal analysis
    monthly_precip: dict     # {JAN..DEC: float}
    source: str = "NASA POWER MERRA-2 2001-2020 climatology"


def _cache_path(lat: float, lon: float) -> Path:
    """Cache key: round to nearest 0.25° grid cell."""
    lat_r = round(round(lat / 0.25) * 0.25, 2)
    lon_r = round(round(lon / 0.25) * 0.25, 2)
    return _CACHE_DIR / f"{lat_r}_{lon_r}.json"


def _fetch_raw(lat: float, lon: float) -> dict:
    """Fetch from NASA POWER API or load from cache."""
    path = _cache_path(lat, lon)
    if path.exists():
        with open(path) as f:
            return json.load(f)

    url = _BASE_URL.format(lat=round(lat, 4), lon=round(lon, 4))
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        with open(path, "w") as f:
            json.dump(data, f)
        return data
    except Exception as exc:
        raise ConnectionError(f"NASA POWER API unavailable: {exc}") from exc


def get_baseline(lat: float, lon: float) -> Optional[NASAPowerBaseline]:
    """
    Fetch observed climatological baseline for (lat, lon).

    Returns NASAPowerBaseline on success, None on API failure
    (caller falls back to regional lookup table).
    """
    try:
        raw = _fetch_raw(lat, lon)
        params = raw["properties"]["parameter"]

        months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                  "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

        return NASAPowerBaseline(
            lat=lat,
            lon=lon,
            t2m_mean_c=params["T2M"]["ANN"],
            t2m_max_c=params["T2M_MAX"]["ANN"],
            precip_mm_day=params["PRECTOTCORR"]["ANN"],
            rh2m_pct=params["RH2M"]["ANN"],
            monthly_t2m_max={m: params["T2M_MAX"][m] for m in months},
            monthly_precip={m: params["PRECTOTCORR"][m] for m in months},
        )
    except Exception:
        return None
