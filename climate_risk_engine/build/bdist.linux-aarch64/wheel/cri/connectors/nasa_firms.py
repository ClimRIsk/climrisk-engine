"""
NASA FIRMS Active Fire Connector.

Queries the NASA Fire Information for Resource Management System (FIRMS)
VIIRS (Suomi-NPP / NOAA-20) active fire product to check whether any fire
detections exist within a given radius of an asset coordinate in the last
N days.

Product used: VIIRS_SNPP_NRT (375 m, near-real-time)
API: https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/VIIRS_SNPP_NRT/
Coverage: global, ~12-hour latency
Resolution: 375 m (VIIRS I-band)
Temporal: rolling 10-day NRT window; requests limited to 10 days max.

Authentication
--------------
Requires a free NASA Earthdata MAP_KEY.  Register at:
  https://firms.modaps.eosdis.nasa.gov/usfs/api/area/

Set the key via environment variable:

    export NASA_FIRMS_MAP_KEY=your_key_here

If the variable is not set, the connector returns None (graceful no-op).

Caching
-------
Results are cached on disk under .cache/nasa_firms/ by a key derived from
(rounded lat, rounded lon, radius_km, days_back).  Cache TTL is 6 hours —
fire detections older than that are stale for operational risk monitoring.

Rate limits
-----------
NASA FIRMS NRT API allows ~1 000 requests / day per MAP_KEY.  The connector
rounds coordinates to 0.1° (~11 km) before forming cache keys so a cluster
of assets in the same area shares one cached response.

Source
------
Giglio, L., Schroeder, W., Justice, C.O. (2016).  The collection 6 MODIS
active fire detection algorithm and fire characterization study.  Remote
Sensing of Environment, 178, 31–41.  https://doi.org/10.1016/j.rse.2016.02.054
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_MAP_KEY_ENV = "NASA_FIRMS_MAP_KEY"
_CACHE_DIR = Path(__file__).parent.parent / ".cache" / "nasa_firms"
_CACHE_TTL_SECONDS = 6 * 3600      # 6 hours — NRT latency ~12 h, so 6 h is safe

# VIIRS NRT CSV endpoint
_BASE_URL = (
    "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
    "/{key}/VIIRS_SNPP_NRT"
    "/{w},{s},{e},{n}"   # bounding box: west,south,east,north
    "/{days}"
)

# Default search radius and window
_DEFAULT_RADIUS_KM: float = 50.0   # 50 km radius around asset
_DEFAULT_DAYS_BACK: int = 7        # 7-day rolling window

# Minimum fire radiative power to count as a "significant" detection (MW)
_MIN_FRP_MW: float = 5.0


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FIRMSObservation:
    """Result of a FIRMS active fire query for an asset location.

    Attributes
    ----------
    lat, lon        : Asset coordinates queried.
    radius_km       : Search radius used.
    days_back       : Rolling window searched.
    active_fire     : True if ≥1 VIIRS detection found within radius/window.
    detection_count : Number of VIIRS pixels detected.
    max_frp_mw      : Peak fire radiative power (MW) across all detections.
    nearest_km      : Distance to nearest detection (km); None if no fire.
    detections      : Raw list of detection dicts (lat, lon, acq_date, frp).
    source          : Data source string for provenance.
    queried_at      : Unix timestamp of this query.
    from_cache      : True if this result was served from disk cache.
    """
    lat: float
    lon: float
    radius_km: float
    days_back: int
    active_fire: bool
    detection_count: int
    max_frp_mw: float
    nearest_km: Optional[float]
    detections: list = field(default_factory=list)
    source: str = "NASA FIRMS VIIRS_SNPP_NRT"
    queried_at: float = field(default_factory=time.time)
    from_cache: bool = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _map_key() -> Optional[str]:
    return os.environ.get(_MAP_KEY_ENV)


def _bounding_box(lat: float, lon: float, radius_km: float
                  ) -> tuple[float, float, float, float]:
    """Compute (west, south, east, north) bounding box around (lat, lon)."""
    # 1° latitude ≈ 111.32 km; 1° longitude varies by latitude
    delta_lat = radius_km / 111.32
    delta_lon = radius_km / (111.32 * math.cos(math.radians(abs(lat))))
    return (
        round(lon - delta_lon, 4),   # west
        round(lat - delta_lat, 4),   # south
        round(lon + delta_lon, 4),   # east
        round(lat + delta_lat, 4),   # north
    )


def _haversine_km(lat1: float, lon1: float,
                  lat2: float, lon2: float) -> float:
    """Great-circle distance between two points (km)."""
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(a))


def _cache_path(lat: float, lon: float,
                radius_km: float, days_back: int) -> Path:
    """Disk cache path; coordinates rounded to 0.1° to share nearby assets."""
    lat_r = round(round(lat / 0.1) * 0.1, 1)
    lon_r = round(round(lon / 0.1) * 0.1, 1)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / f"{lat_r}_{lon_r}_r{int(radius_km)}_d{days_back}.json"


def _is_cache_valid(path: Path) -> bool:
    if not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) < _CACHE_TTL_SECONDS


def _fetch_raw_csv(lat: float, lon: float,
                   radius_km: float, days_back: int,
                   key: str) -> str:
    """Fetch VIIRS NRT CSV from NASA FIRMS API."""
    w, s, e, n = _bounding_box(lat, lon, radius_km)
    url = _BASE_URL.format(key=key, w=w, s=s, e=e, n=n, days=days_back)
    with urllib.request.urlopen(url, timeout=15) as resp:  # noqa: S310
        return resp.read().decode("utf-8")


def _parse_csv(csv_text: str, asset_lat: float, asset_lon: float,
               radius_km: float) -> list[dict]:
    """Parse FIRMS CSV and filter to detections within radius_km of asset."""
    reader = csv.DictReader(io.StringIO(csv_text))
    detections = []
    for row in reader:
        try:
            det_lat = float(row.get("latitude") or row.get("lat", 0))
            det_lon = float(row.get("longitude") or row.get("lon", 0))
            frp = float(row.get("frp", 0) or 0)
            dist_km = _haversine_km(asset_lat, asset_lon, det_lat, det_lon)
            if dist_km <= radius_km and frp >= _MIN_FRP_MW:
                detections.append({
                    "lat": det_lat,
                    "lon": det_lon,
                    "dist_km": round(dist_km, 2),
                    "acq_date": row.get("acq_date", ""),
                    "acq_time": row.get("acq_time", ""),
                    "frp_mw": frp,
                    "confidence": row.get("confidence", ""),
                })
        except (ValueError, KeyError):
            continue
    return detections


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def get_active_fire(
    lat: float,
    lon: float,
    radius_km: float = _DEFAULT_RADIUS_KM,
    days_back: int = _DEFAULT_DAYS_BACK,
) -> Optional[FIRMSObservation]:
    """Query NASA FIRMS for active fire detections near (lat, lon).

    Parameters
    ----------
    lat, lon    : Asset coordinates (decimal degrees, WGS-84).
    radius_km   : Search radius around asset (default 50 km).
    days_back   : Rolling window in days (1–10; default 7).

    Returns
    -------
    FIRMSObservation on success (``active_fire`` may still be False if
    no detections were found).  Returns None if:
      - NASA_FIRMS_MAP_KEY environment variable is not set
      - API call fails (network error, rate limit, etc.)

    Caller should treat None as "data unavailable" and not as "no fire".
    """
    key = _map_key()
    if not key:
        return None     # graceful no-op — key not configured

    days_back = max(1, min(days_back, 10))  # API cap: 10 days
    cpath = _cache_path(lat, lon, radius_km, days_back)

    try:
        from_cache = False
        if _is_cache_valid(cpath):
            with open(cpath) as f:
                detections = json.load(f)
            from_cache = True
        else:
            csv_text = _fetch_raw_csv(lat, lon, radius_km, days_back, key)
            detections = _parse_csv(csv_text, lat, lon, radius_km)
            with open(cpath, "w") as f:
                json.dump(detections, f)

        if not detections:
            return FIRMSObservation(
                lat=lat, lon=lon, radius_km=radius_km, days_back=days_back,
                active_fire=False, detection_count=0,
                max_frp_mw=0.0, nearest_km=None,
                from_cache=from_cache,
            )

        nearest = min(d["dist_km"] for d in detections)
        max_frp = max(d["frp_mw"] for d in detections)

        return FIRMSObservation(
            lat=lat, lon=lon, radius_km=radius_km, days_back=days_back,
            active_fire=True,
            detection_count=len(detections),
            max_frp_mw=max_frp,
            nearest_km=nearest,
            detections=detections,
            from_cache=from_cache,
        )

    except Exception:
        return None
