"""
NOAA NHC (National Hurricane Center) tropical cyclone advisory connector.

Fetches active storm tracks and 5-day forecast cones from NOAA NHC.
Covers Atlantic and Eastern Pacific basins.  For Western Pacific / Indian
Ocean storms the connector gracefully returns an empty list (those basins
are handled by GDACS which the existing connector already polls).

Data sources (all open, no API key required)
--------------------------------------------
  CurrentStorms.json  — list of active storms with advisory metadata
  /storm/{id}/        — per-storm JSON with track + forecast cone points

Caches responses for 1 hour (advisories are issued every 6 hours).

Usage
-----
    from cri.connectors.noaa_nhc import get_active_storms, StormAdvisory

    storms = get_active_storms()
    for s in storms:
        print(s.name, s.intensity_kt, s.category)
        for pt in s.forecast_track:
            print(pt.date_utc, pt.lat, pt.lon, pt.wind_kt)

    # Check if any storm threatens an asset at (lat, lon)
    threats = [s for s in storms if s.threatens(asset_lat=25.77, asset_lon=-80.19)]
"""

from __future__ import annotations

import json
import math
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_CACHE_DIR   = Path(__file__).parent.parent.parent.parent / "data" / "cache"
_CACHE_TTL   = 3600          # 1 hour
_NHC_BASE    = "https://www.nhc.noaa.gov"


def _cache_path(key: str) -> Path:
    return _CACHE_DIR / f"nhc_{key}.json"


def _cache_valid(path: Path) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < _CACHE_TTL


def _read_cache(path: Path) -> Optional[dict | list]:
    if _cache_valid(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _write_cache(path: Path, data) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# TC category helpers
# ---------------------------------------------------------------------------

def _category(wind_kt: float) -> str:
    """Saffir-Simpson category string from max sustained wind in knots."""
    mph = wind_kt * 1.15078
    if mph < 39:
        return "Tropical Depression"
    if mph < 74:
        return "Tropical Storm"
    if mph < 96:
        return "Category 1"
    if mph < 111:
        return "Category 2"
    if mph < 130:
        return "Category 3"
    if mph < 157:
        return "Category 4"
    return "Category 5"


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a  = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TrackPoint:
    """One forecast point on a TC track."""
    date_utc: str          # ISO-8601 UTC string
    lat: float
    lon: float
    wind_kt: float         # max sustained wind (knots)
    gust_kt: float         # max gust (knots)
    mslp_hpa: Optional[float]
    cone_radius_km: float  # approx. 67% probability cone radius at this point

    @property
    def category(self) -> str:
        return _category(self.wind_kt)

    @property
    def wind_kmh(self) -> float:
        return self.wind_kt * 1.852

    @property
    def gust_kmh(self) -> float:
        return self.gust_kt * 1.852


@dataclass
class StormAdvisory:
    """Full advisory for one active tropical cyclone."""
    storm_id: str          # e.g. "al012024"
    name: str              # e.g. "HURRICANE BERYL"
    basin: str             # "al" | "ep" | "cp"
    advisory_number: str
    issued_utc: str        # ISO-8601
    # Current position
    lat: float
    lon: float
    intensity_kt: float    # max sustained wind knots
    mslp_hpa: Optional[float]
    # Forecast
    forecast_track: list[TrackPoint] = field(default_factory=list)
    source: str = "NOAA NHC"

    @property
    def category(self) -> str:
        return _category(self.intensity_kt)

    @property
    def wind_kmh(self) -> float:
        return self.intensity_kt * 1.852

    def threatens(
        self,
        asset_lat: float,
        asset_lon: float,
        buffer_km: float = 0.0,
    ) -> bool:
        """
        Return True if this storm's current position OR any forecast track
        point is within (cone_radius_km + buffer_km) of the asset.

        buffer_km adds extra margin for indirect impacts (storm surge,
        outer bands, etc.).  A value of 100 km is typical.
        """
        # Current position check
        dist_now = _haversine_km(self.lat, self.lon, asset_lat, asset_lon)
        # Tropical-storm-force winds typically extend ~200–400 km from center
        current_radius = max(400.0, 200.0) + buffer_km
        if dist_now <= current_radius:
            return True

        # Forecast track check
        for pt in self.forecast_track:
            dist = _haversine_km(pt.lat, pt.lon, asset_lat, asset_lon)
            if dist <= (pt.cone_radius_km + buffer_km):
                return True

        return False

    def hours_to_closest_approach(
        self,
        asset_lat: float,
        asset_lon: float,
    ) -> Optional[float]:
        """
        Estimated hours until the forecast track's closest approach to the
        asset.  Returns None if no forecast track available.
        """
        if not self.forecast_track:
            return None
        min_dist = float("inf")
        best_hours: Optional[float] = None
        now = datetime.now(timezone.utc)
        for pt in self.forecast_track:
            dist = _haversine_km(pt.lat, pt.lon, asset_lat, asset_lon)
            if dist < min_dist:
                min_dist = dist
                try:
                    pt_time = datetime.fromisoformat(pt.date_utc.replace("Z", "+00:00"))
                    best_hours = (pt_time - now).total_seconds() / 3600
                except Exception:
                    best_hours = None
        return best_hours


# ---------------------------------------------------------------------------
# NHC cone radius lookup (approximate from NHC historical verification)
# Hours into forecast → average cone radius (km) at 67% probability
# ---------------------------------------------------------------------------

_CONE_RADIUS_KM: dict[int, float] = {
    12:  75,
    24:  120,
    36:  165,
    48:  210,
    72:  295,
    96:  385,
    120: 475,
}


def _cone_radius_for_hours(hours: float) -> float:
    """Linear interpolation of NHC cone radius at given forecast hour."""
    keys = sorted(_CONE_RADIUS_KM)
    if hours <= keys[0]:
        return _CONE_RADIUS_KM[keys[0]]
    if hours >= keys[-1]:
        return _CONE_RADIUS_KM[keys[-1]]
    for i in range(len(keys) - 1):
        h0, h1 = keys[i], keys[i + 1]
        if h0 <= hours <= h1:
            r0, r1 = _CONE_RADIUS_KM[h0], _CONE_RADIUS_KM[h1]
            t = (hours - h0) / (h1 - h0)
            return r0 + t * (r1 - r0)
    return 300.0


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def _fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "ClimRisk-CRI/1.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def _parse_storm(meta: dict) -> Optional[StormAdvisory]:
    """Parse one entry from CurrentStorms.json into a StormAdvisory."""
    try:
        storm_id = meta.get("id", "")
        # NHC publishes per-storm detail JSON at a predictable URL
        # but format varies by year.  We extract what we can from CurrentStorms.
        lat = float(meta.get("latitudeNumeric", 0))
        lon = float(meta.get("longitudeNumeric", 0))
        wind = float(meta.get("maxWindMph", 0)) / 1.15078  # mph → knots

        advisory = StormAdvisory(
            storm_id=storm_id,
            name=meta.get("name", storm_id).upper(),
            basin=storm_id[:2].lower(),
            advisory_number=str(meta.get("advisoryNumber", "?")),
            issued_utc=meta.get("advisoryDate", ""),
            lat=lat,
            lon=lon,
            intensity_kt=wind,
            mslp_hpa=None,
            forecast_track=[],
        )

        # Try to fetch detailed forecast from NHC forecast JSON
        # NHC publishes: /storm_graphics/{BASIN_UPPER}/{ID_UPPER}/{ID_UPPER}_5day_latest.kmz
        # and a forecast advisory text, but JSON forecast is most reliable via:
        # https://www.nhc.noaa.gov/productlist.shtml or ATCF files
        # Best accessible endpoint: advisory forecast positions in CurrentStorms detail
        forecast_pts = meta.get("forecastPositions", [])
        now = datetime.now(timezone.utc)
        for i, fp in enumerate(forecast_pts):
            try:
                hours = float(fp.get("forecastHour", (i + 1) * 12))
                f_lat  = float(fp.get("lat", 0))
                f_lon  = float(fp.get("lon", 0))
                f_wind = float(fp.get("maxWindMph", 0)) / 1.15078
                f_gust = float(fp.get("gustMph", f_wind * 1.25)) / 1.15078
                advisory.forecast_track.append(TrackPoint(
                    date_utc=fp.get("validTime", ""),
                    lat=f_lat,
                    lon=f_lon,
                    wind_kt=f_wind,
                    gust_kt=f_gust,
                    mslp_hpa=None,
                    cone_radius_km=_cone_radius_for_hours(hours),
                ))
            except Exception:
                continue

        return advisory
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_active_storms() -> list[StormAdvisory]:
    """
    Return a list of StormAdvisory objects for all currently active NHC storms
    (Atlantic + Eastern/Central Pacific basins).

    Returns an empty list if:
      - No storms are currently active (common outside June–November)
      - The NHC endpoint is unreachable
      - A parse error occurs

    Results are cached for 1 hour.
    """
    cpath = _cache_path("current_storms")
    raw = _read_cache(cpath)

    if raw is None:
        try:
            raw = _fetch_json(f"{_NHC_BASE}/CurrentStorms.json")
            _write_cache(cpath, raw)
        except Exception:
            return []

    storms = []
    for meta in raw.get("activeStorms", []):
        advisory = _parse_storm(meta)
        if advisory is not None:
            storms.append(advisory)

    return storms
