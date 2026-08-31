"""
GDACS Active Flood / Disaster Connector.

Queries the Global Disaster Alert and Coordination System (GDACS) RSS/GeoRSS
feed for currently active flood and storm events and checks whether any event
bounding box overlaps a given asset coordinate.

API: https://www.gdacs.org/xml/rss.xml  (public, no key required)
Update frequency: ~hourly
Coverage: global, GLIDE-registered events only
Event types used: FL (flood), TC (tropical cyclone), EQ (earthquake, for
                  infrastructure risk), TS (tsunami)

Authentication
--------------
None required.  GDACS RSS is publicly available.

Caching
-------
The RSS feed is cached on disk under .cache/gdacs/ for up to 1 hour.
This prevents hammering the GDACS server when many assets are assessed
in a single run.

Event-to-asset matching
-----------------------
Each GDACS event carries a bounding box (georss:where polygon or point +
radius).  We check:
  1. Whether the asset coordinate falls inside the event polygon/bbox.
  2. For point events without a polygon, whether the distance from the asset
     to the event epicentre is ≤ the event's reported affected radius.

Alert severity mapping
----------------------
GDACS assigns a colour-coded alert level:
  Green  → 1   (minor, informational)
  Orange → 2   (moderate impact)
  Red    → 3   (major impact)

We surface the maximum severity among all events that match the asset.

Source
------
De Groeve, T., Vernaccini, L., Annunziato, A. (2015).  GDACS — Global
Disaster Alert and Coordination System.  Disaster Prevention and Management,
24(4), 466–471.  https://doi.org/10.1108/DPM-12-2014-0265
"""

from __future__ import annotations

import json
import math
import time
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_GDACS_RSS_URL = "https://www.gdacs.org/xml/rss.xml"
_CACHE_DIR = Path(__file__).parent.parent / ".cache" / "gdacs"
_CACHE_TTL_SECONDS = 3600       # 1 hour — GDACS updates ~hourly

# Event type filter: flood (FL), tropical cyclone (TC), tsunami (TS)
# Earthquake (EQ) is omitted — handled separately if needed
_ACTIVE_TYPES = {"FL", "TC", "TS"}

# Default search radius when event has no polygon (km)
_DEFAULT_EVENT_RADIUS_KM = 100.0

# Alert colour → severity integer
_SEVERITY_MAP = {"Green": 1, "Orange": 2, "Red": 3}

# XML namespaces used in GDACS GeoRSS
_NS = {
    "gdacs": "http://www.gdacs.org",
    "georss": "http://www.georss.org/georss",
    "geo": "http://www.w3.org/2003/01/geo/wgs84_pos#",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class GDACSEvent:
    """Single GDACS event record."""
    event_id: str
    event_type: str       # FL, TC, TS, EQ, …
    title: str
    alert_level: str      # Green / Orange / Red
    severity: int         # 1 / 2 / 3
    country: str
    # Bounding box or point + radius
    lat: Optional[float] = None
    lon: Optional[float] = None
    bbox: Optional[tuple[float, float, float, float]] = None  # (s, w, n, e)
    affected_radius_km: float = _DEFAULT_EVENT_RADIUS_KM
    date_from: str = ""
    date_to: str = ""


@dataclass
class GDACSObservation:
    """Result of a GDACS event query for an asset location.

    Attributes
    ----------
    lat, lon            : Asset coordinates queried.
    active_flood        : True if ≥1 active FL event overlaps the asset.
    active_cyclone      : True if ≥1 active TC event overlaps the asset.
    active_tsunami      : True if ≥1 active TS event overlaps the asset.
    max_alert_severity  : 0 = no event; 1 = Green; 2 = Orange; 3 = Red.
    matched_events      : List of GDACSEvent objects that match the asset.
    total_active_events : Total active GDACS events at time of query.
    source              : Provenance string.
    queried_at          : Unix timestamp.
    from_cache          : True if served from disk cache.
    """
    lat: float
    lon: float
    active_flood: bool = False
    active_cyclone: bool = False
    active_tsunami: bool = False
    max_alert_severity: int = 0
    matched_events: list = field(default_factory=list)
    total_active_events: int = 0
    source: str = "GDACS RSS feed (https://www.gdacs.org/xml/rss.xml)"
    queried_at: float = field(default_factory=time.time)
    from_cache: bool = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lon1: float,
                  lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(a))


def _cache_path() -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / "rss_current.json"


def _is_cache_valid(path: Path) -> bool:
    if not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) < _CACHE_TTL_SECONDS


def _fetch_rss_xml() -> str:
    with urllib.request.urlopen(_GDACS_RSS_URL, timeout=10) as resp:  # noqa: S310
        return resp.read().decode("utf-8", errors="replace")


def _safe_text(elem: Optional[ET.Element]) -> str:
    if elem is None:
        return ""
    return (elem.text or "").strip()


def _parse_point(point_str: str) -> Optional[tuple[float, float]]:
    """Parse 'lat lon' string from georss:point."""
    parts = point_str.strip().split()
    if len(parts) >= 2:
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            pass
    return None


def _parse_polygon(poly_str: str) -> list[tuple[float, float]]:
    """Parse 'lat lon lat lon ...' georss:polygon into list of (lat, lon)."""
    coords = []
    parts = poly_str.strip().split()
    for i in range(0, len(parts) - 1, 2):
        try:
            coords.append((float(parts[i]), float(parts[i + 1])))
        except ValueError:
            continue
    return coords


def _point_in_polygon(lat: float, lon: float,
                      polygon: list[tuple[float, float]]) -> bool:
    """Ray-casting test for point-in-polygon."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i][1], polygon[i][0]   # lon, lat
        xj, yj = polygon[j][1], polygon[j][0]
        intersect = ((yi > lat) != (yj > lat)) and (
            lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-12) + xi
        )
        if intersect:
            inside = not inside
        j = i
    return inside


def _parse_events(xml_text: str) -> list[GDACSEvent]:
    """Parse GDACS RSS XML and return list of active events."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    events: list[GDACSEvent] = []
    channel = root.find("channel")
    if channel is None:
        return []

    for item in channel.findall("item"):
        # Event type
        etype_el = item.find("gdacs:eventtype", _NS)
        etype = _safe_text(etype_el).upper()
        if etype not in _ACTIVE_TYPES:
            continue

        # Event ID
        eid_el = item.find("gdacs:eventid", _NS)
        eid = _safe_text(eid_el) or "unknown"

        # Title
        title_el = item.find("title")
        title = _safe_text(title_el)

        # Alert level
        alert_el = item.find("gdacs:alertlevel", _NS)
        alert_str = _safe_text(alert_el).capitalize()
        severity = _SEVERITY_MAP.get(alert_str, 1)

        # Country
        country_el = item.find("gdacs:country", _NS)
        country = _safe_text(country_el)

        # Temporal range
        date_from = _safe_text(item.find("gdacs:fromdate", _NS))
        date_to   = _safe_text(item.find("gdacs:todate", _NS))

        # Geometry — try polygon first, then point
        lat: Optional[float] = None
        lon: Optional[float] = None
        bbox: Optional[tuple[float, float, float, float]] = None
        affected_radius_km = _DEFAULT_EVENT_RADIUS_KM

        poly_el = item.find("georss:polygon", _NS)
        if poly_el is not None and poly_el.text:
            coords = _parse_polygon(poly_el.text)
            if len(coords) >= 3:
                lats = [c[0] for c in coords]
                lons = [c[1] for c in coords]
                bbox = (min(lats), min(lons), max(lats), max(lons))
                lat = sum(lats) / len(lats)
                lon = sum(lons) / len(lons)
        else:
            point_el = item.find("georss:point", _NS)
            if point_el is not None and point_el.text:
                parsed = _parse_point(point_el.text)
                if parsed:
                    lat, lon = parsed

        # Affected radius from GDACS severity radius field (km)
        rad_el = item.find("gdacs:severitydata", _NS)
        if rad_el is not None:
            try:
                affected_radius_km = float(
                    rad_el.get("radius", str(_DEFAULT_EVENT_RADIUS_KM))
                )
            except (ValueError, TypeError):
                pass

        events.append(GDACSEvent(
            event_id=eid,
            event_type=etype,
            title=title,
            alert_level=alert_str,
            severity=severity,
            country=country,
            lat=lat,
            lon=lon,
            bbox=bbox,
            affected_radius_km=affected_radius_km,
            date_from=date_from,
            date_to=date_to,
        ))

    return events


def _asset_in_event(asset_lat: float, asset_lon: float,
                    event: GDACSEvent) -> bool:
    """Return True if asset coordinate falls within the event's affected area."""
    # Try bounding box first (fast)
    if event.bbox is not None:
        s, w, n, e = event.bbox
        if s <= asset_lat <= n and w <= asset_lon <= e:
            return True
        return False   # bounding box is more authoritative than point+radius

    # Point + radius fallback
    if event.lat is not None and event.lon is not None:
        dist_km = _haversine_km(asset_lat, asset_lon, event.lat, event.lon)
        return dist_km <= event.affected_radius_km

    return False


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def get_active_disasters(
    lat: float,
    lon: float,
) -> Optional[GDACSObservation]:
    """Query GDACS for active flood / cyclone / tsunami events near (lat, lon).

    Parameters
    ----------
    lat, lon : Asset coordinates (decimal degrees, WGS-84).

    Returns
    -------
    GDACSObservation on success (fields may all be False if no events match).
    Returns None if the GDACS feed is unreachable and no cache exists.
    """
    cpath = _cache_path()

    try:
        from_cache = False
        if _is_cache_valid(cpath):
            with open(cpath) as f:
                raw_events = json.load(f)
            from_cache = True
            # Reconstruct GDACSEvent objects from cached dicts
            events = [GDACSEvent(**e) for e in raw_events]
        else:
            xml_text = _fetch_rss_xml()
            events = _parse_events(xml_text)
            # Serialize to cache (convert dataclasses → dicts)
            cache_data = [
                {
                    "event_id": e.event_id,
                    "event_type": e.event_type,
                    "title": e.title,
                    "alert_level": e.alert_level,
                    "severity": e.severity,
                    "country": e.country,
                    "lat": e.lat,
                    "lon": e.lon,
                    "bbox": list(e.bbox) if e.bbox else None,
                    "affected_radius_km": e.affected_radius_km,
                    "date_from": e.date_from,
                    "date_to": e.date_to,
                }
                for e in events
            ]
            with open(cpath, "w") as f:
                json.dump(cache_data, f)

        matched: list[GDACSEvent] = [
            e for e in events if _asset_in_event(lat, lon, e)
        ]

        obs = GDACSObservation(
            lat=lat,
            lon=lon,
            total_active_events=len(events),
            from_cache=from_cache,
        )

        for evt in matched:
            if evt.event_type == "FL":
                obs.active_flood = True
            elif evt.event_type == "TC":
                obs.active_cyclone = True
            elif evt.event_type == "TS":
                obs.active_tsunami = True
            obs.max_alert_severity = max(obs.max_alert_severity, evt.severity)
            obs.matched_events.append(evt)

        return obs

    except Exception:
        return None
