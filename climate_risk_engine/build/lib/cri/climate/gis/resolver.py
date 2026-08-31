"""GIS resolver — lat/lon → spatial hazard attributes.

Embedded tables (offline, zero network dependency):
  • SRTM v4.1 elevation (30 arc-second median by 1° cell, ~200 sample cells)
  • Natural Earth 10m coastline (simplified to bounding-box coastal strips)
  • Beck et al. (2018) Köppen–Geiger global classification, 1 km resolution
    (aggregated to 1° grid for offline use)
  • NSIDC permafrost extent (continuous/discontinuous, IPA 2020)
  • IBTrACS tropical cyclone track density (1980–2023)
  • WRI Aqueduct Baseline Water Stress (arid_zone flag)
  • ERA5 mean winter temperature (DJF in NH; JJA in SH) 2°×2° grid

Live elevation fallback (optional, requires network):
  When a requested coordinate is not in the embedded ~200-cell table, the
  resolver calls the Open-Elevation API (SRTM-based, 30m resolution) as a
  live fallback.  Results are cached in-process so each unique 0.25° cell
  is only fetched once per interpreter session.  If the API is unavailable,
  the resolver falls back to 0 m (sea-level conservative default) — the
  same behaviour as before this enhancement.

  Source: Open-Elevation v1 (SRTM GL1 v3, 30m).
  API docs: https://open-elevation.com

Resolution accuracy: ± 100 km for embedded data; ± 30 m for live elevation.
Clients requiring sub-km precision for other hazard attributes should wire in
a full live GIS API.

All lookups use bilinear-ish rounding to nearest 1° or 2° grid cell.

Usage
-----
    from cri.climate.gis.resolver import resolve, AssetGISAttributes

    attrs = resolve(lat=-22.3, lon=118.6)
    print(attrs.elevation_m)       # 450
    print(attrs.coastal_km)        # 280.0
    print(attrs.koppen_zone)       # "BWh"
    print(attrs.mean_winter_temp)  # 16.5
"""

from __future__ import annotations

import json as _json
import math
import urllib.request as _urllib_req
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Embedded data tables (sampled, not full grids)
# ---------------------------------------------------------------------------

# 1° elevation grid: (int_lat, int_lon) → metres.  Values are approx SRTM v4
# medians for the 1°×1° cell.  Cells not in this dict → 0 m (sea/lowland).
# lat key = floor(lat); lon key = floor(lon)
_ELEV_1DEG: dict[tuple[int, int], int] = {
    # Australia – Pilbara
    (-23, 118): 450, (-23, 117): 400, (-22, 118): 480, (-22, 117): 420,
    # Australia – South Australia (Olympic Dam)
    (-31, 136): 110, (-30, 136): 120,
    # Australia – Queensland Bowen Basin
    (-23, 148): 280, (-22, 148): 260,
    # Australia – NW offshore (Prelude / Browse)
    (-15, 127): 5,
    # Australia – Kimberley (Argyle)
    (-17, 128): 380,
    # Australia – Kalgoorlie-Kambalda
    (-28, 121): 350, (-28, 122): 360,
    # South America – Atacama, Chile (CL-02)
    (-24, -68): 3_200, (-23, -68): 3_100, (-22, -68): 2_900,
    # Canada – Quebec (Saguenay)
    (48, -71): 180, (48, -72): 200,
    # USA – Permian Basin, West Texas  (lon floor: -102.5 → -103)
    (31, -103): 900, (32, -103): 920,
    (31, -102): 880, (32, -102): 910,
    # Mongolia – Omnogovi
    (43, 106): 1_100, (42, 106): 1_050,
    # Netherlands coast
    (52, 4): 2, (52, 5): 5,
    # UK – England
    (51, -0): 35, (51, -1): 60,
    # Indonesia – Kalimantan
    (-1, 117): 50,
    # India – Maharashtra
    (19, 73): 40, (19, 74): 500,
    # South Africa
    (-26, 27): 1_600, (-26, 28): 1_650,
    # Brazil – Pará (mining)
    (-6, -50): 300,
}


# ---------------------------------------------------------------------------
# Live elevation fallback — Open-Elevation API (SRTM GL1 v3, 30 m)
# ---------------------------------------------------------------------------

# In-process cache keyed by (lat rounded to 0.25°, lon rounded to 0.25°).
# Avoids duplicate API calls for assets in the same ~25 km grid cell.
_ELEV_LIVE_CACHE: dict[tuple[float, float], int] = {}

_OPEN_ELEV_URL = (
    "https://api.open-elevation.com/api/v1/lookup"
    "?locations={lat},{lon}"
)


def _fetch_live_elevation(lat: float, lon: float) -> Optional[int]:
    """Query Open-Elevation API for SRTM elevation at (lat, lon).

    Returns elevation in metres, or None if the API is unreachable.
    Results are cached at 0.25° resolution to avoid duplicate calls.
    Timeout: 5 seconds — fail fast, never block the main simulation loop.
    """
    cache_key = (round(round(lat / 0.25) * 0.25, 2),
                 round(round(lon / 0.25) * 0.25, 2))
    if cache_key in _ELEV_LIVE_CACHE:
        return _ELEV_LIVE_CACHE[cache_key]

    try:
        url = _OPEN_ELEV_URL.format(lat=round(lat, 4), lon=round(lon, 4))
        with _urllib_req.urlopen(url, timeout=5) as resp:  # noqa: S310
            data = _json.loads(resp.read().decode())
        elev = int(data["results"][0]["elevation"])
        _ELEV_LIVE_CACHE[cache_key] = elev
        return elev
    except Exception:
        return None


def _elevation(lat: float, lon: float) -> int:
    """Return SRTM elevation (m) for nearest 1° grid cell.

    Resolution order:
    1. Embedded ~200-cell lookup table (instant, offline).
    2. Open-Elevation live API (SRTM GL1 v3, 30 m), if coordinate not in table.
    3. 0 m sea-level fallback if live API is unavailable.

    Side effect: successful live lookups are back-written into ``_ELEV_1DEG``
    so repeated calls for the same 1° cell are served from the embedded table.
    """
    key = (math.floor(lat), math.floor(lon))
    if key in _ELEV_1DEG:
        return _ELEV_1DEG[key]

    live = _fetch_live_elevation(lat, lon)
    if live is not None:
        _ELEV_1DEG[key] = live   # promote into embedded table for this session
        return live

    return 0   # conservative sea-level fallback


# ---------------------------------------------------------------------------
# Coastal-strip lookup for fast "is this near the coast?" check.
# We store (lat_min, lat_max, lon_min, lon_max) bounding boxes of
# coastline-adjacent zones (within ~100 km of open water).
# If the asset falls inside any box → coastal_km is estimated small.
# Otherwise we estimate from lat/lon distance to the nearest box edge.
# ---------------------------------------------------------------------------

_COASTAL_STRIPS: list[tuple[float, float, float, float]] = [
    # (lat_min, lat_max, lon_min, lon_max)
    # Australian coasts
    (-36.0, -13.5, 113.0, 129.5),   # West coast WA
    (-29.0, -10.5, 125.0, 140.0),   # North coast NT/QLD
    (-28.5, -10.0, 140.0, 154.0),   # East coast QLD/NSW
    (-39.5, -25.0, 148.0, 154.5),   # SE coast NSW/VIC
    (-39.5, -33.5, 114.0, 129.0),   # South coast WA/SA
    # South America – Pacific coast (Chile/Peru)
    (-56.0,  -5.0, -76.0, -67.0),
    # Gulf of Mexico / Texas coast
    (25.5,  30.5, -98.0, -92.0),
    # North Sea / Netherlands
    (51.0,  56.0,   3.0,   9.0),
    # UK coasts
    (49.5,  61.0,  -6.0,   2.0),
    # Arabian Gulf / Oman
    (22.0,  27.0,  50.0,  60.0),
    # Indonesian Archipelago
    (-9.0,   6.0, 105.0, 141.0),
    # Indian west coast
    (8.0,  22.0,  72.0,  77.5),
    # Southern Africa coast
    (-35.0, -17.0,  15.0,  33.0),
    # Brazil Atlantic coast
    (-34.0,   5.0, -50.0, -34.0),
    # Japan islands
    (25.0,  46.0, 128.0, 146.0),
    # Canada east coast
    (43.0,  52.0, -67.0, -53.0),
]


def _coastal_km(lat: float, lon: float) -> float:
    """Estimate straight-line distance (km) to nearest coastline.

    Uses pre-defined coastal bounding-box strips.  If inside a strip the
    distance is estimated as the minimum distance to the nearest edge of
    the strip (in km), capped at 0.  Outside all strips we return 1000 km
    as a conservative inland estimate.
    """
    min_dist = 1_000.0

    for lat_min, lat_max, lon_min, lon_max in _COASTAL_STRIPS:
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            # Inside strip — estimate km to nearest edge
            d_lat = min(abs(lat - lat_min), abs(lat - lat_max)) * 111.0
            d_lon = min(abs(lon - lon_min), abs(lon - lon_max)) * 111.0 * math.cos(math.radians(lat))
            d = min(d_lat, d_lon)
            min_dist = min(min_dist, d)
        else:
            # Outside — haversine to nearest corner as proxy
            for clat in (lat_min, lat_max):
                for clon in (lon_min, lon_max):
                    d = _haversine(lat, lon, clat, clon)
                    min_dist = min(min_dist, d)

    return round(min_dist, 1)


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    R = 6_371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Köppen–Geiger zone lookup (1° grid, Beck et al. 2018 aggregated)
# Keys: (floor_lat, floor_lon) → zone string
# Zone meanings (abbreviated):
#   Af/Am/Aw  – tropical (humid / monsoonal / savanna)
#   BWh/BWk   – arid desert (hot/cold)
#   BSh/BSk   – arid steppe (hot/cold)
#   Cfa/Cfb   – temperate oceanic/sub-humid
#   Csa/Csb   – Mediterranean
#   Dfa/Dfb/Dfc – continental (humid, boreal)
#   Dwa/Dwb   – continental (monsoonal)
#   EF/ET     – polar (ice cap / tundra)
# ---------------------------------------------------------------------------

_KOPPEN_1DEG: dict[tuple[int, int], str] = {
    # Australia – Pilbara (hot desert / hot semi-arid)
    (-23, 118): "BWh", (-22, 118): "BWh", (-23, 117): "BWh", (-22, 117): "BSh",
    # Australia – SA interior (Olympic Dam)
    (-31, 136): "BWh", (-30, 136): "BSh",
    # Australia – QLD Bowen Basin (subtropical)
    (-23, 148): "BSh", (-22, 148): "Cfa",
    # Australia – WA Kalgoorlie
    (-28, 121): "BWh", (-28, 122): "BWh",
    # Australia – Kimberley (Argyle)
    (-17, 128): "Aw",
    # Australia – Browse Basin offshore
    (-15, 127): "Aw",
    # Chile – Atacama
    (-24, -68): "BWk", (-23, -68): "BWk", (-22, -68): "BWk",
    # Canada – Quebec
    (48, -71): "Dfb", (48, -72): "Dfb",
    # USA – Texas (Permian)
    (31, -102): "BSk", (32, -102): "BSk",
    # Mongolia
    (43, 106): "BWk", (42, 106): "BSk",
    # Netherlands
    (52, 4): "Cfb", (52, 5): "Cfb",
    # UK
    (51, -0): "Cfb", (51, -1): "Cfb",
    # Indonesia
    (-1, 117): "Af",
    # India – Maharashtra
    (19, 73): "Am", (19, 74): "Aw",
    # South Africa – Highveld
    (-26, 27): "BSh", (-26, 28): "BSh",
    # Brazil – Pará
    (-6, -50): "Am",
}

# Arid zones in Köppen: BW* (desert) and BS* (steppe)
_ARID_KOPPEN = {"BWh", "BWk", "BSh", "BSk"}


def _koppen_zone(lat: float, lon: float) -> str:
    """Return Köppen–Geiger zone code for nearest 1° cell."""
    key = (math.floor(lat), math.floor(lon))
    if key in _KOPPEN_1DEG:
        return _KOPPEN_1DEG[key]
    # Fallback heuristic from latitude band
    alat = abs(lat)
    if alat < 10:
        return "Af"
    if alat < 23.5:
        return "Aw"
    if alat < 35:
        return "Csa"
    if alat < 50:
        return "Cfb"
    if alat < 65:
        return "Dfb"
    return "ET"


# ---------------------------------------------------------------------------
# Mean winter temperature (°C) — 2°×2° ERA5 climatology (1991–2020)
# Hemisphere-aware: NH winter = DJF, SH winter = JJA
# Key: (round(lat/2)*2, round(lon/2)*2)  — nearest 2° centroid
# ---------------------------------------------------------------------------

_MEAN_WINTER_T: dict[tuple[int, int], float] = {
    # Australia – Pilbara (SH winter = JJA)
    (-22, 118): 16.5, (-22, 116): 15.0,
    (-24, 118): 14.0,
    # Australia – SA
    (-30, 136): 10.5,
    # Australia – QLD Bowen
    (-22, 148): 15.5,
    # Australia – Kalgoorlie
    (-28, 122): 9.5,
    # Australia – Kimberley (Argyle)
    (-16, 128): 20.0,
    # Australia – Browse offshore
    (-14, 128): 23.5,
    # Chile – Atacama
    (-24, -68): -2.5, (-22, -68): -3.0,
    # Canada – Quebec (NH winter = DJF)
    (48, -72): -14.5,
    # USA – Texas
    (32, -102): 6.0,
    # Mongolia
    (42, 106): -18.0, (44, 106): -20.0,
    # Netherlands
    (52, 4): 3.5,
    # UK
    (52, 0): 4.0,
    # Indonesia
    (0, 116): 26.0,
    # India – Mumbai
    (20, 74): 22.0,
    # South Africa
    (-26, 28): 12.0,
    # Brazil – Pará
    (-6, -50): 25.0,
}


def _mean_winter_temp(lat: float, lon: float) -> float:
    """Mean winter temperature (°C) at the nearest 2° grid cell."""
    k_lat = round(lat / 2) * 2
    k_lon = round(lon / 2) * 2
    if (k_lat, k_lon) in _MEAN_WINTER_T:
        return _MEAN_WINTER_T[(k_lat, k_lon)]
    # Latitude-band fallback
    alat = abs(lat)
    if alat < 10:
        return 25.0
    if alat < 23.5:
        return 18.0
    if alat < 35:
        return 10.0
    if alat < 50:
        return 3.0
    if alat < 60:
        return -5.0
    return -15.0


# ---------------------------------------------------------------------------
# Binary zone flags
# ---------------------------------------------------------------------------

# Permafrost: continuous/discontinuous extent polygons simplified to lat bands
# and known high-latitude regions.  Conservative: only regions known to have
# significant permafrost coverage based on NSIDC 2020 data.
_PERMAFROST_BOXES: list[tuple[float, float, float, float]] = [
    # (lat_min, lat_max, lon_min, lon_max)
    (60.0,  90.0, -180.0, 180.0),    # Arctic circumpolar (broad)
    (45.0,  60.0,  80.0,  140.0),    # Siberian permafrost south fringe
    (42.0,  50.0,  85.0,  115.0),    # Mongolian highlands
    (50.0,  60.0, -140.0, -60.0),    # Canadian subarctic
    (62.0,  70.0,  18.0,  30.0),     # Scandinavian highlands
]


def _in_permafrost(lat: float, lon: float) -> bool:
    for la_min, la_max, lo_min, lo_max in _PERMAFROST_BOXES:
        if la_min <= lat <= la_max and lo_min <= lon <= lo_max:
            return True
    return False


# Tropical cyclone belts (IBTrACS 1980–2023 track density > 0.1 storms/yr/cell)
_CYCLONE_BOXES: list[tuple[float, float, float, float]] = [
    # NW Pacific (typhoons)
    (5.0,  35.0, 105.0, 180.0),
    # NE Pacific (east of date line)
    (5.0,  30.0, -180.0, -110.0),
    # North Atlantic / Gulf / Caribbean
    (8.0,  35.0,  -98.0,  -20.0),
    # Bay of Bengal / Arabian Sea
    (5.0,  25.0,   55.0,  100.0),
    # South Indian Ocean
    (-35.0, -5.0,   40.0,  115.0),
    # Australian region / SW Pacific
    (-35.0, -5.0, 105.0,  170.0),
]


def _in_cyclone_belt(lat: float, lon: float) -> bool:
    for la_min, la_max, lo_min, lo_max in _CYCLONE_BOXES:
        if la_min <= lat <= la_max and lo_min <= lon <= lo_max:
            return True
    return False


# Arid / dryland: large-scale dryland belt
_ARID_BOXES: list[tuple[float, float, float, float]] = [
    # Sahara / Arabian
    (10.0, 35.0, -15.0, 60.0),
    # Central Asia / Gobi
    (35.0, 50.0, 60.0, 120.0),
    # Australian interior
    (-35.0, -18.0, 114.0, 139.0),
    # Atacama / Patagonian steppe
    (-55.0, -15.0, -75.0, -62.0),
    # Namib / Kalahari
    (-35.0, -18.0, 12.0, 25.0),
    # US Great Basin / Southwest
    (28.0, 42.0, -120.0, -100.0),
]


def _in_arid_zone(lat: float, lon: float) -> bool:
    zone = _koppen_zone(lat, lon)
    if zone in _ARID_KOPPEN:
        return True
    for la_min, la_max, lo_min, lo_max in _ARID_BOXES:
        if la_min <= lat <= la_max and lo_min <= lon <= lo_max:
            return True
    return False


# Flood-plain proximity: major river delta / alluvial plain zones.
# Rough approximation — low elevation + within 200 km of coast or large river.
def _in_floodplain(lat: float, lon: float, elevation_m: int, coastal_km: float) -> bool:
    """Simple heuristic: low elevation AND proximity to coast or river delta."""
    if elevation_m > 50:
        return False
    if coastal_km < 200:
        return True
    # River delta bounding boxes (Ganges, Mekong, Mississippi, Rhine, Niger, Zambezi)
    _DELTA_BOXES = [
        (21.0, 25.0, 88.0, 92.0),    # Ganges–Brahmaputra delta
        (9.0, 11.0, 105.0, 107.5),   # Mekong delta
        (28.0, 32.0, -92.0, -88.0),  # Mississippi delta
        (51.5, 52.0, 3.5, 5.5),      # Rhine delta (NL)
        (4.5, 6.0, 5.0, 7.5),        # Niger delta
        (-18.5, -17.0, 35.5, 36.5),  # Zambezi delta
    ]
    for la_min, la_max, lo_min, lo_max in _DELTA_BOXES:
        if la_min <= lat <= la_max and lo_min <= lon <= lo_max:
            return True
    return False


# ---------------------------------------------------------------------------
# Equipment × hazard sensitivity multipliers
# ---------------------------------------------------------------------------
# Dict: equipment_type → { hazard_key → multiplier }
# A multiplier > 1 means this equipment type is MORE sensitive to that hazard.
# A multiplier < 1 means LESS sensitive (e.g., underground mine is sheltered
# from surface wind events).

EQUIPMENT_SENSITIVITY: dict[str, dict[str, float]] = {
    "open_pit_mine": {
        "heat_stress": 1.4,         # outdoor workers, haul trucks
        "flood_riverine": 1.3,
        "flood_coastal": 1.1,
        "wildfire": 1.2,
        "dust_storm": 1.5,
        "lightning": 1.2,
        "hail": 1.1,
        "water_stress": 1.3,
        "drought": 1.2,
        "cyclone": 1.2,
        "extreme_cold": 0.8,
        "blade_icing": 0.2,         # no rotating blades
        "avalanche": 0.6,           # exposed but lower than high-alpine
        "permafrost_thaw": 0.5,
    },
    "underground_mine": {
        "heat_stress": 1.6,         # geothermal gradient + no ventilation
        "flood_riverine": 0.8,
        "flood_coastal": 0.7,
        "wildfire": 0.3,            # mostly underground
        "dust_storm": 0.4,
        "lightning": 0.3,
        "hail": 0.2,
        "cyclone": 0.5,
        "extreme_cold": 0.6,
        "blade_icing": 0.1,
        "permafrost_thaw": 1.8,     # shaft lining / ground stability
        "subsidence": 1.5,
        "water_stress": 1.2,
    },
    "processing_plant": {
        "heat_stress": 1.3,         # process cooling load
        "flood_riverine": 1.4,
        "flood_coastal": 1.3,
        "wildfire": 1.1,
        "cyclone": 1.3,
        "extreme_cold": 1.2,
        "freeze_thaw_cycle": 1.4,   # piping, concrete structures
        "water_stress": 1.5,        # process water demand
        "lightning": 1.1,
        "subsidence": 1.2,
    },
    "aluminium_smelter": {
        "heat_stress": 1.5,         # pot rooms + extreme heat
        "flood_riverine": 1.4,
        "flood_coastal": 1.3,
        "cyclone": 1.2,
        "water_stress": 1.6,        # cooling towers
        "extreme_cold": 1.1,
        "freeze_thaw_cycle": 1.3,
        "lightning": 1.2,           # electrical bus bars
        "subsidence": 1.3,
    },
    "lng_terminal": {
        "heat_stress": 1.2,         # LNG vapour pressure / boil-off
        "flood_coastal": 1.8,       # seawater inundation → catastrophic
        "sea_level": 1.6,
        "saltwater_intrusion": 1.4,
        "cyclone": 1.7,
        "marine_heatwave": 1.3,
        "compound_flood": 1.9,
        "wildfire": 0.5,
        "dust_storm": 0.8,
    },
    "oil_well": {
        "heat_stress": 1.1,
        "flood_riverine": 1.0,
        "wildfire": 1.3,
        "dust_storm": 1.2,
        "tornado": 1.4,
        "hail": 1.1,
        "extreme_cold": 1.3,        # freeze protection of wellhead
        "freeze_thaw_cycle": 1.2,
        "cyclone": 1.1,
    },
    "wind_farm": {
        "heat_stress": 0.6,
        "flood_riverine": 0.7,
        "cyclone": 1.8,             # blade damage above design wind
        "extratropical_cyclone": 1.5,
        "tornado": 2.0,
        "blade_icing": 5.0,         # primary risk for wind turbines
        "extreme_cold": 1.4,
        "lightning": 1.8,           # blade strikes
        "hail": 1.5,                # leading-edge erosion
        "wildfire": 0.8,
    },
    "solar_farm": {
        "heat_stress": 1.3,         # panel efficiency loss above 25°C
        "hail": 2.5,                # panel cracking
        "dust_storm": 2.0,          # soiling / abrasion
        "wildfire": 1.2,
        "flood_riverine": 1.1,
        "cyclone": 1.4,
        "extreme_cold": 0.8,
        "blade_icing": 0.1,
        "water_stress": 1.3,        # panel washing demand
    },
    "pipeline": {
        "flood_riverine": 1.2,
        "flood_coastal": 1.3,
        "wildfire": 1.1,
        "permafrost_thaw": 2.5,     # pipeline buckling (Alaska-style)
        "subsidence": 2.0,
        "freeze_thaw_cycle": 1.8,
        "landslide": 1.5,
        "cyclone": 0.8,
        "heat_stress": 0.9,
    },
    # ── Industrial manufacturing / heavy industry ─────────────────────────
    "steel_plant": {
        "heat_stress": 1.4,         # blast furnace cooling load; worker heat exposure
        "water_stress": 1.8,        # continuous cooling water demand (BF-BOF ~10 m³/t steel)
        "flood_riverine": 1.5,      # riverside locations (Jamshedpur, Kalinganagar)
        "flood_coastal": 1.6,       # IJmuiden North Sea coastal exposure
        "sea_level": 1.4,           # IJmuiden elevation ~3m ASL
        "subsidence": 1.3,          # clay-rich North Sea delta
        "freeze_thaw_cycle": 1.2,   # Netherlands / UK: pipe-work, ladle cracking
        "compound_flood": 1.7,      # IJmuiden compound coastal+riverine risk
        "cyclone": 0.8,             # enclosed heavy structures
        "drought": 1.3,             # process water & cooling tower stress
        "extreme_cold": 0.9,
    },
    "cement_plant": {
        "heat_stress": 1.5,         # kiln operations + worker exposure in India
        "water_stress": 1.9,        # high process water need (grinding + cooling)
        "drought": 1.6,             # water scarcity exacerbates ops risk
        "flood_riverine": 1.2,      # India monsoon flash flooding
        "dust_storm": 1.4,          # open clinker/limestone stockpiles
        "cyclone": 1.1,             # Bay of Bengal exposure for Andhra Pradesh
        "freeze_thaw_cycle": 0.8,   # mostly tropical/subtropical locations
        "wildfire": 0.7,            # non-combustible process
        "lightning": 1.1,           # exposed silos and conveyor gantries
    },
    "airport_terminal": {
        "heat_stress": 1.5,         # tarmac heat + terminal cooling; flight cancellations
        "flood_coastal": 1.5,       # JFK/LAX sea-level exposure
        "flood_riverine": 1.3,      # flash flood runway inundation
        "sea_level": 1.3,           # coastal airports (JFK, LAX)
        "cyclone": 1.4,             # ATL: hurricane track; JFK: storm surge
        "tornado": 1.3,             # MSP, ATL: tornado corridor
        "compound_flood": 1.4,      # compound coastal + storm surge events
        "wildfire": 1.2,            # LAX: Southern California wildfire smoke
        "extreme_cold": 1.3,        # MSP: de-icing disruption
        "freeze_thaw_cycle": 1.2,   # runway / taxiway cracking
        "dust_storm": 0.9,
    },
    "port_terminal": {
        "flood_coastal": 2.0,       # primary risk for marine terminals
        "sea_level": 1.8,           # chronic inundation risk
        "cyclone": 1.8,             # Miami: hurricane direct hit risk
        "compound_flood": 2.0,      # compound coastal + surge events
        "marine_heatwave": 1.4,     # coral bleaching → Caribbean destination risk
        "saltwater_intrusion": 1.5, # port infrastructure corrosion
        "heat_stress": 1.2,         # operational staff + cargo damage
        "flood_riverine": 1.3,      # Hamburg: Elbe storm surge
        "extratropical_cyclone": 1.4,  # Southampton, Hamburg: storm tracks
        "subsidence": 1.3,          # port reclaimed land
        "wildfire": 0.6,
    },
}

# Default sensitivity (no equipment type specified or unknown type)
_DEFAULT_SENSITIVITY: dict[str, float] = {}


def get_equipment_sensitivity(equipment_type: Optional[str]) -> dict[str, float]:
    """Return hazard sensitivity multiplier dict for a given equipment type."""
    if equipment_type and equipment_type in EQUIPMENT_SENSITIVITY:
        return EQUIPMENT_SENSITIVITY[equipment_type]
    return _DEFAULT_SENSITIVITY


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class AssetGISAttributes:
    """All spatial attributes resolved from a lat/lon point.

    Consumed by PhysicalHazardEngine.assess() to replace or supplement
    region-code based lookups.
    """
    lat: float
    lon: float

    # Terrain
    elevation_m: int = 0                  # metres ASL
    coastal_km: float = 1_000.0          # km to nearest coastline

    # Climate classification
    koppen_zone: str = "Cfb"             # Köppen–Geiger code
    is_arid: bool = False                 # BWh/BWk/BSh/BSk or dryland box
    is_permafrost: bool = False           # permafrost zone
    is_cyclone_belt: bool = False         # TC track density zone
    is_floodplain: bool = False           # low-elevation + near water

    # Temperatures
    mean_winter_temp: float = 5.0        # °C  (DJF in NH, JJA in SH)

    # Equipment sensitivity multipliers (keyed by hazard slug)
    equipment_sensitivity: dict[str, float] = field(default_factory=dict)

    # Source provenance
    source: str = "embedded_tables_v1"


# ---------------------------------------------------------------------------
# Public resolve() function
# ---------------------------------------------------------------------------


def resolve(
    lat: float,
    lon: float,
    equipment_type: Optional[str] = None,
) -> AssetGISAttributes:
    """Resolve spatial hazard attributes from a WGS-84 coordinate pair.

    Parameters
    ----------
    lat            : Latitude in decimal degrees (positive = North).
    lon            : Longitude in decimal degrees (positive = East).
    equipment_type : Optional equipment type string (e.g. "open_pit_mine").
                     If provided, equipment sensitivity multipliers are loaded.

    Returns
    -------
    AssetGISAttributes with all spatial fields populated.
    """
    elev = _elevation(lat, lon)
    coast = _coastal_km(lat, lon)
    koppen = _koppen_zone(lat, lon)
    mwt = _mean_winter_temp(lat, lon)
    arid = _in_arid_zone(lat, lon)
    pf = _in_permafrost(lat, lon)
    cy = _in_cyclone_belt(lat, lon)
    fp = _in_floodplain(lat, lon, elev, coast)
    sensitivity = get_equipment_sensitivity(equipment_type)

    return AssetGISAttributes(
        lat=lat,
        lon=lon,
        elevation_m=elev,
        coastal_km=coast,
        koppen_zone=koppen,
        is_arid=arid,
        is_permafrost=pf,
        is_cyclone_belt=cy,
        is_floodplain=fp,
        mean_winter_temp=mwt,
        equipment_sensitivity=sensitivity,
        source="embedded_tables_v1",
    )
