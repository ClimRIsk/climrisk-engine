"""
analysis/asset_intelligence.py — Coordinate-Precise Physical Asset Intelligence

Transforms a real-world lat/lon coordinate into a physical climate risk profile:
  1. Elevation from SRTM30m (OpenTopoData, free, no API key)
  2. Flood risk classification from FEMA/JRC-style elevation thresholds
  3. Secondary hazard profile (cyclone, wildfire, heat) from latitude band
  4. Remediation action plan keyed to hazard × severity × asset type
  5. Elevation grid (10×10) for 3D terrain rendering in the dashboard
  6. Satellite tile URL (ESRI World Imagery, no API key)

All external calls use httpx (already in [api] dependencies).
OpenTopoData free tier: no key required, max 100 locations per batch, 1 req/sec.
ESRI World Imagery tiles: no key required, attribution required.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)

# ── Remediation lookup: hazard × severity × asset_type addons ─────────────────

_REMEDIATION: dict[str, dict[str, list[str]]] = {
    "flood": {
        "MINIMAL": [
            "Review drainage capacity annually.",
            "Maintain flood insurance at replacement value.",
        ],
        "LOW": [
            "Review and maintain site drainage annually.",
            "Maintain flood insurance at replacement value.",
            "Establish a flood watch protocol with local emergency services.",
        ],
        "MODERATE": [
            "Install permanent bund walls — minimum 0.5 m above 1-in-100yr flood level.",
            "Elevate all electrical panels and critical equipment above projected flood line.",
            "Install sump pumps with backup generator power.",
            "Waterproof ground-floor walls to 1 m height with tanking membrane.",
            "Commission a site-specific flood risk assessment (FRA) aligned to RCP 8.5.",
        ],
        "HIGH": [
            "Relocate critical equipment and high-value inventory to upper floors.",
            "Install flood gates at all entry points — minimum 1.2 m clear height.",
            "Elevate building slab by 0.8–1.5 m if redevelopment is feasible (cost-benefit analysis required).",
            "Deploy real-time flood monitoring with automated SMS/email alerts.",
            "Secure backup generator on an elevated platform above flood line.",
            "Develop and rehearse 72-hour asset protection and evacuation protocol.",
        ],
        "CRITICAL": [
            "URGENT: Operational continuity review — asset is in extreme flood exposure zone.",
            "Evaluate full asset relocation as primary long-term strategy.",
            "Deploy temporary emergency flood barriers (Tiger Dams or equivalent) as interim measure.",
            "Cap insurance immediately at full replacement cost — review exclusions.",
            "Develop 48-hour evacuation and asset-protection protocol and rehearse quarterly.",
            "Engage a licensed flood resilience engineer within 30 days.",
        ],
    },
    "wildfire": {
        "LOW": [
            "Maintain standard fire suppression systems with annual inspection.",
            "Clear vegetation within 10 m perimeter annually.",
        ],
        "MODERATE": [
            "Clear defensible space to 30 m radius — low fuel vegetation only.",
            "Install ember-resistant vents and roof mesh on all openings.",
            "Replace combustible roof cladding with metal or tile.",
            "Maintain on-site water storage for fire suppression — minimum 50,000 L.",
        ],
        "HIGH": [
            "Clear defensible space to 100 m — remove all combustible vegetation.",
            "Install automatic exterior sprinkler system on roof and walls.",
            "Replace all external cladding with non-combustible materials (BAL-FZ rating).",
            "Install ember-proof roof covering and sealed gutter guards.",
            "Join local fire brigade network with a dedicated alert hotline.",
        ],
        "CRITICAL": [
            "URGENT: Immediate vegetation clearance to 100 m perimeter.",
            "Engage a wildfire resilience specialist for a full structural audit.",
            "Install thermal imaging cameras for early smoke and ember detection.",
            "Develop and rehearse site evacuation plan quarterly.",
            "Evaluate asset hardening to AS 3959 Construction in Bushfire Prone Areas.",
        ],
    },
    "heat": {
        "LOW": [
            "Monitor Wet Bulb Globe Temperature (WBGT) for outdoor workers.",
            "Provide shade structures and hydration stations in outdoor work areas.",
        ],
        "MODERATE": [
            "Upgrade HVAC to handle +3–4 °C above original design temperature.",
            "Install cool-roof coating (high-reflectance membrane) — reduces surface temp by up to 30 °C.",
            "Increase thermal mass in the building envelope (insulation upgrade).",
            "Shift peak-load operations to night hours by 2030 as temperatures increase.",
        ],
        "HIGH": [
            "Full HVAC redesign for RCP 8.5 wet-bulb temperatures.",
            "Install building thermal energy storage to decouple cooling from peak grid load.",
            "Retrofit exterior walls with insulation to U-value < 0.3 W/m²K.",
            "Install automated external shading on south and west facades.",
        ],
        "CRITICAL": [
            "Operational risk assessment for outdoor worker safety — potential regulatory exposure.",
            "Commission engineering study for passive cooling redesign.",
            "Evaluate shift to 24-hour operations to avoid peak heat windows.",
            "Install heat-health early warning system linked to BOM/NWS alerts.",
        ],
    },
    "cyclone": {
        "LOW": [
            "Annual structural inspection by a licensed engineer.",
            "Maintain 72-hour emergency supply stocks on-site.",
        ],
        "MODERATE": [
            "Upgrade roof fixings to cyclone rating — N3/C2 minimum per AS 1170.2.",
            "Install impact-resistant glazing on all openings.",
            "Secure all rooftop equipment (HVAC units, solar panels) with rated fixings.",
            "Maintain a 72-hour emergency supply cache and rehearse shutdown procedure.",
        ],
        "HIGH": [
            "Full structural audit to cyclone category 4 standard.",
            "Replace roof cladding with standing-seam steel — minimum 0.7 mm BMT.",
            "Install rated shutters on all openings.",
            "Underground critical utilities, communications, and power feeds.",
        ],
        "CRITICAL": [
            "URGENT: Structural engineering review to category 5 cyclone standard.",
            "Develop Category 5 response and recovery protocol with insurance pre-approval.",
            "Consider relocating high-value equipment off-site during cyclone season.",
            "Install structural health monitoring sensors on roof and columns.",
        ],
    },
}

_ASSET_TYPE_ADDONS: dict[str, list[str]] = {
    "data_centre": [
        "Install N+1 redundancy for all cooling systems.",
        "Physical flood barriers around all electrical risers and UPS rooms.",
        "Dual-path network routing to avoid single points of failure.",
    ],
    "port": [
        "Develop rerouting protocol for supply chain disruption under each hazard scenario.",
        "Corrosion management program for saltwater spray and storm surge exposure.",
    ],
    "logistics": [
        "Develop rerouting protocol for supply chain disruption.",
        "Inventory resilience buffer — maintain 30-day stock of critical SKUs off-site.",
    ],
    "agriculture": [
        "Install crop weather monitoring and automated irrigation backup.",
        "Diversify crop varieties for climate resilience — engage an agronomist.",
    ],
    "hospital": [
        "Maintain full generator backup with 7-day fuel supply.",
        "Critical care wards must remain operational — assess vertical evacuation routes.",
    ],
}


# ── Elevation ─────────────────────────────────────────────────────────────────

def get_elevation(lat: float, lon: float) -> dict:
    """
    Fetch elevation at a single coordinate via OpenTopoData SRTM30m.
    Free, no API key, ~30 m horizontal resolution globally.
    """
    url = f"https://api.opentopodata.org/v1/srtm30m?locations={lat},{lon}"
    try:
        import httpx
        resp = httpx.get(url, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        r = data["results"][0]
        return {
            "elevation_m": r["elevation"],
            "dataset": "SRTM30m (OpenTopoData)",
            "lat": r["location"]["lat"],
            "lon": r["location"]["lng"],
        }
    except Exception as exc:
        logger.warning("OpenTopoData elevation failed for (%s, %s): %s", lat, lon, exc)
        return {"elevation_m": None, "error": str(exc)}


def get_elevation_grid(
    lat: float,
    lon: float,
    radius_km: float = 2.5,
    grid_n: int = 10,
) -> list[float]:
    """
    Fetch an N×N elevation grid centered on (lat, lon) for 3D terrain rendering.

    OpenTopoData allows up to 100 locations per batch — a 10×10 grid fits exactly.
    Returns a flat list of elevation values in row-major order (row 0 = north).
    """
    step_deg = radius_km / 111.0  # 1 degree latitude ≈ 111 km
    step_lon_deg = radius_km / (111.0 * math.cos(math.radians(lat)))

    locations = []
    for i in range(grid_n):
        for j in range(grid_n):
            dlat = (0.5 - i / (grid_n - 1)) * 2 * step_deg       # north → south
            dlon = (j / (grid_n - 1) - 0.5) * 2 * step_lon_deg   # west → east
            locations.append(f"{round(lat + dlat, 6)},{round(lon + dlon, 6)}")

    url = f"https://api.opentopodata.org/v1/srtm30m?locations={'|'.join(locations)}"
    try:
        import httpx
        resp = httpx.get(url, timeout=20.0)
        resp.raise_for_status()
        data = resp.json()
        return [r.get("elevation") or 0.0 for r in data["results"]]
    except Exception as exc:
        logger.warning("Elevation grid fetch failed: %s", exc)
        return [0.0] * (grid_n * grid_n)


# ── Flood risk classification ─────────────────────────────────────────────────

def classify_flood_risk(elevation_m: Optional[float]) -> dict:
    """
    Classify flood exposure from elevation using FEMA / JRC methodology.

    Thresholds:
      < 0 m  → CRITICAL  (below sea level)
      0–2 m  → CRITICAL  (extreme coastal / storm surge exposure)
      2–5 m  → HIGH      (within 1-in-50yr zone for most coastal areas)
      5–10 m → MODERATE  (1-in-100yr zone for low-gradient river catchments)
      10–20m → LOW       (residual fluvial risk; minimal coastal)
      > 20 m → MINIMAL
    """
    if elevation_m is None:
        return {
            "level": "UNKNOWN",
            "score": 0.5,
            "note": "Elevation data unavailable — treating as moderate risk.",
        }
    if elevation_m < 0:
        return {"level": "CRITICAL", "score": 1.0, "elevation_m": elevation_m,
                "note": "Below sea level — extreme permanent inundation risk."}
    if elevation_m < 2:
        return {"level": "CRITICAL", "score": 0.95, "elevation_m": elevation_m,
                "note": "Within 2 m of sea level — extreme coastal/storm surge exposure."}
    if elevation_m < 5:
        return {"level": "HIGH", "score": 0.75, "elevation_m": elevation_m,
                "note": "2–5 m elevation — within 1-in-50yr flood zone for most coastal areas."}
    if elevation_m < 10:
        return {"level": "MODERATE", "score": 0.50, "elevation_m": elevation_m,
                "note": "5–10 m elevation — 1-in-100yr zone for low-gradient river catchments."}
    if elevation_m < 20:
        return {"level": "LOW", "score": 0.25, "elevation_m": elevation_m,
                "note": "10–20 m elevation — residual fluvial risk only; minimal coastal exposure."}
    return {"level": "MINIMAL", "score": 0.05, "elevation_m": elevation_m,
            "note": "Above 20 m — negligible flood risk from sea level rise or storm surge."}


# ── Secondary hazard profiling ────────────────────────────────────────────────

def _secondary_hazards(lat: float, lon: float, elevation_m: Optional[float]) -> list[dict]:
    """
    Latitude-band heuristics for non-flood physical hazards.
    Production replacement: IPCC AR6 Ch12 hazard exposure tables by climate zone.
    """
    abs_lat = abs(lat)
    hazards = []

    # Cyclone: tropical/sub-tropical
    if abs_lat < 8:
        hazards.append({"hazard": "cyclone", "exposure": "HIGH",
                        "note": "Equatorial — low cyclone frequency but high flood and heat."})
    elif abs_lat < 25:
        hazards.append({"hazard": "cyclone", "exposure": "HIGH",
                        "note": "Tropical cyclone belt (Cat 1–5 exposure)."})
    elif abs_lat < 35:
        hazards.append({"hazard": "cyclone", "exposure": "MODERATE",
                        "note": "Sub-tropical — occasional intense cyclones."})

    # Heat stress
    if abs_lat < 20:
        hazards.append({"hazard": "heat", "exposure": "HIGH",
                        "note": "Tropical — chronic extreme heat, WBGT regularly >32 °C."})
    elif abs_lat < 40:
        hazards.append({"hazard": "heat", "exposure": "MODERATE",
                        "note": "Sub-tropical / temperate — increasing heat wave frequency under RCP 8.5."})
    else:
        hazards.append({"hazard": "heat", "exposure": "LOW",
                        "note": "Temperate / boreal — heat stress increasing but currently low baseline."})

    # Wildfire: higher elevation + mid-latitude
    if elevation_m and elevation_m > 100 and 25 < abs_lat < 55:
        hazards.append({"hazard": "wildfire", "exposure": "MODERATE",
                        "note": "Elevated terrain in wildfire-prone latitude band."})
    elif 30 < abs_lat < 55:
        hazards.append({"hazard": "wildfire", "exposure": "LOW",
                        "note": "Temperate latitude — low-to-moderate wildfire risk."})

    return hazards


# ── Satellite tile URL ────────────────────────────────────────────────────────

def _esri_tile_url(lat: float, lon: float, zoom: int = 15) -> dict:
    """
    Convert lat/lon to an ESRI World Imagery tile URL.
    No API key required. Attribution: © Esri, Maxar, Earthstar Geographics.
    """
    n = 2 ** zoom
    x = int((lon + 180) / 360 * n)
    lat_rad = math.radians(lat)
    y = int((1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * n)
    # Clamp to valid range
    y = max(0, min(n - 1, y))
    x = max(0, min(n - 1, x))
    return {
        "url": (
            "https://services.arcgisonline.com/arcgis/rest/services/"
            f"World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
        ),
        "leaflet_url_template": (
            "https://services.arcgisonline.com/arcgis/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        "attribution": "© Esri, Maxar, Earthstar Geographics",
        "tile_x": x,
        "tile_y": y,
        "zoom": zoom,
    }


# ── Master assessment ─────────────────────────────────────────────────────────

def assess_asset(
    lat: float,
    lon: float,
    asset_name: str = "Unknown Asset",
    asset_type: str = "industrial",
    fetch_grid: bool = True,
) -> dict:
    """
    Full coordinate-precise physical risk assessment.

    Steps
    ─────
    1. Fetch elevation from SRTM30m via OpenTopoData
    2. Classify flood risk from elevation thresholds
    3. Profile secondary hazards (cyclone, heat, wildfire) from latitude band
    4. Build remediation action plan (primary + secondary hazards)
    5. Optionally fetch 10×10 elevation grid for 3D terrain rendering
    6. Compute ESRI satellite tile URL for the coordinate

    Returns
    ───────
    dict with keys: asset_name, asset_type, lat, lon, elevation, flood_risk,
    secondary_hazards, remediation, risk_signal, satellite_tile,
    elevation_grid (if fetch_grid=True)
    """
    out: dict = {
        "asset_name": asset_name,
        "asset_type": asset_type,
        "lat": lat,
        "lon": lon,
    }

    # 1. Elevation
    elev_data = get_elevation(lat, lon)
    out["elevation"] = elev_data
    elevation_m = elev_data.get("elevation_m")

    # 2. Flood risk
    flood = classify_flood_risk(elevation_m)
    out["flood_risk"] = flood
    primary_severity = flood["level"]

    # 3. Secondary hazards
    secondary = _secondary_hazards(lat, lon, elevation_m)
    out["secondary_hazards"] = secondary

    # 4. Remediation plan
    flood_actions = _REMEDIATION["flood"].get(primary_severity, [])
    secondary_actions: list[str] = []
    for h in secondary[:2]:  # Top 2 secondary hazards
        hazard_key = h["hazard"]
        sev = h["exposure"]
        secondary_actions.extend(_REMEDIATION.get(hazard_key, {}).get(sev, [])[:2])

    # Asset-type specific additions
    type_addons = _ASSET_TYPE_ADDONS.get(asset_type, [])[:2]

    out["remediation"] = {
        "flood": flood_actions,
        "secondary": secondary_actions,
        "asset_specific": type_addons,
        "total_actions": len(flood_actions) + len(secondary_actions) + len(type_addons),
    }

    # 5. Elevation grid for 3D rendering
    if fetch_grid:
        out["elevation_grid"] = get_elevation_grid(lat, lon)

    # 6. Satellite tile
    out["satellite_tile"] = _esri_tile_url(lat, lon, zoom=15)

    # 7. Overall risk signal (driven by flood level — most data-backed)
    _signal_map = {
        "CRITICAL": "CRITICAL",
        "HIGH": "HIGH",
        "MODERATE": "MODERATE",
        "LOW": "LOW",
        "MINIMAL": "LOW",
        "UNKNOWN": "MODERATE",
    }
    out["risk_signal"] = _signal_map.get(primary_severity, "MODERATE")

    # 8. Summary
    out["summary"] = (
        f"{asset_name} sits at {elevation_m:.1f} m elevation. "
        f"Flood exposure: {primary_severity}. "
        f"{flood.get('note', '')} "
        f"{len(out['remediation']['flood'])} flood mitigation actions recommended."
    ) if elevation_m is not None else (
        f"{asset_name} — elevation unavailable. Risk treated as MODERATE pending survey."
    )

    return out
