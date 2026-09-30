"""
Biodiversity & Nature Risk Module (TNFD LEAP Framework).

Implements the Taskforce on Nature-related Financial Disclosures (TNFD) LEAP
approach (Locate, Evaluate, Assess, Prepare) for nature-related financial risk
assessment.

Framework coverage
------------------
L — Locate: protected areas mapped in OpenStreetMap (Overpass API, ODbL) containing the site or within
    5 / 25 km, with IUCN category where tagged. KBAs are not included (licensed data).
E — Evaluate: sector-level dependencies on and impacts on ecosystem services
    (ENCORE database mapping).
A — Assess: nature risk score combining location sensitivity and sector profile.
P — Prepare: disclosure flags for TNFD recommended disclosures.

Ecosystem services assessed (ENCORE taxonomy)
---------------------------------------------
Provisioning: freshwater, biomass
Regulating: climate regulation, flood/storm protection, water purification,
            pollination, disease control, pest control, erosion control
Cultural:    not scored quantitatively (requires site-specific assessment)

Sector dependency mapping (abbreviated from ENCORE v1.1)
Each sector has:
  - dependency_score (0–1): how much the business model relies on ecosystem services
  - impact_driver list: key ways the sector affects nature

References
----------
TNFD Framework v1.0 (September 2023). https://tnfd.global/publication/framework/
ENCORE (Exploring Natural Capital Opportunities, Risks and Exposure) v1.1.
  https://encore.naturalcapital.finance
WWF / WRI – Biodiversity Risk Filter.
IUCN Red List spatial data. https://www.iucnredlist.org/resources/spatial-data-download
KBA Partnership. https://www.keybiodiversityareas.org
IPBES Global Assessment (2019).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── Offline fallback only: coarse bounding boxes for ~20 major areas ─────────
# Used when OpenStreetMap / Overpass cannot be reached (see _osm_protected).

_PROTECTED_AREAS: list[dict] = [
    # Format: {name, lat_min, lat_max, lon_min, lon_max, category, iucn_category}
    # Africa
    {"name": "Serengeti NP",   "lat_min": -3.5,  "lat_max": -1.0,  "lon_min": 33.5, "lon_max": 35.5, "category": "national_park",  "iucn": "II"},
    {"name": "Okavango Delta", "lat_min": -20.0, "lat_max": -18.0, "lon_min": 22.0, "lon_max": 24.0, "category": "ramsar_wetland",  "iucn": "III"},
    {"name": "Congo Basin FR",  "lat_min": -3.0,  "lat_max":  2.0,  "lon_min": 15.0, "lon_max": 26.0, "category": "forest_reserve",  "iucn": "VI"},
    # South / Southeast Asia
    {"name": "Sundarbans",     "lat_min": 21.5,  "lat_max": 22.8,  "lon_min": 88.0, "lon_max": 90.0, "category": "ramsar_wetland",  "iucn": "II"},
    {"name": "Mekong Delta",   "lat_min":  9.0,  "lat_max": 11.5,  "lon_min":104.5, "lon_max":106.5, "category": "wetland",         "iucn": "IV"},
    {"name": "Borneo Rainforest","lat_min": 0.0,  "lat_max":  7.0,  "lon_min":108.0, "lon_max":119.0, "category": "tropical_forest", "iucn": "II"},
    # Americas
    {"name": "Amazon Basin",   "lat_min":-15.0,  "lat_max":  5.0,  "lon_min": -74.0,"lon_max": -44.0,"category": "tropical_forest", "iucn": "II"},
    {"name": "Pantanal",       "lat_min":-21.0,  "lat_max": -15.0, "lon_min": -58.0,"lon_max": -52.0,"category": "ramsar_wetland",  "iucn": "III"},
    {"name": "Everglades NP",  "lat_min": 24.5,  "lat_max": 26.0,  "lon_min": -81.5,"lon_max": -80.0,"category": "national_park",   "iucn": "II"},
    {"name": "Pacific Salmon Rivers","lat_min":54.0,"lat_max":60.0,"lon_min":-160.0,"lon_max":-130.0,"category":"freshwater_kba","iucn":"IV"},
    # Europe
    {"name": "Danube Delta",   "lat_min": 44.5,  "lat_max": 45.5,  "lon_min": 28.5, "lon_max": 30.0, "category": "ramsar_wetland",  "iucn": "II"},
    {"name": "Doñana NP",      "lat_min": 36.8,  "lat_max": 37.3,  "lon_min": -7.0, "lon_max": -6.0, "category": "national_park",   "iucn": "II"},
    # Oceania
    {"name": "Great Barrier Reef","lat_min":-25.0,"lat_max":-10.0,"lon_min":143.0,"lon_max":155.0,"category":"marine_protected","iucn":"II"},
    {"name": "Daintree Rainforest","lat_min":-16.5,"lat_max":-15.5,"lon_min":145.0,"lon_max":145.8,"category":"tropical_forest","iucn":"II"},
]


# ── ENCORE sector dependency and impact profile (abbreviated) ─────────────────
# Source: ENCORE v1.1 — https://encore.naturalcapital.finance

@dataclass(frozen=True)
class SectorNatureProfile:
    dependency_score: float       # 0–1: how much operations depend on ecosystem services
    key_dependencies: list[str]   # ecosystem services the sector relies on
    impact_drivers: list[str]     # main impact pathways on nature
    sbtn_sector_flag: bool        # Science Based Targets for Nature high-priority sector


_SECTOR_PROFILES: dict[str, SectorNatureProfile] = {
    "agriculture": SectorNatureProfile(
        dependency_score=0.90,
        key_dependencies=["pollination", "freshwater", "soil_formation", "climate_regulation"],
        impact_drivers=["land_use_change", "water_abstraction", "pollution", "climate_change"],
        sbtn_sector_flag=True,
    ),
    "food_beverage": SectorNatureProfile(
        dependency_score=0.75,
        key_dependencies=["freshwater", "pollination", "soil_formation"],
        impact_drivers=["land_use_change", "water_abstraction", "pollution"],
        sbtn_sector_flag=True,
    ),
    "oil_gas": SectorNatureProfile(
        dependency_score=0.30,
        key_dependencies=["freshwater"],
        impact_drivers=["land_use_change", "pollution", "climate_change", "physical_disturbance"],
        sbtn_sector_flag=False,
    ),
    "mining": SectorNatureProfile(
        dependency_score=0.35,
        key_dependencies=["freshwater"],
        impact_drivers=["land_use_change", "water_abstraction", "pollution", "physical_disturbance"],
        sbtn_sector_flag=True,
    ),
    "metals_mining": SectorNatureProfile(
        dependency_score=0.35,
        key_dependencies=["freshwater"],
        impact_drivers=["land_use_change", "water_abstraction", "pollution", "physical_disturbance"],
        sbtn_sector_flag=True,
    ),
    "chemicals": SectorNatureProfile(
        dependency_score=0.40,
        key_dependencies=["freshwater", "biomass"],
        impact_drivers=["pollution", "water_abstraction", "land_use_change"],
        sbtn_sector_flag=False,
    ),
    "utilities": SectorNatureProfile(
        dependency_score=0.60,
        key_dependencies=["freshwater", "climate_regulation"],
        impact_drivers=["land_use_change", "water_abstraction", "pollution"],
        sbtn_sector_flag=False,
    ),
    "energy": SectorNatureProfile(
        dependency_score=0.45,
        key_dependencies=["freshwater", "climate_regulation"],
        impact_drivers=["land_use_change", "pollution", "climate_change"],
        sbtn_sector_flag=False,
    ),
    "real_estate": SectorNatureProfile(
        dependency_score=0.55,
        key_dependencies=["flood_storm_protection", "climate_regulation", "freshwater"],
        impact_drivers=["land_use_change", "physical_disturbance"],
        sbtn_sector_flag=False,
    ),
    "construction": SectorNatureProfile(
        dependency_score=0.45,
        key_dependencies=["freshwater", "biomass"],
        impact_drivers=["land_use_change", "physical_disturbance", "water_abstraction"],
        sbtn_sector_flag=False,
    ),
    "transport": SectorNatureProfile(
        dependency_score=0.30,
        key_dependencies=["climate_regulation"],
        impact_drivers=["land_use_change", "pollution", "physical_disturbance"],
        sbtn_sector_flag=False,
    ),
    "industrials": SectorNatureProfile(
        dependency_score=0.35,
        key_dependencies=["freshwater", "biomass"],
        impact_drivers=["pollution", "water_abstraction", "land_use_change"],
        sbtn_sector_flag=False,
    ),
    "financials": SectorNatureProfile(
        dependency_score=0.15,
        key_dependencies=[],
        impact_drivers=["financed_impacts_via_portfolio"],
        sbtn_sector_flag=False,
    ),
    "technology": SectorNatureProfile(
        dependency_score=0.20,
        key_dependencies=["freshwater"],   # data center cooling
        impact_drivers=["resource_extraction", "e_waste"],
        sbtn_sector_flag=False,
    ),
    "healthcare": SectorNatureProfile(
        dependency_score=0.25,
        key_dependencies=["biomass"],       # pharmaceuticals from natural compounds
        impact_drivers=["pollution"],
        sbtn_sector_flag=False,
    ),
    "consumer": SectorNatureProfile(
        dependency_score=0.50,
        key_dependencies=["biomass", "freshwater", "pollination"],
        impact_drivers=["land_use_change", "pollution"],
        sbtn_sector_flag=False,
    ),
    "automotive": SectorNatureProfile(
        dependency_score=0.30,
        key_dependencies=["freshwater"],
        impact_drivers=["pollution", "resource_extraction"],
        sbtn_sector_flag=False,
    ),
    "default": SectorNatureProfile(
        dependency_score=0.30,
        key_dependencies=["freshwater"],
        impact_drivers=["pollution"],
        sbtn_sector_flag=False,
    ),
}


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class BiodiversityRiskResult:
    company_id: str
    company_name: str
    sector_key: str
    asset_lat: float
    asset_lon: float
    # Location layer (LEAP — Locate)
    nearby_protected_areas: list[dict]   # list of {name, category, iucn, distance_km_approx}
    kba_proximity: str                   # "within" | "adjacent" | "none"
    # Dependency layer (LEAP — Evaluate)
    sector_dependency_score: float
    key_dependencies: list[str]
    key_impact_drivers: list[str]
    # Risk assessment (LEAP — Assess)
    location_sensitivity_score: float    # 0–1
    combined_nature_risk_score: float    # 0–1
    risk_tier: str                       # "critical" | "high" | "medium" | "low"
    sbtn_priority: bool                  # Science Based Targets for Nature
    # Disclosure (LEAP — Prepare)
    tnfd_disclosure_recommended: bool
    disclosure_metrics: list[str]        # recommended TNFD metrics to report
    notes: list[str] = field(default_factory=list)
    location_source: str = "osm"             # "osm" | "fallback"


# ── OpenStreetMap protected areas (live, ODbL) ────────────────────────────────
# Protected areas mapped in OpenStreetMap (boundary=protected_area / national_park, leisure=nature_reserve),
# queried through the Overpass API. Many national agencies' boundaries are imported into OSM, but OSM is not
# the WDPA and KBAs are not in OSM: WDPA / KBA data require a licence from UNEP-WCMC / BirdLife.
_OVERPASS = "https://overpass-api.de/api/interpreter"
_OSM_CACHE = __import__("pathlib").Path.home() / ".climrisk" / "cache" / "osm_protected"
_STRICT = {"1a", "1b", "1", "2", "3"}
_IUCN = {"1a": "Ia", "1b": "Ib", "1": "I", "2": "II", "3": "III", "4": "IV", "5": "V", "6": "VI"}


def _osm_protected(lat: float, lon: float, timeout: int = 45) -> Optional[dict]:
    """{'within': [...], 'within_5km': [...], 'within_25km': [...]} or None when Overpass is unreachable."""
    import json
    import urllib.parse
    import urllib.request
    key = f"{lat:.3f}_{lon:.3f}"
    f = _OSM_CACHE / f"{key}.json"
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:
            pass
    sel = lambda r: (f'(nwr(around:{r},{lat},{lon})["boundary"~"^(protected_area|national_park)$"];'
                     f'nwr(around:{r},{lat},{lon})["leisure"="nature_reserve"];);out tags;')
    q = (f'[out:json][timeout:{timeout - 5}];is_in({lat},{lon})->.a;'
         '(area.a["boundary"~"^(protected_area|national_park)$"];area.a["leisure"="nature_reserve"];);out tags;'
         'make marker name="near5";out;' + sel(5000) + 'make marker name="near25";out;' + sel(25000))
    try:
        req = urllib.request.Request(_OVERPASS, data=urllib.parse.urlencode({"data": q}).encode(),
                                     headers={"User-Agent": "ClimRisk-engine/1.0 (+https://climrisk.io)"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            els = json.loads(r.read().decode())["elements"]
    except Exception:
        return None
    out = {"within": [], "within_5km": [], "within_25km": []}
    bucket, seen = "within", {}
    for e in els:
        if e["type"] == "marker":
            bucket = {"near5": "within_5km", "near25": "within_25km"}[e["tags"]["name"]]
            continue
        t = e.get("tags", {})
        name = t.get("name:en") or t.get("name")
        if not name:
            continue
        osm_id = e["id"] - 3600000000 if e["type"] == "area" else e["id"]
        pc = (t.get("protect_class") or "").lower()
        rec = {"name": name, "designation": t.get("protection_title") or t.get("designation") or
               ("National park" if t.get("boundary") == "national_park" else "Nature reserve"
                if t.get("leisure") == "nature_reserve" else "Protected area"),
               "iucn": _IUCN.get(pc), "strict": pc in _STRICT or t.get("boundary") == "national_park",
               "osm": f"{'relation' if e['type'] in ('area', 'relation') else e['type']}/{osm_id}"}
        k = (name, rec["osm"])
        if k in seen:
            continue
        seen[k] = bucket
        out[bucket].append(rec)
    try:
        _OSM_CACHE.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(out))
    except Exception:
        pass
    return out


def compute_biodiversity_risk(
    company_id: str,
    company_name: str,
    sector_key: str,
    asset_lat: float,
    asset_lon: float,
    revenue_usd_m: float = 100.0,
    use_live: bool = True,
) -> BiodiversityRiskResult:
    """
    Compute nature-related risk for an asset using the TNFD LEAP approach.

    Parameters
    ----------
    company_id, company_name : str
    sector_key : str
        CRI sector key.
    asset_lat, asset_lon : float
        Asset coordinates (WGS84).
    revenue_usd_m : float
        Annual revenue — used for materiality context (not scored directly).

    Returns
    -------
    BiodiversityRiskResult
    """
    profile = _SECTOR_PROFILES.get(sector_key, _SECTOR_PROFILES["default"])

    # ── L: Locate — protected area proximity ─────────────────────────────────
    nearby: list[dict] = []
    kba_within = False
    osm = _osm_protected(asset_lat, asset_lon) if use_live else None
    location_source = "osm"
    if osm is not None:
        for band, lab in (("within", "within"), ("within_5km", "within 5 km"), ("within_25km", "within 25 km")):
            for pa in osm[band]:
                nearby.append(dict(pa, overlap=lab))
        within = [p for p in nearby if p["overlap"] == "within"]
        near5 = [p for p in nearby if p["overlap"] == "within 5 km"]
        near25 = [p for p in nearby if p["overlap"] == "within 25 km"]
        kba_within = bool(within)
        kba_proximity = "within" if within else ("adjacent" if near5 else ("nearby" if near25 else "none"))
        # ── E: Evaluate — location sensitivity score ─────────────────────────
        if within:
            loc_score = 1.0 if any(p["strict"] for p in within) else 0.8
        elif near5:
            loc_score = 0.65 if any(p["strict"] for p in near5) else 0.5
        elif near25:
            loc_score = 0.35 if any(p["strict"] for p in near25) else 0.25
        else:
            loc_score = 0.15 if profile.dependency_score > 0.5 else 0.05
        # keep the response compact: all 'within' and 5 km areas, the strictest 15 of the 25 km ring
        near25_sorted = sorted(near25, key=lambda p: (not p["strict"], p["name"]))
        nearby = within + near5 + near25_sorted[:15]
        n25_total = len(near25)
    else:
        location_source = "fallback"
        n25_total = 0
        for pa in _PROTECTED_AREAS:
            if (pa["lat_min"] <= asset_lat <= pa["lat_max"] and pa["lon_min"] <= asset_lon <= pa["lon_max"]):
                nearby.append({"name": pa["name"], "category": pa["category"], "iucn": pa["iucn"], "overlap": "within"})
                kba_within = True
            elif (pa["lat_min"] - 1.0 <= asset_lat <= pa["lat_max"] + 1.0 and
                  pa["lon_min"] - 1.0 <= asset_lon <= pa["lon_max"] + 1.0):
                nearby.append({"name": pa["name"], "category": pa["category"], "iucn": pa["iucn"], "overlap": "adjacent"})
        kba_proximity = "within" if kba_within else ("adjacent" if nearby else "none")
        if kba_proximity == "within":
            loc_score = 1.0
        elif kba_proximity == "adjacent":
            loc_score = 0.55
        else:
            loc_score = 0.15 if profile.dependency_score > 0.5 else 0.05

    # ── A: Assess — combined score ────────────────────────────────────────────
    combined = round(0.5 * loc_score + 0.5 * profile.dependency_score, 3)

    if combined >= 0.65:
        tier = "critical"
    elif combined >= 0.40:
        tier = "high"
    elif combined >= 0.20:
        tier = "medium"
    else:
        tier = "low"

    # ── P: Prepare — disclosure ───────────────────────────────────────────────
    tnfd_flag = tier in ("critical", "high")

    metrics: list[str] = []
    if tnfd_flag:
        metrics += [
            "Land use and land-use change (ha affected)",
            "Water withdrawal in water-stressed areas (m³)",
            "Species at risk in operational area (IUCN Red List count)",
        ]
    if profile.sbtn_sector_flag:
        metrics.append("Science Based Targets for Nature commitment / target year")
    if "freshwater" in profile.key_dependencies:
        metrics.append("Water consumption intensity (m³ / unit revenue)")
    if kba_within:
        metrics.append("Operations within or adjacent to KBAs / protected areas (% revenue)")

    notes: list[str] = []
    if kba_within:
        names = [p["name"] for p in nearby if p["overlap"] == "within"]
        notes.append(f"Asset located WITHIN: {', '.join(names)}")
    if profile.sbtn_sector_flag:
        notes.append(f"Sector '{sector_key}' is an SBTn high-priority sector — nature targets expected by 2025-2030")
    if location_source == "osm":
        if not nearby:
            notes.append("No protected area mapped in OpenStreetMap within 25 km — location sensitivity from sector profile only")
        elif n25_total > 15:
            notes.append(f"{n25_total} protected areas mapped within 25 km; the strictest 15 are listed")
        notes.append("Protected areas from OpenStreetMap (© OpenStreetMap contributors, ODbL). OSM is not the WDPA; "
                     "Key Biodiversity Areas are not included (KBA/WDPA data require a UNEP-WCMC / BirdLife licence).")
    else:
        notes.append("OpenStreetMap unreachable — coarse built-in bounding boxes for ~20 major areas used instead; "
                     "re-run online for a site-level answer.")

    return BiodiversityRiskResult(
        company_id=company_id,
        company_name=company_name,
        sector_key=sector_key,
        asset_lat=asset_lat,
        asset_lon=asset_lon,
        nearby_protected_areas=nearby,
        kba_proximity=kba_proximity,
        sector_dependency_score=profile.dependency_score,
        key_dependencies=list(profile.key_dependencies),
        key_impact_drivers=list(profile.impact_drivers),
        location_sensitivity_score=round(loc_score, 3),
        combined_nature_risk_score=combined,
        risk_tier=tier,
        sbtn_priority=profile.sbtn_sector_flag,
        tnfd_disclosure_recommended=tnfd_flag,
        disclosure_metrics=metrics,
        notes=notes,
        location_source=location_source,
    )


def biodiversity_to_dict(result: BiodiversityRiskResult) -> dict:
    """Serialise biodiversity result to JSON-compatible dict."""
    return {
        "company_id":   result.company_id,
        "company_name": result.company_name,
        "sector_key":   result.sector_key,
        "location": {
            "lat":              result.asset_lat,
            "lon":              result.asset_lon,
            "protected_area_proximity": result.kba_proximity,
            "source":           result.location_source,
            "nearby_protected_areas": result.nearby_protected_areas,
        },
        "nature_dependency": {
            "sector_dependency_score": result.sector_dependency_score,
            "key_dependencies":        result.key_dependencies,
            "key_impact_drivers":      result.key_impact_drivers,
        },
        "risk_assessment": {
            "location_sensitivity_score": result.location_sensitivity_score,
            "combined_nature_risk_score": result.combined_nature_risk_score,
            "risk_tier":                  result.risk_tier,
            "sbtn_priority_sector":       result.sbtn_priority,
        },
        "disclosure": {
            "tnfd_disclosure_recommended": result.tnfd_disclosure_recommended,
            "recommended_metrics":         result.disclosure_metrics,
        },
        "notes": result.notes,
        "methodology": (
            "TNFD LEAP (Locate, Evaluate, Assess, Prepare). "
            + ("Location: protected areas mapped in OpenStreetMap (Overpass API, ODbL) — containing the site, "
               "within 5 km and within 25 km; IUCN category from protect_class where tagged. KBAs not included. "
               if result.location_source == "osm" else
               "Location: offline fallback — coarse bounding boxes for ~20 major areas. ") +
            "Sector dependency from ENCORE v1.1. "
            "Combined score = 0.5 × location_sensitivity + 0.5 × sector_dependency."
        ),
    }
