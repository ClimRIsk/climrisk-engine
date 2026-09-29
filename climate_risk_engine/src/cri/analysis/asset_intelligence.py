"""
analysis/asset_intelligence.py — Evidence-based physical asset intelligence.

Fuses three independent observation layers for a real-world coordinate and
turns them into findings that each carry their own measured evidence:

  Terrain    TerrainEngine — AWS terrain tiles (~10 m), pysheds D8 flow
             accumulation, slope, TPI terrain position
             → flood routing, debris flow, landslide
  Imagery    Sentinel-2 L2A pixels (AWS Open Data) — NDVI, MNDWI, water
             frequency over up to 8 dates in ~2 years
             → observed inundation, distance to water, vegetation state
             Optional Prithvi-EO-2.0 flood segmentation (HF endpoint)
  Land cover ESA WorldCover 2021 (10 m, AWS Open Data)
             → built-up / impervious share, flammable vegetation, wetlands

Each finding = {hazard, severity, score, mechanism, evidence[], actions[]}.
Evidence items name the metric, value and source dataset, so every statement
in the dashboard or LLM narrative can be traced back to an observation.

Scoring is transparent and additive (points → severity) and documented inline.
It is a screening model: it prioritises assets for site survey and does not
replace a hydraulic flood model or a geotechnical investigation.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

log = logging.getLogger(__name__)

SEVERITY_ORDER = ["NEGLIGIBLE", "LOW", "MODERATE", "HIGH", "CRITICAL"]
_TERRAIN_TO_SEV = {"NEGLIGIBLE": "NEGLIGIBLE", "LOW": "LOW", "MEDIUM": "MODERATE", "HIGH": "HIGH"}


def _sev_from_points(pts: float) -> str:
    if pts >= 6:
        return "CRITICAL"
    if pts >= 4:
        return "HIGH"
    if pts >= 2:
        return "MODERATE"
    if pts >= 1:
        return "LOW"
    return "NEGLIGIBLE"


def _ev(metric: str, value, source: str, unit: str = "") -> dict:
    return {"metric": metric, "value": value, "unit": unit, "source": source}


def _pct(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(100 * x, 1)


# ── Remediation library, keyed by mechanism ──────────────────────────────────

_ACTIONS = {
    "fluvial": [
        "Commission a site flood risk assessment with hydraulic modelling of the adjacent channel (1-in-100yr + climate uplift).",
        "Install demountable flood barriers or permanent bund walls at openings facing the channel.",
        "Raise electrical switchgear, transformers and control rooms above the modelled design flood level.",
        "Fit non-return valves on drainage outfalls to stop backflow when the channel is in flood.",
    ],
    "observed_inundation": [
        "Satellite record shows standing water near the site: survey finished-floor levels against observed flood extent.",
        "Relocate high-value stock and critical spares off the ground floor or to higher ground on site.",
        "Set up automated water-level monitoring with alerts tied to a written shutdown procedure.",
    ],
    "pluvial": [
        "High impervious cover drives surface-water flooding: audit on-site drainage capacity against 1-in-30yr rainfall.",
        "Retrofit sustainable drainage (permeable paving, attenuation tanks, green roofs) to cut peak runoff.",
        "Keep inlets and gullies clear before the wet season; install trash screens on culverts.",
    ],
    "low_lying_water": [
        "Asset sits low relative to a permanent water body: check exposure to storm surge and high-water events.",
        "Elevate or waterproof ground-floor services (dry-proofing to ~1 m).",
    ],
    "debris_flow": [
        "Map upslope source areas and install debris nets or check dams in contributing channels.",
        "Keep the channel between source zone and site clear; design a deflection berm if the path is confirmed.",
        "Install rainfall-triggered early warning (threshold alerts) with an evacuation route away from the channel.",
    ],
    "landslide": [
        "Commission a geotechnical slope stability assessment (boreholes, inclinometers) for the site slope.",
        "Improve slope drainage (horizontal drains, interceptor ditches) — water is the main failure trigger.",
        "Consider retaining structures or soil nailing on over-steep cut slopes.",
    ],
    "wildfire": [
        "Create and maintain defensible space: 30 m low-fuel zone, 100 m reduced-fuel zone around structures.",
        "Upgrade to non-combustible cladding and ember-resistant vents, gutter guards and roof mesh.",
        "Maintain dedicated firefighting water storage and access for fire appliances.",
        "Join regional fire-danger alerting and pre-plan shutdown on extreme fire-danger days.",
    ],
    "cyclone_wind": [
        "Design or retrofit the building envelope to the local cyclone wind code for the modelled 1-in-100-year "
        "wind (roof tie-downs, rated cladding and fixings, impact-rated openings).",
        "Secure or remove rooftop plant and loose external items; inspect roof fixings before each season.",
        "Prepare a cyclone season plan: 72-hour shutdown procedure, backup power and fuel, staff shelter.",
        "Review windstorm insurance limits against the worst historical storm at this site.",
    ],
    "urban_heat": [
        "Apply high-albedo cool-roof coatings to large roof areas (cuts roof surface temperatures significantly).",
        "Increase shade and tree canopy on site; low canopy cover amplifies local heat.",
        "Re-rate HVAC and cooling for higher design temperatures; add heat-stress protocols for outdoor staff.",
    ],
}

_ASSET_ADDONS = {
    "data_centre": ["Verify N+1 cooling capacity at elevated ambient temperatures and flood-protect UPS/battery rooms."],
    "port": ["Review quay and yard drainage against combined storm-surge and rainfall events."],
    "logistics": ["Pre-agree alternative routing and a secondary site for disruption to access roads."],
    "agriculture": ["Diversify water sources and irrigation backup; monitor NDVI seasonally for crop stress."],
    "hospital": ["Confirm 72h+ generator fuel and that critical wards are above flood level."],
}


# ── Hazard evaluators ────────────────────────────────────────────────────────

def _flood(t: dict, s: dict, lc: dict, sar: Optional[dict] = None,
           belev: Optional[dict] = None) -> dict:
    sar = sar or {}
    pts, ev, mech = 0.0, [], []
    TER, S2, WC = "Terrain (AWS 10 m DEM, pysheds D8)", "Sentinel-2 L2A", "ESA WorldCover 2021"

    tr = t.get("flood_exposure")
    if tr:
        # evidence only — terrain routing showed no skill against FEMA claims
        ev.append(_ev("Terrain flood routing rating", tr, TER))
        ev.append(_ev("Terrain position", t.get("terrain_position"), TER))
        ev.append(_ev("Upstream drainage area", t.get("upslope_area_km2"), TER, "km²"))
        pass

    # ── Observed inundation: optical (Sentinel-2) and radar (Sentinel-1) ──────
    # Scored per sensor, then the stronger one counts (no double counting).
    def _obs_pts(at_asset, near500):
        p = 0.0
        if at_asset is not None:
            p += 4 if at_asset >= 0.3 else 3 if at_asset >= 0.1 else 0
        if near500 is not None:
            p += 2 if near500 >= 0.05 else 1 if near500 >= 0.01 else 0
        return p

    opt_pts = 0.0
    if s.get("temporal_scenes_used"):
        wa, e500 = s.get("water_freq_at_asset"), s.get("ephemeral_water_within_500m")
        ev.append(_ev("Optical water history: clear dates / months covered",
                      f"{s['temporal_scenes_used']} / {len(s.get('temporal_months_covered') or [])} of 12",
                      f"{S2} ({s.get('temporal_date_range')})"))
        if wa is not None:
            ev.append(_ev("Water seen at asset pixel (optical)", _pct(wa), S2, "% of clear dates"))
        if e500 is not None:
            ev.append(_ev("Intermittently wet land within 500 m (optical)", _pct(e500), S2, "%"))
        opt_pts = _obs_pts(wa, e500)

    sar_pts = 0.0
    S1 = "Sentinel-1 radar (Planetary Computer RTC)"
    if sar.get("available"):
        ev.append(_ev("Radar water history: dates / months covered",
                      f"{sar['scenes_used']} / {len(sar.get('months_covered') or [])} of 12",
                      f"{S1} ({sar.get('date_range')})"))
        ev.append(_ev("Water seen at asset pixel (radar)", _pct(sar.get("water_freq_at_asset")), S1, "% of dates"))
        ev.append(_ev("Land under water on ≥1 date within 500 m (radar)", _pct(sar.get("ever_wet_within_500m")), S1, "%"))
        if (sar.get("peak_wet_within_500m") or 0) >= 0.01:
            ev.append(_ev("Largest flood extent within 500 m (radar)",
                          f"{_pct(sar['peak_wet_within_500m'])}% on {sar.get('peak_wet_date')}", S1))
        high_alt = (t.get("elevation_m") or 0) > 2500
        if high_alt:
            ev.append(_ev("Radar flood signal discounted", "site > 2,500 m: wet snow mimics water", S1))
        else:
            at_asset = sar.get("water_freq_at_asset")
            # Pin dropped on open water (optical sees water most clear dates) with no radar
            # flooding of the land around it: radar misses wind-roughened water on some dates,
            # so its at-pixel frequency is the sea/lake, not land flooding (Kwinana pin in
            # Cockburn Sound). A river char that goes under (Jamuna) keeps its signal.
            if (s.get("water_freq_at_asset") or 0) >= 0.5 and (sar.get("intermittent_water_within_500m") or 0) < 0.01:
                at_asset = None
                ev.append(_ev("Radar at-pixel signal ignored", "pin is on open water — move it onto the site", S1))
            sar_pts = _obs_pts(at_asset, sar.get("intermittent_water_within_500m"))

    # Only radar-observed inundation scores. Optical "water" is evidence only: dark roofs,
    # shadows (Johannesburg) and snow/slag (Norilsk) produced false water in the world test.
    if sar_pts > 0:
        pts += sar_pts; mech.append("observed_inundation")

    nw = s.get("nearest_water_m")
    if nw is not None:
        ev.append(_ev("Distance to nearest open water", nw, f"{S2} ({s.get('water_method')})", "m"))
        # evidence only (MNDWI water can be dark roofs / asphalt)

    # ── Validated component: elevation of built-up land within 1 km ──────────
    # 180 census tracts in 3 US coastal-plain counties vs 95k FEMA NFIP claims
    # (scripts/backtest_flood_area.py): built-up land below 3 m → ~2.7× median
    # claim rate, 67 % of such tracts in the top-claim fifth. 3–40 m: little signal.
    if belev and belev.get("median_elev_m") is not None:
        me = belev["median_elev_m"]
        ev.append(_ev("Median elevation of built-up land within 1 km (validated signal)", me,
                      f"AWS 10 m DEM × ESA WorldCover ({belev.get('basis')})", "m"))
        if me < -2.0:
            # Below sea level: dike-protected polder (Rotterdam) or inland basin (Caspian) —
            # outside the validated range; flag for survey rather than score as coastal.
            pts += 2; mech.append("low_lying_water")
            ev.append(_ev("Below sea level", "protected polder or inland basin — needs site survey",
                          "AWS 10 m DEM"))
        elif me < 3.0:
            pts += 4; mech.append("low_lying_water"); mech.append("fluvial")
        elif me < 6.0:
            pts += 2; mech.append("low_lying_water")

    elev = t.get("elevation_m")
    if elev is not None and (s.get("permanent_water_fraction") or 0) > 0.02 and elev < 10:
        ev.append(_ev("Site elevation next to permanent water", elev, TER, "m"))

    bu = lc.get("built_up_within_250m")
    if bu is not None:
        ev.append(_ev("Built-up / impervious cover within 250 m", _pct(bu), WC, "%"))
        if bu > 0.6:
            mech.append("pluvial")
    wl = lc.get("wetland_within_1km")
    if wl and wl > 0.1:
        ev.append(_ev("Wetland / mangrove within 1 km", _pct(wl), WC, "%"))

    caveats = [
        "Validation status (FEMA flood-insurance claims, 3 US coastal-plain counties, 180 census tracts): "
        "severity comes from the elevation of built-up land (held-out AUC 0.64–0.94) plus radar-observed "
        "flooding; terrain routing, drainage height, optical water and distance to water did not separate "
        "high- from low-claim areas and are shown as supporting evidence only (no points). Not yet validated "
        "inland or outside the US."
    ]
    if not sar.get("available"):
        missing = sorted(set(f"{m:02d}" for m in range(1, 13)) - set(s.get("temporal_months_covered") or []))
        if len(missing) >= 3:
            caveats.append(f"No radar history and no clear optical imagery for months {', '.join(missing)} — "
                           "floods in those months would not be seen.")
    if (bu or 0) > 0.6:
        caveats.append("Street-level waterlogging in dense built-up areas is not visible to radar or optical "
                       "satellites (buildings mask it) — check municipal drainage and insurance claim records.")
    return {"hazard": "flood", "score": round(pts, 1), "severity": _sev_from_points(pts),
            "mechanisms": sorted(set(mech)), "evidence": ev,
            **({"caveat": " ".join(caveats)} if caveats else {})}


def _debris_landslide(t: dict, s: dict) -> list[dict]:
    TER = "Terrain (AWS 10 m DEM, pysheds D8)"
    out = []
    df = t.get("debris_flow_risk")
    if df:
        sev = _TERRAIN_TO_SEV.get(df, "LOW")
        # The DEM is a surface model: tanks, chimneys and buildings look like cliffs
        # (flat reclaimed Jurong Island, Singapore, scored HIGH in the world test).
        # Require real natural relief before rating debris flow above LOW.
        pct25 = t.get("upslope_pct_over_25deg") or 0.0
        head = t.get("elevation_head_m") or 0.0
        artefact = sev in ("MODERATE", "HIGH") and pct25 < 3.0 and head < 100.0
        out.append({
            "hazard": "debris_flow", "severity": sev,
            "score": {"NEGLIGIBLE": 0, "LOW": 1, "MODERATE": 2.5, "HIGH": 4.5}.get(sev, 0),
            "mechanisms": ["debris_flow"] if sev in ("MODERATE", "HIGH") else [],
            "evidence": [
                _ev("Max upslope gradient", t.get("upslope_max_slope_deg"), TER, "°"),
                _ev("Upslope area steeper than 25°", t.get("upslope_pct_over_25deg"), TER, "%"),
                _ev("Elevation head above site", t.get("elevation_head_m"), TER, "m"),
                _ev("Effective upstream catchment", t.get("upslope_area_km2"), TER, "km²"),
            ] + ([_ev("Steep cells likely structures, not hillslopes",
                      f"only {pct25:.1f}% of upslope area > 25° and {head:.0f} m relief — capped at LOW", TER)]
                 if artefact else []),
        })
        if artefact:
            out[-1].update(severity="LOW", score=1, mechanisms=[])
    ls = t.get("landslide_susceptibility")
    if ls:
        sev = _TERRAIN_TO_SEV.get(ls, "LOW")
        ev = [_ev("Slope at asset", t.get("slope_at_asset_deg"), TER, "°")]
        nd = s.get("ndvi_at_asset")
        if nd is not None:
            ev.append(_ev("Vegetation (NDVI) at asset", nd, "Sentinel-2 L2A"))
            if sev in ("MODERATE", "HIGH") and nd < 0.3:
                ev.append(_ev("Sparse vegetation on slope", "reduces root reinforcement", "Sentinel-2 L2A"))
        out.append({
            "hazard": "landslide", "severity": sev,
            "score": {"NEGLIGIBLE": 0, "LOW": 1, "MODERATE": 2.5, "HIGH": 4.5}.get(sev, 0),
            "mechanisms": ["landslide"] if sev in ("MODERATE", "HIGH") else [],
            "evidence": ev,
        })
    return out


def _wildfire(t: dict, s: dict, lc: dict) -> dict:
    pts, ev = 0.0, []
    WC, S2 = "ESA WorldCover 2021", "Sentinel-2 L2A"
    fv = lc.get("flammable_veg_within_1km")
    if fv is not None:
        ev.append(_ev("Tree / shrub / grass cover within 1 km", _pct(fv), WC, "%"))
        pts += 2 if fv >= 0.6 else 1 if fv >= 0.3 else 0
    tc = lc.get("tree_cover_within_1km")
    if tc and tc >= 0.3:
        pts += 0.5
        ev.append(_ev("Tree cover within 1 km (crown-fire fuel)", _pct(tc), WC, "%"))
    nd = s.get("ndvi_area_mean")
    if nd is not None:
        ev.append(_ev("Mean NDVI of surrounding area", nd, S2))
        if 0.12 <= nd <= 0.45 and (fv or 0) >= 0.3:
            pts += 1   # sparse / cured vegetation — fine dry fuel
            ev.append(_ev("Fuel state", "sparse or cured vegetation (fine dry fuels)", S2))
    ms = t.get("upslope_mean_slope_deg")
    if ms and ms > 15:
        pts += 0.5
        ev.append(_ev("Mean upslope gradient (fire runs faster upslope)", ms, "Terrain (AWS 10 m DEM)", "°"))
    bu = lc.get("built_up_within_250m") or 0
    if bu > 0.8:
        pts -= 1
        ev.append(_ev("Dense built-up surroundings", _pct(bu), WC, "%"))
    # High-altitude dampener: WorldCover labels alpine meadow as grassland, but
    # above ~3,000 m cold, short fire seasons make large fires rare (e.g. Himalayan
    # forest fires concentrate in sub-2,500 m chir-pine belts).
    elev = t.get("elevation_m")
    if elev is not None and elev > 3000:
        pts -= 2
        ev.append(_ev("High-altitude site (short, cold fire season)", elev, "Terrain (AWS 10 m DEM)", "m"))
    pts = max(pts, 0)
    sev = _sev_from_points(pts)
    return {"hazard": "wildfire", "score": round(pts, 1), "severity": sev,
            "mechanisms": ["wildfire"] if sev in ("MODERATE", "HIGH", "CRITICAL") else [],
            "evidence": ev,
            "caveat": "Fuel-based screening; add NASA FIRMS fire history (needs FIRMS MAP_KEY) for ignition frequency."}


def _tropical_cyclone(lat: float, lon: float) -> dict:
    """
    Track-based cyclone wind hazard (climate.tropical_cyclone): IBTrACS 1980–2025,
    Holland wind field, validated against measured station winds.
    Severity from the 1-in-100-year open-terrain 1-min wind (Saffir–Simpson scale).
    """
    from ..climate.tropical_cyclone import site_hazard
    h = site_hazard(lat, lon)
    SRC = "IBTrACS 1980–2025 + Holland wind model"
    rp100 = h.rp_wind_ms.get(100, 0.0) or 0.0
    rp50 = h.rp_wind_ms.get(50, 0.0) or 0.0
    if rp100 >= 58:
        sev = "CRITICAL"
    elif rp100 >= 50:
        sev = "HIGH"
    elif rp100 >= 33:
        sev = "MODERATE"
    elif rp100 >= 17.5:
        sev = "LOW"
    else:
        sev = "NEGLIGIBLE"
    ev = [
        _ev("Cyclones passing within 200 km since 1980", h.storms_within_200km, SRC),
        _ev("Tropical-storm-force winds at site", h.annual_rate_ge_ts, SRC, "per year"),
        _ev("Hurricane-force (Cat 1+) events at site since 1980", h.events_ge_cat1, SRC),
        _ev("1-in-50-year wind (open terrain, 1-min)", f"{rp50:.0f} m/s — {h.rp_category.get(50)}", SRC),
        _ev("1-in-100-year wind (open terrain, 1-min)", f"{rp100:.0f} m/s — {h.rp_category.get(100)}", SRC),
    ]
    if h.worst:
        w = h.worst
        ev.append(_ev("Worst storm at this site", f"{w['name']} {w['season']}: {w['peak_wind_ms']:.0f} m/s "
                      f"({w['category_at_site']}, passed {w['closest_km']:.0f} km away)", SRC))
    if h.station_equiv_factor and rp100:
        ev.append(_ev("1-in-100-year wind as a typical weather station would record it",
                      round(rp100 * h.station_equiv_factor, 1), "Calibrated to NOAA ISD station measurements", "m/s"))
    v = h.validation or {}
    caveat = (f"Validated against measured peak winds at {v.get('stations')} weather stations in {v.get('storms')} "
              f"storms across all basins: pattern r = {v.get('pattern_r')}; after the land/exposure factor, "
              f"{int(100 * (v.get('heldout_within_30pct') or 0))}% of held-out stations within ±30%. "
              "Weakest in the West Pacific (r ≈ 0.3). Historical climate (1980–2025); no climate-change uplift applied."
              ) if v else "Model not yet validated in this build."
    pts = {"NEGLIGIBLE": 0, "LOW": 1, "MODERATE": 2.5, "HIGH": 4.5, "CRITICAL": 6}[sev]
    return {"hazard": "tropical_cyclone", "score": pts, "severity": sev,
            "mechanisms": ["cyclone_wind"] if sev in ("MODERATE", "HIGH", "CRITICAL") else [],
            "evidence": ev, "caveat": caveat,
            "detail": {"rp_wind_ms": h.rp_wind_ms, "rp_category": h.rp_category,
                       "top_events": h.top_events, "method": h.method}}


def _urban_heat(lat: float, lc: dict) -> dict:
    pts, ev = 0.0, []
    WC = "ESA WorldCover 2021"
    b1 = (lc.get("within_1km") or {}).get("built_up")
    tc = lc.get("tree_cover_within_1km")
    if b1 is not None:
        ev.append(_ev("Built-up cover within 1 km", _pct(b1), WC, "%"))
        pts += 2 if b1 >= 0.6 else 1 if b1 >= 0.35 else 0
    if tc is not None:
        ev.append(_ev("Tree canopy within 1 km", _pct(tc), WC, "%"))
        if tc < 0.1 and (b1 or 0) >= 0.35:
            pts += 1
    if abs(lat) < 35 and pts > 0:
        pts += 1
        ev.append(_ev("Latitude band", f"{abs(lat):.1f}° (warm climate)", "Asset coordinates"))
    sev = _sev_from_points(pts)
    if sev in ("HIGH", "CRITICAL"):                  # land-cover proxy only — no temperatures yet
        sev, pts = "MODERATE", min(pts, 3.0)
    return {"hazard": "urban_heat", "score": round(pts, 1), "severity": sev,
            "mechanisms": ["urban_heat"] if sev in ("MODERATE", "HIGH", "CRITICAL") else [],
            "evidence": ev,
            "caveat": "Land-cover amplification only (capped at MODERATE); the temperature-based heat model is the next build."}


# ── Orchestrator ─────────────────────────────────────────────────────────────

def assess_asset(
    lat: float,
    lon: float,
    asset_name: str = "Asset",
    asset_type: str = "industrial",
    radius_km: float = 2.5,
    include_visuals: bool = True,
    viz_px: int = 160,
    sar_timeout_s: float = 75.0,
    fetch_grid: Optional[bool] = None,   # backward-compat alias for include_visuals
) -> dict:
    """
    Evidence-based physical risk assessment for a real asset location.

    Runs terrain, Sentinel-2 and WorldCover analyses in parallel, derives
    findings with traceable evidence, and (optionally) returns a visual payload
    for the 3D dashboard: DEM grid, true-colour texture and overlay layers, all
    on the same geographic bounds.
    """
    if fetch_grid is not None:
        include_visuals = fetch_grid
    t0 = time.time()

    from ..connectors.sentinel2_pixels import asset_bbox, get_satellite_evidence
    from ..connectors.sentinel1_sar import get_sar_water_history
    from ..connectors.worldcover import get_land_cover
    from ..terrain.pysheds_analyzer import TerrainEngine

    def _terrain():
        return TerrainEngine(radius_km=radius_km, viz_px=viz_px).assess(
            lat=lat, lon=lon, asset_name=asset_name).to_dict()

    def _sat():
        return get_satellite_evidence(lat, lon, radius_km=radius_km,
                                      include_visuals=include_visuals)

    def _lc():
        return get_land_cover(lat, lon, radius_km=radius_km, include_layer=include_visuals)

    def _sar():
        return get_sar_water_history(lat, lon, radius_km=radius_km, include_layer=include_visuals)

    errors = {}
    pool = ThreadPoolExecutor(max_workers=4)
    futs = {"terrain": pool.submit(_terrain), "satellite": pool.submit(_sat),
            "land_cover": pool.submit(_lc), "radar": pool.submit(_sar)}
    results = {}
    for k, f in futs.items():
        try:
            # Radar is capped so a slow Azure read can't stall the whole assessment;
            # it keeps running in the background and fills the cache for next time.
            results[k] = f.result(timeout=sar_timeout_s if k == "radar" else None)
        except Exception as exc:
            log.warning("asset_intelligence: %s failed: %s", k, exc)
            errors[k] = "timed out (cached for next run)" if k == "radar" and not str(exc) else str(exc)
            results[k] = None
    pool.shutdown(wait=False)

    terrain = results["terrain"] or {}
    sat_obj = results["satellite"]
    sat = sat_obj.to_dict(include_visuals=False) if sat_obj else {}
    lc = results["land_cover"] or {}
    sar = results["radar"] or {}

    belev = None
    try:
        from ..terrain.hand import built_elevation_stats
        belev = built_elevation_stats(lat, lon, radius_km=radius_km)
    except Exception as exc:                      # noqa: BLE001
        errors["built_elevation"] = str(exc)

    findings = [_flood(terrain, sat, lc, sar, belev), *_debris_landslide(terrain, sat),
                _wildfire(terrain, sat, lc), _urban_heat(lat, lc)]
    try:
        findings.append(_tropical_cyclone(lat, lon))
    except Exception as exc:                      # noqa: BLE001
        errors["tropical_cyclone"] = str(exc)
    for f in findings:
        acts = []
        if f["hazard"] == "flood" and f["severity"] == "CRITICAL":
            acts.append("Site floods repeatedly on satellite record: treat as unsuitable for permanent high-value "
                        "assets without major engineered protection — evaluate relocation or elevated construction "
                        "(plinth above the largest observed flood extent) before further capex.")
        mechs = f.get("mechanisms", [])
        if f["hazard"] == "flood" and f["severity"] in ("NEGLIGIBLE", "LOW"):
            mechs = []                                  # no flood actions for immaterial flood
        for m in mechs:
            acts.extend(_ACTIONS.get(m, []))
        f["actions"] = list(dict.fromkeys(acts))
    findings.sort(key=lambda f: (SEVERITY_ORDER.index(f["severity"]), f.get("score", 0)), reverse=True)

    top = findings[0] if findings else {"severity": "NEGLIGIBLE"}
    risk_signal = top["severity"] if top["severity"] != "NEGLIGIBLE" else "LOW"

    material = [f for f in findings if f["severity"] in ("MODERATE", "HIGH", "CRITICAL")]
    if material and asset_type in _ASSET_ADDONS:
        material[0]["actions"] = material[0]["actions"] + _ASSET_ADDONS[asset_type]

    headline = _headline(asset_name, terrain, sat, lc, material)

    out = {
        "asset_name": asset_name,
        "asset_type": asset_type,
        "lat": lat,
        "lon": lon,
        "risk_signal": risk_signal,
        "headline": headline,
        "findings": findings,
        "terrain": {k: v for k, v in terrain.items() if k != "viz"},
        "satellite": sat,
        "land_cover": {k: v for k, v in lc.items() if k != "layer"},
        "radar": {k: v for k, v in sar.items() if k != "layer"},
        "sources": [
            "AWS Terrain Tiles (Terrarium, ~10 m) — s3://elevation-tiles-prod",
            "Sentinel-2 L2A COGs — AWS Open Data via Earth Search STAC",
            "ESA WorldCover 2021 v200 — s3://esa-worldcover",
            "NOAA IBTrACS v04r01 tropical-cyclone tracks 1980–2025 (validated vs NOAA ISD station winds)",
        ] + (["Sentinel-1 RTC radar — Microsoft Planetary Computer"] if sar.get("available") else []) + (["Prithvi-EO-2.0-300M-TL-Sen1Floods11 (Hugging Face endpoint)"]
             if sat.get("prithvi_water_fraction") is not None else []),
        "caveats": [
            "Screening-grade: prioritises assets for survey; not a substitute for hydraulic flood modelling or geotechnical investigation.",
            "Terrain from a digital surface model — buildings and trees can raise apparent ground height in dense urban areas.",
        ] + terrain.get("data_caveats", []),
        "errors": errors | ({"satellite": sat["errors"]} if sat.get("errors") else {}),
        "elapsed_s": round(time.time() - t0, 1),
    }

    if include_visuals:
        out["visual"] = _build_visual(
            bounds=list(asset_bbox(lat, lon, radius_km)),
            viz=terrain.get("viz") or {},
            sat_obj=sat_obj,
            scene_date=sat.get("scene_date"),
            lc=lc,
            sar=sar,
        )
    return out


# ── Compact visual payload ───────────────────────────────────────────────────

def _pack(values: Optional[list], shape, lo: float, hi: float, dtype: str = "uint8") -> Optional[dict]:
    """
    Quantise a flat list into base64 bytes. Reserved code (255 / 65535) = nodata.
    JS decode: v = lo + code / (maxcode - 1) * (hi - lo).
    """
    import base64
    import numpy as np
    if not values:
        return None
    a = np.array([np.nan if v is None else v for v in values], dtype=np.float64)
    maxcode = 255 if dtype == "uint8" else 65535
    span = (hi - lo) or 1.0
    q = np.round((np.clip(a, lo, hi) - lo) / span * (maxcode - 1))
    q[~np.isfinite(a)] = maxcode
    q = q.astype(np.uint8 if dtype == "uint8" else "<u2")
    return {"b64": base64.b64encode(q.tobytes()).decode(), "dtype": dtype,
            "lo": float(lo), "hi": float(hi), "nodata": maxcode, "shape": list(shape)}


def _build_visual(bounds, viz: dict, sat_obj, scene_date, lc: dict, sar: Optional[dict] = None) -> dict:
    import numpy as np
    vis: dict = {"bounds": bounds, "scene_date": scene_date, "layers": {}}

    if viz.get("elevations"):
        shp = (viz["rows"], viz["cols"])
        el = [v for v in viz["elevations"] if v is not None]
        lo, hi = (min(el), max(el)) if el else (0.0, 1.0)
        vis["dem"] = {
            "rows": viz["rows"], "cols": viz["cols"], "cell_size_m": viz.get("cell_size_m"),
            "asset_row": viz.get("asset_row"), "asset_col": viz.get("asset_col"),
            "min_m": lo, "max_m": hi,
            "elevation": _pack(viz["elevations"], shp, lo, hi, "uint16"),
        }
        vis["layers"]["slope"] = _pack(viz.get("slope"), shp, 0.0, 60.0)
        fa = [v for v in (viz.get("flow_acc") or []) if v is not None]
        vis["layers"]["flow_acc"] = _pack(viz.get("flow_acc"), shp, 0.0, max(fa) if fa else 1.0)

    if sat_obj is not None:
        vis["texture_jpeg_b64"] = getattr(sat_obj, "true_color_jpeg_b64", None)
        shp = tuple(getattr(sat_obj, "layer_shape", (0, 0)))
        for name, (lo, hi) in {"ndvi": (-1, 1), "mndwi": (-1, 1), "ndbi": (-1, 1),
                               "water_freq": (0, 1)}.items():
            vals = (getattr(sat_obj, "layers", {}) or {}).get(name)
            if vals:
                vis["layers"][name] = _pack(vals, shp, lo, hi)

    if sar and sar.get("layer"):
        vis["layers"]["sar_water"] = _pack(sar["layer"], tuple(sar.get("layer_shape", (0, 0))), 0.0, 1.0)

    if lc.get("layer"):
        codes = np.array([255 if v is None else int(v) for v in lc["layer"]], dtype=np.uint8)
        import base64
        vis["layers"]["land_cover"] = {
            "b64": base64.b64encode(codes.tobytes()).decode(), "dtype": "uint8",
            "categorical": True, "nodata": 255, "shape": list(lc.get("layer_shape", (0, 0))),
        }

    vis["layers"] = {k: v for k, v in vis["layers"].items() if v}
    return vis


def _headline(name: str, t: dict, s: dict, lc: dict, material: list[dict]) -> str:
    parts = []
    if t.get("elevation_m") is not None:
        parts.append(f"{name} sits at {t['elevation_m']:.0f} m on {t.get('terrain_position', 'unknown')} terrain")
    if lc.get("class_at_asset"):
        parts.append(f"in {lc['class_at_asset'].replace('_', ' ')} land cover")
    s1 = " ".join(parts) + "." if parts else f"{name}."
    if not material:
        return s1 + " No hazard reached MODERATE on the combined terrain, satellite and land-cover evidence."
    top = material[0]
    lead = {
        "flood": "Flood is the leading exposure",
        "debris_flow": "Debris flow from upslope terrain is the leading exposure",
        "landslide": "Slope instability is the leading exposure",
        "wildfire": "Wildfire fuel load is the leading exposure",
        "urban_heat": "Urban heat amplification is the leading exposure",
        "tropical_cyclone": "Tropical cyclone wind is the leading exposure",
    }.get(top["hazard"], f"{top['hazard']} is the leading exposure")
    others = [f["hazard"].replace("_", " ") for f in material[1:]]
    tail = f"; also {', '.join(others)}" if others else ""
    return f"{s1} {lead} ({top['severity']}){tail}."
