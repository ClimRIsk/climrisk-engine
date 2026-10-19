"""
cri/hazardsurface.py — one interface that turns a hazard model into a map layer for the 3D viewer.

  surface(hazard, lat, lon, radius_km=2.5, polygons=None, **params) -> dict
  catalogue() -> list of every hazard with whether it has a spatial surface, and why not when it does not

A layer is a classified raster in a WGS84 box (a PNG the viewer drapes on the terrain), a legend, the share of cells in each class (over the whole box and
inside the study polygon), the method, a validation status and the caveats. Hazards whose model gives ONE number per site (heat, drought, earthquake PGA,
wildfire weather, cyclone wind) have no surface: asking for one returns {"available": False, "reason": ...} and the site value, never an invented map.
Null is not zero: cells the model did not evaluate are transparent and counted as "not evaluated", not as "low".
"""
from __future__ import annotations

import base64
import io
import math
from typing import Callable, Optional

import numpy as np

# colour, label per class index (0 = worst). Same four-step scale for every hazard, plus a transparent "clear" class.
PALETTE = ["#b71c1c", "#f57c00", "#fdd835", "#2e7d32"]
LABELS = ["Very high", "High", "Moderate", "Low"]
MAX_PX = 700


RASTERS: dict = {}          # last class raster per hazard: (cls array, box, labels); lets buildings() read the class under a footprint


class NoSurface(Exception):
    pass


# ── geometry ────────────────────────────────────────────────────────────────

def _box(lat, lon, rows, cols, ar, ac, cell_m):
    cl = cell_m / 111320.0
    co = cell_m / (111320.0 * math.cos(math.radians(lat)))
    north, west = lat + ar * cl, lon - ac * co
    return {"west": west, "east": west + cols * co, "north": north, "south": north - rows * cl, "cell_lat": cl, "cell_lon": co}


def _radius_for(lat, lon, radius_km, polygons):
    if not polygons:
        return lat, lon, radius_km
    pts = [p for r in polygons for p in r]
    w, e = min(p[0] for p in pts), max(p[0] for p in pts)
    s, n = min(p[1] for p in pts), max(p[1] for p in pts)
    clat, clon = (s + n) / 2, (w + e) / 2
    half_km = max((n - s) * 110.574 / 2, (e - w) * 111.32 * math.cos(math.radians(clat)) / 2)
    return clat, clon, float(min(6.0, max(1.0, half_km * 1.8 + 0.3)))


def _classify(v, edges):
    """edges ascending: v < e0 -> class 0 (worst) ... v >= last -> last class. NaN stays -1."""
    out = np.full(v.shape, -1, int)
    ok = np.isfinite(v)
    out[ok] = np.digitize(v[ok], edges)
    return out


def _png(cls, n_classes, clear_class=None, alpha=190):
    from PIL import Image
    h, w = cls.shape
    step = max(1, int(math.ceil(max(h, w) / MAX_PX)))
    c = cls[::step, ::step]
    rgba = np.zeros(c.shape + (4,), np.uint8)
    for k in range(n_classes):
        hexs = PALETTE[k]
        rgb = [int(hexs[i:i + 2], 16) for i in (1, 3, 5)]
        m = c == k
        rgba[m] = rgb + [alpha]
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _package(hazard, title, cls, n_classes, labels, box, unit_note, method, validated, caveats, polygons, extra=None, clear_label=None, alpha=190):
    h, w = cls.shape
    valid = cls >= 0
    lat_c = np.linspace(box["north"] - box["cell_lat"] / 2, box["south"] + box["cell_lat"] / 2, h)
    lon_c = np.linspace(box["west"] + box["cell_lon"] / 2, box["east"] - box["cell_lon"] / 2, w)
    inside = None
    if polygons:
        from .geo_io import points_in_polygons
        LON, LAT = np.meshgrid(lon_c, lat_c)
        inside = points_in_polygons(LON, LAT, polygons)

    def shares(mask):
        n = int((mask & valid).sum())
        if n == 0:
            return None
        d = {labels[k]: round(float(((cls == k) & mask).sum()) / n, 4) for k in range(len(labels))}
        d["cells"] = n
        d["not_evaluated_cells"] = int((mask & ~valid).sum())
        return d

    out = {"available": True, "hazard": hazard, "title": title, "bounds": {k: round(box[k], 7) for k in ("west", "south", "east", "north")},
           "image_png_b64": _png(cls, n_classes, alpha=alpha), "alpha": alpha / 255.0, "legend": [{"label": labels[k], "color": PALETTE[k]} for k in range(len(labels))],
           "shares_box": shares(np.ones(cls.shape, bool)), "shares_study_area": shares(inside) if inside is not None else None,
           "unit_note": unit_note, "method": method, "validated": validated, "caveats": caveats, "grid": {"rows": h, "cols": w}}
    RASTERS[("last", hazard)] = (cls, dict(box), list(labels))
    if clear_label:
        out["legend"].append({"label": clear_label, "color": None})
    if extra:
        out["details"] = extra
    return out


# ── adapters ────────────────────────────────────────────────────────────────

def _dem(lat, lon, radius_km, dem=None):
    if dem is not None:
        return dem
    from .connectors.terrain_tiles import get_elevation_grid
    z = 14 if radius_km <= 4.0 else 13
    return get_elevation_grid(lat, lon, radius_km=radius_km, zoom=z)


def _landslide(lat, lon, radius_km, polygons, return_period=100, c_kpa=5.0, phi_deg=30.0, gamma=18.0, depth_m=2.0, porosity=0.40, cn=75.0, dem=None, rain_fetch=None, **_):
    from .modellab import hydrology, landslide as LS
    from scipy import ndimage as ndi
    d = _dem(lat, lon, radius_km, dem)
    z = np.asarray(d.elevations, float)
    zf = ndi.median_filter(z, size=3, mode="nearest")
    sl = LS.slope_rad(zf, float(d.cell_size_m))
    land = zf > 0.5
    try:
        if rain_fetch is None:
            from .modellab.flood_study import hydrology_fetch as rain_fetch
        am = rain_fetch(lat, lon)
        mu, beta = hydrology.gumbel_fit(am["maxima_mm"])
        d_T, src = hydrology.gumbel_quantile(mu, beta, return_period), am["source"]
    except Exception as e:                                       # noqa: BLE001
        d_T, src = 150.0, f"generic fallback rainfall ({type(e).__name__}): the saturation, and so the map, is a placeholder"
    ante = 1.5 * d_T
    inf = max(ante - hydrology.scs_cn_runoff(ante, cn), 0.0)
    m = LS.saturation_fraction(inf, depth_m, porosity)
    fs = LS.factor_of_safety(sl, c_kpa, phi_deg, gamma, depth_m, m)
    v = np.where(land, fs, np.nan)
    cls = _classify(v, [1.0, 1.25, 1.5])
    box = _box(d.lat, d.lon, d.rows, d.cols, d.asset_row, d.asset_col, d.cell_size_m)
    cav = ["Generic soil strength (not a soil map) and a single soil depth.", "Initiation only: no runout, no earthquake trigger, no vegetation root strength.",
           "The class edges (FS 1.0 / 1.25 / 1.5) are a common engineering convention, not calibrated to a landslide inventory.", f"Rainfall: {return_period}-year daily rain {d_T:.0f} mm from {src}."]
    return _package("landslide", f"Landslide susceptibility after a {return_period}-year storm (factor of safety)", cls, 4, LABELS, box, "FS < 1.0 very high; 1.0-1.25 high; 1.25-1.5 moderate; above 1.5 low",
                    "Infinite-slope stability with a rainfall-driven pore-pressure term (modellab.landslide)", "verified against hand calculations only; not validated against real landslides unless an inventory is supplied", cav, polygons,
                    {"saturation": round(m, 3), "rain_mm": round(d_T, 1), "return_period": return_period, "cell_m": round(float(d.cell_size_m), 1)})


def _flood_hand(lat, lon, radius_km, polygons, **_):
    from .terrain.hand import hand_grid
    hg = hand_grid(lat, lon, radius_km=radius_km)
    key = min(hg.hand) if hg.hand else None
    if key is None:
        raise NoSurface("no drainage network could be derived from the terrain here (flat or tiny area)")
    h = np.where(hg.valid, hg.hand[key], np.nan)
    cls = _classify(h, [1.0, 2.0, 5.0])
    box = _box(hg.lat, hg.lon, h.shape[0], h.shape[1], hg.asset_rc[0], hg.asset_rc[1], hg.cell_m)
    cav = ["Terrain susceptibility only: height above the nearest stream. It is not a flood depth and has no return period.", "Ignores rainfall, drainage works, defences, culverts and coastal water.",
           "A backtest against FEMA NFIP claims showed point-in-cell values have little skill at tract level; use the share of an area, not a single cell."]
    return _package("flood_terrain", "Flood susceptibility from terrain (height above nearest drainage)", cls, 4, LABELS, box, "HAND < 1 m very high; 1-2 m high; 2-5 m moderate; above 5 m low",
                    "Height Above Nearest Drainage from D8 flow routing on the 10 m AWS terrain (terrain.hand)", "area-level backtest against NFIP claims exists; cell-level values are not validated", cav, polygons, {"channel_threshold_km2": key})


def _slr(lat, lon, radius_km, polygons, water_level_m=None, scenario="ssp245", year=2100, dem=None, **_):
    d = _dem(lat, lon, radius_km, dem)
    z = np.asarray(d.elevations, float)
    sea = z <= 0.0
    if not sea.any():
        raise NoSurface("no sea inside this box: coastal inundation needs the box to include open water (try a larger radius at the coast)")
    src = "water level supplied by the user"
    if water_level_m is None:
        try:
            from .modellab import slr
            loc = slr.nearest(lat, lon)
            water_level_m = float(slr.rsl(loc["i"])[scenario][int(year)][2])
            src = f"AR6 median relative sea-level rise {scenario} {year} at {loc['name']}"
        except Exception as e:                                   # noqa: BLE001
            raise NoSurface(f"no sea-level projection could be loaded ({type(e).__name__}); supply water_level_m") from e
    from scipy import ndimage as ndi
    low = z <= water_level_m
    lab, n = ndi.label(low)
    keep = np.unique(lab[sea & low])
    flooded = np.isin(lab, keep[keep > 0])
    depth = np.where(flooded, water_level_m - z, 0.0)
    v = np.where(np.isfinite(z), depth, np.nan)
    # worst class = deepest. classify on negative depth so edges ascend
    cls = np.full(v.shape, -1, int)
    ok = np.isfinite(v)
    cls[ok & (v >= 1.0)] = 0
    cls[ok & (v >= 0.3) & (v < 1.0)] = 1
    cls[ok & (v > 0.0) & (v < 0.3)] = 2
    cls[ok & (v <= 0.0)] = 3
    box = _box(d.lat, d.lon, d.rows, d.cols, d.asset_row, d.asset_col, d.cell_size_m)
    cav = ["Bathtub model: still water at one level connected to the sea. No tides, storm surge, waves, defences, drainage or subsidence beyond what the water level already includes.",
           "Terrain tiles have vertical errors of several metres in places (worst in cities and forest); the map is a screen.", f"Water level: {water_level_m:.2f} m ({src})."]
    pkg = _package("sea_level_rise", f"Coastal inundation at {water_level_m:.2f} m above today's mean sea level", cls, 4, ["Depth 1 m or more", "Depth 0.3-1 m", "Depth under 0.3 m", "Not inundated"], box,
                   "depth of connected still water on the 10 m terrain", "Bathtub inundation with hydrological connectivity to open sea; water level from the AR6 projections (modellab.slr)",
                   "verified only; the AR6 rise is backtested against tide gauges where one is near, the inundation map is not validated", cav, polygons, {"water_level_m": round(water_level_m, 3), "water_level_source": src})
    pkg["legend"] = [{"label": "Depth 1 m or more", "color": PALETTE[0]}, {"label": "Depth 0.3-1 m", "color": PALETTE[1]}, {"label": "Depth under 0.3 m", "color": PALETTE[2]}, {"label": "Not inundated", "color": None}]
    return pkg


def _plume(lat, lon, radius_km, polygons, Q_g_s=None, H_m=None, u10_ms=None, wind_from_deg=None, half_m=None, stability=None, **_):
    if None in (Q_g_s, H_m, u10_ms, wind_from_deg):
        raise NoSurface("a release needs its inputs: Q_g_s (grams per second), H_m (release height), u10_ms (10 m wind) and wind_from_deg. None is assumed.")
    from . import release_plume
    half = float(half_m or min(1500.0, radius_km * 1000.0))
    f = release_plume.field(float(Q_g_s), float(H_m), float(u10_ms), float(wind_from_deg), cls=stability, half_m=half, ground_n=128)
    g = np.array([[np.nan if v is None else v for v in row] for row in f["ground"]["ug_m3"]], float)[::-1]       # row 0 = north
    with np.errstate(divide="ignore"):
        lg = np.where(g > 0, np.log10(g), np.nan)
    cls = np.full(g.shape, -1, int)
    ok = np.isfinite(g)
    cls[ok & (g >= 1000)] = 0
    cls[ok & (g >= 100) & (g < 1000)] = 1
    cls[ok & (g >= 10) & (g < 100)] = 2
    cls[ok & (g < 10)] = 3
    n = g.shape[0]
    cl = 2 * half / (n - 1) / 111320.0
    co = cl / math.cos(math.radians(lat))
    box = {"north": lat + half / 111320.0, "south": lat - half / 111320.0, "west": lon - half / (111320.0 * math.cos(math.radians(lat))), "east": lon + half / (111320.0 * math.cos(math.radians(lat))), "cell_lat": cl, "cell_lon": co}
    cav = ["Screening Gaussian plume for ONE declared release: flat terrain, steady wind, no buildings, no chemistry or deposition.", "Not a health threshold: the classes are decades of ground-level concentration.",
           "Cells within 30 m downwind of the source and upwind of it are not evaluated (transparent)."]
    return _package("plume", "Ground-level concentration from a declared release (micrograms per cubic metre)", cls, 4, ["1000 or more", "100 to 1000", "10 to 100", "Under 10"], box,
                    "ground-level concentration, decade classes", f["method"], "verified against closed-form benchmarks; not validated against a measured release", cav, polygons,
                    {"inputs": f["inputs"], "peak_ground": f["peak_ground"], "levels_ug_m3": f["levels_ug_m3"]})


def _slope(lat, lon, radius_km, polygons, dem=None, **_):
    from .modellab import landslide as LS
    from scipy import ndimage as ndi
    d = _dem(lat, lon, radius_km, dem)
    z = np.asarray(d.elevations, float)
    zf = ndi.median_filter(z, size=3, mode="nearest")
    deg = np.degrees(LS.slope_rad(zf, float(d.cell_size_m)))
    v = np.where(zf > 0.5, deg, np.nan)
    cls = _classify(v, [10.0, 20.0, 30.0])
    cls = np.where(cls >= 0, 3 - cls, cls)                      # steeper = worse
    box = _box(d.lat, d.lon, d.rows, d.cols, d.asset_row, d.asset_col, d.cell_size_m)
    return _package("slope", "Terrain slope", cls, 4, ["Over 30 deg", "20-30 deg", "10-20 deg", "Under 10 deg"], box, "slope in degrees on the 10-30 m terrain", "finite-difference slope of the median-filtered terrain",
                    "exact on a plane (unit test); the terrain data itself has vertical error", ["A description of the ground, not a hazard on its own."], polygons)


REGISTRY: dict = {
    "landslide": {"label": "Landslide susceptibility", "fn": _landslide},
    "flood_terrain": {"label": "Flood susceptibility (terrain)", "fn": _flood_hand},
    "sea_level_rise": {"label": "Coastal inundation (sea-level rise)", "fn": _slr},
    "plume": {"label": "Release plume (needs declared release)", "fn": _plume},
    "slope": {"label": "Terrain slope", "fn": _slope},
}
POINT_ONLY = {
    "heat": "heat stress is modelled as a time series at the site, not as a map; the viewer shows the site value",
    "drought": "drought is a regional rainfall and flow record, with no sub-kilometre pattern",
    "earthquake": "ground-motion estimates are one value per site at this scale",
    "wildfire": "fire weather is one value per site; the engine has no validated fire-spread surface",
    "cyclone_wind": "design wind is one value per site; terrain exposure factors are not modelled",
    "air_quality": "PM2.5 comes from monitors and a regional model, not a site-scale surface",
    "water_stress": "basin-scale index, one value for the site",
    "river_flood_depth": "a depth-for-return-period map needs the flood solver run on the site (hydraulics), which the twin does; not exposed here yet",
}


def catalogue() -> list:
    rows = [{"hazard": k, "label": v["label"], "spatial_surface": True} for k, v in REGISTRY.items()]
    rows += [{"hazard": k, "label": k.replace("_", " ").title(), "spatial_surface": False, "reason": r} for k, r in POINT_ONLY.items()]
    return rows


def surface(hazard: str, lat: float, lon: float, radius_km: float = 2.5, polygons=None, **params) -> dict:
    hazard = (hazard or "").lower()
    if hazard in POINT_ONLY:
        return {"available": False, "hazard": hazard, "reason": POINT_ONLY[hazard]}
    if hazard not in REGISTRY:
        raise ValueError(f"unknown hazard {hazard!r}; choose one of {sorted(list(REGISTRY) + list(POINT_ONLY))}")
    clat, clon, rad = _radius_for(lat, lon, radius_km, polygons)
    try:
        return REGISTRY[hazard]["fn"](clat, clon, rad, polygons, **params)
    except NoSurface as e:
        return {"available": False, "hazard": hazard, "reason": str(e)}


# ── 3D buildings (OpenStreetMap), optionally coloured by the hazard class under each footprint ───────────────────────────────

OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter", "https://overpass.private.coffee/api/interpreter"]
_UA = "ClimRisk/1.0 (climate risk research; contact via project owner)"
MAX_BUILDINGS = 1800


def _cache_dir():
    import os
    from pathlib import Path
    return Path(os.environ.get("CRI_CACHE_DIR", Path.home() / ".climrisk" / "cache")) / "osm"


def _overpass_buildings(lat, lon, radius_km, fetch=None):
    import hashlib
    import json as _j
    import time
    import urllib.request
    from pathlib import Path
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * math.cos(math.radians(lat)))
    s, w, n, e = lat - dlat, lon - dlon, lat + dlat, lon + dlon
    q = f'[out:json][timeout:40];way["building"]({s},{w},{n},{e});out tags geom;'
    if fetch is not None:
        return fetch(q)
    key = _cache_dir() / ("b_" + hashlib.md5(f"{s:.4f}{w:.4f}{n:.4f}{e:.4f}".encode()).hexdigest() + ".json")
    if key.exists():
        try:
            return _j.loads(key.read_text())
        except Exception:                                       # noqa: BLE001
            pass
    last = None
    for ep, attempt in [(e, a) for a in (0, 1) for e in OVERPASS]:   # every mirror once, then every mirror again (public servers rate-limit and recover quickly)
        req = urllib.request.Request(ep, data=("data=" + urllib.request.quote(q)).encode(), headers={"User-Agent": _UA, "Content-Type": "application/x-www-form-urlencoded"})
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                out = _j.loads(r.read())
            try:
                key.parent.mkdir(parents=True, exist_ok=True)
                key.write_text(_j.dumps(out, separators=(",", ":")))
            except Exception:                                   # noqa: BLE001
                pass
            return out
        except Exception as exc:                                # noqa: BLE001
            last = exc
            time.sleep(1.0 + 2.0 * attempt)
    raise RuntimeError(f"OpenStreetMap buildings unavailable ({type(last).__name__})")


def _height(tags: dict):
    """(metres, source). Real tag if present; storeys x 3.2 m next; otherwise a flagged default of 7 m."""
    import re as _re

    def num(v):
        m = _re.match(r"\s*([0-9]+(?:\.[0-9]+)?)", str(v or ""))
        return float(m.group(1)) if m else None
    h = num(tags.get("height"))
    if h and 1 <= h <= 600:
        return h, "height tag"
    lv = num(tags.get("building:levels"))
    if lv and 1 <= lv <= 150:
        return lv * 3.2 + 0.5, "storeys x 3.2 m"
    return 7.0, "assumed (no height in OpenStreetMap)"


def class_at(raster, lon: float, lat: float):
    """Class index at a point from a raster tuple (cls, box, labels), or -1 outside / not evaluated."""
    cls, box, _ = raster
    h, w = cls.shape
    c = int((lon - box["west"]) / (box["east"] - box["west"]) * w)
    r = int((box["north"] - lat) / (box["north"] - box["south"]) * h)
    if 0 <= r < h and 0 <= c < w:
        return int(cls[r, c])
    return -1


def buildings(lat: float, lon: float, radius_km: float = 1.0, hazard: str = None, polygons=None, fetch=None) -> dict:
    """OpenStreetMap building footprints with heights. If a hazard layer for the same hazard was just computed, each building gets the WORST class among its corners and centre
    (a terrain-cell value under the footprint: it is not the building's own vulnerability). Heights are flagged by source."""
    radius_km = float(min(max(radius_km, 0.2), 1.5))
    osm = _overpass_buildings(lat, lon, radius_km, fetch)
    raster = RASTERS.get(("last", hazard)) if hazard else None
    labels = raster[2] if raster else None
    out, counts, hs = [], {}, {}
    from .geo_io import points_in_polygons
    for el in osm.get("elements", []):
        g = el.get("geometry")
        if el.get("type") != "way" or not g or len(g) < 4:
            continue
        ring = [[round(p["lon"], 6), round(p["lat"], 6)] for p in g]
        if ring[0] == ring[-1]:
            ring = ring[:-1]
        if len(ring) < 3:
            continue
        h, hsrc = _height(el.get("tags") or {})
        hs[hsrc] = hs.get(hsrc, 0) + 1
        item = {"ring": ring, "h": round(h, 1), "h_src": hsrc}
        if raster:
            cx = sum(p[0] for p in ring) / len(ring)
            cy = sum(p[1] for p in ring) / len(ring)
            cs = [class_at(raster, px, py) for px, py in ring + [[cx, cy]]]
            ok = [c for c in cs if c >= 0]
            item["cls"] = (min(ok) if ok else None)
            item["label"] = labels[item["cls"]] if item["cls"] is not None and item["cls"] < len(labels) else None
            counts[item["label"] or "not evaluated"] = counts.get(item["label"] or "not evaluated", 0) + 1
        out.append(item)
        if len(out) >= MAX_BUILDINGS:
            break
    if polygons:
        cen = [(sum(p[0] for p in b["ring"]) / len(b["ring"]), sum(p[1] for p in b["ring"]) / len(b["ring"])) for b in out]
        if cen:
            m = points_in_polygons([c[0] for c in cen], [c[1] for c in cen], polygons)
            for b, inside in zip(out, m):
                b["in_study_area"] = bool(inside)
    return {"available": True, "count": len(out), "capped": len(out) >= MAX_BUILDINGS, "buildings": out, "height_sources": hs,
            "hazard": hazard if raster else None, "legend": ([{"label": l, "color": PALETTE[i]} for i, l in enumerate(labels)] if raster else None), "class_counts": counts or None,
            "legend_colors": PALETTE, "source": "OpenStreetMap contributors (ODbL)",
            "caveat": "Footprints and heights come from OpenStreetMap volunteers and are incomplete in many places; heights marked 'assumed' are a flat 7 m. The hazard class is the terrain-cell value under the footprint, not an assessment of the building."}
