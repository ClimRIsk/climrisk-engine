"""Shared coastal data access: sea-floor and land elevation from the open terrain tiles, NOAA CO-OPS tide-gauge records (public domain), great-circle geometry.

Everything is injectable for tests and cached on disk. NOAA CO-OPS data are US Government works (public domain); the terrain tiles carry open licences (GEBCO and national DEMs, attribution in the report).
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from . import data as D

COOPS = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
COOPS_STATIONS = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json?type=waterlevels"
R_EARTH_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(np.asarray(lon2) - lon1) / 2) ** 2
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(a))


def destination(lat, lon, bearing_deg, dist_km):
    """Point reached from (lat, lon) travelling dist_km on the initial bearing (great circle)."""
    d = dist_km / R_EARTH_KM; b = math.radians(bearing_deg); p1 = math.radians(lat); l1 = math.radians(lon)
    p2 = math.asin(math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(b))
    l2 = l1 + math.atan2(math.sin(b) * math.sin(d) * math.cos(p1), math.cos(d) - math.sin(p1) * math.sin(p2))
    return math.degrees(p2), (math.degrees(l2) + 540) % 360 - 180


_MEM: dict = {}


def _cached_tile(z, x, y):
    """Terrain tile with a disk cache (tiles never change), so a study that samples hundreds of points downloads each tile once per machine."""
    from ..connectors import terrain_tiles as T
    key = (z, x, y)
    if key in _MEM:
        return _MEM[key]
    p = D._cache_path(f"terrarium:{z}:{x}:{y}").with_suffix(".npy")
    if p.exists():
        try:
            _MEM[key] = np.load(p)
            return _MEM[key]
        except Exception:                                              # noqa: BLE001
            pass
    arr = T._fetch_tile(z, x, y)
    if arr is not None:
        _MEM[key] = arr
        try:
            np.save(p, arr)
        except Exception:                                              # noqa: BLE001
            pass
    return arr


# ── terrain / bathymetry ──────────────────────────────────────────────────────────────────────────────────────────────────────
def elevation(lat: float, lon: float, z: int = 9, tile_fetch: Optional[Callable] = None) -> Optional[float]:
    """Elevation in metres (negative = sea floor) from Terrarium tiles; None when the tile cannot be fetched. tile_fetch(z, x, y) -> uint8 RGB array, for tests."""
    from ..connectors import terrain_tiles as T
    n = 2 ** z
    xf = (lon + 180.0) / 360.0 * n
    yf = (1.0 - math.asinh(math.tan(math.radians(max(-85.0, min(85.0, lat))))) / math.pi) / 2.0 * n
    tx, ty = int(xf) % n, min(n - 1, max(0, int(yf)))
    rgb = (tile_fetch or _cached_tile)(z, tx, ty)
    if rgb is None:
        return None
    el = T._decode_terrarium(rgb)
    px, py = min(255, int((xf - int(xf)) * 256)), min(255, int((yf - int(yf)) * 256))
    return float(el[py, px])


def profile(lat: float, lon: float, bearing: float, max_km: float = 300.0, step_km: float = 5.0, z: int = 8, tile_fetch=None) -> np.ndarray:
    """Elevation (m; negative = depth) at the site and every step_km along the bearing."""
    out = []
    for k in np.arange(0.0, max_km + 1e-9, step_km):
        la, lo = destination(lat, lon, bearing, k) if k else (lat, lon)
        e = elevation(la, lo, z, tile_fetch)
        out.append(np.nan if e is None else e)
    return np.array(out)


# ── NOAA CO-OPS tide gauges ─────────────────────────────────────────────────────────────────────────────────────────────────────
def stations(fetch=None) -> list:
    d = D._get(COOPS_STATIONS) if fetch is None else fetch(COOPS_STATIONS)
    return [{"id": s["id"], "name": s["name"], "lat": float(s["lat"]), "lon": float(s["lng"]), "state": s.get("state")} for s in d["stations"]]


def _cached_json(url: str, fetch) -> dict:
    import json
    p = D._cache_path(url)
    if fetch is None and p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:                                              # noqa: BLE001
            pass
    d = D._get(url, 90) if fetch is None else fetch(url)
    if fetch is None and "error" not in d:
        try:
            p.write_text(json.dumps(d))
        except Exception:                                              # noqa: BLE001
            pass
    return d


def nearest_station(lat: float, lon: float, max_km: float = 150.0, fetch=None, min_years: int = 0) -> Optional[dict]:
    """Closest CO-OPS water-level station within max_km (None outside the US coast)."""
    st = _cached_json(COOPS_STATIONS, fetch)
    best = None
    for s in st["stations"]:
        d = float(haversine_km(lat, lon, float(s["lat"]), float(s["lng"])))
        if d <= max_km and (best is None or d < best["km"]):
            best = {"id": s["id"], "name": s["name"], "lat": float(s["lat"]), "lon": float(s["lng"]), "km": d}
    return best


def monthly(station: str, y0: int, y1: int, fetch=None) -> list:
    """Monthly mean sea level and highest water level (metres above MSL), one record per month with data."""
    url = f"{COOPS}?product=monthly_mean&application=ClimRisk&begin_date={y0}0101&end_date={y1}1231&datum=MSL&station={station}&time_zone=gmt&units=metric&format=json"
    d = _cached_json(url, fetch)
    if "error" in d:
        raise RuntimeError("NOAA CO-OPS: " + str(d["error"].get("message", d["error"]))[:120])
    out = []
    for r in d.get("data", []):
        try:
            out.append({"year": int(r["year"]), "month": int(r["month"]), "msl": float(r["MSL"]), "highest": float(r["highest"]) if r.get("highest") not in ("", None) else None})
        except (ValueError, KeyError):
            continue
    return out


def annual(station: str, y0: int, y1: int, fetch=None, min_months: int = 10) -> dict:
    """Annual mean sea level (mm) and annual maximum water level (m above MSL) for years with at least min_months of monthly data."""
    rows = monthly(station, y0, y1, fetch)
    by: dict = {}
    for r in rows:
        by.setdefault(r["year"], []).append(r)
    years, msl, hi = [], [], []
    for y in sorted(by):
        rs = by[y]
        if len(rs) >= min_months:
            years.append(y); msl.append(1000.0 * float(np.mean([r["msl"] for r in rs])))
            h = [r["highest"] for r in rs if r["highest"] is not None]
            hi.append(max(h) if h else float("nan"))
    return {"years": years, "msl_mm": msl, "highest_m": hi}


def window(station: str, t0: str, t1: str, fetch=None) -> dict:
    """Hourly observed and predicted (tide-only) water level between two YYYYMMDD dates (<= 1 year). Surge = observed - predicted."""
    ob = _cached_json(f"{COOPS}?product=hourly_height&application=ClimRisk&begin_date={t0}&end_date={t1}&datum=MSL&station={station}&time_zone=gmt&units=metric&format=json", fetch)
    pr = _cached_json(f"{COOPS}?product=predictions&application=ClimRisk&begin_date={t0}&end_date={t1}&datum=MSL&station={station}&time_zone=gmt&units=metric&interval=h&format=json", fetch)
    o = {r["t"]: float(r["v"]) for r in ob.get("data", []) if r.get("v") not in ("", None)}
    p = {r["t"]: float(r["v"]) for r in pr.get("predictions", []) if r.get("v") not in ("", None)}
    ts = sorted(set(o) & set(p))
    return {"t": ts, "observed": [o[t] for t in ts], "predicted": [p[t] for t in ts], "surge": [o[t] - p[t] for t in ts]}
