"""
cryosphere.py — site-level cold-region hazards from open data.

Replaces the parametric regional scores for these perils with models that read a real
time series or a published map at the site, and say where they have and have not been
tested (docs/CRYOSPHERE_BENCHMARK.md, docs/VALIDATION_LEDGER.md).

  freeze_thaw(lat, lon)     frost days, ice days and freeze-thaw days per year from 20 years of ERA5
                            daily temperature (Open-Meteo archive). Tested against GHCN-Daily stations.
  avalanche_terrain(lat, lon)  share of the surrounding terrain that can release a dry-snow slab
                            (28-50 degrees) and where the steepest part faces. Tested against the SLF
                            Swiss avalanche-accident record. It is a TERRAIN screen: no snowpack, no runout.
  permafrost(lat, lon)      permafrost probability from the published Obu et al.
                            (2019) map, resampled to 10 km. Not tested here; a published product.

  glof_history(lat, lon)    reported glacial lake outburst floods near the site (Luetzow & Veh database). A record
                            of what happened, not a hazard model; no unburst lakes, no probability.

Not covered: blade icing (no open icing observations to test against), avalanche runout and snowpack,
permafrost thaw settlement. Those stay "Not assessed".
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Optional

import numpy as np

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FIRST_YEAR = 2004
LAST_YEAR = 2023            # fixed 20-year window so results reproduce; the window is stated in every result

# Screening bands for freeze-thaw days per year. Judgement, not calibrated to damage: they only order sites.
FT_BANDS = ((10.0, "Low"), (40.0, "Moderate"), (80.0, "High"), (1e9, "Very high"))


def _cache_dir() -> Path:
    d = Path(os.environ.get("CRI_CACHE_DIR", Path.home() / ".climrisk" / "cache")) / "cryosphere"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get(url: str, tries: int = 3, timeout: int = 40) -> dict:
    last: Optional[Exception] = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "climrisk-engine/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as e:                      # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"could not read {url.split('?')[0]}: {last}")


# --------------------------------------------------------------------------- freeze-thaw

def freeze_thaw_counts(tmin, tmax) -> dict:
    """Counts for one record of daily min/max air temperature (deg C, None or NaN = missing).
    frost day: tmin < 0.  ice day: tmax < 0.  freeze-thaw day: tmin < 0 and tmax > 0."""
    a = np.array([np.nan if v is None else v for v in tmin], dtype=float)
    b = np.array([np.nan if v is None else v for v in tmax], dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    return {"days": int(ok.sum()),
            "frost_days": int(((a < 0) & ok).sum()),
            "ice_days": int(((b < 0) & ok).sum()),
            "freeze_thaw_days": int(((a < 0) & (b > 0) & ok).sum())}


def freeze_thaw_band(ft_days_per_year: Optional[float]) -> Optional[str]:
    if ft_days_per_year is None:
        return None
    for edge, name in FT_BANDS:
        if ft_days_per_year < edge:
            return name
    return None


def daily_temperature(lat: float, lon: float, y0: int = FIRST_YEAR, y1: int = LAST_YEAR,
                      fetch: Callable[[str], dict] = _get) -> dict:
    q = urllib.parse.urlencode({"latitude": round(lat, 3), "longitude": round(lon, 3),
                                "start_date": f"{y0}-01-01", "end_date": f"{y1}-12-31",
                                "daily": "temperature_2m_min,temperature_2m_max", "timezone": "UTC"})
    f = _cache_dir() / f"daily_{lat:.2f}_{lon:.2f}_{y0}_{y1}.json"
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:                           # noqa: BLE001
            pass
    d = fetch(f"{ARCHIVE}?{q}")
    if "daily" not in d:
        raise RuntimeError("archive returned no daily series: " + str(d)[:120])
    out = {"time": d["daily"]["time"], "tmin": d["daily"]["temperature_2m_min"], "tmax": d["daily"]["temperature_2m_max"],
           "grid_elevation_m": d.get("elevation")}
    f.write_text(json.dumps(out))
    return out


def freeze_thaw(lat: float, lon: float, fetch: Callable[[str], dict] = _get) -> dict:
    """Freeze-thaw exposure at a point. Raises if the temperature record cannot be read (never returns a neutral value)."""
    d = daily_temperature(lat, lon, fetch=fetch)
    years = {}
    for t, a, b in zip(d["time"], d["tmin"], d["tmax"]):
        years.setdefault(t[:4], ([], []))
        years[t[:4]][0].append(a)
        years[t[:4]][1].append(b)
    per = {y: freeze_thaw_counts(v[0], v[1]) for y, v in sorted(years.items())}
    full = [c for c in per.values() if c["days"] >= 350]
    if len(full) < 10:
        raise RuntimeError("fewer than 10 complete years in the temperature record")
    m = lambda k: round(float(np.mean([c[k] for c in full])), 1)          # noqa: E731
    ft = [c["freeze_thaw_days"] for c in full]
    ftd = m("freeze_thaw_days")
    return {
        "frost_days_per_year": m("frost_days"), "ice_days_per_year": m("ice_days"), "freeze_thaw_days_per_year": ftd,
        "freeze_thaw_days_p10_p90": [round(float(np.percentile(ft, 10)), 0), round(float(np.percentile(ft, 90)), 0)],
        "band": freeze_thaw_band(ftd), "years": [FIRST_YEAR, LAST_YEAR], "complete_years": len(full),
        "grid_elevation_m": d.get("grid_elevation_m"),
        "source": "ERA5 reanalysis daily temperature, Open-Meteo archive",
        "basis": ("Freeze-thaw day = daily minimum below 0 and daily maximum above 0 degrees C, averaged over 2004-2023. "
                  "Air temperature at a reanalysis cell, not ground or surface temperature; bands are screening judgement, not damage thresholds."),
    }


# --------------------------------------------------------------------------- avalanche terrain

RELEASE_MIN_DEG, RELEASE_MAX_DEG = 28.0, 50.0     # dry-slab release slopes (Schweizer et al. 2003; Maggioni & Gruber 2003)
ASPECTS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


def slope_aspect(dem: np.ndarray, cell_m: float):
    """Slope (deg) and aspect (deg clockwise from north, direction the slope FACES). dem rows run north to south."""
    dz_dr, dz_dc = np.gradient(dem, cell_m, cell_m)           # per metre southwards, eastwards
    slope = np.degrees(np.arctan(np.hypot(dz_dr, dz_dc)))
    aspect = (np.degrees(np.arctan2(-dz_dc, dz_dr)) + 360.0) % 360.0   # downslope direction: east component = -dz/dc, north = +dz/dr (rows go south)
    return slope, aspect


def aspect_name(deg: float) -> str:
    return ASPECTS[int(((deg + 22.5) % 360) // 45)]


def release_zone_share(dem: np.ndarray, cell_m: float, centre: tuple, radius_m: float) -> dict:
    """Share of cells within radius_m of `centre` (row, col) whose slope is in the release range, plus the mean slope of those cells."""
    slope, aspect = slope_aspect(dem, cell_m)
    rr, cc = np.indices(dem.shape)
    inside = np.hypot((rr - centre[0]) * cell_m, (cc - centre[1]) * cell_m) <= radius_m
    valid = inside & np.isfinite(slope)
    if not valid.any():
        return {"share": None, "cells": 0}
    rel = valid & (slope >= RELEASE_MIN_DEG) & (slope <= RELEASE_MAX_DEG)
    out = {"share": float(rel.sum() / valid.sum()), "cells": int(valid.sum()), "release_cells": int(rel.sum()),
           "max_slope_deg": float(np.nanmax(slope[valid]))}
    if rel.any():
        a = np.radians(aspect[rel])
        out["dominant_aspect"] = aspect_name(float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360.0))
    return out


AVALANCHE_TEST = {"data": "SLF Swiss avalanche accidents since 1970/71 (backcountry and off-piste, start zone located within 100 m), 300 events",
                  "controls": "300 points 2-8 km from each event; 77 of them within 200 m of the event's elevation",
                  "auc_150m_share": 0.62, "auc_150m_share_elevation_matched": 0.59, "auc_ci_95_elevation_matched": [0.52, 0.66],
                  "start_zones_with_release_slope_within_60m": 0.37,
                  "reading": ("Weak skill: start zones have slightly more 28-50 degree terrain around them than nearby terrain, but the screen misses most "
                              "start zones and cannot tell them from the steep slopes around them. Accident locations are where people go, not where "
                              "snow slides.")}


def avalanche_terrain(lat: float, lon: float, radius_m: float = 150.0, elevation_grid=None) -> dict:
    """Avalanche release-terrain screen. Returns terrain facts and a band; says what it does not cover.
    The 150 m radius and 10 m grid are the settings that were tested (AVALANCHE_TEST)."""
    if elevation_grid is None:
        from .connectors.terrain_tiles import get_elevation_grid
        elevation_grid = get_elevation_grid(lat, lon, radius_km=0.5, zoom=14)
    g = elevation_grid
    dem, cell, centre = np.asarray(g.elevations, dtype=float), float(g.cell_size_m), (g.asset_row, g.asset_col)
    if cell < 7.0:                              # LiDAR: slope at 1 m picks up walls and embankments; the screen was tested on ~10 m cells
        k = int(round(10.0 / cell))
        h, w = (dem.shape[0] // k) * k, (dem.shape[1] // k) * k
        dem = np.nanmean(dem[:h, :w].reshape(h // k, k, w // k, k), axis=(1, 3))
        cell, centre = cell * k, (min(centre[0] // k, dem.shape[0] - 1), min(centre[1] // k, dem.shape[1] - 1))
    r = release_zone_share(dem, cell, centre, radius_m)
    if r["share"] is None:
        raise RuntimeError("no valid elevation cells around the site")
    sh = r["share"]
    band = "None" if sh < 0.02 else "Low" if sh < 0.10 else "Moderate" if sh < 0.25 else "High"
    return {"release_terrain_share": round(sh, 3), "radius_m": radius_m, "cell_m": round(cell, 1),
            "max_slope_deg": round(r["max_slope_deg"], 1), "dominant_release_aspect": r.get("dominant_aspect"),
            "band": band, "source": getattr(g, "source", "terrain tiles"), "tested": AVALANCHE_TEST,
            "basis": (f"Share of terrain within {int(radius_m)} m with slopes of {RELEASE_MIN_DEG:.0f}-{RELEASE_MAX_DEG:.0f} degrees, the range in which dry-snow slabs release. "
                      "Terrain only: no snowpack, snow depth, forest cover or runout path, and no risk from avalanches that start outside the radius. "
                      "Bands are screening judgement.")}


# --------------------------------------------------------------------------- glacial lake outburst: record of past events

_GLOF = None


def _glof_table():
    global _GLOF
    if _GLOF is None:
        import csv
        import gzip
        p = Path(__file__).parent / "data" / "glof_events_v4_2.csv.gz"
        if not p.exists():
            raise RuntimeError("GLOF event table is not installed (data/glof_events_v4_2.csv.gz)")
        with gzip.open(p, "rt", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["lat"], r["lon"] = float(r["lat"]), float(r["lon"])
        _GLOF = rows
    return _GLOF


def glof_history(lat: float, lon: float, radius_km: float = 50.0, limit: int = 10) -> dict:
    """Reported glacial lake outburst floods within radius_km of a point (Luetzow and Veh, global GLOF database v4.2, CC BY 4.0).
    This is a RECORD, not a hazard model: it says what has been reported near the site. An empty list means no reported event
    in the database, not that no glacial lake exists upstream; lakes that have not yet burst are not in it. Distance is to the
    lake, not to the flood path, which can travel far further down a valley."""
    rows = _glof_table()
    la0, lo0 = np.radians(lat), np.radians(lon)
    hits = []
    for r in rows:
        la, lo = np.radians(r["lat"]), np.radians(r["lon"])
        a = np.sin((la - la0) / 2) ** 2 + np.cos(la0) * np.cos(la) * np.sin((lo - lo0) / 2) ** 2
        km = 6371.0 * 2 * np.arcsin(np.sqrt(a))
        if km <= radius_km:
            hits.append((km, r))
    hits.sort(key=lambda t: t[0])
    lakes = {(r["lake"] or r["glacier"] or f"{r['lat']:.3f},{r['lon']:.3f}") for _, r in hits}
    years = [int(r["year"][:4]) for _, r in hits if r.get("year") and r["year"][:4].isdigit()]

    def fat(v):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return 0
    return {"radius_km": radius_km, "events": len(hits), "lakes": len(lakes),
            "first_year": min(years) if years else None, "latest_year": max(years) if years else None,
            "reported_fatalities": sum(fat(r.get("fatalities")) for _, r in hits),
            "nearest": [{"lake": r["lake"] or None, "glacier": r["glacier"] or None, "country": r["country"], "distance_km": round(float(km), 1),
                         "date": r["date"] or None, "mechanism": r["mechanism"] or None,
                         "peak_discharge_m3s": round(float(r["qp_m3s"]), 0) if r.get("qp_m3s") not in (None, "") else None}
                        for km, r in hits[:limit]],
            "source": "Luetzow and Veh, Global GLOF database v4.2 (Zenodo 7330344; Earth Syst. Sci. Data 15, 2983, 2023), CC BY 4.0",
            "basis": ("Reported past events only. The database is compiled from literature and news, so remote events are under-reported; "
                      "it contains no unburst lakes and gives no probability.")}


# --------------------------------------------------------------------------- permafrost

_PF = None


def _pf_grid():
    global _PF
    if _PF is None:
        p = Path(__file__).parent / "data" / "permafrost_obu2019_10km.npz"
        if not p.exists():
            raise RuntimeError("permafrost grid is not installed (data/permafrost_obu2019_10km.npz)")
        z = np.load(p)
        _PF = {k: z[k] for k in z.files}
    return _PF


def permafrost(lat: float, lon: float) -> dict:
    """Permafrost probability from the Obu et al. (2019) map (2000-2016).
    Outside the map (south of 25 N, southern hemisphere, ocean) returns covered=False; never a zero."""
    g = _pf_grid()
    from pyproj import Transformer
    x, y = Transformer.from_crs(4326, 3995, always_xy=True).transform(lon, lat)
    res_x, res_y = float(g["res_x"]), float(g["res_y"])
    col = int((x - float(g["x0"])) / res_x)
    row = int((float(g["y0"]) - y) / res_y)
    prob = g["prob"]
    if lat < 25.0 or not (0 <= row < prob.shape[0] and 0 <= col < prob.shape[1]):
        return {"covered": False, "reason": "outside the map's northern-hemisphere extent (25 N and above)"}
    v = int(prob[row, col])
    if v == 255:
        return {"covered": False, "reason": "no permafrost estimate at this cell (water, glacier or outside land mask)"}
    p = v / 100.0
    zone = "Continuous" if p >= 0.9 else "Discontinuous" if p >= 0.5 else "Sporadic" if p >= 0.1 else "Isolated or none"
    return {"covered": True, "permafrost_probability": round(p, 2), "zone": zone,
            "resolution_km": round((res_x + res_y) / 2000.0, 1), "period": "2000-2016",
            "source": "Obu et al. (2019), Earth-Science Reviews 193, TTOP model, PANGAEA 10.1594/PANGAEA.888600 (CC BY 3.0), resampled to 10 km",
            "basis": ("Published model map, not measured at the site. Mountain and valley permafrost varies over tens of metres and a 10 km cell averages it. "
                      "Probability is the model's, not a count of boreholes. Ground ice content and thaw settlement are not given.")}
