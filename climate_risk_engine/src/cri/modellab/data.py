"""Data access for studies: free, key-less sources, every call injectable for tests and every failure explicit."""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Callable, Optional

_UA = {"User-Agent": "ClimRisk-ModelLab/1.0"}


def _get(url: str, timeout: int = 40) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=timeout) as r:
        return json.loads(r.read())


def annual_max_daily_rain(lat: float, lon: float, y0: int = 1991, y1: int = 2020, fetch: Callable = _get) -> dict:
    """ERA5 daily precipitation (Open-Meteo archive) → annual maxima (mm/day). ERA5 smooths extremes, so this is a floor."""
    url = ("https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(
        {"latitude": round(lat, 4), "longitude": round(lon, 4), "start_date": f"{y0}-01-01", "end_date": f"{y1}-12-31", "daily": "precipitation_sum", "timezone": "UTC"}))
    d = fetch(url)
    days, vals = d["daily"]["time"], d["daily"]["precipitation_sum"]
    best: dict = {}
    for t, v in zip(days, vals):
        if v is None:
            continue
        y = int(t[:4]); best[y] = max(best.get(y, 0.0), float(v))
    if len(best) < 10:
        raise RuntimeError(f"only {len(best)} years of rainfall returned")
    return {"years": sorted(best), "maxima_mm": [best[y] for y in sorted(best)], "source": "ERA5 reanalysis via Open-Meteo archive", "url": url}


def hourly_met_and_pm(lat: float, lon: float, days: int = 21, fetch: Callable = _get) -> dict:
    """Hourly wind, boundary-layer height (forecast API, past days) and CAMS PM2.5 (air-quality API)."""
    q = {"latitude": round(lat, 4), "longitude": round(lon, 4), "past_days": days, "forecast_days": 1, "timezone": "UTC"}
    met = fetch("https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode({**q, "hourly": "wind_speed_10m,boundary_layer_height,cloud_cover,is_day", "wind_speed_unit": "ms"}))
    aq = fetch("https://air-quality-api.open-meteo.com/v1/air-quality?" + urllib.parse.urlencode({**q, "hourly": "pm2_5"}))
    tm, ta = met["hourly"]["time"], aq["hourly"]["time"]
    pm = dict(zip(ta, aq["hourly"]["pm2_5"]))
    rows = []
    for i, t in enumerate(tm):
        p = pm.get(t)
        u, h = met["hourly"]["wind_speed_10m"][i], met["hourly"]["boundary_layer_height"][i]
        if p is None or u is None or h is None:
            continue
        rows.append({"t": t, "wind": float(u), "blh": float(h), "cloud": float(met["hourly"]["cloud_cover"][i] or 0) / 100, "day": bool(met["hourly"]["is_day"][i]), "pm25": float(p)})
    if len(rows) < 72:
        raise RuntimeError(f"only {len(rows)} usable hours of meteorology and PM2.5 returned")
    return {"rows": rows, "source": "Open-Meteo: ECMWF/GFS meteorology and CAMS (Copernicus) PM2.5 analysis"}
