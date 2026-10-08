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


# ── cached archive access (ERA5 / ERA5-Land via Open-Meteo) ──────────────────

def _cache_path(key: str):
    import hashlib, os
    from pathlib import Path
    d = Path(os.environ.get("CRI_CACHE_DIR", Path.home() / ".climrisk" / "cache")) / "modellab"
    d.mkdir(parents=True, exist_ok=True)
    return d / (hashlib.md5(key.encode()).hexdigest() + ".json")


def archive(lat: float, lon: float, y0: int, y1: int, daily: str = "", hourly: str = "", fetch: Callable = _get, cache: bool = True) -> dict:
    """ERA5/ERA5-Land time series. Results are cached on disk, so a hazard is only downloaded once per place."""
    q = {"latitude": round(lat, 3), "longitude": round(lon, 3), "start_date": f"{y0}-01-01", "end_date": f"{y1}-12-31", "timezone": "UTC"}
    if daily:
        q["daily"] = daily
    if hourly:
        q["hourly"] = hourly
    url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(q)
    p = _cache_path(url)
    if cache and p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:                                              # noqa: BLE001
            pass
    d = fetch(url, 150) if fetch is _get else fetch(url)
    if cache and "error" not in d:
        try:
            p.write_text(json.dumps(d))
        except Exception:                                              # noqa: BLE001
            pass
    if "error" in d:
        raise RuntimeError(str(d.get("reason", d["error"]))[:200])
    return d


def river_discharge(lat: float, lon: float, y0: int, y1: int, fetch: Callable = _get) -> dict:
    url = "https://flood-api.open-meteo.com/v1/flood?" + urllib.parse.urlencode({"latitude": round(lat, 3), "longitude": round(lon, 3), "start_date": f"{y0}-01-01", "end_date": f"{y1}-12-31", "daily": "river_discharge"})
    p = _cache_path(url)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:                                              # noqa: BLE001
            pass
    d = fetch(url, 120)
    try:
        p.write_text(json.dumps(d))
    except Exception:                                                  # noqa: BLE001
        pass
    return d


def usgs_gauge_near(lat: float, lon: float, y0: int, y1: int, radius_deg: float = 0.25, fetch: Callable = _get) -> Optional[dict]:
    """Nearest USGS stream gauge with daily discharge in the window (United States only). Returns None if there is none."""
    if not (24 <= lat <= 50 and -125 <= lon <= -66):
        return None
    bbox = f"{lon - radius_deg:.4f},{lat - radius_deg:.4f},{lon + radius_deg:.4f},{lat + radius_deg:.4f}"
    try:
        req = urllib.request.Request("https://waterservices.usgs.gov/nwis/site/?" + urllib.parse.urlencode({"format": "rdb", "bBox": bbox, "parameterCd": "00060", "siteType": "ST", "hasDataTypeCd": "dv", "siteStatus": "all"}), headers=_UA)
        txt = urllib.request.urlopen(req, timeout=40).read().decode()
    except Exception:                                                  # noqa: BLE001
        return None
    best = None
    for ln in txt.splitlines():
        if ln.startswith("#") or ln.startswith("agency_cd") or ln.startswith("5s"):
            continue
        c = ln.split("\t")
        if len(c) < 8:
            continue
        try:
            sla, slo = float(c[4]), float(c[5])
        except ValueError:
            continue
        d = ((sla - lat) ** 2 + (slo - lon) ** 2) ** 0.5
        if best is None or d < best[0]:
            best = (d, c[1], c[2], sla, slo)
    if not best:
        return None
    try:
        j = fetch("https://waterservices.usgs.gov/nwis/dv/?" + urllib.parse.urlencode({"format": "json", "sites": best[1], "parameterCd": "00060", "startDT": f"{y0}-01-01", "endDT": f"{y1}-12-31"}), 90)
        vals = j["value"]["timeSeries"][0]["values"][0]["value"]
        series = {v["dateTime"][:10]: float(v["value"]) * 0.0283168 for v in vals if v["value"] not in ("", None) and float(v["value"]) >= 0}   # ft³/s → m³/s
    except Exception:                                                  # noqa: BLE001
        return None
    if len(series) < 365 * 3:
        return None
    return {"site": best[1], "name": best[2], "lat": best[3], "lon": best[4], "km": best[0] * 111.0, "series": series, "source": "USGS National Water Information System (daily mean discharge)"}
