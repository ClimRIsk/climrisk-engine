"""Observed extreme-heat exposure at a point: counts of very hot days from the ERA5 reanalysis daily maximum temperature.

Why this exists: the older heat zone was a land-cover and latitude indicator and rated Phoenix, Riyadh and Kuwait City only MODERATE.
This reads a real temperature record (Open-Meteo archive of ECMWF ERA5, 1994-2023, cached on disk) and rates from how often the
site exceeds fixed, documented thresholds. It is a historical-climate screen, not a projection, and a 0.1-0.25 degree reanalysis
cell, so it can differ from a nearby weather station by a few degrees (urban heat islands and coasts especially).

Thresholds (days per year, 30-year mean):
  35 C  - widely used outdoor-work and infrastructure stress marker (ISO 7243 / national heat-health plans start near here)
  40 C  - severe: equipment derating, rail/road buckling, labour stoppage rules
Rating: CRITICAL >=60 days at 40 C or >=150 days at 35 C; HIGH >=10 days at 40 C or >=45 days at 35 C;
        MODERATE >=15 days at 35 C; LOW >=3 days at 35 C; NEGLIGIBLE otherwise.
"""
from __future__ import annotations

import hashlib
import json
import math
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Optional

API = "https://archive-api.open-meteo.com/v1/archive"
Y0, Y1 = 1994, 2023
CACHE = Path.home() / ".cache" / "climrisk" / "heat"


def _get(url: str, timeout: float = 60.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "ClimRisk/1.0 (climate risk research)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def fetch_tmax(lat: float, lon: float, fetch: Optional[Callable] = None) -> dict:
    q = {"latitude": round(lat, 2), "longitude": round(lon, 2), "start_date": f"{Y0}-01-01", "end_date": f"{Y1}-12-31",
         "daily": "temperature_2m_max", "timezone": "UTC"}
    url = API + "?" + urllib.parse.urlencode(q)
    p = CACHE / (hashlib.md5(url.encode()).hexdigest() + ".json")
    if fetch is None and p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:                                      # noqa: BLE001
            pass
    d = (fetch or _get)(url)
    if "error" in d:
        raise RuntimeError(str(d.get("reason", d["error"]))[:200])
    if fetch is None:
        try:
            CACHE.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(d, separators=(",", ":")))
        except Exception:                                      # noqa: BLE001
            pass
    return d


def rate(d35: float, d40: float) -> str:
    if d40 >= 60 or d35 >= 150:
        return "CRITICAL"
    if d40 >= 10 or d35 >= 45:
        return "HIGH"
    if d35 >= 15:
        return "MODERATE"
    if d35 >= 3:
        return "LOW"
    return "NEGLIGIBLE"


def site_heat(lat: float, lon: float, fetch: Optional[Callable] = None) -> dict:
    """Returns {available, rating, days_ge_35, days_ge_40, annual_max_mean, hottest_year_max, years, ...}. available False on any read failure."""
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return {"available": False, "reason": "non-finite coordinates"}
    try:
        d = fetch_tmax(lat, lon, fetch)
        t = d["daily"]["temperature_2m_max"]
        days = d["daily"]["time"]
    except Exception as e:                                     # noqa: BLE001
        return {"available": False, "reason": f"temperature record could not be read ({type(e).__name__})"}
    by: dict = {}
    for day, v in zip(days, t):
        if v is None or not math.isfinite(float(v)):
            continue
        b = by.setdefault(int(day[:4]), [0, 0, -1e9, 0])
        b[3] += 1
        b[0] += float(v) >= 35.0
        b[1] += float(v) >= 40.0
        b[2] = max(b[2], float(v))
    ys = [y for y, b in by.items() if b[3] >= 330]             # years with at least 330 valid days
    if len(ys) < 15:
        return {"available": False, "reason": "temperature record too short or gappy at this point"}
    d35 = sum(by[y][0] for y in ys) / len(ys)
    d40 = sum(by[y][1] for y in ys) / len(ys)
    mx = [by[y][2] for y in ys]
    return {"available": True, "rating": rate(d35, d40), "days_ge_35": round(d35, 1), "days_ge_40": round(d40, 1),
            "annual_max_mean_c": round(sum(mx) / len(mx), 1), "record_max_c": round(max(mx), 1), "years": f"{min(ys)}-{max(ys)}", "n_years": len(ys),
            "source": "ERA5 reanalysis daily maximum 2 m temperature via Open-Meteo archive"}
