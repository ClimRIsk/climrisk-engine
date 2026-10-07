"""
cri/wind_extreme.py — all-cause extreme wind at a point (extratropical storms, monsoon squalls, cyclones as the reanalysis sees them).

Source   ERA5 reanalysis via the Open-Meteo archive API: daily maximum 10 m wind gust, 1950 onward, 0.25° grid.
Method   annual maxima → Gumbel (EV1) by L-moments → return levels for 10, 50, 100 and 250 years; the largest daily
         gusts on record with date and dominant wind direction.

Limits (stated in the output):
  • ERA5 is a 0.25° (~25 km) model. It smooths local peaks and under-represents tropical-cyclone winds, thunderstorm
    outflows and tornadoes. Treat the result as a floor for ordinary storm winds, not a ceiling.
  • 75 annual maxima support the 10- and 50-year levels; 100 and 250 years are extrapolations and are flagged.
  • The cyclone model (cri.climate.tropical_cyclone) remains the better source where tropical cyclones dominate.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.request
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np

CACHE = Path(os.environ.get("CRI_CACHE_DIR", Path.home() / ".climrisk" / "cache")) / "wind"
API = "https://archive-api.open-meteo.com/v1/archive"
EULER = 0.5772156649015329
RPS = (10, 50, 100, 250)


def fetch_daily(lat: float, lon: float, start_year: int = 1950, end_year: Optional[int] = None, timeout: float = 40.0) -> dict:
    end_year = end_year or date.today().year - 1
    key = CACHE / (hashlib.md5(f"{lat:.2f}{lon:.2f}{start_year}{end_year}".encode()).hexdigest() + ".json")
    if key.exists():
        try:
            return json.loads(key.read_text())
        except Exception:                                      # noqa: BLE001
            pass
    url = (f"{API}?latitude={lat:.4f}&longitude={lon:.4f}&start_date={start_year}-01-01&end_date={end_year}-12-31"
           "&daily=wind_gusts_10m_max,wind_speed_10m_max,wind_direction_10m_dominant&wind_speed_unit=ms")
    req = urllib.request.Request(url, headers={"User-Agent": "ClimRisk/1.0 (climate risk research)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    try:
        CACHE.mkdir(parents=True, exist_ok=True)
        key.write_text(json.dumps(out, separators=(",", ":")))
    except Exception:                                          # noqa: BLE001
        pass
    return out


def gumbel_lmom(x: np.ndarray) -> tuple:
    """Gumbel location and scale from the first two L-moments."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    b0 = x.mean()
    b1 = float(np.sum((np.arange(n) / (n - 1)) * x) / n)
    l2 = 2 * b1 - b0
    a = l2 / math.log(2.0)
    return b0 - EULER * a, a


def return_level(xi: float, a: float, T: float) -> float:
    return xi - a * math.log(-math.log(1.0 - 1.0 / T))


def analyse(daily: dict) -> dict:
    d = daily.get("daily") or {}
    t, g = d.get("time") or [], d.get("wind_gusts_10m_max") or []
    if not t or not g:
        return {"available": False, "reason": "no ERA5 series returned"}
    spd, dr = d.get("wind_speed_10m_max") or [None] * len(t), d.get("wind_direction_10m_dominant") or [None] * len(t)
    by = {}
    for i, (ti, gi) in enumerate(zip(t, g)):
        if gi is None:
            continue
        y = int(ti[:4])
        if y not in by or gi > by[y][0]:
            by[y] = (float(gi), ti, spd[i], dr[i])
    years = sorted(by)
    if len(years) < 20:
        return {"available": False, "reason": f"only {len(years)} years of data"}
    am = np.array([by[y][0] for y in years])
    xi, a = gumbel_lmom(am)
    n = len(years)
    rp = {T: round(float(return_level(xi, a, T)), 1) for T in RPS}
    # rough sampling spread of the Gumbel level (delta method on the 2 parameters, Kotz & Nadarajah)
    se = {T: round(float(a) * math.sqrt(1.109 + 0.514 * (-math.log(-math.log(1 - 1 / T))) + 0.608 * (-math.log(-math.log(1 - 1 / T))) ** 2) / math.sqrt(n), 1)
          for T in RPS}
    top = sorted(((v[0], v[1], v[2], v[3]) for v in by.values()), reverse=True)[:8]
    events = [{"date": dt, "year": int(dt[:4]), "gust_ms": round(gu, 1), "mean_wind_ms": None if sp is None else round(float(sp), 1),
               "from_deg": None if di is None else round(float(di), 0), "wind_to_deg": None if di is None else round((float(di) + 180) % 360, 0)}
              for gu, dt, sp, di in top]
    return {"available": True, "source": "ERA5 reanalysis via Open-Meteo archive (daily max 10 m gust, 0.25° grid)",
            "years": f"{years[0]}–{years[-1]}", "n_years": n, "method": "annual maxima, Gumbel by L-moments",
            "rp_gust_ms": rp, "rp_std_error_ms": se, "extrapolated": [T for T in RPS if T > n * 1.3],
            "annual_max_gust_ms": {str(y): round(by[y][0], 1) for y in years}, "top_events": events,
            "mean_annual_max_ms": round(float(am.mean()), 1),
            "caveat": "ERA5 smooths local peaks (0.25° ≈ 25 km) and under-represents tropical cyclones, thunderstorm outflows and "
                      "tornadoes: read these as ordinary-storm winds and a floor, not a ceiling."}


def site_extreme_wind(lat: float, lon: float, _daily: Optional[dict] = None) -> dict:
    try:
        return analyse(_daily if _daily is not None else fetch_daily(lat, lon))
    except Exception as exc:                                   # noqa: BLE001
        return {"available": False, "reason": f"{type(exc).__name__}: {str(exc)[:120]}"}
