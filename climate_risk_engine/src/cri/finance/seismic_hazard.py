"""Earthquake hazard at a point, from two published sources, labelled by which one was used.

1. Contiguous United States: USGS National Seismic Hazard Model 2018 (hazard curves from the USGS web service, site class B/C boundary Vs30 = 760 m/s).
   Peak ground acceleration with a 10 % chance of exceedance in 50 years (475-year return period) and 2 % in 50 years (2,475 years).
   This is ground SHAKING on rock, not building damage; soil amplification and structure type are not included.
2. Everywhere else: a SEISMICITY screen from the USGS ComCat catalogue: how many earthquakes of magnitude 5.5 or more were recorded within 150 km since
   1973 (the global catalogue is complete to about M5 from then). Observed history only: it misses faults that have not ruptured in 50 years and says
   nothing about ground motion at the site.

Rating (indicative; NEGLIGIBLE < LOW < MODERATE < HIGH < CRITICAL):
  PGA 475-year in g:   < 0.04 NEGLIGIBLE, < 0.08 LOW, < 0.16 MODERATE, < 0.32 HIGH, otherwise CRITICAL (the bands used by seismic design codes to separate very low,
                       low, moderate and high seismicity zones)
  Catalogue count:     0 NEGLIGIBLE, 1-3 LOW, 4-15 MODERATE, 16-60 HIGH, more than 60 CRITICAL
"""
from __future__ import annotations

import hashlib
import json
import math
import urllib.request
from pathlib import Path
from typing import Callable, Optional

CACHE = Path.home() / ".cache" / "climrisk" / "seismic"
NSHM = "https://earthquake.usgs.gov/ws/nshmp/conus-2018/dynamic/hazard/{lon}/{lat}/{vs30}"
COMCAT = "https://earthquake.usgs.gov/fdsnws/event/1/count?starttime=1973-01-01&minmagnitude=5.5&latitude={lat}&longitude={lon}&maxradiuskm=150"
CATALOGUE_YEARS = 53.0


def _get(url: str, timeout: float = 60.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "ClimRisk/1.0 (climate risk research)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8")


def _cached(url: str, fetch: Optional[Callable]) -> str:
    p = CACHE / (hashlib.md5(url.encode()).hexdigest() + ".txt")
    if fetch is None and p.exists():
        return p.read_text()
    txt = (fetch or _get)(url)
    if fetch is None:
        try:
            CACHE.mkdir(parents=True, exist_ok=True)
            p.write_text(txt)
        except Exception:                                       # noqa: BLE001
            pass
    return txt


def in_conus(lat: float, lon: float) -> bool:
    return 24.4 <= lat <= 50.0 and -125.0 <= lon <= -65.0


def pga_rating(pga475_g: float) -> str:
    return "NEGLIGIBLE" if pga475_g < 0.04 else "LOW" if pga475_g < 0.08 else "MODERATE" if pga475_g < 0.16 else "HIGH" if pga475_g < 0.32 else "CRITICAL"


def count_rating(n: int) -> str:
    return "NEGLIGIBLE" if n == 0 else "LOW" if n <= 3 else "MODERATE" if n <= 15 else "HIGH" if n <= 60 else "CRITICAL"


def _loglog(xs, ys, target):
    """Ground motion (xs, g) at which the annual exceedance rate (ys) equals target, by log-log interpolation. None if outside the curve."""
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if x > 0 and y > 0]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        t = math.log(target)
        if (y0 >= t >= y1) and y0 != y1:
            return math.exp(x0 + (t - y0) * (x1 - x0) / (y1 - y0))
    return None


def _pga_from_service(lat, lon, vs30, fetch):
    d = json.loads(_cached(NSHM.format(lon=round(lon, 3), lat=round(lat, 3), vs30=vs30), fetch))
    if d.get("status") != "success":
        raise RuntimeError("USGS hazard service returned status " + str(d.get("status")))
    for block in (d.get("response") or {}).get("hazardCurves", []):
        if (block.get("imt") or {}).get("value") == "PGA":
            for dat in block.get("data", []):
                if dat.get("component") == "Total":
                    xs, ys = dat["values"]["xs"], dat["values"]["ys"]
                    r475, r2475 = -math.log(1 - 0.10) / 50.0, -math.log(1 - 0.02) / 50.0
                    a475, a2475 = _loglog(xs, ys, r475), _loglog(xs, ys, r2475)
                    if a475 is None and ys[0] < r475:
                        a475 = 0.0                              # even the smallest tabulated shaking is rarer than the 475-year rate
                    return a475, a2475
    raise RuntimeError("PGA curve not found in the USGS response")


def site_seismic(lat: float, lon: float, vs30: int = 760, fetch: Optional[Callable] = None) -> dict:
    """{available, method, rating, ...}; available False with a reason when neither source can be read."""
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return {"available": False, "reason": "non-finite coordinates"}
    err = None
    if in_conus(lat, lon):
        try:
            a475, a2475 = _pga_from_service(lat, lon, vs30, fetch)
            if a475 is not None:
                return {"available": True, "method": "USGS NSHM 2018 hazard curve", "pga_475yr_g": round(a475, 3),
                        "pga_2475yr_g": None if a2475 is None else round(a2475, 3), "vs30_m_s": vs30, "rating": pga_rating(a475),
                        "source": "USGS National Seismic Hazard Model 2018 (peak ground acceleration on rock, not building damage)"}
        except Exception as e:                                   # noqa: BLE001
            err = f"{type(e).__name__}"
    try:
        n = int(_cached(COMCAT.format(lat=round(lat, 3), lon=round(lon, 3)), fetch).strip())
    except Exception as e:                                       # noqa: BLE001
        return {"available": False, "reason": f"earthquake data could not be read ({type(e).__name__}" + (f"; hazard model also failed: {err}" if err else "") + ")"}
    return {"available": True, "method": "seismicity screen (recorded earthquakes)", "m55_within_150km_since_1973": n,
            "rate_per_year": round(n / CATALOGUE_YEARS, 3), "rating": count_rating(n),
            "source": "USGS ComCat catalogue, magnitude 5.5 or more within 150 km, 1973 onward (observed history; not a ground-motion hazard)"}
