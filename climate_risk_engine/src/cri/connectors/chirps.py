"""CHIRPS v3 daily rainfall (satellite infrared plus rain-gauge blend, 0.05 degrees, 1981 to near present) for checking a known storm day.  (17 Nov 2026)

Why: ERA5 and NASA POWER are model-based area averages and can read far below a storm's point rainfall (Chennai, 1 Dec 2015: ERA5 about 101 mm, gauge 344.9 mm). CHIRPS blends
satellite cloud-top data with station gauges at 0.05 degrees (about 5.5 km), so it is a third, differently built view of the same day. It is NOT a replacement for a gauge, and one day
is a check, not a design rainfall: reading 40 years of annual maxima is 14,000 single-day reads (about 3 s each), which this module does not do.

Data: Climate Hazards Center, UC Santa Barbara, https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/sat/ (open; cite Funk et al. 2015 for CHIRPS and the CHC for v3). Each daily file
is a global GeoTIFF; only a small window is read with an HTTP range request.
"""
from __future__ import annotations

from typing import Callable, Optional

URL = "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/sat/{y}/chirps-v3.0.sat.{y}.{m}.{d}.tif"
ATTRIBUTION = "CHIRPS v3 (Climate Hazards Center, UC Santa Barbara): satellite and rain-gauge blended daily rainfall, 0.05 degrees. Funk et al. 2015, Scientific Data 2:150066 (CHIRPS)."


def _read(url: str, lat: float, lon: float, half_deg: float) -> dict:
    import rasterio
    from rasterio.windows import from_bounds
    with rasterio.Env(GDAL_HTTP_MAX_RETRY="3", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
        with rasterio.open("/vsicurl/" + url) as ds:
            r, c = ds.index(lon, lat)
            cell = float(ds.read(1, window=((r, r + 1), (c, c + 1)))[0, 0])
            a = ds.read(1, window=from_bounds(lon - half_deg, lat - half_deg, lon + half_deg, lat + half_deg, ds.transform), masked=True)
            nd = ds.nodata
    if nd is not None and cell == nd:
        cell = float("nan")
    return {"cell_mm": cell, "max_within_mm": float(a.max()) if a.size and not a.mask.all() else float("nan")}


def day_total(lat: float, lon: float, date: str, half_deg: float = 0.1, reader: Optional[Callable] = None) -> dict:
    """CHIRPS rainfall (mm) for one date 'YYYY-MM-DD' at the cell holding the point, and the maximum within +-half_deg. A failure is returned as a reason, never as zero."""
    y, m, d = date[:4], date[5:7], date[8:10]
    url = URL.format(y=y, m=m, d=d)
    try:
        r = (reader or _read)(url, lat, lon, half_deg)
    except Exception as e:                                           # noqa: BLE001
        return {"available": False, "date": date, "reason": f"{type(e).__name__}: {str(e)[:100]}", "url": url}
    if r["cell_mm"] != r["cell_mm"] or r["cell_mm"] < -900:          # NaN or CHIRPS fill value
        return {"available": False, "date": date, "reason": "no CHIRPS value at this cell (ocean, ice or no data)", "url": url}
    return {"available": True, "date": date, "cell_mm": round(r["cell_mm"], 1), "max_within_mm": round(r["max_within_mm"], 1), "half_deg": half_deg, "url": url, "source": ATTRIBUTION}
