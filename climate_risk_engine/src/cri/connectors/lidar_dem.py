"""National airborne-LiDAR bare-earth terrain, used instead of the global 10-30 m tiles wherever it is open.

Covered today (all keyless, open licence, GeoTIFF):
  England  - Environment Agency LIDAR Composite DTM 1 m (Open Government Licence v3), EPSG:27700
  Netherlands - AHN DTM 0.5 m (PDOK / Rijkswaterstaat, CC0), EPSG:28992
  Contiguous United States - USGS 3DEP bare-earth DEM (public domain) through the 3DEP ImageServer, EPSG:5070. The service mosaics the best available
    source (1 m LiDAR-derived in most cities, 10-30 m elsewhere); the native resolution at the pin is read from the service and the grid is used only
    when it is 3 m or finer, otherwise the global tiles (which already carry 10 m 3DEP) are kept.
Anything else, any failure, or CRI_LIDAR=0 returns None and the caller keeps the global tiles. The grid says which source it came from.

Why it matters: LiDAR bare earth resolves kerbs, embankments, dikes and floodplain steps that 30 m global terrain smooths away, which is what decides
whether a river or surge reaches a building. The box is fetched at most ~800 cells wide (so 5-6 m cells at a 2.5 km radius, 1 m cells under 0.4 km).
The projected grid is treated as north-up (grid convergence is under 2 degrees in both countries).
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)
CACHE = Path.home() / ".cache" / "climrisk" / "lidar"
_FAILED: set = set()          # points confirmed not to have usable LiDAR (permanent for the process)
_SLOW: dict = {}               # source key -> time of the last total-deadline timeout: the whole service is skipped for 10 minutes so one slow service cannot add 45 s to every call
_DOWN: dict = {}               # points where a service call failed: retried after 120 s so a blip does not switch LiDAR off

THREEDEP = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer"
USA_BBOX = (24.4, 49.5, -125.0, -66.9)

SOURCES = {
    "england": {
        "label": "Environment Agency LIDAR Composite DTM 1 m (England, Open Government Licence v3)", "native_m": 1.0, "epsg": 27700,
        "bbox": (49.8, 55.9, -6.5, 1.9),
        "url": "https://environment.data.gov.uk/spatialdata/lidar-composite-digital-terrain-model-dtm-1m-2022/wcs",
        "cov": "13787b9a-26a4-4775-8523-806d13af58fc__Lidar_Composite_Elevation_DTM_1m", "axes": ("E", "N"),
        "url_dsm": "https://environment.data.gov.uk/spatialdata/lidar-composite-digital-surface-model-last-return-dsm-1m/wcs",
        "cov_dsm": "9ba4d5ac-d596-445a-9056-dae3ddec0178__Lidar_Composite_Elevation_LZ_DSM_1m",
    },
    "netherlands": {
        "label": "AHN DTM 0.5 m (Netherlands, Rijkswaterstaat, CC0)", "native_m": 0.5, "epsg": 28992,
        "bbox": (50.7, 53.6, 3.2, 7.4),
        "url": "https://service.pdok.nl/rws/ahn/wcs/v1_0", "cov": "dtm_05m", "axes": ("x", "y"), "url_dsm": "https://service.pdok.nl/rws/ahn/wcs/v1_0", "cov_dsm": "dsm_05m",
    },
}


def _which(lat: float, lon: float) -> Optional[str]:
    for k, s in SOURCES.items():
        a, b, c, d = s["bbox"]
        if a <= lat <= b and c <= lon <= d:
            return k
    if USA_BBOX[0] <= lat <= USA_BBOX[1] and USA_BBOX[2] <= lon <= USA_BBOX[3]:
        return "usa"
    return None


def _usa_native_m(lat: float, lon: float) -> Optional[float]:
    """Native ground sample distance (m) of the best 3DEP source at the point, or None if it cannot be read."""
    import json
    q = urllib.parse.urlencode({"geometry": json.dumps({"x": lon, "y": lat, "spatialReference": {"wkid": 4326}}), "geometryType": "esriGeometryPoint",
                                "returnCatalogItems": "true", "returnGeometry": "false", "f": "json"})
    d = json.loads(_get(THREEDEP + "/identify?" + q, 40))
    items = [f["attributes"] for f in (d.get("catalogItems") or {}).get("features", []) if f["attributes"].get("Category") == 1]
    if not items:
        return None
    best = min(items, key=lambda a: a.get("Best") if a.get("Best") is not None else 1e12)      # lowest rank number = the source the mosaic shows
    ps = best.get("LowPS")
    return float(ps) if ps else None


def _get(url: str, timeout: float = 45.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "ClimRisk/1.0 (climate risk research)"})
    import time as _t
    deadline = _t.time() + timeout                                # a hard TOTAL deadline: urlopen's timeout only bounds each socket read, so a slowly trickling service could hold a report for minutes
    with urllib.request.urlopen(req, timeout=min(timeout, 20.0)) as r:
        chunks = []
        while True:
            b = r.read(1 << 16)
            if not b:
                return b"".join(chunks)
            chunks.append(b)
            if _t.time() > deadline:
                raise TimeoutError(f"service too slow: no complete response within {timeout:g} s")


def _grid_from_bytes(raw: bytes, key: str, lat: float, lon: float, radius_km: float, centre_en: tuple, surface: bool = False):
    import rasterio
    from rasterio.io import MemoryFile
    from .terrain_tiles import ElevationGrid
    with MemoryFile(raw) as mf, mf.open() as ds:
        a = ds.read(1).astype(np.float64)
        t = ds.transform
        res = abs(t.a)
    a[(a < -500) | (a > 9000) | ~np.isfinite(a)] = np.nan
    if np.isnan(a).mean() > 0.85:
        return None
    row = int((t.f - centre_en[1]) / res)
    col = int((centre_en[0] - t.c) / res)
    if not (0 <= row < a.shape[0] and 0 <= col < a.shape[1]):
        return None
    if np.isnan(a[row, col]):                                   # pin on masked ground (water, or just outside the survey): nearest valid cell within 25 m
        k = max(1, int(25 / res))
        win = a[max(0, row - k): row + k + 1, max(0, col - k): col + k + 1]
        if np.isnan(win).all():
            return None
    src = SOURCES[key]
    g = ElevationGrid(lat=lat, lon=lon, radius_km=radius_km, cell_size_m=float(res), rows=a.shape[0], cols=a.shape[1], asset_row=row, asset_col=col, elevations=a)
    g.source = (src["label"].replace("DTM", "DSM") + " - surface model, buildings and trees included") if surface else src["label"]
    g.native_res_m = src["native_m"]
    g.epsg = src.get("epsg", 5070)
    g.centre_en = centre_en
    return g


def _fetch_cached(url: str, timeout: float = 90.0) -> bytes:
    p = CACHE / (hashlib.md5(url.encode()).hexdigest() + ".tif")
    if p.exists():
        return p.read_bytes()
    raw = _get(url, timeout)
    if raw[:4] not in (b"II*\x00", b"MM\x00*"):
        raise ValueError("service did not return a GeoTIFF: " + raw[:120].decode("utf-8", "replace"))
    try:
        CACHE.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
    except Exception:                                           # noqa: BLE001
        pass
    return raw


def try_grid(lat: float, lon: float, radius_km: float, surface: bool = False):
    """An ElevationGrid from open LiDAR, or None (not covered, disabled, or any failure). surface=True asks for the first/last-return surface model (England, Netherlands only)."""
    if os.environ.get("CRI_LIDAR", "1") == "0" or not (math.isfinite(lat) and math.isfinite(lon)):
        return None
    key = _which(lat, lon)
    if key is None:
        return None
    if surface and key not in ('england', 'netherlands'):
        return None
    ck = (key, round(lat, 3), round(lon, 3), round(radius_km, 2), bool(surface))
    if ck in _FAILED:
        return None
    import time as _t
    if _t.time() - _SLOW.get(key, -1e9) < 600:
        return None
    if _t.time() - _DOWN.get(ck, -1e9) < 120:
        return None
    try:
        from pyproj import Transformer
        half = radius_km * 1000.0
        if key == "usa":
            native = _usa_native_m(lat, lon)
            if native is None or native > 3.0:                  # only 10-30 m data here: the global tiles already carry it
                _FAILED.add(ck)
                return None
            e, n = Transformer.from_crs(4326, 5070, always_xy=True).transform(lon, lat)
            cell = max(native, 2.0 * half / 800.0)
            size = int(round(2.0 * half / cell))
            q = urllib.parse.urlencode({"bbox": f"{e - half},{n - half},{e + half},{n + half}", "bboxSR": 5070, "imageSR": 5070, "size": f"{size},{size}",
                                        "format": "tiff", "pixelType": "F32", "interpolation": "RSP_BilinearInterpolation", "f": "image"})
            raw = _fetch_cached(THREEDEP + "/exportImage?" + q)
            SOURCES["usa"] = {"label": f"USGS 3DEP bare-earth DEM, {native:g} m (US public domain)", "native_m": native}
            g = _grid_from_bytes(raw, "usa", lat, lon, radius_km, (e, n))
        else:
            s = SOURCES[key]
            e, n = Transformer.from_crs(4326, s["epsg"], always_xy=True).transform(lon, lat)
            target = max(s["native_m"], 2.0 * half / 800.0)
            sf = round(s["native_m"] / target, 4)
            ax, ay = s["axes"]
            x0, x1, y0, y1 = int(e - half), int(e + half), int(n - half), int(n + half)
            q = (f"service=WCS&version=2.0.1&request=GetCoverage&coverageId={urllib.parse.quote(s['cov_dsm'] if surface else s['cov'])}"
                 f"&subset={ax}({x0},{x1})&subset={ay}({y0},{y1})" + (f"&scalefactor={sf}" if sf < 1 else "") + "&format=image/tiff")
            raw = _fetch_cached((s["url_dsm"] if surface else s["url"]) + "?" + q, 45.0)
            g = _grid_from_bytes(raw, key, lat, lon, radius_km, (e, n), surface=surface)
        if g is None:
            _FAILED.add(ck)
        return g
    except Exception as exc:                                    # noqa: BLE001
        log.warning("LiDAR terrain unavailable at (%g, %g): %s; falling back to global tiles", lat, lon, str(exc)[:160])
        _DOWN[ck] = _t.time()
        if isinstance(exc, TimeoutError):
            _SLOW[key] = _t.time()
        return None


def sample_lonlat(g, lons, lats) -> np.ndarray:
    """Heights (m) of a LiDAR grid at geographic points (NaN where the grid does not reach or has no data). Nearest cell; the grid is north-up in its own projection."""
    from pyproj import Transformer
    e, n = Transformer.from_crs(4326, g.epsg, always_xy=True).transform(np.asarray(lons, float), np.asarray(lats, float))
    e0, n0 = g.centre_en
    col = np.round((e - e0) / g.cell_size_m + g.asset_col).astype(int)
    row = np.round(-(n - n0) / g.cell_size_m + g.asset_row).astype(int)
    out = np.full(np.shape(e), np.nan)
    ok = (row >= 0) & (row < g.elevations.shape[0]) & (col >= 0) & (col < g.elevations.shape[1])
    out[ok] = g.elevations[row[ok], col[ok]]
    return out
