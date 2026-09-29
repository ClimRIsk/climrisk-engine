"""
finance/flood_hazard.py — flood depth by return period over an asset footprint.

River  JRC CEMS-GloFAS global river flood hazard maps v2.1 (LISFLOOD + LISFLOOD-FP,
       3 arc-sec ≈ 90 m), return periods 10–500 yr, UNDEFENDED. Cloud-optimised
       GeoTIFFs are range-read; permanent water bodies (river channel, lakes) are
       masked so a pin on the river does not count the channel as flooded land.
Coast  Deltares Global Flood Maps (GTSMip6 surge + tide, attenuated inland,
       NASADEM 90 m; Microsoft Planetary Computer), return periods 2–250 yr at 2018
       and 2050 sea level, UNDEFENDED. The engine ships a compact chunk index
       (cri/data/deltares_coastal_index.npz) and range-reads the one ~0.17° chunk
       around the site from the public blob (SAS token from Planetary Computer).

Defences are applied later from FLOPROS (finance/geo.py): events rarer than the
protection standard overtop, more frequent ones do not.

Results are cached on disk (~/.climrisk/cache/flood) — a site is fetched once.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

_DATA = Path(__file__).resolve().parent.parent / "data"
CACHE = Path(os.environ.get("CRI_CACHE_DIR", Path.home() / ".climrisk" / "cache")) / "flood"
PX_DEG = 1.0 / 1200.0                       # 3 arc-seconds
_GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                 GDAL_HTTP_TIMEOUT="30", GDAL_HTTP_MAX_RETRY="2", GDAL_HTTP_RETRY_DELAY="1")


@lru_cache(maxsize=1)
def _jrc() -> dict:
    return json.loads((_DATA / "jrc_flood_tiles.json").read_text())


@lru_cache(maxsize=1)
def _delt() -> dict:
    z = np.load(_DATA / "deltares_coastal_index.npz")
    return {k: z[k] for k in z.files}


def _tile_name(lat: float, lon: float) -> str:
    top = int(math.floor(lat / 10.0) * 10) + 10
    left = int(math.floor(lon / 10.0) * 10)
    ns = f"N{top}" if top >= 0 else f"S{abs(top)}"
    ew = f"E{left}" if left > 0 else f"W{abs(left)}"          # JRC names the 0–10°E column "W0"
    return f"{ns}_{ew}"


def _disk(lat, lon, r_px):
    """Pixel offsets inside a footprint radius (in pixels) around the centre pixel."""
    n = int(math.ceil(r_px))
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    return (yy ** 2 + xx ** 2) <= max(r_px, 0.5) ** 2, n


def _read_jrc(url: str, lat: float, lon: float, n: int) -> Optional[np.ndarray]:
    import rasterio
    from rasterio.windows import Window
    with rasterio.Env(**_GDAL_ENV):
        with rasterio.open("/vsicurl/" + url) as ds:
            r, c = ds.index(lon, lat)
            a = ds.read(1, window=Window(c - n, r - n, 2 * n + 1, 2 * n + 1), boundless=True,
                        fill_value=ds.nodata if ds.nodata is not None else 0)
            nod = ds.nodata
    a = a.astype(np.float32)
    if nod is not None:
        a[a == nod] = np.nan
    return a


def river_depths(lat: float, lon: float, radius_m: float = 250.0) -> dict:
    j = _jrc()
    name = _tile_name(lat, lon)
    tid = j["tiles"].get(name)
    out = {"source": j["source"], "rps": j["rps"], "radius_m": radius_m, "tile": name}
    if tid is None:
        out.update(available=False, reason="no JRC tile (ocean or outside coverage)")
        return out
    r_px = radius_m / (111320.0 * PX_DEG)
    mask, n = _disk(lat, lon, r_px)
    base = j["base"]
    urls = {rp: f"{base}RP{rp}/ID{tid}_{name}_RP{rp}_depth.tif" for rp in j["rps"]}
    urls["pw"] = f"{base}Permanent_WaterBodies/ID{tid}_{name}_permanent_water.tif"
    with ThreadPoolExecutor(len(urls)) as pool:
        res = dict(zip(urls, pool.map(lambda u: _safe(_read_jrc, u, lat, lon, n), urls.values())))
    pw = res.pop("pw")
    land = mask.copy()
    if pw is not None:
        land &= ~(np.nan_to_num(pw, nan=0) == 1)
    if not land.any():
        out.update(available=True, footprint_on_water=True, depths={str(rp): [] for rp in j["rps"]},
                   note="Footprint lies on a permanent water body; move the pin onto the site.")
        return out
    depths = {}
    for rp in j["rps"]:
        a = res.get(rp)
        if a is None:
            out.update(available=False, reason=f"read failed for RP{rp}")
            return out
        depths[str(rp)] = np.round(np.nan_to_num(a[land], nan=0.0), 2).tolist()
    out.update(available=True, footprint_on_water=False, depths=depths, n_pixels=int(land.sum()),
               permanent_water_share=round(float(1 - land.sum() / mask.sum()), 3))
    return out


# ── coastal (Deltares) ────────────────────────────────────────────────────────

_TOK = {"t": None, "exp": 0.0}
_TOK_LOCK = threading.Lock()
_NC = ("https://deltaresfloodssa.blob.core.windows.net/floods/v2021.06/global/NASADEM/90m/"
       "GFM_global_NASADEM90m_{y}slr_rp{rp:04d}_masked.nc")


def _sas() -> str:
    with _TOK_LOCK:
        if not _TOK["t"] or time.time() > _TOK["exp"]:
            with urllib.request.urlopen("https://planetarycomputer.microsoft.com/api/sas/v1/token/deltaresfloodssa/floods",
                                        timeout=20) as r:
                _TOK["t"] = json.loads(r.read())["token"]
            _TOK["exp"] = time.time() + 40 * 60
        return _TOK["t"]


def _chunk(y: int, rp: int, i: int, j: int, n: int) -> Optional[np.ndarray]:
    """(2n+1)² window centred on global pixel (i, j) — chunk size differs between files."""
    D = _delt()
    c = int(D[f"chunk_{y}_{rp}"])
    ci, cj, ii, jj = i // c, j // c, i % c, j % c
    cid = ci * int(D[f"nx_{y}_{rp}"]) + cj
    ids = D[f"cid_{y}_{rp}"]
    k = np.searchsorted(ids, cid)
    if k >= len(ids) or ids[k] != cid:
        return None                                        # empty chunk: sea/fill or dry land
    off, ln = int(D[f"off_{y}_{rp}"][k]), int(D[f"ln_{y}_{rp}"][k])
    req = urllib.request.Request(_NC.format(y=y, rp=rp) + "?" + _sas(),
                                 headers={"Range": f"bytes={off}-{off + ln - 1}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = zlib.decompress(r.read())
    a = np.frombuffer(raw, np.uint8).reshape(4, -1).T.copy().view("<f4").ravel()   # un-shuffle (4-byte)
    a = a.reshape(c, c)
    win = np.full((2 * n + 1, 2 * n + 1), np.nan, np.float32)
    r0, r1 = max(ii - n, 0), min(ii + n + 1, c)
    c0, c1 = max(jj - n, 0), min(jj + n + 1, c)
    win[(r0 - ii + n):(r1 - ii + n), (c0 - jj + n):(c1 - jj + n)] = a[r0:r1, c0:c1]
    return win


def coastal_depths(lat: float, lon: float, radius_m: float = 250.0) -> dict:
    D = _delt()
    i = int(round((lat + 90.0) / PX_DEG))                   # lat ascending from −90
    j = int(round((lon + 180.0) / PX_DEG))
    r_px = radius_m / (111320.0 * PX_DEG)
    mask, n = _disk(lat, lon, r_px)
    out = {"source": "Deltares Global Flood Maps v2021.06 (GTSMip6 surge+tide, NASADEM 90 m), undefended",
           "rps": [int(x) for x in D["rps"]], "years": [int(x) for x in D["years"]], "radius_m": radius_m}
    jobs = [(int(y), int(rp)) for y in D["years"] for rp in D["rps"]]

    def one(yr):
        y, rp = yr
        return yr, _chunk(y, rp, i, j, n)

    try:
        with ThreadPoolExecutor(8) as pool:
            res = dict(pool.map(lambda x: _safe_pair(one, x), jobs))
    except Exception as exc:                               # noqa: BLE001
        out.update(available=False, reason=repr(exc)[:200])
        return out
    depths = {}
    for (y, rp), w in res.items():
        if w is None:
            depths[f"{y}_{rp}"] = []
            continue
        v = w[mask]
        v = np.where((v < -9000) | ~np.isfinite(v), 0.0, v)
        depths[f"{y}_{rp}"] = np.round(v, 2).tolist()
    out.update(available=True, depths=depths, n_pixels=int(mask.sum()))
    return out


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception:                                      # noqa: BLE001
        return None


def _safe_pair(fn, x):
    try:
        return fn(x)
    except Exception:                                      # noqa: BLE001
        return x, None


def site_flood(lat: float, lon: float, radius_m: float = 250.0, coastal: Optional[bool] = None,
               elev_m: Optional[float] = None) -> dict:
    """River + coastal depths (cached). Coastal is read when the site is low (< 30 m) or unknown."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = CACHE / f"{lat:.4f}_{lon:.4f}_{int(radius_m)}.json"
    if key.exists():
        try:
            return json.loads(key.read_text())
        except Exception:                                  # noqa: BLE001
            pass
    t = time.time()
    want_coast = coastal if coastal is not None else (elev_m is None or elev_m < 30)
    with ThreadPoolExecutor(2) as pool:
        fr = pool.submit(river_depths, lat, lon, radius_m)
        fc = pool.submit(coastal_depths, lat, lon, radius_m) if want_coast else None
        river = fr.result()
        coast = fc.result() if fc else {"available": False, "reason": "site above 30 m — coastal flooding not assessed"}
    res = {"river": river, "coastal": coast, "elapsed_s": round(time.time() - t, 1)}
    if river.get("available") is not False or coast.get("available") is not False:
        try:
            key.write_text(json.dumps(res, separators=(",", ":")))
        except Exception:                                  # noqa: BLE001
            pass
    return res
