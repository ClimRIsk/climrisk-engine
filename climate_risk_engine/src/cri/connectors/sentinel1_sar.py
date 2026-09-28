"""
connectors/sentinel1_sar.py — Sentinel-1 radar flood history (sees through cloud).

Why
───
Optical Sentinel-2 cannot see floods during the monsoon: for Dhaka the last two
years contain zero clear scenes from June to September — the flood season. C-band
radar images through cloud, so it records inundation when it actually happens.

Data
────
Microsoft Planetary Computer `sentinel-1-rtc`: radiometrically terrain-corrected
gamma0 backscatter, 10 m COGs, free with an anonymous SAS token. (The raw
Sentinel-1 archive on AWS is requester-pays and needs calibration +
terrain correction first — RTC is the analysis-ready option.)

Method
──────
1. One scene per calendar month over the lookback, up to `max_scenes`.
2. Windowed read of VV onto the same EPSG:4326 grid as the other layers (20 m).
3. 3×3 median filter (speckle), convert to dB.
4. Water where VV < −18 dB (smooth open water is a specular reflector).
5. Per-pixel water frequency across dates.

Only *intermittent* water (wet on some dates, dry on others) is used as flood
evidence. Always-dark pixels — dry sand, radar shadow on steep slopes — would
otherwise be mistaken for permanent water, so permanent water comes from the
optical / WorldCover layers instead. Wet snow can mimic seasonal water; the
caller should discount this evidence at high altitude.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

from .sentinel2_pixels import (_disk_mask, _downsample, _flat, _grid_shape, _read_band,
                               _spread_in_time, asset_bbox)

log = logging.getLogger(__name__)

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
TOKEN = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-1-rtc"
WATER_DB = -18.0

_token: dict = {"value": None, "exp": 0.0}
_CACHE: dict[tuple, tuple[float, dict]] = {}
_CACHE_TTL_S = 6 * 3600


def _sas() -> str:
    if _token["value"] and time.time() < _token["exp"]:
        return _token["value"]
    with urllib.request.urlopen(TOKEN, timeout=15) as r:
        tok = json.loads(r.read())["token"]
    _token.update(value=tok, exp=time.time() + 45 * 60)
    return tok


def _search(lat: float, lon: float, days_back: int) -> list[dict]:
    import datetime as dt
    end = dt.datetime.utcnow().date()
    start = end - dt.timedelta(days=days_back)
    body = {
        "collections": ["sentinel-1-rtc"],
        "intersects": {"type": "Point", "coordinates": [lon, lat]},
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "limit": 250,
    }
    req = urllib.request.Request(STAC, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        feats = json.loads(r.read()).get("features", [])
    # _spread_in_time keeps the first scene per month → prefer VV-bearing scenes
    return [f for f in feats if "vv" in f.get("assets", {})]


def get_sar_water_history(
    lat: float,
    lon: float,
    radius_km: float = 2.5,
    days_back: int = 730,
    max_scenes: int = 10,
    layer_px: int = 128,
    include_layer: bool = True,
) -> dict:
    key = (round(lat, 4), round(lon, 4), radius_km, max_scenes, include_layer)
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < _CACHE_TTL_S:
        return hit[1]

    t0 = time.time()
    out: dict = {"available": False, "source": "Sentinel-1 RTC (VV) · Microsoft Planetary Computer",
                 "threshold_db": WATER_DB}
    try:
        feats = _search(lat, lon, days_back)
        picks = _spread_in_time(feats, max_scenes)
        if not picks:
            out["error"] = "no Sentinel-1 scenes"
            return out

        bbox = asset_bbox(lat, lon, radius_km)
        shape = _grid_shape(lat, radius_km, 20.0)
        tok = _sas()

        def read(f):
            href = f["assets"]["vv"]["href"]
            href = f"{href}{'&' if '?' in href else '?'}{tok}"
            try:
                return _read_band(href, bbox, shape, "average", decimate=2)[0]
            except Exception as exc:          # one slow/broken scene must not sink the history
                log.info("SAR scene %s skipped: %s", f.get("id"), exc)
                return None

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(read, picks))
        kept = [(a, p) for a, p in zip(results, picks) if a is not None]
        out["scenes_failed"] = len(picks) - len(kept)
        if len(kept) < 3:
            out["error"] = f"only {len(kept)} radar scenes readable"
            return out
        arrs = [a for a, _ in kept]
        picks = [p for _, p in kept]

        from scipy import ndimage
        wet_stack, obs_stack = [], []
        for a in arrs:
            valid = np.isfinite(a) & (a > 0)
            filt = ndimage.median_filter(np.where(valid, a, np.nanmedian(a[valid]) if valid.any() else 1.0), size=3)
            db = 10 * np.log10(np.maximum(filt, 1e-6))
            wet_stack.append((db < WATER_DB) & valid)
            obs_stack.append(valid)
        wet, obs = np.stack(wet_stack), np.stack(obs_stack)
        n = obs.sum(0)
        freq = np.where(n >= 3, wet.sum(0) / np.maximum(n, 1), np.nan)
        has = np.isfinite(freq)

        # Exclude permanent water bodies (WorldCover class 80). Wind, currents and
        # ship wakes roughen rivers above the threshold on some dates, which would
        # otherwise read as "intermittent" flooding of the channel itself.
        try:
            from .worldcover import lc_grid
            lc10, _ = lc_grid(lat, lon, radius_km)
            lc20 = lc10[::2, ::2][:shape[0], :shape[1]]
            if lc20.shape == shape:
                perm = lc20 == 80
                has = has & ~perm
                out["permanent_water_excluded_fraction"] = round(float(perm.mean()), 4)
        except Exception as exc:
            log.debug("WorldCover mask unavailable for SAR: %s", exc)

        ephem = (freq >= 0.1) & (freq < 0.8) & has
        ctr = (shape[0] // 2, shape[1] // 2)
        ring = _disk_mask(shape, ctr, 25)       # 500 m at 20 m cells
        ever = wet.any(0) & has & (freq < 0.8)
        # Date of maximum wet extent within 500 m — "the worst flood we saw"
        extents = [(float((w & ring & has).sum() / max((o & ring & has).sum(), 1)), p["properties"]["datetime"][:10])
                   for w, o, p in zip(wet_stack, obs_stack, picks)]
        peak_frac, peak_date = max(extents) if extents else (0.0, None)

        def frac(m, v):
            s = v.sum()
            return round(float((m & v).sum() / s), 4) if s else None

        fa = freq[ctr]
        out.update({
            "available": True,
            "scenes_used": len(picks),
            "date_range": f"{picks[0]['properties']['datetime'][:10]} → {picks[-1]['properties']['datetime'][:10]}",
            "months_covered": sorted({p["properties"]["datetime"][5:7] for p in picks}),
            "intermittent_water_fraction": frac(ephem, has),
            "intermittent_water_within_500m": frac(ephem & ring, has & ring),
            "ever_wet_within_500m": frac(ever & ring, has & ring),
            "peak_wet_within_500m": round(peak_frac, 4),
            "peak_wet_date": peak_date,
            "water_freq_at_asset": round(float(fa), 3) if np.isfinite(fa) else None,
        })
        if include_layer:
            lay = np.where(has & (freq < 0.8), freq, np.nan)   # hide always-dark pixels
            out["layer"] = _flat(_downsample(lay, layer_px))
            out["layer_shape"] = (layer_px, layer_px)
    except Exception as exc:
        log.warning("Sentinel-1 water history failed: %s", exc)
        out["error"] = str(exc)
    out["elapsed_s"] = round(time.time() - t0, 1)
    _CACHE[key] = (time.time(), out)
    return out
