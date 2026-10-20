"""True-colour satellite image of a box from Copernicus Sentinel-2 (10 m), for draping on the 3D terrain mesh.

Source: Sentinel-2 L2A true-colour (TCI) cloud-optimised GeoTIFFs on AWS, found through the Earth Search STAC catalogue (no key). Free and open licence:
"Contains modified Copernicus Sentinel data". The clearest scene (lowest scene-wide cloud cover) of the last ~18 months is used; the date and cloud cover are returned so the
viewer can say what the picture is. 10 m pixels: roads, fields, water, tree cover and large buildings read; individual houses do not. Where better open aerial imagery exists
(Netherlands 8-25 cm, United States NAIP) the viewer prefers it.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

STAC = "https://earth-search.aws.element84.com/v1/search"
CACHE = Path.home() / ".cache" / "climrisk" / "s2rgb"


def _post(url: str, body: dict, timeout: float = 40.0) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "User-Agent": "ClimRisk/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def bounds_for(lat: float, lon: float, radius_km: float):
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * max(0.05, math.cos(math.radians(lat))))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def find_scene(lat: float, lon: float, radius_km: float, months: int = 18, max_cloud: float = 30.0, post=_post) -> Optional[dict]:
    w, s, e, n = bounds_for(lat, lon, radius_km)
    end = datetime.now(timezone.utc)
    q = {"collections": ["sentinel-2-l2a"], "bbox": [w, s, e, n], "datetime": f"{(end - timedelta(days=30 * months)).strftime('%Y-%m-%dT00:00:00Z')}/{end.strftime('%Y-%m-%dT00:00:00Z')}",
         "limit": 12, "query": {"eo:cloud_cover": {"lt": max_cloud}}, "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}]}
    feats = post(STAC, q).get("features", [])
    for f in feats:                                              # the box must sit inside the scene footprint, not straddle its edge
        bb = f.get("bbox") or [-181, -91, 181, 91]
        if bb[0] <= w and bb[1] <= s and bb[2] >= e and bb[3] >= n and "visual" in f.get("assets", {}):
            return {"id": f["id"], "date": f["properties"]["datetime"][:10], "cloud_pct": round(float(f["properties"].get("eo:cloud_cover", float("nan"))), 1), "href": f["assets"]["visual"]["href"]}
    return None


def true_colour(lat: float, lon: float, radius_km: float, max_px: int = 1024, post=_post):
    """(png bytes, {west, south, east, north}, meta) in geographic coordinates, or None when no usable scene exists."""
    key = hashlib.md5(f"{lat:.3f},{lon:.3f},{radius_km:.2f}".encode()).hexdigest()
    cp = CACHE / (key + ".png")
    cm = CACHE / (key + ".json")
    if cp.exists() and cm.exists():
        m = json.loads(cm.read_text())
        return cp.read_bytes(), m["bounds"], m["meta"]
    sc = find_scene(lat, lon, radius_km, post=post)
    if sc is None:
        return None
    import rasterio
    from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds
    from rasterio.windows import from_bounds
    from PIL import Image
    w, s, e, n = bounds_for(lat, lon, radius_km)
    with rasterio.open(sc["href"]) as ds:
        ub = transform_bounds(4326, ds.crs, w, s, e, n)
        win = from_bounds(*ub, transform=ds.transform).round_offsets().round_lengths()
        arr = ds.read([1, 2, 3], window=win, boundless=False)
        tr = ds.window_transform(win)
        crs = ds.crs
    if arr.size == 0 or arr.shape[1] < 8:
        return None
    px = min(max_px, max(arr.shape[1], arr.shape[2]))
    dst_tr, dw, dh = calculate_default_transform(crs, "EPSG:4326", arr.shape[2], arr.shape[1], *rasterio.transform.array_bounds(arr.shape[1], arr.shape[2], tr))
    out = np.zeros((3, dh, dw), np.uint8)
    for i in range(3):
        reproject(arr[i], out[i], src_transform=tr, src_crs=crs, dst_transform=dst_tr, dst_crs="EPSG:4326", resampling=Resampling.bilinear)
    west, north = dst_tr.c, dst_tr.f
    east, south = west + dst_tr.a * dw, north + dst_tr.e * dh
    img = Image.fromarray(np.moveaxis(out, 0, -1), "RGB")
    if max(img.size) > px:
        img.thumbnail((px, px))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    bounds = {"west": west, "south": south, "east": east, "north": north}
    meta = {"scene": sc["id"], "date": sc["date"], "cloud_pct": sc["cloud_pct"], "resolution_m": 10, "source": "Copernicus Sentinel-2 L2A true colour (contains modified Copernicus Sentinel data)"}
    try:
        CACHE.mkdir(parents=True, exist_ok=True)
        cp.write_bytes(buf.getvalue())
        cm.write_text(json.dumps({"bounds": bounds, "meta": meta}))
    except Exception:                                            # noqa: BLE001
        pass
    return buf.getvalue(), bounds, meta
