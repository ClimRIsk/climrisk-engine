"""
connectors/sentinel2_pixels.py — Real Sentinel-2 L2A pixel analysis from AWS Open Data.

Why this exists
───────────────
The older `sentinel2_stac.py` connector only reads STAC *metadata*. It expects
per-scene band statistics in `properties.statistics`, which Earth Search does
not publish, so its NDVI is always None and its flags are driven by cloud-cover
metadata alone. This module reads the actual pixels.

How
───
1. STAC search on Earth Search (AWS Open Data, no key) for scenes covering the
   asset.
2. Windowed HTTP range reads of the Cloud-Optimised GeoTIFFs with rasterio —
   only the ~5 km × 5 km window around the asset is transferred, never the
   full 110 km tile.
3. Each band is reprojected onto a fixed EPSG:4326 grid whose bounds match the
   AWS terrain-tile DEM used by `terrain.pysheds_analyzer.TerrainEngine`, so
   imagery drapes exactly onto the 3D surface.
4. Scale / offset come from the STAC `raster:bands` metadata, which handles the
   post-2022 processing-baseline BOA offset correctly.

Evidence produced
─────────────────
Single clearest scene (10 m):
  • True-colour image (from the `visual` TCI asset) — base64 JPEG for 3D draping
  • NDVI   (nir − red)/(nir + red)          vegetation health
  • MNDWI  (green − swir16)/(green + swir16) open water (Xu 2006)
  • NDBI   (swir16 − nir)/(swir16 + nir)     built-up / impervious
  • Distance from the asset to the nearest open water
  • Impervious fraction within 250 m of the asset (runoff generation)

Multi-temporal stack (up to 8 scenes spread across ~2 years, SCL at 20 m):
  • Water frequency per pixel = water observations / cloud-free observations
  • Permanent water   : frequency ≥ 0.8
  • Ephemeral water   : 0.1 ≤ frequency < 0.8  → observed inundation, i.e.
    direct satellite evidence of flooding / seasonal floodplain
  • Ephemeral water within 500 m of the asset

References
──────────
  AWS registry : https://registry.opendata.aws/sentinel-2-l2a-cogs/
  STAC API     : https://earth-search.aws.element84.com/v1
  SCL classes  : 3 cloud shadow, 4 vegetation, 5 bare, 6 water,
                 8/9 cloud, 10 cirrus, 11 snow
"""
from __future__ import annotations

import base64
import datetime as _dt
import io
import json
import logging
import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Optional

import warnings

import numpy as np

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message="Dataset has no geotransform")

STAC_URL = "https://earth-search.aws.element84.com/v1/search"
COLLECTION = "sentinel-2-l2a"

_GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    AWS_NO_SIGN_REQUEST="YES",
    GDAL_HTTP_MULTIPLEX="YES",
    GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff,.TIF,.TIFF",
    VSI_CACHE="TRUE",
)

_SCL_WATER = 6
_SCL_INVALID = {0, 1, 3, 8, 9, 10, 11}  # nodata, saturated, shadow, cloud, cirrus, snow

# In-process cache: (lat, lon, radius) → SatelliteEvidence, 6h TTL
_CACHE: dict[tuple, tuple[float, "SatelliteEvidence"]] = {}
_CACHE_TTL_S = 6 * 3600


# ── Geometry (identical formula to connectors/terrain_tiles.py) ──────────────

def asset_bbox(lat: float, lon: float, radius_km: float) -> tuple[float, float, float, float]:
    """(west, south, east, north) — same maths as the terrain-tile DEM crop."""
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * math.cos(math.radians(lat)))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def _grid_shape(lat: float, radius_km: float, cell_m: float) -> tuple[int, int]:
    n = int(round(2 * radius_km * 1000 / cell_m))
    return n, n


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class SatelliteEvidence:
    lat: float
    lon: float
    radius_km: float
    bounds: tuple[float, float, float, float] = (0, 0, 0, 0)  # W, S, E, N

    # Clearest scene
    scene_id: Optional[str] = None
    scene_date: Optional[str] = None
    scene_cloud_pct: Optional[float] = None

    ndvi_at_asset: Optional[float] = None
    ndvi_area_mean: Optional[float] = None
    vegetation_fraction: Optional[float] = None      # NDVI > 0.4
    builtup_fraction: Optional[float] = None         # NDBI > 0 and NDVI < 0.2
    impervious_within_250m: Optional[float] = None
    open_water_fraction: Optional[float] = None      # MNDWI > 0, or Prithvi if available
    nearest_water_m: Optional[float] = None
    water_method: str = "MNDWI > 0 (spectral index)"
    prithvi_water_fraction: Optional[float] = None
    mndwi_water_fraction: Optional[float] = None

    # Multi-temporal water history
    temporal_scenes_used: int = 0
    temporal_date_range: Optional[str] = None
    temporal_months_covered: list = field(default_factory=list)
    permanent_water_fraction: Optional[float] = None
    ephemeral_water_fraction: Optional[float] = None
    ephemeral_water_within_500m: Optional[float] = None
    water_freq_at_asset: Optional[float] = None

    # Visual payloads (for the 3D renderer)
    true_color_jpeg_b64: Optional[str] = field(default=None, repr=False)
    layers: dict = field(default_factory=dict, repr=False)   # name → flat list
    layer_shape: tuple[int, int] = (0, 0)

    elapsed_s: float = 0.0
    source: str = "Sentinel-2 L2A COGs · AWS Open Data · Earth Search STAC"
    errors: list[str] = field(default_factory=list)

    def to_dict(self, include_visuals: bool = True) -> dict:
        d = asdict(self)
        if not include_visuals:
            d.pop("true_color_jpeg_b64", None)
            d.pop("layers", None)
        return d


# ── STAC search ───────────────────────────────────────────────────────────────

def _stac_search(lat: float, lon: float, days_back: int, max_cloud: float, limit: int) -> list[dict]:
    end = _dt.datetime.utcnow().date()
    start = end - _dt.timedelta(days=days_back)
    payload = {
        "collections": [COLLECTION],
        "intersects": {"type": "Point", "coordinates": [lon, lat]},
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        "limit": limit,
    }
    req = urllib.request.Request(
        STAC_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "ClimRisk-Engine/1.0"},
    )
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.loads(r.read()).get("features", [])


def _spread_in_time(features: list[dict], n: int) -> list[dict]:
    """Pick up to n scenes, at most one per calendar month, lowest cloud first."""
    by_month: dict[str, dict] = {}
    for f in features:  # already sorted by cloud ascending
        m = f["properties"]["datetime"][:7]
        by_month.setdefault(m, f)
    picks = sorted(by_month.values(), key=lambda f: f["properties"]["datetime"])
    if len(picks) <= n:
        return picks
    idx = np.linspace(0, len(picks) - 1, n).round().astype(int)
    return [picks[i] for i in sorted(set(idx))]


# ── Raster reading ────────────────────────────────────────────────────────────

def _read_band(href: str, bbox, shape, resampling: str = "bilinear",
               band_meta: Optional[dict] = None, count: int = 1,
               decimate: int = 1) -> np.ndarray:
    """
    Windowed read of a COG, reprojected onto a fixed EPSG:4326 grid.
    Returns float32 array (count, H, W) with NaN for nodata, scale/offset applied.

    decimate > 1 reads the window at 1/decimate resolution, letting GDAL serve
    it from the COG's internal overviews (≈ decimate² fewer bytes transferred).
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds as tf_from_bounds
    from rasterio.warp import reproject, transform_bounds
    from rasterio.windows import from_bounds as win_from_bounds

    H, W = shape
    dst_tf = tf_from_bounds(*bbox, W, H)
    out = np.full((count, H, W), np.nan, dtype=np.float32)
    rs = getattr(Resampling, resampling)

    with rasterio.Env(**_GDAL_ENV):
        with rasterio.open(href) as ds:
            # Native-CRS window with a small margin so reprojection has full coverage
            w, s, e, n = transform_bounds("EPSG:4326", ds.crs, *bbox, densify_pts=21)
            pad = 4 * ds.res[0]
            win = win_from_bounds(w - pad, s - pad, e + pad, n + pad, transform=ds.transform)
            kw = {}
            if decimate > 1:
                oh = max(1, int(round(win.height / decimate)))
                ow = max(1, int(round(win.width / decimate)))
                kw = {"out_shape": (count, oh, ow), "resampling": Resampling.average}
            src = ds.read(indexes=list(range(1, count + 1)), window=win, boundless=True,
                          fill_value=ds.nodata or 0, **kw).astype(np.float32)
            nodata = ds.nodata if ds.nodata is not None else 0
            src[src == nodata] = np.nan
            src_tf = ds.window_transform(win)
            if decimate > 1:
                from affine import Affine
                src_tf = src_tf * Affine.scale(win.width / src.shape[2], win.height / src.shape[1])
            for b in range(count):
                reproject(
                    source=src[b], destination=out[b],
                    src_transform=src_tf, src_crs=ds.crs,
                    dst_transform=dst_tf, dst_crs="EPSG:4326",
                    src_nodata=np.nan, dst_nodata=np.nan, resampling=rs,
                )

    if band_meta:
        scale = band_meta.get("scale", 1.0)
        offset = band_meta.get("offset", 0.0)
        out = out * scale + offset
    return out


def _band_meta(feature: dict, key: str) -> dict:
    """
    DN → surface reflectance.

    Earth Search lists offset −0.1 in raster:bands for every scene, but when
    `earthsearch:boa_offset_applied` is true the +1000 DN offset introduced in
    processing baseline 04.00 has already been removed from the pixels. Applying
    −0.1 again drives dark targets negative and inflates NDVI (urban Dhaka read
    0.96 before this fix). So the offset is applied only when the baseline is
    ≥ 04.00 AND Earth Search has not already harmonised the scene.
    """
    rb = (feature["assets"][key].get("raster:bands") or [{}])[0]
    props = feature.get("properties", {})
    baseline = str(props.get("s2:processing_baseline", "00.00"))
    applied = bool(props.get("earthsearch:boa_offset_applied", False))
    try:
        needs_offset = float(baseline) >= 4.0 and not applied
    except ValueError:
        needs_offset = False
    return {"scale": rb.get("scale", 1e-4), "offset": -0.1 if needs_offset else 0.0}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (a - b) / (a + b)
    r[~np.isfinite(r)] = np.nan
    return np.clip(r, -1, 1)


def _disk_mask(shape, center, radius_px) -> np.ndarray:
    H, W = shape
    yy, xx = np.ogrid[:H, :W]
    return (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius_px ** 2


def _frac(mask: np.ndarray, valid: np.ndarray) -> Optional[float]:
    v = valid.sum()
    return round(float((mask & valid).sum() / v), 4) if v else None


def _downsample(arr: np.ndarray, n: int) -> np.ndarray:
    from scipy import ndimage
    H, W = arr.shape
    if max(H, W) <= n:
        return arr
    filled = np.where(np.isnan(arr), np.nanmean(arr) if np.isfinite(arr).any() else 0, arr)
    return ndimage.zoom(filled, (n / H, n / W), order=1)


def _flat(arr: np.ndarray, nd: int = 3) -> list:
    return [None if not np.isfinite(v) else round(float(v), nd) for v in arr.ravel()]


def _jpeg_b64(rgb: np.ndarray, max_px: int, quality: int = 85) -> str:
    from PIL import Image
    img = Image.fromarray(rgb.astype(np.uint8), "RGB")
    if max(img.size) > max_px:
        img = img.resize((max_px, max_px), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


# ── Main entry point ─────────────────────────────────────────────────────────

def get_satellite_evidence(
    lat: float,
    lon: float,
    radius_km: float = 2.5,
    days_back: int = 365,
    temporal_scenes: int = 8,
    temporal_days_back: int = 730,
    layer_px: int = 128,
    texture_px: int = 1024,
    include_visuals: bool = True,
) -> SatelliteEvidence:
    """
    Pixel-level Sentinel-2 evidence for an asset.

    Parameters
    ----------
    radius_km          : half-width of the analysis window (match TerrainEngine)
    days_back          : search window for the single clearest scene
    temporal_scenes    : number of scenes for the water-frequency stack (0 disables)
    temporal_days_back : search window for the temporal stack
    layer_px           : output resolution of overlay layers for the 3D view
    texture_px         : output resolution of the true-colour JPEG
    """
    key = (round(lat, 4), round(lon, 4), radius_km, temporal_scenes, include_visuals)
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < _CACHE_TTL_S:
        return hit[1]

    t0 = time.time()
    bbox = asset_bbox(lat, lon, radius_km)
    ev = SatelliteEvidence(lat=lat, lon=lon, radius_km=radius_km, bounds=bbox)
    shape10 = _grid_shape(lat, radius_km, 10.0)
    shape20 = _grid_shape(lat, radius_km, 20.0)
    ctr10 = (shape10[0] // 2, shape10[1] // 2)

    # ── 1. Clearest recent scene ──────────────────────────────────────────────
    try:
        feats = _stac_search(lat, lon, days_back, max_cloud=20, limit=10)
        if not feats:
            feats = _stac_search(lat, lon, days_back, max_cloud=60, limit=10)
        if not feats:
            ev.errors.append("no Sentinel-2 scene found")
        else:
            f = feats[0]
            ev.scene_id = f["id"]
            ev.scene_date = f["properties"]["datetime"][:10]
            ev.scene_cloud_pct = round(float(f["properties"].get("eo:cloud_cover", 0)), 2)

            from ..ml.prithvi_flood import endpoint_configured, segment_water
            use_prithvi = endpoint_configured() and all(
                b in f["assets"] for b in ("blue", "nir08", "swir22"))

            jobs = {
                "green": ("green", shape10, "bilinear"),
                "red": ("red", shape10, "bilinear"),
                "nir": ("nir", shape10, "bilinear"),
                "swir16": ("swir16", shape10, "bilinear"),
                "scl": ("scl", shape10, "nearest"),
            }
            if use_prithvi:
                for b in ("blue", "nir08", "swir22"):
                    jobs[b] = (b, shape10, "bilinear")
            with ThreadPoolExecutor(max_workers=6) as pool:
                futs = {
                    k: pool.submit(_read_band, f["assets"][a]["href"], bbox, shp, rs,
                                   None if k == "scl" else _band_meta(f, a))
                    for k, (a, shp, rs) in jobs.items()
                }
                vis_fut = (pool.submit(_read_band, f["assets"]["visual"]["href"], bbox,
                                       shape10, "bilinear", None, 3)
                           if include_visuals and "visual" in f["assets"] else None)
                bands = {k: fu.result()[0] for k, fu in futs.items()}
                visual = vis_fut.result() if vis_fut else None

            scl = bands["scl"]
            clear = np.isfinite(scl) & ~np.isin(scl, list(_SCL_INVALID))
            ndvi = _safe_ratio(bands["nir"], bands["red"])
            mndwi = _safe_ratio(bands["green"], bands["swir16"])
            ndbi = _safe_ratio(bands["swir16"], bands["nir"])
            valid = clear & np.isfinite(ndvi) & np.isfinite(mndwi)

            water = (mndwi > 0.0) & valid
            ev.mndwi_water_fraction = _frac(water, valid)

            # Prithvi-EO-2.0 flood segmentation (HF Inference Endpoint) — preferred
            # over MNDWI when available: it detects turbid floodwater and flooded
            # vegetation that a single spectral threshold misses.
            if use_prithvi:
                chip = np.stack([bands[b] for b in ("blue", "green", "red", "nir08", "swir16", "swir22")])
                mask = segment_water(chip)
                if mask is not None:
                    pw = (mask == 1) & valid
                    ev.prithvi_water_fraction = _frac(pw, valid)
                    water = pw
                    ev.water_method = "Prithvi-EO-2.0 Sen1Floods11 (HF endpoint)"

            veg = (ndvi > 0.4) & valid
            built = (ndbi > 0.0) & (ndvi < 0.2) & ~water & valid

            ev.ndvi_area_mean = round(float(np.nanmean(np.where(valid, ndvi, np.nan))), 3) if valid.any() else None
            a = ndvi[max(ctr10[0] - 1, 0):ctr10[0] + 2, max(ctr10[1] - 1, 0):ctr10[1] + 2]
            ev.ndvi_at_asset = round(float(np.nanmean(a)), 3) if np.isfinite(a).any() else None
            ev.vegetation_fraction = _frac(veg, valid)
            ev.builtup_fraction = _frac(built, valid)
            ev.open_water_fraction = _frac(water, valid)
            ring250 = _disk_mask(shape10, ctr10, 25)            # 25 px × 10 m
            ev.impervious_within_250m = _frac(built & ring250, valid & ring250)

            if water.any():
                from scipy import ndimage
                dist_px = ndimage.distance_transform_edt(~water)
                ev.nearest_water_m = round(float(dist_px[ctr10]) * 10.0, 0)

            if include_visuals:
                ev.layer_shape = (layer_px, layer_px)
                ev.layers["ndvi"] = _flat(_downsample(np.where(valid, ndvi, np.nan), layer_px))
                ev.layers["mndwi"] = _flat(_downsample(np.where(valid, mndwi, np.nan), layer_px))
                ev.layers["ndbi"] = _flat(_downsample(np.where(valid, ndbi, np.nan), layer_px))
                if visual is not None:
                    rgb = np.nan_to_num(np.moveaxis(visual, 0, -1), nan=0)
                    # TCI is 0–255 but typically dark; apply a gentle percentile stretch
                    lo, hi = np.percentile(rgb[rgb > 0], [2, 98]) if (rgb > 0).any() else (0, 255)
                    rgb = np.clip((rgb - lo) / max(hi - lo, 1) * 255, 0, 255)
                    ev.true_color_jpeg_b64 = _jpeg_b64(rgb, texture_px)
    except Exception as exc:
        log.warning("Sentinel-2 clearest-scene analysis failed: %s", exc)
        ev.errors.append(f"clearest scene: {exc}")

    # ── 2. Multi-temporal water frequency (SCL, 20 m) ─────────────────────────
    if temporal_scenes > 0:
        try:
            feats = _stac_search(lat, lon, temporal_days_back, max_cloud=50, limit=100)
            picks = _spread_in_time(feats, temporal_scenes)
            if picks:
                with ThreadPoolExecutor(max_workers=8) as pool:
                    scls = list(pool.map(
                        lambda f: _read_band(f["assets"]["scl"]["href"], bbox, shape20, "nearest")[0],
                        picks,
                    ))
                stack = np.stack(scls)
                obs = np.isfinite(stack) & ~np.isin(stack, list(_SCL_INVALID))
                wet = (stack == _SCL_WATER) & obs
                n_obs = obs.sum(0)
                freq = np.where(n_obs >= 2, wet.sum(0) / np.maximum(n_obs, 1), np.nan)
                has = np.isfinite(freq)
                # Tidal / turbid edges of permanent rivers flicker between classes;
                # judge "intermittent" only on land (WorldCover class ≠ 80).
                perm_wc = None
                try:
                    from .worldcover import lc_grid
                    lc10, _ = lc_grid(lat, lon, radius_km)
                    lc20 = lc10[::2, ::2][:shape20[0], :shape20[1]]
                    if lc20.shape == shape20:
                        perm_wc = lc20 == 80
                except Exception:
                    pass

                ev.temporal_scenes_used = len(picks)
                ev.temporal_months_covered = sorted({p["properties"]["datetime"][5:7] for p in picks})
                ev.temporal_date_range = (f"{picks[0]['properties']['datetime'][:10]} → "
                                          f"{picks[-1]['properties']['datetime'][:10]}")
                perm = (freq >= 0.8) & has
                ephem = (freq >= 0.1) & (freq < 0.8) & has
                if perm_wc is not None:
                    ephem &= ~perm_wc
                ev.permanent_water_fraction = _frac(perm, has)
                ev.ephemeral_water_fraction = _frac(ephem, has)
                ctr20 = (shape20[0] // 2, shape20[1] // 2)
                ring500 = _disk_mask(shape20, ctr20, 25)          # 25 px × 20 m
                ev.ephemeral_water_within_500m = _frac(ephem & ring500, has & ring500)
                fa = freq[ctr20]
                ev.water_freq_at_asset = round(float(fa), 3) if np.isfinite(fa) else None
                if include_visuals:
                    ev.layers["water_freq"] = _flat(_downsample(freq, layer_px))
        except Exception as exc:
            log.warning("Sentinel-2 temporal water analysis failed: %s", exc)
            ev.errors.append(f"temporal stack: {exc}")

    ev.elapsed_s = round(time.time() - t0, 1)
    _CACHE[key] = (time.time(), ev)
    return ev
