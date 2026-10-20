"""cri/api/hazard3d_api.py — HTTP surface for the 3D hazard viewer.

  GET  /hazard3d                 the viewer page
  GET  /hazard3d/catalogue       every hazard, whether it has a spatial surface, and why not when it has none
  POST /hazard3d/surface         one classified hazard layer for a place (and optional study polygon)
  POST /hazard3d/study-area      KML / KMZ / GeoJSON upload -> polygons
  GET  /hazard3d/terrain/{z}/{x}/{y}.png   elevation tile for the mesh: open LiDAR where it exists (z 14-15), else the global tiles
  GET  /hazard3d/coverage        which open LiDAR / aerial imagery covers a point
  GET  /hazard3d/imagery/aerial/{z}/{x}/{y}.jpg   open aerial photo tiles (Netherlands PDOK, United States USGS NAIP), proxied
  POST /hazard3d/imagery/sentinel2   Sentinel-2 true-colour image of the box (anywhere)
  GET  /hazard3d/config          whether a Google Maps Platform key is stored (the key is returned to loopback callers only: the page needs it to stream the tiles)
  POST /hazard3d/config          store or clear that key, in ~/.climrisk with owner-only permissions. It never leaves the machine except in requests to Google.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

HERE = Path(__file__).parent
_LOCK = threading.Lock()
_CACHE: dict = {}


def _home() -> Path:
    return Path(os.environ.get("CRI_HOME") or (Path.home() / ".climrisk"))


def _key_file() -> Path:
    return _home() / "maps_key.json"


def _read_key() -> Optional[str]:
    env = os.environ.get("GOOGLE_MAPS_TILES_KEY")
    if env:
        return env.strip()
    try:
        return (json.loads(_key_file().read_text()).get("google_maps_key") or "").strip() or None
    except Exception:                                            # noqa: BLE001
        return None


def _merge_lidar(z: int, x: int, y: int, global_png: Optional[bytes]) -> Optional[bytes]:
    """Terrarium tile with open-LiDAR heights where they exist and the global tile elsewhere. Returns global_png unchanged when LiDAR does not cover the tile."""
    try:
        import io

        import numpy as np
        from PIL import Image

        from ..connectors import terrain_mesh as TM
        got = TM.lidar_tile(z, x, y)
        if got is None:
            return global_png
        h, mask = got
        if global_png is not None:
            g = TM.decode_terrarium(np.asarray(Image.open(io.BytesIO(global_png)).convert("RGB")))
            h = np.where(mask, h, g)
        elif not mask.all():
            return None
        return TM.tile_png(h)
    except Exception:                                            # noqa: BLE001
        return global_png


AERIAL = {"pdok": {"bbox": (50.7, 53.6, 3.2, 7.4), "max_zoom": 19, "label": "Luchtfoto Beeldmateriaal Nederland, 8-25 cm (PDOK, CC BY 4.0)",
                   "url": "https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0/Actueel_orthoHR/EPSG:3857/{z}/{x}/{y}.jpeg"},
          "usgs": {"bbox": (24.4, 49.5, -125.0, -66.9), "max_zoom": 17, "label": "USGS National Map imagery (NAIP aerial photography, US public domain)",
                   "url": "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"}}


def _aerial_for(lat: float, lon: float) -> Optional[str]:
    for k, a in AERIAL.items():
        s_, n_, w_, e_ = a["bbox"]
        if s_ <= lat <= n_ and w_ <= lon <= e_:
            return k
    return None


class SurfaceReq(BaseModel):
    hazard: str
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    radius_km: float = Field(2.5, gt=0.2, le=6.0)
    polygons: Optional[list] = None
    params: dict = Field(default_factory=dict)


class ConfigReq(BaseModel):
    google_maps_key: Optional[str] = Field(None, max_length=200)


def make_router(auth):
    pub, prot = APIRouter(tags=["3D hazard viewer"]), APIRouter(tags=["3D hazard viewer"], dependencies=[auth])

    @pub.get("/hazard3d", response_class=HTMLResponse)
    def page():
        return HTMLResponse((HERE / "hazard3d.html").read_text(encoding="utf-8"))

    @pub.get("/hazard3d/terrain/{z}/{x}/{y}.png")
    def terrain_tile(z: int, x: int, y: int):
        """Open AWS Terrarium elevation tile (same source the hazard models use), proxied so the page can read its pixels. Cached in memory."""
        from fastapi.responses import Response
        if not (0 <= z <= 15 and 0 <= x < 2 ** z and 0 <= y < 2 ** z):
            raise HTTPException(404, "tile out of range")
        k = ("t", z, x, y)
        with _LOCK:
            b = _CACHE.get(k)
        if b is None:
            import requests
            try:
                r = requests.get(f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png", timeout=15)
            except requests.RequestException as e:
                raise HTTPException(502, f"terrain tile unavailable: {type(e).__name__}") from e
            if r.status_code != 200:
                lid = _merge_lidar(z, x, y, None) if 14 <= z <= 15 else None
                if lid is None:
                    raise HTTPException(502, f"terrain tile unavailable: HTTP {r.status_code}")
                with _LOCK:
                    _CACHE[k] = lid
                return Response(lid, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})
            b = r.content
            if 14 <= z <= 15:
                b = _merge_lidar(z, x, y, b)
            with _LOCK:
                if len(_CACHE) > 600:
                    for kk in [kk for kk in _CACHE if isinstance(kk, tuple)][:100]:
                        _CACHE.pop(kk, None)
                _CACHE[k] = b
        return Response(b, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})

    @pub.get("/hazard3d/coverage")
    def coverage(lat: float, lon: float) -> dict:
        """What open high-resolution data exist at this point: LiDAR terrain (and whether a surface model gives measured building heights) and aerial photography."""
        from ..connectors import lidar_dem as LD
        k = LD._which(lat, lon)
        a = _aerial_for(lat, lon)
        return {"lidar": None if k is None else (LD.SOURCES[k]["label"] if k != "usa" else "USGS 3DEP bare earth (1 m where LiDAR-derived; checked when the layer loads)"),
                "lidar_building_heights": k in ("england", "netherlands"),
                "aerial": None if a is None else {"provider": a, "label": AERIAL[a]["label"], "max_zoom": AERIAL[a]["max_zoom"]},
                "satellite": "Copernicus Sentinel-2 true colour, 10 m, worldwide (clearest scene of the last 18 months)"}

    @pub.get("/hazard3d/imagery/aerial/{z}/{x}/{y}.jpg")
    def aerial_tile(z: int, x: int, y: int):
        from fastapi.responses import Response
        if not (0 <= z <= 20 and 0 <= x < 2 ** z and 0 <= y < 2 ** z):
            raise HTTPException(404, "tile out of range")
        from ..connectors.terrain_mesh import tile_bounds
        w, s_, e, n = tile_bounds(z, x, y)
        prov = _aerial_for((s_ + n) / 2, (w + e) / 2)
        if prov is None or z > AERIAL[prov]["max_zoom"]:
            raise HTTPException(404, "no open aerial imagery for this tile")
        k = ("a", z, x, y)
        with _LOCK:
            b = _CACHE.get(k)
        if b is None:
            import requests
            try:
                r = requests.get(AERIAL[prov]["url"].format(z=z, x=x, y=y), timeout=15, headers={"User-Agent": "ClimRisk/1.0"})
            except requests.RequestException as ex:
                raise HTTPException(502, f"aerial tile unavailable: {type(ex).__name__}") from ex
            if r.status_code != 200 or not r.headers.get("content-type", "").startswith("image"):
                raise HTTPException(404, f"aerial tile unavailable: HTTP {r.status_code}")
            b = r.content
            with _LOCK:
                if len(_CACHE) > 900:
                    for kk in [kk for kk in _CACHE if isinstance(kk, tuple)][:150]:
                        _CACHE.pop(kk, None)
                _CACHE[k] = b
        return Response(b, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})

    @prot.post("/hazard3d/imagery/sentinel2")
    def sentinel2(req: SurfaceReq) -> dict:
        import base64

        from ..connectors import sentinel2_rgb as S2
        try:
            got = S2.true_colour(req.lat, req.lon, min(req.radius_km, 3.0))
        except Exception as ex:                                  # noqa: BLE001
            raise HTTPException(502, f"Sentinel-2 imagery unavailable: {type(ex).__name__}: {str(ex)[:120]}") from ex
        if got is None:
            return {"available": False, "reason": "no clear Sentinel-2 scene covers this box in the last 18 months"}
        png, bounds, meta = got
        return {"available": True, "image_png_b64": base64.b64encode(png).decode(), "bounds": bounds, "meta": meta}

    @pub.get("/hazard3d/catalogue")
    def catalogue() -> dict:
        from .. import hazardsurface as H
        return {"hazards": H.catalogue()}

    @pub.get("/hazard3d/config")
    def config(request: Request) -> dict:
        key = _read_key()
        local = (request.client.host if request.client else "") in ("127.0.0.1", "::1", "localhost", "testclient")
        return {"google_key_present": bool(key), "google_maps_key": key if (key and local) else None,
                "note": "Photorealistic 3D Tiles are Google's service and need your own Google Maps Platform key with billing and the Map Tiles API enabled."}

    @prot.post("/hazard3d/config")
    def set_config(req: ConfigReq) -> dict:
        k = (req.google_maps_key or "").strip()
        _home().mkdir(parents=True, exist_ok=True)
        f = _key_file()
        if not k:
            f.unlink(missing_ok=True)
            return {"google_key_present": False}
        if not all(c.isalnum() or c in "-_" for c in k):
            raise HTTPException(400, "that does not look like a Google API key (letters, digits, - and _ only)")
        f.write_text(json.dumps({"google_maps_key": k}))
        try:
            f.chmod(0o600)
        except OSError:
            pass
        return {"google_key_present": True}

    @prot.post("/hazard3d/study-area")
    async def study_area(file: UploadFile = File(...)) -> dict:
        from ..geo_io import parse_study_area
        data = await file.read()
        if len(data) > 15_000_000:
            raise HTTPException(413, "file is larger than 15 MB")
        try:
            return parse_study_area(data, file.filename or "")
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    @prot.post("/hazard3d/buildings")
    def buildings(req: SurfaceReq) -> dict:
        from .. import hazardsurface as H
        polys = [[(float(p[0]), float(p[1])) for p in r] for r in (req.polygons or []) if len(r) >= 3] or None
        try:
            return H.buildings(req.lat, req.lon, min(req.radius_km, 1.5), hazard=req.hazard if req.hazard != "none" else None, polygons=polys)
        except Exception as e:                                   # noqa: BLE001
            raise HTTPException(502, f"{type(e).__name__}: {e}") from e

    @prot.post("/hazard3d/surface")
    def surface(req: SurfaceReq) -> dict:
        from .. import hazardsurface as H
        polys = [[(float(p[0]), float(p[1])) for p in r] for r in (req.polygons or []) if len(r) >= 3] or None
        key = json.dumps([req.hazard, round(req.lat, 5), round(req.lon, 5), req.radius_km, polys, req.params], sort_keys=True, default=str)
        with _LOCK:
            if key in _CACHE:
                return _CACHE[key]
        try:
            out = H.surface(req.hazard, req.lat, req.lon, req.radius_km, polygons=polys, **req.params)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        except TypeError as e:
            raise HTTPException(400, f"bad parameters for {req.hazard}: {e}") from e
        except Exception as e:                                   # noqa: BLE001
            raise HTTPException(502, f"{req.hazard} could not be computed here: {type(e).__name__}: {e}") from e
        with _LOCK:
            if len(_CACHE) > 24:
                _CACHE.pop(next(iter(_CACHE)))
            _CACHE[key] = out
        return out

    return prot, pub
