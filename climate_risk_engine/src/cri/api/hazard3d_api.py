"""cri/api/hazard3d_api.py — HTTP surface for the 3D hazard viewer.

  GET  /hazard3d                 the viewer page
  GET  /hazard3d/catalogue       every hazard, whether it has a spatial surface, and why not when it has none
  POST /hazard3d/surface         one classified hazard layer for a place (and optional study polygon)
  POST /hazard3d/study-area      KML / KMZ / GeoJSON upload -> polygons
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
                raise HTTPException(502, f"terrain tile unavailable: HTTP {r.status_code}")
            b = r.content
            with _LOCK:
                if len(_CACHE) > 600:
                    for kk in [kk for kk in _CACHE if isinstance(kk, tuple)][:100]:
                        _CACHE.pop(kk, None)
                _CACHE[k] = b
        return Response(b, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})

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
