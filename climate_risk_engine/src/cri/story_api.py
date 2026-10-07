"""cri/story_api.py — HTTP surface for the story-format report suite (/story-reports/*)."""
from __future__ import annotations

import copy
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field

from . import story as S
from . import story_reports as SR

HERE = Path(__file__).parent
_CACHE: dict = {}
_LOCK = threading.Lock()


class BuildReq(BaseModel):
    type: str
    payload: dict = Field(default_factory=dict)
    template: Optional[str] = None
    brand: Optional[dict] = None


class RenderReq(BaseModel):
    token: str
    sections: Optional[list] = None
    brand: Optional[dict] = None
    format: str = "html"


class GenReq(BuildReq):
    sections: Optional[list] = None
    format: str = "json"


def _put(story: dict) -> str:
    with _LOCK:
        for k in [k for k, v in _CACHE.items() if time.time() - v[0] > 3600]:
            _CACHE.pop(k, None)
        while len(_CACHE) >= 30:
            _CACHE.pop(min(_CACHE, key=lambda k: _CACHE[k][0]))
        t = uuid.uuid4().hex[:12]
        _CACHE[t] = (time.time(), story)
        return t


def _render(story: dict, sections: Optional[list], brand: Optional[dict], fmt: str):
    st = copy.deepcopy(story)
    if brand:
        st["brand"] = {**(st.get("brand") or {}), **{k: v for k, v in brand.items() if v}}
    out = S.pick_sections(st, sections)
    safe = "".join(c if c.isalnum() else "-" for c in st["title"])[:60].strip("-") or "report"
    if fmt == "html":
        return HTMLResponse(S.html(out))
    if fmt == "markdown":
        return PlainTextResponse(S.markdown(out), headers={"Content-Disposition": f'attachment; filename="{safe}.md"'})
    if fmt == "pdf":
        return Response(S.pdf(out), media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="{safe}.pdf"'})
    if fmt == "json":
        return JSONResponse({"story": out, "sections": S.sections_of(story)})
    raise HTTPException(status_code=400, detail="format must be html, pdf, markdown or json")


def make_router(auth_dep) -> tuple:
    prot = APIRouter(prefix="/story-reports", tags=["Story Reports"], dependencies=[auth_dep])
    pub = APIRouter(prefix="/story-reports", tags=["Story Reports"])

    @prot.get("/catalogue")
    def catalogue() -> dict:
        return {"reports": [c for c in SR.catalogue() if not c["id"].startswith("_")], "templates": {k: v["label"] for k, v in SR.TEMPLATES.items()}}

    def _build(req: BuildReq) -> dict:
        try:
            return SR.build(req.type, req.payload, req.template, req.brand)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown report type; choose one of {list(SR.REPORTS)}")
        except (ValueError, TypeError) as e:
            raise HTTPException(status_code=422, detail=str(e)[:300])
        except Exception as e:                                        # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"Could not build the report: {type(e).__name__}: {str(e)[:250]}")

    @prot.post("/build")
    def build(req: BuildReq) -> dict:
        st = _build(req)
        return {"token": _put(st), "title": st["title"], "standfirst": st["standfirst"], "sections": S.sections_of(st), "template": SR.template_for(req.payload.get("sector"), req.template)["key"]}

    @prot.post("/render")
    def render(req: RenderReq):
        with _LOCK:
            hit = _CACHE.get(req.token)
        if not hit:
            raise HTTPException(status_code=404, detail="report expired — build it again")
        return _render(hit[1], req.sections, req.brand, req.format)

    @prot.post("/generate")
    def generate(req: GenReq):
        st = _build(req)
        return _render(st, req.sections, None, req.format)

    @pub.get("/dashboard")
    def dashboard() -> HTMLResponse:
        return HTMLResponse((HERE / "api" / "story_reports_dashboard.html").read_text(encoding="utf-8"))

    return prot, pub
