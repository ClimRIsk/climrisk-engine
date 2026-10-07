"""cri/resident/api.py — HTTP surface of the resident agent (/resident/*)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from . import agent as A
from . import autopilot as AP
from . import linkedin as LI
from . import memory as M
from . import nlu as N

HERE = Path(__file__).parent


class ChatReq(BaseModel):
    text: str = Field(..., max_length=2000)
    session: str = "default"
    intent: Optional[str] = None

class FeedbackReq(BaseModel):
    text: str
    predicted: str = ""
    intent: str

class PublishReq(BaseModel):
    approved: bool = False

class ReadReq(BaseModel):
    ids: Optional[list] = None

class SiteReq(BaseModel):
    lat: float
    lon: float
    name: str = ""
    sector: Optional[str] = None

class EditReq(BaseModel):
    text: str = Field(..., max_length=3000)


def make_router(auth_dep) -> tuple:
    """Returns (protected_router, public_router). The public one serves static UI (dock.js, dashboard)."""
    prot = APIRouter(prefix="/resident", tags=["Resident Agent"], dependencies=[auth_dep])
    pub = APIRouter(prefix="/resident", tags=["Resident Agent"])

    @prot.post("/chat")
    def chat(req: ChatReq) -> dict:
        return A.get_agent().handle(req.session, req.text, req.intent)

    @prot.get("/job/{jid}")
    def job(jid: str) -> dict:
        return A.job_status(jid)

    @prot.get("/status")
    def status() -> dict:
        m = N.get_model()
        return {"model": {"classes": len(m.classes), **m.meta}, "linkedin": LI.status(), "facts": len(M.facts(None, 500)),
                "feedback_rows": len(M.feedback_rows()), "posts": len(M.posts(100)), "runs": len(M.runs(100))}

    @prot.get("/facts")
    def facts(q: Optional[str] = None) -> dict:
        return {"facts": M.facts(q, 100)}

    @prot.delete("/facts/{fid}")
    def del_fact(fid: int) -> dict:
        M.forget_fact(fid); return {"ok": True}

    @prot.get("/history")
    def history() -> dict:
        return {"runs": M.runs(30), "posts": M.posts(30)}

    @prot.post("/feedback")
    def feedback(req: FeedbackReq) -> dict:
        if req.intent not in N.get_model().classes:
            raise HTTPException(status_code=400, detail=f"unknown intent; choose one of {N.get_model().classes}")
        M.add_feedback(req.text, req.predicted, req.intent)
        return {"ok": True, "feedback_rows": len(M.feedback_rows())}

    @prot.post("/retrain")
    def retrain() -> dict:
        rows = [(r["text"], r["intent"]) for r in M.feedback_rows()]
        rep = N.train(extra=rows)
        M.mark_feedback_used()
        return rep

    @prot.post("/research/scan")
    def scan() -> dict:
        from . import research as R
        sc = R.scan()
        M.set_state("last_scan", sc)
        return sc

    @prot.get("/research/last")
    def last_scan() -> dict:
        return M.get_state("last_scan") or {"events": []}

    @prot.get("/post/{pid}")
    def post(pid: int) -> dict:
        p = M.get_post(pid)
        if not p:
            raise HTTPException(status_code=404, detail="no such post")
        p.pop("image", None)
        return p

    @prot.get("/post/{pid}/image")
    def post_image(pid: int):
        p = M.get_post(pid)
        if not p or not p.get("image"):
            raise HTTPException(status_code=404, detail="no image")
        return Response(p["image"], media_type="image/png")

    @prot.put("/post/{pid}")
    def edit(pid: int, req: EditReq) -> dict:
        if not M.get_post(pid):
            raise HTTPException(status_code=404, detail="no such post")
        M.update_post(pid, text=req.text)
        return {"ok": True}

    @prot.post("/post/{pid}/publish")
    def publish(pid: int, req: PublishReq) -> JSONResponse:
        r = LI.publish(pid, approved=req.approved)
        return JSONResponse(r, status_code=200 if r.get("ok") else (400 if r.get("needs_approval") else 502))

    # ── autopilot: the agent working in the background ──
    @prot.get("/autopilot")
    def ap_status() -> dict:
        return AP.status()

    @prot.post("/autopilot/run")
    def ap_run() -> dict:
        import threading
        threading.Thread(target=AP.run_cycle, args=("run now",), daemon=True).start()
        return {"ok": True, "note": "cycle started; results appear in the inbox"}

    @prot.get("/inbox")
    def ap_inbox(unread: bool = False) -> dict:
        return {"items": AP.inbox(unread), "unread": len(AP.inbox(True)), "status": AP.status()}

    @prot.post("/inbox/read")
    def ap_read(req: ReadReq) -> dict:
        return {"marked": AP.mark_read(req.ids)}

    @prot.get("/sites")
    def sites() -> dict:
        return {"sites": AP.sites()}

    @prot.post("/sites")
    def add_site(req: SiteReq) -> dict:
        return AP.register_site(req.lat, req.lon, req.name, req.sector)

    @prot.delete("/sites/{name}")
    def del_site(name: str) -> dict:
        AP.remove_site(name); return {"ok": True}

    @pub.get("/dock.js")
    def dock_js() -> Response:
        return Response((HERE / "dock.js").read_text(encoding="utf-8"), media_type="application/javascript", headers={"Cache-Control": "no-cache"})

    @pub.get("/dashboard")
    def dashboard() -> HTMLResponse:
        return HTMLResponse('<!doctype html><html><head><meta charset="utf-8"><title>ClimRisk Resident Agent</title>'
                            '<meta name="viewport" content="width=device-width,initial-scale=1"></head>'
                            '<body style="margin:0;background:#0b1220"><script>window.__RESIDENT_FULL=true</script>'
                            '<script src="/resident/dock.js"></script></body></html>')

    return prot, pub


INJECT = b'<script src="/resident/dock.js" defer></script>'
