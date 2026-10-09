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

class BriefReq(BaseModel):
    assessment: dict

class LearnReq(BaseModel):
    url: Optional[str] = None
    topic: Optional[str] = None
    force: bool = False

class AskReq(BaseModel):
    question: str = Field(..., max_length=1000)
    context: Optional[str] = None

class LabReq(BaseModel):
    domain: str = Field(..., pattern="^[a-z]{3,12}$")
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    name: str = Field("Study site", max_length=80)
    return_period: Optional[int] = Field(None, ge=2, le=1000)
    radius_km: Optional[float] = Field(None, ge=0.5, le=500.0)
    city_km: Optional[float] = Field(None, ge=5.0, le=80.0)
    manning: Optional[float] = Field(None, ge=0.01, le=0.3)
    cn: Optional[float] = Field(None, ge=40, le=98)
    observed_depths: Optional[list] = None
    pm25_series: Optional[list] = None

class PortfolioReq(BaseModel):
    csv: str = Field(..., max_length=200000)
    hazards: Optional[list] = None

class EditReq(BaseModel):
    text: str = Field(..., max_length=3000)


def make_router(auth_dep) -> tuple:
    """Returns (protected_router, public_router). The public one serves static UI (dock.js, dashboard)."""
    prot = APIRouter(prefix="/resident", tags=["Resident Agent"], dependencies=[auth_dep])
    pub = APIRouter(prefix="/resident", tags=["Resident Agent"])

    @prot.post("/chat")
    def chat(req: ChatReq) -> dict:
        return A.get_agent().handle(req.session, req.text, req.intent)

    @prot.get("/selfcheck")
    def selfcheck() -> dict:
        """Can this machine reach every data source the engine needs? Plain-language result."""
        from . import selfcheck as SC
        return SC.run()

    @prot.post("/brief")
    def brief(req: BriefReq) -> dict:
        """Keyless 'AI deep dive': a written site brief from a computed assessment. No LLM, no key."""
        from .brief import site_brief
        if not isinstance(req.assessment, dict) or not req.assessment.get("findings"):
            raise HTTPException(422, "assessment with findings required")
        return site_brief(req.assessment)

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

    # ── what the agent has learned ──
    @prot.get("/knowledge")
    def kn_stats() -> dict:
        from . import knowledge as K, answer as AN
        return {**K.stats(), "language_model": AN.llm_status()}

    @prot.post("/knowledge/learn")
    def kn_learn(req: LearnReq) -> dict:
        from . import knowledge as K
        if req.url:
            return K.learn_url(req.url)
        if req.topic:
            return K.learn_topic(req.topic)
        return K.learn_seed(force=req.force)

    @prot.post("/ask")
    def ask(req: AskReq) -> dict:
        from . import answer as AN
        return AN.ask(req.question, context=req.context)

    @prot.post("/reason")
    def reason(req: dict) -> dict:
        """Scenario in (company financials, loss or insurer-run loss, covenants) -> engine-computed effect + cited, checked reasoning steps + verdict."""
        from . import answer as AN
        return AN.ask_reasoned(req.get("scenario") or req, int(req.get("question_index", 0) or 0), bool(req.get("use_llm", True)))

    # ── the case library: real events, company filings and financials ──
    @prot.get("/cases")
    def cases_stats() -> dict:
        from . import casebase as CB, answer as AN
        if not CB.available():
            return {"available": False, "note": "no case index installed yet (run Run-V4-Mac.command)"}
        return {"available": True, **CB.stats(), "language_model": AN.llm_status()}

    @prot.post("/cases/analogs")
    def cases_analogs(req: dict) -> dict:
        """{hazard, assets, channels?, region?, text?} -> closest real cases, checked and cited. Used by reports and the insurer view."""
        from . import answer as AN
        text = str(req.get("text") or f"closest cases for {req.get('hazard', '')} at {req.get('assets', '')}")
        return AN.ask_analogs(text, assets=str(req.get("assets") or ""), hazard=str(req.get("hazard") or ""), channels=req.get("channels"))

    # ── risk-manager tools (cri.riskmgmt) ─────────────────────────────────────────────────────────────────────────────────────
    @prot.get("/audit")
    def audit(n: int = 50, kind: Optional[str] = None) -> dict:
        """Recent audit records: every insurer run and case answer, with input hash, seed, engine and index versions."""
        from .. import riskmgmt as RM
        return {"records": RM.audit_recent(max(1, min(n, 500)), kind)}

    @prot.post("/risk/translate")
    def risk_translate(req: dict) -> dict:
        """{fin:{revenue,ebitda,net_debt,interest_expense,equity,total_assets}, loss:{asset_damage,insured_recovery,bi_revenue_lost,...}, covenants:{max_net_debt_to_ebitda,...}}"""
        from .. import riskmgmt as RM
        try:
            return RM.translate(req.get("fin") or {}, req.get("loss") or {}, req.get("covenants"))
        except ValueError as e:
            raise HTTPException(422, str(e))

    @prot.post("/risk/exposure-quality")
    def risk_exposure_quality(req: dict) -> dict:
        from .. import riskmgmt as RM
        sites = req.get("sites")
        if not isinstance(sites, list):
            raise HTTPException(422, "sites (a list) required")
        return RM.exposure_quality(sites)

    @prot.post("/risk/issb-draft")
    def risk_issb(req: dict) -> dict:
        from .. import riskmgmt as RM
        return RM.issb_s2_draft(req.get("outputs") or {}, req.get("org"))

    @prot.get("/risk/validation-pack")
    def risk_validation(demo: Optional[str] = None, evaluation: Optional[str] = None, tests: Optional[str] = None) -> dict:
        """Paths are read only from the engine's own output folder (CRI_HOME), never from arbitrary locations."""
        import os
        from .. import riskmgmt as RM
        root = RM._home().resolve()

        def safe(p):
            if not p:
                return None
            q = (root / p).resolve()
            return str(q) if str(q).startswith(str(root)) else None
        return RM.validation_pack(safe(demo) or safe("llm/demo_report.json"), safe(evaluation) or safe("llm/eval.json"), tests)

    # ── insurer and bank bridges (cri.oasis, cri.creditbridge) ─────────────────────────────────────────────────────────────────
    def _rows(v):
        from .. import oasis as OA
        if isinstance(v, str):
            return OA.parse_csv(v)
        if isinstance(v, list):
            return v
        return []

    @prot.post("/oed/import")
    def oed_import(req: dict) -> dict:
        """{location: CSV text or rows, account?: CSV text or rows, occupancy_map?: {code: sector}, fx?: {EUR: usd_per_unit}} -> sites + hierarchy + every row/term not carried over."""
        from .. import oasis as OA
        loc = _rows(req.get("location"))
        if not loc:
            raise HTTPException(422, "location (OED location file as CSV text or a list of rows) required")
        if len(loc) > 20000:
            raise HTTPException(422, "at most 20,000 locations per call")
        return OA.read_oed(loc, _rows(req.get("account")) or None, req.get("occupancy_map"), req.get("fx"))

    @prot.post("/oed/run")
    def oed_run(req: dict) -> dict:
        """Import an OED exposure file, run the insurer simulation, and return the result with ORD-style tables (and their CSV text)."""
        from .. import oasis as OA, insurance as INS
        imp = oed_import(req)
        if not imp["sites"]:
            raise HTTPException(422, {"error": "no location could be loaded", "warnings": imp["warnings"][:20]})
        try:
            res = INS.run(imp["sites"], n_years=int(req.get("n_years", 5000)), seed=int(req.get("seed", 11)), reinsurance_program=req.get("reinsurance"))
        except INS.InsuranceError as e:
            raise HTTPException(422, str(e))
        t = OA.ord_tables(res, imp["meta"])
        return {"import": {k: imp[k] for k in ("read", "loaded", "skipped", "hierarchy", "warnings")}, "result": res, "ord": t, "ord_csv": {k: OA.to_csv(v) for k, v in t.items() if isinstance(v, list) and v and isinstance(v[0], dict)}}

    @prot.get("/credit/schema")
    def credit_schema() -> dict:
        from .. import creditbridge as CR
        return CR.SCHEMA

    @prot.post("/credit/assess")
    def credit_assess(req: dict) -> dict:
        """{entities: [...Entity -> Facility -> Collateral...], return_period?, n_years?, seed?} -> LTV, covenant effects and portfolio view. No PD/LGD is produced."""
        from .. import creditbridge as CR
        try:
            return CR.assess(req.get("entities") or [], int(req.get("return_period", 100)), int(req.get("n_years", 5000)), int(req.get("seed", 11)), float(req.get("corr_scale", 1.0)))
        except ValueError as e:
            raise HTTPException(422, str(e))

    @prot.post("/cases/warm")
    def cases_warm() -> dict:
        from . import answer as AN
        return AN.warm(force=True)

    @prot.get("/lab/domains")
    def lab_domains() -> dict:
        from ..modellab import lab as L
        return L.DOMAINS

    @prot.post("/lab/run")
    def lab_run(req: LabReq) -> dict:
        """Start a Model Lab study as a background job; poll /resident/job/{id}. Result blocks carry the report links."""
        from ..modellab import lab as L
        if req.domain not in L.DOMAINS:
            raise HTTPException(status_code=422, detail=f"unknown domain; available: {', '.join(L.DOMAINS)}")
        opts: dict = {}
        if req.domain == "flood":
            for k in ("return_period", "radius_km", "manning", "cn"):
                if getattr(req, k) is not None:
                    opts[k] = getattr(req, k)
            if "radius_km" in opts:
                opts["radius_km"] = min(5.0, opts["radius_km"])
            if req.observed_depths:
                opts["observed"] = {"points": [p for p in req.observed_depths if isinstance(p, dict) and "depth_m" in p], "source": "supplied by the user"}
        elif req.domain != "air":
            if req.radius_km is not None and req.domain in ("landslide", "earthquake"):
                opts["radius_km"] = req.radius_km
            if req.return_period is not None and req.domain == "landslide":
                opts["return_period"] = req.return_period
        else:
            if req.city_km is not None:
                opts["city_km"] = req.city_km
            if req.pm25_series:
                opts["obs"] = {"pm25": req.pm25_series, "source": "supplied ground-station series"}
        holder: dict = {}

        def work():
            out = L.run_study(req.domain, req.lat, req.lon, req.name, opts, progress=lambda m: holder["j"].__setitem__("progress", m))
            r = out["result"]
            return {"reply": out["story"]["title"] + "\n" + out["story"].get("standfirst", ""),
                    "blocks": [{"type": "files", "items": out["files"]}, {"type": "table", "head": ["", ""], "rows": L.summary(r)}], "actions": []}
        holder["j"] = A._job(work, f"{req.domain} model study")
        return {"job": holder["j"]["id"]}

    @prot.post("/portfolio/run")
    def portfolio_run(req: PortfolioReq) -> dict:
        """Screen a CSV of assets (name, lat, lon[, value]) for several hazards as a background job. Assets in the same 0.25° cell share one study; results are stored so a re-run resumes."""
        from ..modellab import portfolio as PF, lab as L
        try:
            assets = PF.parse_assets(req.csv)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        hz = [h for h in (req.hazards or PF.DEFAULT_HAZARDS) if h in L.DOMAINS and h not in ("flood", "air")] or list(PF.DEFAULT_HAZARDS)
        holder: dict = {}

        def work():
            import os, time as _t
            from pathlib import Path
            store = Path(os.environ.get("CRI_CACHE_DIR", Path.home() / ".climrisk" / "cache")) / "portfolio.jsonl"
            store.parent.mkdir(parents=True, exist_ok=True)
            res = PF.screen(assets, hz, store=str(store), progress=lambda m: holder["j"].__setitem__("progress", m))
            head, rows = PF.table(res)
            name = "portfolio-" + _t.strftime("%Y%m%d-%H%M%S")
            (A.report_dir() / (name + ".csv")).write_text(PF.to_csv(res), encoding="utf-8")
            d = PF.digest(res)
            lines = [f"{len(assets)} assets in {res['cells']} grid cells; {res['runs']} new studies run, {res['reused']} reused."]
            if d["movers"]:
                lines.append("Largest projected change by mid-century (SSP2-4.5): " + "; ".join(f"{n} ({h}) {c:+.1f}{u}" for n, h, c, u in d["movers"][:3]) + ".")
            if d["weak"]:
                lines.append(f"{len(d['weak'])} asset-hazard results have thin evidence (grade C/D or a failed hindcast): check the table.")
            return {"reply": "\n".join(lines), "blocks": [{"type": "files", "items": [{"label": "Download the table (CSV)", "url": f"/resident/file/{name}.csv"}]}, {"type": "table", "head": head, "rows": rows[:200]}], "actions": []}
        holder["j"] = A._job(work, f"portfolio screening of {len(assets)} assets")
        return {"job": holder["j"]["id"]}

    @pub.get("/file/{name}")
    def report_file(name: str) -> Response:
        import re as _re
        if not _re.fullmatch(r"[A-Za-z0-9_\-]+\.(html|pdf|csv)", name):
            raise HTTPException(status_code=404, detail="no such file")
        p = A.report_dir() / name
        if not p.exists():
            raise HTTPException(status_code=404, detail="no such file")
        if name.endswith(".csv"):
            return Response(p.read_text(encoding="utf-8"), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{name}"'})
        if name.endswith(".pdf"):
            return Response(p.read_bytes(), media_type="application/pdf", headers={"Content-Disposition": f'inline; filename="{name}"'})
        return Response(p.read_text(encoding="utf-8"), media_type="text/html")

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
