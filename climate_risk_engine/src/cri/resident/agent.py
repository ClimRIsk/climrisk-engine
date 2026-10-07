"""
cri/resident/agent.py — the resident agent: understands a request (trained NLU), plans an engine tool chain, runs it, remembers.

No external language model is involved. Understanding = cri.resident.nlu (trained in-engine); answers are composed from engine
results with templates, so every number in a reply comes from a tool call and is shown with its source. Long jobs (twin builds, hazard
research, risk analysis) run in background threads; the dock polls the job.
"""
from __future__ import annotations

import datetime as dt
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

from . import linkedin as LI
from . import memory as M
from . import nlu as N

_POOL = ThreadPoolExecutor(max_workers=3)
JOBS: dict = {}
_GEO: dict = {}

REPORT_TYPES = {"report_physical": "physical", "report_transition": "transition", "report_financial": "financial", "report_lca": "lca",
                "report_iso14067": "iso14067", "report_carbon": "carbon_audit", "report_cbam": "cbam"}

HELP = ("I can: **build a 3D twin** of a site and replay flood and wind on it; **find protection** and verify it; **build scenarios** and rank solutions; "
        "**write story reports** (physical, transition, financial, LCA, ISO 14067, carbon audit, CBAM); **run the risk analyst** on a company or coordinates; "
        "**list a company's assets**; **research** live hazards and map them onto the asset register; **draft LinkedIn posts** (and publish only after you approve); "
        "and **remember** facts about your sites. Try: \"build a 3D twin of 22.2645, 91.805\", \"how do I protect it?\", \"scan this week's events\", \"write the CBAM report for Holcim\".")

FAQ = [
    (r"solver|hydraul|how.*flood.*(model|calcul)", "Flood depths come from a 2D shallow-water solver (local-inertial form, Bates 2010) on 9 m terrain, with every building a solid obstruction. "
     "It is forced with the water level implied by the JRC/Deltares flood map, runs 3 hours (1.5 h rise, then hold), and is checked for mass conservation. "
     "Limits: the map is the ceiling for the level, event duration is assumed, and rainfall, drains, pumps and waves are not modelled."),
    (r"design wind|beyond design", "Design wind is the gust a structure type is assumed to be built for (e.g. silo, stack, roof sheeting). When the gust at the structure's height — wind rises with height — "
     "exceeds it, the state moves from minor to severe to failure by the ratio. The design values are assumptions by structure kind; edit them in the viewer."),
    (r"rating|score|rank", "Ratings in the engine come from screening hazards (cyclone return level, ground elevation, flood-map depths) and, in the risk analyst, Monte Carlo loss distributions. Open the finding for its inputs."),
    (r"assum|limit|reliab|accura", "Known limits: flood level is capped by the map; terrain is 9 m (Terrarium) not survey grade; heights, equipment levels and design winds are assumed by structure kind until you edit them; "
     "ERA5 wind smooths peaks, so extreme-wind figures are floors; tornado and downburst are not modelled."),
]


def _job(fn: Callable, label: str) -> dict:
    jid = uuid.uuid4().hex[:10]
    JOBS[jid] = {"id": jid, "label": label, "status": "running", "t0": time.time(), "result": None, "error": None}

    def run():
        try:
            JOBS[jid]["result"] = fn()
            JOBS[jid]["status"] = "done"
        except Exception as e:                                       # noqa: BLE001
            JOBS[jid]["status"] = "failed"; JOBS[jid]["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    _POOL.submit(run)
    return JOBS[jid]


def job_status(jid: str) -> dict:
    j = JOBS.get(jid)
    if not j:
        return {"status": "unknown"}
    return {k: j[k] for k in ("id", "label", "status", "error")} | {"elapsed_s": round(time.time() - j["t0"], 1), "reply": (j.get("result") or {}).get("reply") if j["status"] == "done" else None,
                                                                    "blocks": (j.get("result") or {}).get("blocks") if j["status"] == "done" else None,
                                                                    "actions": (j.get("result") or {}).get("actions") if j["status"] == "done" else None}


# ── place resolution ─────────────────────────────────────────────────────────

def geocode(q: str) -> Optional[dict]:
    q = q.strip()
    if q in _GEO:
        return _GEO[q]
    try:
        import requests
        r = requests.get("https://nominatim.openstreetmap.org/search", params={"q": q, "format": "json", "limit": 1},
                         headers={"User-Agent": "ClimRisk-resident/1.0"}, timeout=12)
        j = r.json()
        v = {"lat": float(j[0]["lat"]), "lon": float(j[0]["lon"]), "name": j[0].get("display_name", q)[:80]} if j else None
    except Exception:                                                # noqa: BLE001
        v = None
    _GEO[q] = v
    return v


def _place_words(text: str, ent: dict) -> Optional[str]:
    m = re.search(r"\b(?:at|in|of|for|on|near|around)\s+(?:the\s+)?([A-Z][\w\-']+(?:\s+[A-Z][\w\-']+){0,3})", text)
    return m.group(1) if m else None


def resolve_place(text: str, ent: dict, session: str) -> Optional[dict]:
    if "lat" in ent:
        return {"lat": ent["lat"], "lon": ent["lon"], "name": f'{ent["lat"]:.4f}, {ent["lon"]:.4f}', "source": "coordinates in the message"}
    w = _place_words(text, ent)
    if w:
        for f in M.facts(w, 5):
            kv = f.get("kv") or {}
            if "lat" in kv:
                return {"lat": kv["lat"], "lon": kv["lon"], "name": w, "source": "your saved note"}
        g = geocode(w)
        if g:
            return {**g, "name": w, "source": "OpenStreetMap geocoder"}
    last = M.get_state("last_place")
    if last and re.search(r"\b(it|this|there|here|that site|the site|my plant|the plant)\b", text.lower()):
        return {**last, "source": "the site we just discussed"}
    if last and not w:
        return {**last, "source": "the last site we worked on"}
    return None


# ── the agent ────────────────────────────────────────────────────────────────

def _blk(kind: str, **kw) -> dict:
    return {"type": kind, **kw}


class Agent:
    def __init__(self, twin_builder: Optional[Callable] = None, analyser: Optional[Callable] = None, scanner: Optional[Callable] = None, lab: Optional[Callable] = None):
        from .. import asset_twin as AT
        from . import research as R, scenario as SC
        self.AT, self.R, self.SC = AT, R, SC
        self.twin_builder = twin_builder or (lambda **k: AT.build_twin(**k))
        self.analyser = analyser or AT.analyse
        self.scanner = scanner or R.scan
        self.lab = lab or SC.lab

    # entry
    def handle(self, session: str, text: str, intent: Optional[str] = None) -> dict:
        text = (text or "").strip()
        if not text:
            return {"reply": "Ask me something — for example \"build a 3D twin of 22.2645, 91.805\".", "blocks": [], "actions": [], "intent": "empty"}
        quick = self._autopilot_commands(session, text)
        if quick:
            return quick
        model = N.get_model()
        pred = model.predict(text)
        ent = N.entities(text)
        comp = N.find_company(text)
        if intent:                                                     # user confirmed an intent from the clarification buttons
            M.add_feedback(text, pred["intent"], intent)
            pred = {"intent": intent, "confidence": 1.0, "top": pred["top"]}
        M.add_message(session, "user", text, pred["intent"], {"entities": ent, "company": comp, "top": pred["top"]})
        try:
            out = self._dispatch(session, text, pred, ent, comp)
        except Exception as e:                                         # noqa: BLE001
            out = {"reply": f"That failed inside the engine: {type(e).__name__}: {str(e)[:200]}. Nothing was changed.", "blocks": [], "actions": []}
        out["intent"] = pred["intent"]; out["confidence"] = pred["confidence"]; out["top"] = pred["top"]; out["entities"] = ent
        M.add_message(session, "agent", out["reply"], pred["intent"], {"actions": out.get("actions"), "job": out.get("job")})
        return out

    def _autopilot_commands(self, session, text) -> Optional[dict]:
        """Inbox, autopilot and watch commands: keyword-routed, no model needed."""
        import re as _re
        from . import autopilot as AP
        t = text.lower().strip()
        out = None
        if _re.search(r"\b(inbox|while i was (away|gone)|what.?s new|what did you (do|find)|any news|briefing|catch me up|autopilot|background)\b", t):
            if _re.search(r"run|now|again|refresh", t) and "autopilot" in t:
                import threading
                threading.Thread(target=AP.run_cycle, args=("asked in chat",), daemon=True).start()
                out = {"reply": "Started a cycle: scanning hazards, checking watched sites and companies. The findings will appear here in a minute or two.", "blocks": [], "actions": []}
            else:
                items = AP.inbox(False)[:6]; st = AP.status()
                if not items:
                    out = {"reply": "The inbox is empty — the autopilot has not finished a cycle yet" + (f' (next in {max(0, round((st["next_run"] - time.time()) / 60))} min)' if st.get("next_run") else "") + ". Say “run the autopilot now” to start one.", "blocks": [], "actions": []}
                else:
                    blocks = []
                    for i in items:
                        blocks.append(_blk("note", text=f'{i["title"]} — {i["detail"]}'))
                        blocks += i.get("blocks", [])[:1]
                    acts = [a for i in items[:2] for a in i.get("actions", [])][:4]
                    out = {"reply": f'{st["unread"]} new item(s). Autopilot: {st["cycles"]} cycle(s) run, every {st["interval_min"]} min; I started the first one when the app opened.', "blocks": blocks, "actions": acts}
                    AP.mark_read()
        m = _re.match(r"\s*(?:please\s+)?(?:watch|monitor|keep an eye on)\s+(?:the\s+)?(.+?)\s*$", t)
        if out is None and m and not _re.search(r"\d+\.\d+", t):
            name = text.strip()[len(text.strip()) - len(m.group(1)):]
            comp = N.find_company(name)
            try:
                from ..api.agents import _load_watchlist, _save_watchlist
                wl = _load_watchlist()
                label = comp["name"] if comp else name
                if not any(w["company_name"].lower() == label.lower() for w in wl):
                    wl.append({"company_name": label, "sector": None, "alert_email": None, "notes": "added by the resident agent", "added_at": dt.datetime.utcnow().isoformat()})
                    _save_watchlist(wl)
                out = {"reply": f"Watching {label}. The autopilot re-assesses it every cycle and tells you when expected loss moves by 10% or more.", "blocks": [], "actions": [{"type": "say", "label": "Run the autopilot now", "text": "run the autopilot now"}]}
            except Exception as e:                                       # noqa: BLE001
                out = {"reply": f"Could not add it to the watchlist: {e}", "blocks": [], "actions": []}
        if out:
            out.setdefault("intent", "autopilot"); out.setdefault("confidence", 1.0); out.setdefault("top", []); out.setdefault("entities", {})
            M.add_message(session, "user", text, "autopilot", {}); M.add_message(session, "agent", out["reply"], "autopilot", {})
        return out

    def _dispatch(self, session, text, pred, ent, comp) -> dict:
        it = pred["intent"]
        if it == "unknown":
            top = [t for t in pred["top"][:3] if t["p"] > 0.08]
            return {"reply": "I'm not sure what you want. Did you mean one of these? (Your choice teaches me.)", "blocks": [],
                    "actions": [{"type": "intent", "intent": t["intent"], "label": t["intent"].replace("_", " "), "text": text} for t in top]}
        if it == "smalltalk":
            return {"reply": "Hello — ready when you are. " + "Say what you want to assess, or ask what I can do.", "blocks": [], "actions": []}
        if it == "help":
            return {"reply": HELP, "blocks": [], "actions": []}
        if it == "remember":
            return self._remember(text, ent, comp)
        if it == "recall":
            return self._recall(text, comp)
        if it == "history":
            return self._history()
        if it == "explain":
            return self._explain(text)
        if it == "assets":
            return self._assets(comp, text)
        if it == "risk_analysis":
            return self._risk(session, text, ent, comp)
        if it == "research":
            return self._research(session, text, comp)
        if it == "draft_post":
            return self._draft(session, text)
        if it == "publish_post":
            return self._publish_request()
        if it in ("twin", "protect", "scenario"):
            return self._site(session, text, it, ent)
        if it in REPORT_TYPES:
            return self._report(session, text, it, ent, comp)
        return {"reply": "I recognised the request but have no handler for it yet.", "blocks": [], "actions": []}

    # memory
    def _remember(self, text, ent, comp):
        pl = _place_words(text, ent)
        subject = (comp["name"] if comp else pl or "general")
        kv = {}
        if "lat" in ent:
            kv.update(lat=ent["lat"], lon=ent["lon"])
        for k in ("metres", "wind_ms", "year"):
            if k in ent:
                kv[k] = ent[k]
        M.add_fact(subject, text, kv)
        if "lat" in ent:
            M.set_state("last_place", {"lat": ent["lat"], "lon": ent["lon"], "name": subject})
        return {"reply": f"Noted under “{subject}”: {text}. I'll use it next time — for site values, edit the component in the 3D viewer so the assessment itself changes.", "blocks": [], "actions": []}

    def _recall(self, text, comp):
        pl = _place_words(text, {}) or (comp["name"] if comp else None)
        fs = M.facts(pl, 12)
        if not fs:
            return {"reply": "I have nothing saved" + (f" about {pl}" if pl else "") + " yet. Tell me with \"remember that …\".", "blocks": [], "actions": []}
        return {"reply": "Here is what I have:", "blocks": [_blk("list", items=[f'{f["subject"]}: {f["text"]}' for f in fs])], "actions": []}

    def _history(self):
        rs = M.runs(10); ps = M.posts(6)
        items = [f'{time.strftime("%d %b %H:%M", time.localtime(r["ts"]))} · {r["intent"]} · {r["summary"]}' for r in rs] + [f'post #{p["id"]} · {p["status"]} · {p["title"]}' for p in ps]
        return {"reply": "Recent activity:" if items else "Nothing yet.", "blocks": [_blk("list", items=items)] if items else [], "actions": []}

    def _explain(self, text):
        low = text.lower()
        for pat, ans in FAQ:
            if re.search(pat, low):
                return {"reply": ans, "blocks": [], "actions": []}
        last = M.get_state("last_result")
        if last:
            return {"reply": "About the last result: " + last.get("explain", last.get("summary", "")), "blocks": [], "actions": []}
        return {"reply": FAQ[3][1], "blocks": [], "actions": []}

    # data
    def _assets(self, comp, text):
        if not comp:
            return {"reply": "Which company? Name it and I'll list its assets from the register.", "blocks": [], "actions": []}
        from ..ai.tools import execute_tool
        r = execute_tool("get_company_assets", {"company_name": comp["query"]})
        M.add_run("assets", {"company": comp["query"]}, [], bool(r.get("found")), f'{comp["name"]}: {r.get("n_assets", 0)} assets')
        if not r.get("found"):
            return {"reply": f'No assets found for “{comp["query"]}” in the register (heavy industry only).', "blocks": [], "actions": []}
        rows = [[a["name"], a["country"], a["sector"], a["screen_level"], a["cyclone_1in100"]] for a in r["most_exposed_assets"]]
        s = r["summary"]
        M.set_state("last_result", {"summary": f'{comp["name"]} assets', "explain": "Screening only: cyclone from IBTrACS + Holland model, flood from point elevation. Open the twin for site detail."})
        return {"reply": f'{comp["name"]}: {r["n_assets"]} assets; {round(100 * (s.get("share_assets_high_or_critical") or 0))}% screen HIGH or CRITICAL.',
                "blocks": [_blk("table", head=["Asset", "Country", "Sector", "Screen", "1-in-100 cyclone"], rows=rows)],
                "actions": []}

    def _risk(self, session, text, ent, comp):
        args = {}
        if "lat" in ent:
            args = {"lat": ent["lat"], "lon": ent["lon"]}
        elif comp:
            args = {"company_name": comp["query"]}
        else:
            return {"reply": "Give me a company in the asset register or coordinates, and I'll run physical → policy → transition → financial.", "blocks": [], "actions": []}

        def work():
            from ..ai.tools import execute_tool
            r = execute_tool("run_risk_analyst", args)
            if r.get("error") or r.get("found") is False:
                return {"reply": r.get("error") or r.get("note"), "blocks": [], "actions": []}
            M.add_run("risk_analysis", args, [], True, (r.get("headline") or str(r.get("companies")))[:200])
            if "scenarios_pv_usd_2026_2050" in r:
                rows = [[k, f'{v.get("mean", v.get("total_mean", 0)) / 1e6:,.1f}', f'{v.get("p95", v.get("total_p95", 0)) / 1e6:,.1f}'] for k, v in r["scenarios_pv_usd_2026_2050"].items()]
            else:
                rows = []
            M.set_state("last_result", {"summary": "risk analysis", "explain": "Losses are Monte Carlo distributions of present value 2026–2050 under NGFS Phase 5 scenarios; see the report for inputs."})
            return {"reply": (r.get("headline") or "Risk analysis complete.") + " (PV to 2050, US$ m: mean and 95th percentile below.)",
                    "blocks": [_blk("table", head=["Scenario", "Mean", "P95"], rows=rows)] if rows else [], "actions": []}
        j = _job(work, "risk analysis")
        return {"reply": "Running the risk analyst (physical → policy → transition → Monte Carlo). I'll post the result here.", "blocks": [], "actions": [], "job": j["id"]}

    # research + posts
    def _research(self, session, text, comp):
        keys = [comp["key"]] if comp and not comp["key"].startswith("group:") else None

        def work():
            sc = self.scanner(company_keys=keys)
            M.set_state("last_scan", sc)
            ev = self.R.pick_story_event(sc)
            M.add_run("research", {"company": comp["name"] if comp else None}, [sc["feeds"]], True, f'{len(sc["events"])} events; story event: {ev["title"] if ev else "none with assets in range"}')
            if not ev:
                return {"reply": f'Scanned {sc["asof"]}: {len(sc["events"])} GDACS events, none with register assets in range. Feeds: {sc["feeds"]}.', "blocks": [], "actions": []}
            rows = [[e["title"], e["alert"], e.get("country") or "", str(e["near"]["n"]), f'{(e["near"]["emissions_t"] or 0) / 1e6:.1f} Mt'] for e in sc["events"][:8] if e["near"]["n"]]
            M.set_state("last_place", {"lat": ev["lat"], "lon": ev["lon"], "name": ev["title"]})
            return {"reply": f'Top story: {ev["title"]} ({ev["alert"]}) — {ev["near"]["n"]} register assets within {ev["near"]["radius_km"]:.0f} km. Say "write a LinkedIn post" and I\'ll draft it.',
                    "blocks": [_blk("table", head=["Event", "Alert", "Country", "Assets in range", "CO₂e/yr"], rows=rows),
                               _blk("note", text=sc["note"] + " Feeds: " + ", ".join(f"{k} {v}" for k, v in sc["feeds"].items()))],
                    "actions": [{"type": "say", "label": "Draft the LinkedIn post", "text": "write a LinkedIn post about this"},
                                {"type": "open", "label": "Open 3D twin at event centre", "url": f'/asset-twin/dashboard?lat={ev["lat"]:.4f}&lon={ev["lon"]:.4f}&auto=1&name=' + ev["title"].replace(" ", "%20")}]}
        j = _job(work, "research scan")
        return {"reply": "Scanning GDACS, USGS and NOAA and mapping events onto the asset register…", "blocks": [], "actions": [], "job": j["id"]}

    def _draft(self, session, text):
        tw = M.get_state("last_twin_story")
        sc = M.get_state("last_scan")
        use_twin = tw and re.search(r"\b(twin|site|finding|findings|flood|wind|protect)\b", text.lower()) and not re.search(r"\b(event|storm|week)\b", text.lower())
        if use_twin or (tw and not sc):
            post = {"text": tw["post_text"], "title": tw["title"]}
            img = None
            try:
                from .. import story as S
                img = S.chart_card(tw["title"], [tuple(x) for x in tw["numbers"]], footer="ClimRisk Asset Twin · hydraulic solver + design-wind check")
            except Exception:                                          # noqa: BLE001
                pass
            pid = M.add_post(post["title"], post["text"], img, {"from": "twin"})
        elif sc:
            ev = self.R.pick_story_event(sc)
            if not ev:
                return {"reply": "The last scan found no event with assets in range, so there is nothing grounded to post about. Run another scan or build a twin first.", "blocks": [], "actions": []}
            story = self.R.build_story(sc, ev)
            post = self.R.post_from_story(story, ev, sc)
            pid = M.add_post(post["title"], post["text"], post["image"], {"from": "research", "event": ev["id"]})
        else:
            return {"reply": "I have nothing to write from yet. Run a research scan (“scan this week's events”) or build a twin first, then ask for the post.", "blocks": [], "actions": [{"type": "say", "label": "Scan this week", "text": "scan this week's events"}]}
        M.set_state("pending_post", pid)
        return {"reply": "Draft ready. Nothing is posted until you approve it.", "blocks": [_blk("post", id=pid, text=post["text"], has_image=True, image_url=f"/resident/post/{pid}/image")],
                "actions": [{"type": "publish", "label": "Approve & publish to LinkedIn", "post_id": pid}, {"type": "copy", "label": "Copy text", "post_id": pid}],
                "linkedin": LI.status()}

    def _publish_request(self):
        pid = M.get_state("pending_post")
        p = M.get_post(pid) if pid else None
        if not p:
            return {"reply": "There is no draft waiting. Ask me to draft a post first.", "blocks": [], "actions": []}
        return {"reply": ("I don't publish from a chat message — press the button to approve this exact text and image." if LI.configured()
                          else "Publishing needs LinkedIn credentials in the engine .env (see below); the draft is saved and can be copied."),
                "blocks": [_blk("post", id=pid, text=p["text"], has_image=bool(p.get("image")), image_url=f"/resident/post/{pid}/image")],
                "actions": [{"type": "publish", "label": "Approve & publish to LinkedIn", "post_id": pid}], "linkedin": LI.status()}

    # site work
    def _site(self, session, text, it, ent):
        pl = resolve_place(text, ent, session)
        if not pl:
            return {"reply": "Which site? Give coordinates (e.g. 22.2645, 91.805), a place name, or a plant name I know.", "blocks": [], "actions": []}
        M.set_state("last_place", pl)
        sector = ent.get("sector")
        try:                                                           # sites the user works on are watched by the autopilot
            from . import autopilot as AP
            AP.register_site(pl["lat"], pl["lon"], pl["name"], sector)
        except Exception:                                              # noqa: BLE001
            pass
        url = f'/asset-twin/dashboard?lat={pl["lat"]:.5f}&lon={pl["lon"]:.5f}&auto=1&name={pl["name"].replace(" ", "%20")}' + (f'&sector={sector}' if sector else "")
        open_act = {"type": "open", "label": "Open 3D twin", "url": url}
        if it == "twin":
            return {"reply": f'Opening the 3D twin at {pl["name"]} ({pl["source"]}). It builds from OpenStreetMap footprints and terrain, then runs the hydraulic solver.', "blocks": [], "actions": [open_act]}

        def work():
            twin = self.twin_builder(lat=pl["lat"], lon=pl["lon"], name=pl["name"], sector=sector)
            an = self.analyser(twin)
            fnd = an["findings"]
            M.add_run(it, {"place": pl}, an.get("transcript", [])[:12], True, f'{len(fnd)} findings at {pl["name"]}')
            M.set_state("last_result", {"summary": f'twin analysis at {pl["name"]}', "explain": "; ".join(an.get("notes", [])[:2]) or FAQ[0][1]})
            story_pts = [f["title"] for f in fnd[:3]]
            M.set_state("last_twin_story", {"title": f'What a flood and a storm do to {pl["name"]}', "numbers": [[str(len(twin["components"])), "components modelled"], [str(len(fnd)), "findings"],
                                            [str(sum(1 for f in fnd if f["hazard"] == "flood")), "flood findings"]],
                                            "post_text": self._twin_post(pl, twin, an)})
            blocks = []
            if fnd:
                blocks.append(_blk("list", items=[f["title"] for f in fnd[:6]]))
            if it in ("protect", "scenario"):
                lab = self.lab(twin, an)
                rows = [[s["name"], "yes" if s["resolves_base"] else "no", f'+{s["flood_margin_m"]} m' if s["flood_margin_m"] is not None else "–",
                         f'+{s["wind_margin_pct"]}%' if s["wind_margin_pct"] is not None else "–"] for s in lab["solutions"][:6]]
                blocks.append(_blk("table", head=["Solution", "Resolves base", "Flood margin", "Wind margin"], rows=rows, caption="Ranked by physical performance; no costs invented."))
                blocks.append(_blk("note", text=lab["recommendation"] + " " + " ".join(lab["limits"])))
                reply = f'{pl["name"]}: {len(fnd)} finding(s). ' + lab["recommendation"]
            else:
                reply = f'{pl["name"]}: {len(fnd)} finding(s).' if fnd else f'{pl["name"]}: no component is reached by the modelled water or exceeds design wind.'
            if an.get("notes"):
                blocks.append(_blk("note", text=" ".join(an["notes"][:3])))
            return {"reply": reply, "blocks": blocks, "actions": [open_act, {"type": "say", "label": "Draft a LinkedIn post from this", "text": "write a LinkedIn post about the twin findings"}]}
        j = _job(work, f"{it} at {pl['name']}")
        return {"reply": f'Working on {pl["name"]} ({pl["source"]}): building the twin, running the solver and verifying protection — about a minute.', "blocks": [], "actions": [open_act], "job": j["id"]}

    @staticmethod
    def _twin_post(pl, twin, an) -> str:
        fnd = an["findings"]
        lines = [f'What happens to {pl["name"]} when the water and the wind come?', ""]
        for i, f in enumerate(fnd[:3], 1):
            lines += [f"{i}. {f['title']}.", ""]
        if not fnd:
            lines += ["In the modelled scenarios nothing is reached by water or beyond design wind — and the margins are what matter.", ""]
        lines += ["Modelled with a 2D hydraulic solver on real footprints and terrain; heights and design winds are assumptions until the operator edits them.", "",
                  "#ClimateRisk #PhysicalRisk #Resilience"]
        return "\n".join(lines)

    # reports
    def _report(self, session, text, it, ent, comp):
        rtype = REPORT_TYPES[it]
        q = {"type": rtype}
        pl = None
        if rtype == "physical":
            pl = resolve_place(text, ent, session)
            if pl:
                q.update(lat=f'{pl["lat"]:.5f}', lon=f'{pl["lon"]:.5f}', name=pl["name"])
                M.set_state("last_place", pl)
        if comp:
            q["company"] = comp["query"]
        if ent.get("sector"):
            q["sector"] = ent["sector"]
        if rtype == "physical" and not pl:
            return {"reply": "Which site is the physical risk report for? Give coordinates or a place.", "blocks": [], "actions": []}
        qs = "&".join(f"{k}={str(v).replace(' ', '%20')}" for k, v in q.items())
        M.add_run(it, q, [], True, f"opened {rtype} report workbench")
        return {"reply": f"Opening the {rtype.replace('_', ' ')} story report workbench with what I understood ({', '.join(f'{k}={v}' for k, v in q.items() if k != 'type') or 'nothing yet — fill in the form'}). "
                         "Pick the sector template, sections and client branding, then generate HTML, PDF or JSON.",
                "blocks": [], "actions": [{"type": "open", "label": f"Open {rtype.replace('_', ' ')} report", "url": f"/story-reports/dashboard?{qs}"}]}


_AGENT: Optional[Agent] = None


def get_agent() -> Agent:
    global _AGENT
    if _AGENT is None:
        _AGENT = Agent()
    return _AGENT
