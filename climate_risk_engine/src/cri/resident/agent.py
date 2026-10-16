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
        "**build and test physics models on my own** (flood inundation, air quality: I choose the equations, verify them, calibrate, validate and write the methods); "
        "**recall real precedents**: what happened to company assets and finances in past events (from SEC filings and checked case studies, every figure quoted with its source, and I say when the sources do not cover something); "
        "and **remember** facts about your sites. Try: \"how did Allstate respond to Hurricane Ian?\", \"closest past cases for a flood at a chemical plant on a river\",\"build a flood model for 22.2645, 91.805\", \"air quality model for 28.6139, 77.2090\", \"build a 3D twin of 22.2645, 91.805\", \"how do I protect it?\", \"scan this week's events\", \"write the CBAM report for Holcim\".")

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


def report_dir():
    from pathlib import Path
    d = Path(M.db_path()).parent / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def profile_rows(p: dict) -> list:
    """Table rows for a site profile. A null hazard is written as null, never as 0 or low."""
    order = {"modelled": 0, "constructed": 1, "none_found": 2, "null": 3}
    rows = []
    for e in sorted(p["hazards"].values(), key=lambda e: (order.get(e["status"], 4), e["label"])):
        if e["status"] == "null":
            val, sev = "null", "not assessed"
        elif e["status"] == "none_found":
            val, sev = "0 (absent here)", "NEGLIGIBLE"
        else:
            val = "-" if e["value"] is None else (f'{e["value"]:g} {e.get("unit") or ""}').strip()
            sev = e["severity"] or (f'grade {e["evidence_grade"]}' if e.get("evidence_grade") else "no severity scale")
        note = e.get("reason") or e.get("caveat") or e.get("method") or ""
        rows.append([e["label"], e["status"], val, sev, note[:140]])
    return rows


def run_site_profile(pl: dict, progress: Optional[Callable] = None, twin: Optional[dict] = None) -> dict:
    """Every hazard for one coordinate (siteprofile.profile) as an agent reply: summary, status table, history, null rules."""
    from .. import siteprofile as SP
    if twin is None:
        twin = SP.cached_twin(pl["lat"], pl["lon"])
    p = SP.profile(pl["lat"], pl["lon"], pl.get("name") or "Site", twin=twin, progress=progress)
    c = p["counts"]
    blocks = [_blk("table", head=["Hazard", "Status", "Value", "Severity", "Note"], rows=profile_rows(p),
                   caption=f'{c["modelled"]} modelled, {c["constructed"]} constructed, {c["none_found"]} established absent, {c["null"]} null (not assessed).')]
    hs = (p.get("history") or {})
    fe = hs.get("flood_events") or {}
    if fe.get("events"):
        blocks.append(_blk("table", head=["Recorded flood event", "Date", "Source"], rows=[[e.get("name") or e.get("id") or "", str(e.get("date") or "")[:10], e.get("source") or ""] for e in fe["events"][:10]],
                           caption=fe.get("note") or ""))
    elif fe.get("outside_dense_coverage"):
        blocks.append(_blk("note", text="No recorded flood events are listed for this place, but it lies outside the regions where the engine holds event records, so that is not evidence it has never flooded."))
    al = (p["hazards"].get("air_local") or {})
    srcs = (al.get("metrics") or {}).get("sources") or []
    if srcs:
        blocks.append(_blk("table", head=["Found nearby", "Km / bearing", "How it was found", "Confidence", "PM2.5 at the site, central (annual / worst day)"],
                           rows=[[f'{x["kind"]}: {x.get("name") or ""}'[:60], f'{x.get("distance_km")} / {x.get("bearing_deg")}', str(x.get("provenance"))[:90], x.get("confidence"),
                                  (f'{x["scenarios"]["central"]["pm25_annual_ug_m3"]["max"]} / {x["scenarios"]["central"]["pm25_worst_24h_ug_m3_max"]} µg/m3' if x.get("modelled") else "not modelled")] for x in srcs[:10]],
                           caption="Sources the engine found on its own. Tell me if one is wrong or missing and I will correct it."))
    blocks.append(_blk("note", text=" ".join(p["rules"])))
    return {"reply": SP.summarise(p), "blocks": blocks, "actions": [{"type": "say", "label": "Build a 3D twin here", "text": f'build a 3D twin of {pl["lat"]:.4f}, {pl["lon"]:.4f}'}], "profile": {"counts": c, "null_hazards": p["null_hazards"], "top_severity": p["top_severity"]}}


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
    return {k: j[k] for k in ("id", "label", "status", "error")} | {"progress": j.get("progress"), "partial": j.get("partial"), "elapsed_s": round(time.time() - j["t0"], 1), "reply": (j.get("result") or {}).get("reply") if j["status"] == "done" else None,
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
        quick = self._autopilot_commands(session, text) or self._knowledge_commands(session, text) or self._stress_commands(session, text) or self._scenario_commands(session, text) or self._source_commands(session, text) or self._profile_commands(session, text) or self._lab_commands(session, text) or self._case_commands(session, text) or self._extension_commands(session, text)
        if quick:
            if quick.get("intent") in (None, "knowledge", "learn"):      # autopilot commands log themselves
                quick.setdefault("intent", "knowledge")
                M.add_message(session, "user", text, quick["intent"], {}); M.add_message(session, "agent", quick["reply"], quick["intent"], {"job": quick.get("job")})
            quick.setdefault("confidence", 1.0); quick.setdefault("top", []); quick.setdefault("entities", {})
            return quick
        model = N.get_model()
        pred = model.predict(text)
        ent = N.entities(text)
        comp = N.find_company(text)
        if intent:                                                     # user confirmed an intent from the clarification buttons
            M.add_feedback(text, pred["intent"], intent)
            pred = {"intent": intent, "confidence": 1.0, "top": pred["top"]}
        elif self._is_info_question(text, ent, comp) and pred["intent"] in self._TOOLISH:
            pred = {"intent": "explain", "confidence": max(pred["confidence"], 0.6), "top": pred["top"]}   # a question about the world, not a request for a tool
        M.add_message(session, "user", text, pred["intent"], {"entities": ent, "company": comp, "top": pred["top"]})
        try:
            out = self._converse(session, text, pred, ent, comp) or self._dispatch(session, text, pred, ent, comp)
        except Exception as e:                                         # noqa: BLE001
            out = {"reply": f"That failed inside the engine: {type(e).__name__}: {str(e)[:200]}. Nothing was changed.", "blocks": [], "actions": []}
        out["intent"] = pred["intent"]; out["confidence"] = pred["confidence"]; out["top"] = pred["top"]; out["entities"] = ent
        M.add_message(session, "agent", out["reply"], pred["intent"], {"actions": out.get("actions"), "job": out.get("job")})
        return out

    _CTX_WORDS = re.compile(r"\b(it|this|that|these|those|here|there|my|our|we|the (?:site|plant|result|numbers?|profile|air|water|heat|flood)|why|so what|what should|what does (?:that|this)|what do you (?:think|make)|you said|earlier|again|more|detail|worried|concern(?:ed)?|problem)\b", re.I)

    def _converse(self, session, text, pred, ent, comp) -> Optional[dict]:
        """Free conversation through the local language model (no key, no cloud): greetings, follow-ups about the site under discussion,
        thinking out loud. Tool requests, coordinates, companies and plain factual questions keep their own routes."""
        from . import converse as CV, answer as AN
        it = pred["intent"]
        llm = AN.llm_status()["available"]
        if it == "smalltalk":
            return CV.converse(session, text) if llm else CV.no_model_reply(AN.llm_status()["why"], smalltalk=True)
        if not llm:
            return None
        if "lat" in ent or comp or self._ACT.search(text):
            return None
        has_ctx = bool(M.get_state("last_place") or M.get_state("last_result"))
        if it == "unknown" and not self._Q.search(text):
            return CV.converse(session, text)
        if has_ctx and it in ("unknown", "explain", "history", "recall") and self._CTX_WORDS.search(text):
            return CV.converse(session, text)
        return None

    _SSP_WORDS = (("ssp126", r"ssp\s*-?\s*1[\s.\-]*2\.?6|strong(?:ly)? mitigat\w*|paris|net[\s\-]*zero pathway|low emissions?"), ("ssp245", r"ssp\s*-?\s*2[\s.\-]*4\.?5|middle[\s\-]*of[\s\-]*the[\s\-]*road|moderate emissions?|current policies"),
                  ("ssp370", r"ssp\s*-?\s*3[\s.\-]*7\.?0|high emissions?|regional rivalry"), ("ssp585", r"ssp\s*-?\s*5[\s.\-]*8\.?5|very high emissions?|worst[\s\-]*case|business[\s\-]*as[\s\-]*usual|fossil[\s\-]*fuel(?:ed)? (?:future|pathway)"))

    def _stress_commands(self, session, text) -> Optional[dict]:
        """A staged scenario that chains a physical shock, a supplier, freight and a tariff (\"typhoon hits the hub, the Tier-2 supplier is down, air freight triggers a border tariff\").
        Pulls out the numbers the text states, runs the linked stress test on them (hazard, operations, supply, freight, tariff, cash) and says what it still needs.
        Missing inputs stay null; nothing is guessed."""
        from .. import stresstest as ST
        low = (text or "").lower().strip()
        if len(low) < 140 and re.search(r"stress[\s-]?test", low) and re.search(r"\b(open|show|launch|start|go to|take me|run|use|where)\b", low) and not ST.looks_like_stress_scenario(text):
            return {"reply": "The Stress Test runs one scenario through storm, plant, supplier, freight, tariff and cash, and each piece also runs on its own. Opening it now; "
                             "describe a scenario here in full if you would rather have me read the numbers out of your text.",
                    "intent": "stress_test", "blocks": [], "actions": [{"type": "open", "label": "Open the Stress Test", "url": "/stress-test/dashboard"}]}
        if not ST.looks_like_stress_scenario(text):
            return None
        spec = ST.extract_spec(text)
        ST.save_draft(spec)
        r = ST.run(spec, hazard_fn=lambda lat, lon, name: (_ for _ in ()).throw(RuntimeError("no hazard run in chat")))
        got = [k for k in ("hazard", "event", "supplier", "batch", "tariff") if spec.get(k)]
        rows = [[a["question"].split(":")[0], a["status"], a["answer"][:260] + ("…" if len(a["answer"]) > 260 else "")] for a in r["answers"]]
        need = r["inputs_missing"]
        reply = ("Read as a linked stress test. I took these from your text: " + (", ".join(got) if got else "nothing numeric") + ". "
                 "Hazard, operations, supplier delay, freight emissions, tariff and cash run together in the Stress Test screen, where each piece also runs on its own. "
                 + (("Still needed to finish it, and left null until you give them: " + "; ".join(need[:8]) + ". ") if need else "") +
                 "Coordinates for the hub and supplier let the engine place the storm on each site's own return-period curve.")
        return {"reply": reply, "intent": "stress_test", "blocks": [_blk("table", head=["Question", "Status", "Answer so far"], rows=rows)],
                "actions": [{"type": "open", "label": "Open the Stress Test", "url": "/stress-test/dashboard?draft=1"}]}

    def _scenario_commands(self, session, text) -> Optional[dict]:
        """Free-form climate scenarios: "what does the SSP5-8.5 world look like at Chattogram in 2070 for a 1-in-100 flood", "worst case sea level and storm by mid-century at 22.26, 91.80".
        Composes rain, river depth, sea level, heat and cyclone wind from the engine's own results, says how sure each is, and opens it in the 3D twin."""
        t = text.strip(); low = t.lower()
        ssp = next((k for k, rx in self._SSP_WORDS if re.search(rx, low)), None)
        yr = re.search(r"\b(20[3-9]\d|2100)\b", low)
        year = int(yr.group(1)) if yr else (2050 if re.search(r"mid[\s\-]*century", low) else 2090 if re.search(r"end of (?:the )?century|late[\s\-]*century", low) else None)
        climate = ssp or re.search(r"climate (?:change|scenario|future)|warmer (?:world|climate)|future (?:flood|climate|risk)", low)
        if not (climate and (year or ssp)) or re.match(r"\s*(what is|what are|explain|define|why|how does|how do)\b", low) and not re.search(r"\b(at|in|for|near)\b.*(\d+\.\d+|[A-Z][a-z]+)", t):
            return None
        if not re.search(r"flood|inundation|surge|sea[\s\-]*level|storm|rain|wind|cyclone|hurricane|typhoon|heat|drought|water[\s\-]*stress|wildfire|fire|scenario|risk|site|asset|plant|facility", low):
            return None
        ent = N.entities(t)
        pl = resolve_place(t, ent, session)
        if not pl:
            return {"reply": "Which place is the climate scenario for? Give coordinates or a city name, plus a pathway (for example SSP2-4.5) and a year.", "blocks": [], "actions": []}
        carried = pl.get("source", "").startswith(("the last site", "the site we just"))
        M.set_state("last_place", pl)
        ssp = ssp or "ssp245"; year = year or 2070
        rp = int(ent.get("rp") or 100)

        def work():
            from .. import climate_scenario as CS
            c = CS.compose(pl["lat"], pl["lon"], ssp, year)
            rows = [["Heavier extreme rain (1-day)", f"{100 * (c['rain']['median'] - 1):+.0f} %", f"{100 * (c['rain']['p10'] - 1):+.0f} to {100 * (c['rain']['p90'] - 1):+.0f} %", next(x["level"] for x in c["confidence"] if x["factor"] == "Heavier extreme rain")],
                    ["River flood depth", f"×{c['river_depth_factor']['median']:.2f}", f"×{c['river_depth_factor']['p10']:.2f} to ×{c['river_depth_factor']['p90']:.2f}", "low"]]
            if c["sea"].get("available"):
                a = c["sea"]["add_m"]; rows.append(["Sea level since 2020", f"{a['median']:+.2f} m", f"{a['p17']:+.2f} to {a['p83']:+.2f} m (likely range)", next(x["level"] for x in c["confidence"] if x["factor"] == "Sea-level rise")])
            if c["heat"]:
                rows.append(["Hottest day of the year", f"{c['heat']['median']:+.1f} °C", f"{c['heat']['p10']:+.1f} to {c['heat']['p90']:+.1f} °C", "medium-high"])
            rows.append(["Cyclone peak wind", f"×{c['wind']['scale']['median']:.3f}" if c["wind"]["tc_exposed"] else "unchanged", f"×{c['wind']['scale']['low']:.3f} to ×{c['wind']['scale']['high']:.3f}" if c["wind"]["tc_exposed"] else "not projected here", "low" if c["wind"]["tc_exposed"] else "n/a"])
            url = f"/asset-twin/dashboard?lat={pl['lat']:.5f}&lon={pl['lon']:.5f}&name={(pl.get('name') or 'Site').replace(' ', '%20')}&auto=1&climate={ssp}:{year}:{rp}"
            return {"reply": f"{c['ssp_label']}, {year} at {pl.get('name') or 'the site'} ({c['warming_k']:.1f} °C global warming against 1995–2014). Medians with the model range, and how sure the engine is of each. "
                             "Scenarios are not forecasts; the 3D twin replays the mapped 1-in-%d event with these changes applied and re-runs the flood solver." % rp,
                    "blocks": [_blk("table", head=["Factor", "Median change", "Range", "Confidence"], rows=rows), _blk("note", text=" ".join(c["limits"][:2]))],
                    "actions": [{"type": "open", "label": "Open in 3D (re-runs the flood solver)", "url": url}]}
        j = _job(work, "climate scenario")
        note = (f"Your message names no place, so I used {pl.get('name') or 'the last site'} ({pl.get('source')}); give coordinates or a city to change it. ") if carried else ""
        return {"reply": f"{note}Composing the {ssp.upper().replace('SSP', 'SSP')} scenario for {year} at {pl.get('name') or 'the site'} from the climate-model ensemble and sea-level projections …", "blocks": [], "actions": [], "job": j["id"], "intent": "scenario"}

    def _extension_commands(self, session, text) -> Optional[dict]:
        """Chat commands from private extensions (modules listed in CRI_EXTENSIONS or ~/.climrisk/extensions); none are loaded in a normal or commercial install."""
        from .. import extensions as X
        for m in X.modules():
            h = getattr(m, "handle", None)
            if h:
                try:
                    r = h(session, text)
                except Exception:                                  # noqa: BLE001
                    r = None
                if r:
                    return r
        return None

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
        if it == "unknown" and self._Q.search(text):                   # a real question the router has no tool for: answer it from what has been learned
            return self._ask(text)
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
            if re.search(pat, low) and re.search(r"\b(this|the|your|our|my)\b.*\b(engine|model|solver|result|twin|report|rating|score|finding)\b|\b(engine|solver|twin)\b", low):
                return {"reply": ans, "blocks": [], "actions": []}
        if re.search(r"\b(last|previous|that|this)\s+(result|finding|analysis|run)\b", low):
            last = M.get_state("last_result")
            if last:
                return {"reply": "About the last result: " + last.get("explain", last.get("summary", "")), "blocks": [], "actions": []}
        return self._ask(text)

    # open questions: answered from what the agent has learned, with sources; learns the topic first if it has not met it
    _Q = re.compile(r"^\s*(what|why|how|when|where|who|which|does|do|is|are|can|could|should|would|will|explain|describe|tell me|compare|define|give me|summari[sz]e|difference)\b|\?\s*$", re.I)

    _TOOLISH = {"twin", "protect", "scenario", "risk_analysis", "research", "assets", "explain", "unknown", "draft_post", "report_physical", "report_transition", "report_financial",
                "report_lca", "report_iso14067", "report_carbon", "report_cbam"}
    _ACT = re.compile(r"\b(build|run|create|make|write|generate|open|scan|draft|publish|post|protect|model|simulate|assess|analy[sz]e|list|show me|remember|watch|monitor|learn|forget|replay|retrain|find)\b", re.I)

    def _is_info_question(self, text: str, ent: dict, comp) -> bool:
        """'What is a storm surge?' is a question to answer from what has been learned, not a request to open a tool."""
        return bool(self._Q.search(text)) and not self._ACT.search(text) and "lat" not in ent and not comp

    def _ask(self, text: str) -> dict:
        prev = M.get_state("last_question")
        ctx = None
        q = text
        from . import knowledge as K
        if prev and len(K._terms(text)) <= 2:                       # a short follow-up: carry the subject of the last question
            q = prev["q"] + " " + text
            ctx = f'The user previously asked: "{prev["q"]}".'

        def work():
            from . import answer as AN
            try:
                if K._due("engine://method", 10 ** 9):                # the engine's own method notes are part of what it knows
                    K.learn_text("engine://method", "ClimRisk engine method notes", " ".join(a for _, a in FAQ) + " " + HELP.replace("**", ""))
                if K.stats()["passages"] < 50:                          # first question on a fresh install: read the seed sources now
                    K.learn_seed()
            except Exception:                                          # noqa: BLE001
                pass
            r = AN.ask(q, context=ctx)
            M.set_state("last_question", {"q": q, "ts": time.time()})
            M.add_run("ask", {"q": q[:200]}, [r["mode"]], bool(r["ok"]), (r["reply"] or "")[:200])
            blocks = []
            if r["sources"]:
                blocks.append(_blk("sources", items=[{"n": s["n"], "title": s["title"], "url": s["url"], "learned": s["learned"]} for s in r["sources"]]))
            note = []
            if r["learned"]:
                note.append("I had not read about this yet, so I just learned: " + "; ".join(r["learned"]) + ".")
            if r["ok"]:
                note.append("Written by: " + r["mode"] + ".")
                if r["mode"] == "extractive":
                    note.append("These are the most relevant sentences from the sources above." + (
                        f' The local model was running but {r["llm_note"]}.' if r.get("llm_note") else
                        " For fluent written answers, install a free local model (Ollama) — the engine detects it automatically."))
            if note:
                blocks.append(_blk("note", text=" ".join(note)))
            acts = [] if r["ok"] else [{"type": "say", "label": "Learn about this", "text": "learn about " + " ".join(K._terms(q)[:6])}]
            return {"reply": r["reply"], "blocks": blocks, "actions": acts}
        j = _job(work, "answering from what I have learned")
        return {"reply": "Looking through what I have learned" + (" — and reading up on it if it is new to me" if True else "") + "…", "blocks": [], "actions": [], "job": j["id"]}

    # real events and what happened to company assets and finances: answered from the case library (checked, cited), never from memory
    _ANALOG = re.compile(r"\b(closest|similar|comparable|analog(?:ue|y|ies)?|precedents?|historical (?:cases?|events?|examples?)|past (?:cases?|events?)|what (?:has )?worked|lessons? from|has this (?:ever )?happened)\b", re.I)
    _NOT_CASE = re.compile(r"\b(build|run|simulate|twin|publish|post|draft|scan|protect|retrain|forget|learn from)\b", re.I)

    def _case_commands(self, session, text) -> Optional[dict]:
        try:
            from . import casebase as CB
            if not CB.available() or self._NOT_CASE.search(text):
                return None
            analog = bool(self._ANALOG.search(text)) and len(text.split()) >= 5
            if not analog and not (self._Q.search(text) and CB.is_case_question(text)):
                return None
        except Exception:                                              # noqa: BLE001
            return None
        prev = M.get_state("last_question")
        from . import knowledge as K
        q = text
        ctx = None
        if not analog and prev and len(K._terms(text)) <= 2:
            q = prev["q"] + " " + text
            ctx = f'The user previously asked: "{prev["q"]}".'

        def work():
            from . import answer as AN
            r = AN.ask_analogs(text, assets=text) if analog else (AN.ask_cases(q, context=ctx) or AN.ask(q, context=ctx))
            M.set_state("last_question", {"q": q, "ts": time.time()})
            M.add_run("cases", {"q": q[:200], "analog": analog}, [r["mode"]], bool(r["ok"]), (r["reply"] or "")[:200])
            blocks = []
            if r["sources"]:
                blocks.append(_blk("sources", items=[{"n": s["n"], "title": s["title"], "url": s["url"], "learned": s["learned"]} for s in r["sources"]]))
            if r["ok"]:
                blocks.append(_blk("note", text="Written by: " + r["mode"] + ". Every figure above is quoted from the numbered source; the one line marked Analysis is my inference, not a source statement."))
            return {"reply": r["reply"], "blocks": blocks, "actions": []}
        j = _job(work, "looking up the case library")
        return {"reply": "Checking the case library of real events and company filings…", "blocks": [], "actions": [], "job": j["id"], "intent": "knowledge"}

    def _lab_commands(self, session, text) -> Optional[dict]:
        """"build a flood model for <place>", "air quality model for <city>", "what can the model lab do" — the engine runs a full study on its own."""
        from ..modellab import lab as L
        t = text.strip(); low = t.lower()
        if re.search(r"\b(model lab|modelling lab|modeling lab)\b", low) and re.search(r"\b(what|which|list|can|do)\b", low):
            rows = [[k, v["label"], v["needs"], v["validates_with"]] for k, v in L.DOMAINS.items()]
            names = ", ".join(v["label"].split(" (")[0].lower() for k, v in L.DOMAINS.items() if "dossier" not in v["label"])
            return {"reply": ("The Model Lab builds a physics model the way a researcher would, without being walked through it: it chooses the equations, fetches the data, builds and runs the model, "
                              "proves it is solved correctly against analytic solutions and conservation laws, calibrates on part of the record, tests on part it never saw, finds which assumption matters most, and writes the methods and results. "
                              "It says plainly whether the result is only verified, or also calibrated and validated, and for a hazard it has no verified solver for it researches the physics and data and tells you what is missing instead of guessing. "
                              f"It covers {names}. Say, for example: \"build a flood model for Chattogram\", \"earthquake hazard for Tokyo\" or \"air quality model for Delhi\"."),
                    "blocks": [_blk("table", head=["Domain", "Physics", "Needs", "Validates with"], rows=rows)], "actions": []}
        domain = L.detect_domain(t)
        verb = re.search(r"\b(build|construct|develop|create|make|run|simulate|simulation|calibrate|validate|set up|prepare|do|model|study)\b", low)
        short_place = None
        if domain not in (None, "flood", "air"):                         # flood/air phrases overlap the ordinary risk tools: those need an explicit build verb
            verb = verb or re.search(r"\b(hazard|return levels?|assess\w*|analy[sz]e)\b", low)
            short_place = len(low.split()) <= 9 and re.search(r"\b(for|in|near|at|around)\b|\d+\.\d+", low)
        if not domain or not (verb or short_place) or re.match(r"\s*(what|why|how|explain|define|tell|describe|list|which|is|are|does|do you)\b", low):
            return None
        ent = N.entities(t)
        pl = resolve_place(t, ent, session)
        if not pl:
            return {"reply": f"Which place is the {L.DOMAINS[domain]['label'].split(' (')[0].lower()} model for? Give coordinates or a city name and I will take it from there.", "blocks": [], "actions": []}
        M.set_state("last_place", pl)
        opts: dict = {}
        m = re.search(r"(\d{2,3})[\s\-]*(?:year|yr)", low)
        if m and domain in ("flood", "landslide"):
            opts["return_period"] = max(2, min(1000, int(m.group(1))))
        m = re.search(r"(\d+(?:\.\d+)?)\s*km", low)
        if m and domain == "flood":
            opts["radius_km"] = max(0.5, min(5.0, float(m.group(1))))
        if m and domain in ("landslide", "earthquake"):
            opts["radius_km"] = max(1.0, min(5.0 if domain == "landslide" else 500.0, float(m.group(1))))
        if m and domain == "air":
            opts["city_km"] = max(5.0, min(80.0, float(m.group(1))))
        holder: dict = {}

        def work():
            def prog(msg):
                if holder.get("j") is not None:
                    holder["j"]["progress"] = msg
            out = L.run_study(domain, pl["lat"], pl["lon"], pl["name"], opts, progress=prog)
            r = out["result"]
            M.add_run("modellab_" + domain, {"place": pl["name"], **opts}, [r["status"]], bool(r["ok"]), out["story"]["title"][:160])
            M.set_state("last_result", {"summary": out["story"]["title"], "explain": out["story"].get("standfirst", "")})
            blocks = [_blk("files", items=out["files"]), _blk("table", head=["", ""], rows=L.summary(r), caption="What the engine established, and what it did not.")]
            if out["basis"]:
                blocks.append(_blk("sources", items=[{"n": i + 1, "title": b["title"], "url": b["url"], "learned": False} for i, b in enumerate(out["basis"])]))
            blocks.append(_blk("list", items=[f'{n["step"]}. {n["title"]}' for n in r["notebook"]]))
            return {"reply": f'**{out["story"]["title"]}**\n{out["story"].get("standfirst", "")}', "blocks": blocks,
                    "actions": [{"type": "say", "label": lab_, "text": txt_ + pl["name"]} for lab_, txt_ in (("Flood model for this place", "flood model for "), ("Air quality model for this place", "air quality model for "), ("Earthquake hazard for this place", "earthquake hazard for "), ("Extreme heat for this place", "extreme heat model for ")) if not txt_.startswith(domain)][:3]}
        holder["j"] = _job(work, f"{domain} model study")
        steps = {"flood": "physics choice, terrain and rainfall data, model run, five verification tests, sensitivity", "air": "physics choice, three competing model structures, validation on unseen hours, verification tests, sensitivity"}.get(
            domain, "physics choice, public data, model fit, verification tests, and a hindcast on data the model never saw" if domain in L.DOMAINS and "dossier" not in L.DOMAINS[domain]["label"] else "reading up on the physics and the public datasets, and checking what a verified model would need")
        return {"reply": f"Starting a {L.DOMAINS[domain]['label'].split(' (')[0].lower()} study for {pl['name']}. I will work through: {steps}, then write the methods and results. This takes a few minutes; I will post it here.",
                "blocks": [], "actions": [], "job": holder["j"]["id"], "intent": "modellab"}

    _SRC_RX = re.compile(r"\b(?:cement|thermal|coal|power (?:plant|station)|captive power|fire\s?works?|crackers?|pyrotech\w*|sivakasi|match (?:factory|units?)|kiln)\b", re.I)
    _SRC_CTX = re.compile(r"\b(?:there (?:is|are)|we (?:have|are)|i (?:live|am)|near(?:by)?|next to|close to|beside|around|locality|neighbou?rhood|surrounded)\b", re.I)

    def _source_commands(self, session, text) -> Optional[dict]:
        """"We are close to a cement unit and a 13 MW thermal plant, and fireworks industries": the user tells me what is near the site.
        I store it as a DECLARED source (never as verified), re-run air quality and water stress with it, and say what is assumed."""
        t = text.strip()
        if not (self._SRC_RX.search(t) and self._SRC_CTX.search(t)) or re.match(r"\s*(what|why|how|explain|define|is|are|does|do you)\b", t.lower()):
            return None
        from ..analysis import local_air as LA
        srcs = LA.parse_declaration(t)
        if not srcs:
            return None
        ent = N.entities(t)
        pl = resolve_place(t, ent, session)
        if not pl:
            return {"reply": "Which place is that near? Give coordinates or a city name and I will add the sources to it and re-run air quality and water stress.", "blocks": [], "actions": []}
        M.set_state("last_place", pl)
        LA.declare(pl["lat"], pl["lon"], srcs)
        holder: dict = {}

        def work():
            from .. import siteprofile as SP
            p = SP.profile(pl["lat"], pl["lon"], pl.get("name") or "Site", only=["air_local", "water_stress"], use_cache=False)
            out = {"reply": "", "blocks": [], "actions": []}
            e = p["hazards"].get("air_local")
            lines = ["Stored as declared sources (not verified in any database): " + ", ".join(s["kind"] for s in srcs) + "."]
            if e and e["status"] != "null":
                a = e["metrics"]
                rec = a["record"]
                inc = a["increment"]["central"]
                lines.append(f'Measured regional background (CAMS, {rec["years_used"][0]}-{rec["years_used"][-1]}): PM2.5 {rec["annual_pm25"]} µg/m3 a year, {rec["days_above_who_24h"]:.0f} days a year above the WHO 24-hour guideline of 15, and {rec["days_above_naaqs_24h"]:.1f} above the Indian 60.')
                if any(x.get("modelled") for x in a["sources"]):
                    lines.append(f'Stack screen of the declared sources: +{inc["min"]} to +{inc["max"]} µg/m3 (central emissions), up to +{a["increment"]["high"]["max"]} in the high scenario, because the direction and distance were not given. Giving them narrows this.')
                dw = rec.get("diwali") or []
                if dw:
                    lines.append("Diwali-day test: " + ", ".join(f'{d["year"]} x{d["ratio"]}' for d in dw) + " against the surrounding days; a regional 10 km record only partly sees local burning.")
                lines.append("Not modelled: fugitive dust, secondary particles, and fireworks burning. Those are likely larger than the stacks, so this is a lower bound; a ground monitor at the site is the way to settle it.")
                out["blocks"].append(_blk("table", head=["Source", "Provenance", "PM2.5 central, annual (µg/m3)", "High scenario"],
                                          rows=[[x["kind"], str(x.get("provenance")), (f'{x["scenarios"]["central"]["pm25_annual_ug_m3"]["min"]} to {x["scenarios"]["central"]["pm25_annual_ug_m3"]["max"]}' if x.get("modelled") else "not modelled"),
                                                 (str(x["scenarios"]["high"]["pm25_annual_ug_m3"]["max"]) if x.get("modelled") else "")] for x in a["sources"]]))
            else:
                lines.append("The air-quality record could not be read just now, so air quality is null (not assessed). " + (e.get("reason") or ""))
            w = p["hazards"].get("water_stress")
            if w and w["status"] != "null" and w.get("caveat") and "Declared nearby industrial water users" in w["caveat"]:
                lines.append(f'Water stress is {w["severity"]} at basin scale ({w["value"]}); the declared thermal and cement users draw on the same basin, so the local picture can be worse.')
            out["reply"] = " ".join(lines)
            out["actions"] = [{"type": "say", "label": "Correct a source", "text": "the cement plant is 2 km north-east"}]
            M.add_run("local_sources", {"place": pl["name"], "sources": [s["kind"] for s in srcs]}, [], True, out["reply"][:160])
            return out
        holder["j"] = _job(work, f'air quality with local sources at {pl["name"]}')
        return {"reply": f'Noted: {", ".join(s["kind"] for s in srcs)} near {pl["name"]}. I am reading the measured air-quality record, then screening these sources against the local wind. Your statement overrides what the engine found on its own, and I will say what is assumed.', "blocks": [], "actions": [], "job": holder["j"]["id"], "intent": "local_sources"}

    _PROFILE_RX = re.compile(r"\b(?:all|every|full|complete)\s+(?:the\s+)?hazards?\b|\b(?:hazard|site|risk|location)\s+profile\b|\bfull (?:profile|screen\w*)\b|\bwhat(?:'s| is| are) the (?:hazards?|risks?) (?:at|for|in|near)\b|\bhow risky (?:is|are)\b|\bprofile\b.*\d+\.\d+", re.I)

    _HAZ_Q = re.compile(r"\b(heat|hot|humid\w*|flood\w*|cyclone|hurricane|typhoon|storm|wind|water stress|drought|wildfire|fire|earthquake|seismic|freez\w*|cold|air quality|pm ?2\.?5|landslide|subsidence|risk\w*|hazards?|exposed|exposure|safe)\b", re.I)

    def _profile_commands(self, session, text) -> Optional[dict]:
        """Any coordinate runs every hazard by default: "9.47, 77.91", "all hazards at 22.26, 91.80", "hazard profile for Chattogram".
        Hazards with no data are built from a fallback; if that fails they are null (not assessed), never zero."""
        t = text.strip()
        ent = N.entities(t)
        bare = "lat" in ent and len(t.split()) <= 4 and not re.search(r"[a-zA-Z]{4,}", re.sub(r"(?i)\b(profile|hazards?|risk)\b", "", t))
        site_q = "lat" in ent and bool(self._Q.search(t)) and bool(self._HAZ_Q.search(t)) and not self._ACT.search(t)    # "what is the heat risk at 29.7, -95.0?": a question about a place, answered from that place's numbers, never from unrelated literature
        if not (bare or site_q or self._PROFILE_RX.search(t)):
            return None
        pl = resolve_place(t, ent, session)
        if not pl:
            return {"reply": "Which place? Give coordinates or a city name and I will run every hazard for it.", "blocks": [], "actions": []}
        M.set_state("last_place", pl)
        holder: dict = {}

        def work():
            def prog(k, e):
                j = holder.get("j")
                if j is not None:
                    part = j.setdefault("partial", {})
                    part[k] = {"status": e["status"], "value": e["value"], "severity": e["severity"], "label": e["label"]}
                    j["progress"] = f'{len(part)} hazards done; last: {e["label"]} ({e["status"]})'
            out = run_site_profile(pl, progress=prog)
            M.add_run("site_profile", {"place": pl["name"]}, [str(out["profile"]["counts"])], True, out["reply"][:160])
            M.set_state("last_result", {"summary": "site hazard profile", "explain": out["reply"]})
            return out
        holder["j"] = _job(work, f'all hazards at {pl["name"]}')
        return {"reply": f'Running every hazard for {pl["name"]} ({pl["lat"]:.4f}, {pl["lon"]:.4f}). Where a source has no data I try a fallback; if that fails the hazard is reported as null (not assessed), never as zero. First results appear within seconds; the deep studies take a few minutes.',
                "blocks": [], "actions": [], "job": holder["j"]["id"], "intent": "site_profile"}

    def _knowledge_commands(self, session, text) -> Optional[dict]:
        """learn from <url> / learn about <topic> / what do you know / forget <url> — keyword-routed."""
        from . import knowledge as K
        t = text.strip()
        low = t.lower()
        m = re.match(r"(?i)\s*(?:please\s+)?(?:learn|read|study|ingest)\s+(?:from\s+)?(https?://\S+)", t)
        if m:
            url = m.group(1).rstrip(".,)")

            def work():
                r = K.learn_url(url)
                st = K.stats()
                return {"reply": (f'Read it: {r["passages"]} passages from “{r["title"]}”. I now know {st["passages"]:,} passages from {st["sources"]} sources, and I will quote this page when it answers a question.'
                                  if r.get("ok") else f'I could not learn from that address: {r.get("error")}.'), "blocks": [], "actions": []}
            j = _job(work, "learning a page")
            return {"reply": f"Reading {url} …", "blocks": [], "actions": [], "job": j["id"], "intent": "learn"}
        m = re.match(r"(?i)\s*(?:please\s+)?(?:learn|study|read up)\s+(?:about|on)\s+(.{3,120})$", t)
        if m:
            topic = m.group(1).strip(" .?")

            def work2():
                r = K.learn_topic(topic)
                st = K.stats()
                return {"reply": (f'Learned: {", ".join(r["learned"])} ({r["passages"]} passages). I now know {st["passages"]:,} passages in total. Ask me about it.' if r.get("ok")
                                  else f'I could not find a good public source for “{topic}”. Give me a web address and I will read it.'), "blocks": [], "actions": []}
            j = _job(work2, "learning a topic")
            return {"reply": f"Looking up “{topic}” in public sources …", "blocks": [], "actions": [], "job": j["id"], "intent": "learn"}
        if re.search(r"\b(what (do|have) you (know|learn(ed|t))|how much do you know|your knowledge|what.?s in your (knowledge|memory) base|what sources)\b", low):
            st = K.stats()
            from . import answer as AN
            ls = AN.llm_status()
            rows = [[k, str(v)] for k, v in sorted(st["kinds"].items())]
            when = time.strftime("%d %b %Y %H:%M", time.localtime(st["last_learned"])) if st["last_learned"] else "never"
            return {"reply": f'I have learned {st["passages"]:,} passages from {st["sources"]} public sources (last learned {when}). The autopilot keeps reading new material and refreshes it.',
                    "blocks": [_blk("table", head=["Source kind", "Pages"], rows=rows),
                               _blk("note", text="Language model: " + (f'{ls["model"]} (local, no key)' if ls["available"] else "none running — answers are assembled from the sources. " + ls["why"] + "."))],
                    "actions": [{"type": "say", "label": "Learn about CBAM for cement", "text": "learn about carbon border adjustment mechanism cement"}], "intent": "knowledge"}
        m = re.match(r"(?i)\s*forget\s+(https?://\S+)", t)
        if m:
            n = K.forget(m.group(1))
            return {"reply": f"Forgot {n} passages from that page." if n else "I had nothing from that address.", "blocks": [], "actions": [], "intent": "knowledge"}
        return None

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
            if self._Q.search(text) and not self._ACT.search(text):    # a question with no site in it: answer it instead of asking for coordinates
                return self._ask(text)
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
            try:                                                       # the site profile reuses this twin's event history and routed flood scenarios
                from .. import siteprofile as _SP
                _SP.save_twin(pl["lat"], pl["lon"], twin)
            except Exception:                                          # noqa: BLE001
                pass
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
        open_act = {"type": "open", "label": "Open in the workbench to customise", "url": f"/story-reports/dashboard?{qs}"}
        # Reports whose inputs are only a site or a company are written right here; the others need plant data, so the workbench opens with the form.
        payload = None
        if rtype == "physical" and pl:
            payload = {"name": pl["name"], "lat": pl["lat"], "lon": pl["lon"]}
            if ent.get("sector"):
                payload["sector"] = ent["sector"]
        elif rtype in ("transition", "financial"):
            pl2 = resolve_place(text, ent, session) if "lat" in ent else None
            if comp:
                payload = {"company": comp["query"]}
            elif pl2:
                payload = {"lat": pl2["lat"], "lon": pl2["lon"]}
        if payload is None:
            need = {"transition": "a company from the asset register, or coordinates", "financial": "a company from the asset register, or coordinates",
                    "lca": "the plant's clinker ratio, kiln fuel and mix, and electricity use", "iso14067": "the plant data and the study options (boundary, reference year, commissioner)",
                    "carbon_audit": "the emissions inventory", "cbam": "production, EU share of sales, origin country and the origin carbon price"}.get(rtype, "a few inputs")
            M.add_run(it, q, [], True, f"opened {rtype} report workbench")
            return {"reply": f"The {rtype.replace('_', ' ')} report needs {need}. I opened the workbench with what I understood ({', '.join(f'{k}={v}' for k, v in q.items() if k != 'type') or 'nothing yet'}); "
                             "fill the rest and I will not guess any of it.", "blocks": [], "actions": [open_act]}

        def work():
            import uuid as _u
            from .. import story as S, story_reports as SR
            st = SR.build(rtype, payload)
            slug = re.sub(r"[^A-Za-z0-9]+", "-", str(payload.get("name") or payload.get("company") or rtype)).strip("-").lower()[:40] or rtype
            base = f"{rtype}_{slug}_{_u.uuid4().hex[:8]}"
            d = report_dir()
            (d / f"{base}.html").write_text(S.html(st), encoding="utf-8")
            files = [{"label": "Open the report (HTML)", "url": f"/resident/file/{base}.html"}]
            try:
                (d / f"{base}.pdf").write_bytes(S.pdf(st))
                files.append({"label": "Download PDF", "url": f"/resident/file/{base}.pdf"})
            except Exception:                                          # noqa: BLE001
                pass
            secs = S.sections_of(st)
            facts = next((b["items"] for b in st["blocks"] if b["type"] == "facts"), [])
            nfig = sum(1 for b in st["blocks"] if b["type"] == "figure")
            M.add_run(it, payload, [f"{len(secs)} sections", f"{nfig} figures"], True, st["title"][:160])
            M.set_state("last_result", {"summary": st["title"], "explain": st.get("standfirst", "")})
            blocks = [_blk("files", items=files)]
            if facts:
                blocks.append(_blk("table", head=["Number", "What it is"], rows=[[f["value"], f["label"]] for f in facts[:6]], caption="The six numbers on the report's card, all computed by the engine."))
            blocks.append(_blk("list", items=[s["title"] for s in secs]))
            return {"reply": f'**{st["title"]}**\n{st.get("standfirst", "")}\n\n{len(secs)} sections, {nfig} figures, {len(st.get("sources", {}))} cited sources. The full report is in the links below.',
                    "blocks": blocks, "actions": [open_act, {"type": "say", "label": "Draft a LinkedIn post from this", "text": "write a LinkedIn post about this"}]}
        j = _job(work, f"{rtype} report")
        what = payload.get("name") or payload.get("company") or f'{payload.get("lat")}, {payload.get("lon")}'
        return {"reply": f"Writing the {rtype.replace('_', ' ')} report for {what}: building the evidence, drawing the figures and setting the story. About two minutes for a site; I will post it here.",
                "blocks": [], "actions": [open_act], "job": j["id"]}


_AGENT: Optional[Agent] = None


def get_agent() -> Agent:
    global _AGENT
    if _AGENT is None:
        _AGENT = Agent()
    return _AGENT
