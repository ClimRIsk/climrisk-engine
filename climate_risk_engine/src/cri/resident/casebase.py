"""
cri/resident/casebase.py — real climate events and what happened to company assets and financials, as a retrieval index.

Two kinds of record, both built from text that came from a named source and nothing else:
  curated   hand-built case cards (cri.llmtrain.cases_data); each fact was checked against its source page
  filing    sentences quoted verbatim from SEC filings (cri.llmtrain.edgar_cases): company, form, filing date, address

The index is a JSONL file shipped with the package (data/case_index.jsonl) and searched in memory with BM25, boosted when the
question names an event or a company in the index. Retrieval takes a few milliseconds; there is no network call and no model.

  available()                          is an index present?
  is_case_question(q)                  does the question name an indexed event/company, or ask for precedent?
  retrieve(q, k)                       best passages, shaped like knowledge.search() results, in the SAME text format the model was trained on
  analogs(hazard, assets, channels..)  closest curated cases for an exposure (used by the insurer view and the agent)
  check(answer, passages)              mechanical honesty check: every number is in the sources, every [n] exists
  compose(q, passages)                 answer assembled from the passages themselves (used when no model is running or the model fails the check)

Build the index:  python -m cri.resident.casebase build VERIFIED.json EDGAR_CASES.jsonl [OUT.jsonl] [--allow-unverified] [--xbrl XBRL_ITEMS.jsonl] [--macro MACRO_ITEMS.jsonl] [--reports REPORT_CASES.jsonl]
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import threading
from collections import defaultdict
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).parent
_CO_KINDS = ("filing", "xbrl")
PACKAGED = _HERE / "data" / "case_index.jsonl"

# ── text normalisation shared with the knowledge index ──────────────────────

_STOP = set("a an the and or of to in on at for from by with without is are was were be been being do does did how why what when where which who whom whose can could should would will it its this that these "
            "those as about into than then so such not no yes me my we our you your i tell explain describe give show please more most any some there their they them his her he she company companies "
            "report reported reports filing filings filed say said according sec form events event cases case happened happen did affect affected effect effects impact impacts".split())


def _stem(t: str) -> str:
    for suf in ("ing", "ed", "es", "ly", "s"):
        if t.endswith(suf) and len(t) - len(suf) >= 4:
            return t[:-len(suf)]
    return t


def _tokens(s: str) -> list:
    s = re.sub(r"[\-‐-―]", " ", s.lower())
    return [_stem(w) for w in re.findall(r"[a-z0-9][a-z0-9]+", s) if w not in _STOP]


_CO_SUFFIX = re.compile(r"[,.]?\s*\b(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|llc|lp|l\.p|holdings?|group|the|/[a-z]{2}/|n\.?v|s\.?a|ag|se|& co)\b\.?", re.I)


def _co_norm(name: str) -> str:
    n = re.sub(r"/[A-Za-z]{2}/", " ", name)
    for _ in range(3):
        n = _CO_SUFFIX.sub(" ", n)
    return re.sub(r"[^a-z0-9& ]+", " ", n.lower()).strip()


_EVENT_PREFIX = re.compile(r"^(hurricane|typhoon|super ?typhoon|cyclone|tropical storm|superstorm|winter storm|storm|the)\s+", re.I)

# ── the index ───────────────────────────────────────────────────────────────

SEM_W = float(os.environ.get("CRI_RAG_SEM_W", "0.35"))               # weight of the semantic score in the text-relevance term (0 = BM25 only)
_LOCK = threading.RLock()
_IDX: dict = {"path": None, "mtime": None, "items": [], "inv": {}, "df": {}, "avgdl": 1.0, "events": {}, "companies": {}, "cards": {}, "ev_text": {}}


def _vectors_match(p: Path) -> bool:
    """True when index file p is byte-identical to the index the shipped vectors were built from (sha256 stored next to them)."""
    import hashlib
    try:
        want = (PACKAGED.parent / "case_embeddings.sha256").read_text().strip()
        return hashlib.sha256(Path(p).read_bytes()).hexdigest() == want
    except OSError:
        return False


def index_path() -> Path:
    env = os.environ.get("CRI_CASE_INDEX")
    if env:
        return Path(env)
    user = Path(os.environ.get("CRI_HOME") or (Path.home() / ".climrisk")) / "case_index.jsonl"
    return user if user.exists() else PACKAGED


def available() -> bool:
    return index_path().exists() and index_path().stat().st_size > 0


def _load() -> dict:
    p = index_path()
    with _LOCK:
        if not p.exists():
            _IDX.update(path=None, mtime=None, items=[], inv={}, df={}, events={}, companies={}, cards={})
            return _IDX
        mt = p.stat().st_mtime
        if _IDX["path"] == str(p) and _IDX["mtime"] == mt:
            return _IDX
        items, inv, df, events, companies, cards = [], defaultdict(list), defaultdict(int), {}, defaultdict(set), {}
        total = 0
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            toks = _tokens(r["text"] + " " + r.get("event_name", "") + " " + (r.get("company") or ""))
            r["_len"] = len(toks) or 1
            total += r["_len"]
            i = len(items)
            tf: dict = defaultdict(int)
            for t in toks:
                tf[t] += 1
            for t, n in tf.items():
                inv[t].append((i, n))
                df[t] += 1
            items.append(r)
            ev = r["event"]
            nm = r.get("event_name") or ev
            meta = events.setdefault(ev, {"name": nm, "aliases": set(), "year": r.get("year"), "hazard": r.get("hazard"), "region": r.get("region")})
            if r.get("year") and not meta.get("year"):
                meta["year"] = r["year"]
            short = _EVENT_PREFIX.sub("", nm).strip().lower()
            meta["aliases"].add(nm.lower())
            if short and short != nm.lower() and len(short) >= 3:
                meta["aliases"].add(short)                            # "ian" for "Hurricane Ian"
            if r["kind"] in _CO_KINDS and r.get("company"):
                co = _co_norm(r["company"])
                if len(co) >= 3:
                    companies[co].add(ev)
            if r["kind"] == "frame":
                cards[ev] = r
        ev_text: dict = defaultdict(str)
        for r in items:
            if r["kind"] not in _CO_KINDS:
                ev_text[r["event"]] += " " + r["text"].lower()
        sem = None
        if os.environ.get("CRI_RAG_SEMANTIC", "1") != "0" and _vectors_match(p):          # the shipped vectors belong to the exact index they were built from (by content hash)
            from . import semantic as _S
            sem = _S.Dense.load(len(items))
        _IDX.update(path=str(p), mtime=mt, items=items, inv=dict(inv), df=dict(df), avgdl=total / max(1, len(items)), events=events, companies=dict(companies), cards=cards, ev_text=dict(ev_text), sem=sem)
        return _IDX


def stats() -> dict:
    ix = _load()
    kinds: dict = defaultdict(int)
    for r in ix["items"]:
        kinds[r["kind"]] += 1
    pairs = {(r["event"], _co_norm(r["company"])) for r in ix["items"] if r["kind"] in _CO_KINDS and r.get("company") and r["event"] != "climate-reports"}
    reps = {_co_norm(r["company"]) for r in ix["items"] if r["event"] == "climate-reports" and r.get("company")}          # published climate reports are not events: counted apart from the real-event cases
    return {"items": len(ix["items"]), "events": len(ix["events"]) - (1 if reps else 0), "curated_cards": len(ix["cards"]), "company_event_cases": len(pairs), "company_climate_reports": len(reps), "kinds": dict(kinds), "path": ix["path"]}


# ── retrieval ───────────────────────────────────────────────────────────────

_ASK_PRECEDENT = re.compile(r"\b(precedent|analog(?:ue|y|ies)?|similar (?:case|event|situation)s?|past (?:event|case)s?|historical(?:ly)?|lessons? from|has (?:this|it) happened|what happened (?:to|when|after|in)|"
                            r"how did (?:\w+ ){0,4}(?:respond|cope|deal|handle|recover|manage|react|pay|fund|finance)|financial (?:impact|effect|cost|hit)s? of|what did .{2,60} (?:report|disclose|say|lose|cost|pay))\b", re.I)
_WANT_FIG = re.compile(r"\b(how much|cost|loss(?:es)?|figure|amount|\$|million|billion|impairment|charge|claims?|insured)\b", re.I)
_WANT_RESP = re.compile(r"\b(respond|response|deal with|dealt|handle|tackle|recover|action|actions|mitigat|did .{0,40} do|steps|measures)\b", re.I)


def _match_entities(q: str, ix: dict) -> tuple:
    low = " " + re.sub(r"[^a-z0-9& ]+", " ", q.lower()) + " "
    evs = []
    for ev, meta in ix["events"].items():
        for a in meta["aliases"]:
            if re.search(rf"(?<![a-z0-9]){re.escape(a)}(?![a-z0-9])", low):
                # a bare short alias ("ian", "uri") must not fire on common words: it needs a storm word or the year nearby, or to be written as a proper noun
                if (" " not in a or len(a) <= 4) and not (re.search(r"hurricane|typhoon|cyclone|storm|flood|fire|drought|heat|freeze|" + str(meta.get("year") or "x"), low)
                                        or (re.search(rf"(?<![A-Za-z]){re.escape(a.title())}(?![A-Za-z])", q)
                                            and re.search(r"\b(cost|costs|loss|losses|damage|insur\w*|claims?|outage|power|grid|impact\w*|affect\w*|respond\w*|financ\w*|earnings|revenue|report\w*|liabilit\w*|impair\w*)\b", low))):
                    continue
                evs.append(ev)
                break
    cos = []
    for co, evset in ix["companies"].items():
        if f" {co} " in low:
            cos.append(co)
    return evs, cos


SEM_ROUTE = float(os.environ.get("CRI_RAG_ROUTE", "0.65"))          # cosine of the best curated card above which a question is routed to the case library


def semantic_best(q: str) -> float:
    """Cosine similarity of the question to its closest curated card (frame / fact / response). 0.0 when the dense layer is not available."""
    ix = _load()
    if ix.get("sem") is None:
        return 0.0
    cs = ix["sem"].scores(q)
    if cs is None:
        return 0.0
    best = 0.0
    for i, r in enumerate(ix["items"]):
        if r["kind"] in ("frame", "fact", "resp") and cs[i] > best:
            best = float(cs[i])
    return best


def is_case_question(q: str) -> bool:
    if not available():
        return False
    ix = _load()
    evs, cos = _match_entities(q, ix)
    if evs or cos:
        return True
    if _ASK_PRECEDENT.search(q):
        return True
    return semantic_best(q) >= SEM_ROUTE


def coverage_gap(q: str) -> Optional[dict]:
    """The question names a company and an event, and the library holds filings by that company for OTHER events only. Returns what it does hold, so the reply can say plainly
    what is missing (correct by construction: no model needed)."""
    if not available():
        return None
    ix = _load()
    evs, cos = _match_entities(q, ix)
    if not evs or not cos:
        return None
    ev_set = set(evs)
    if any(ix["companies"][co] & ev_set for co in cos):
        return None
    if any(co in " ".join(ix["ev_text"].get(e, "") for e in ev_set) for co in cos):      # the curated case card itself names the company
        return None
    co = cos[0]
    own, seen = [], set()
    for r in ix["items"]:
        if r["kind"] in _CO_KINDS and _co_norm(r.get("company") or "") == co and r["event"] not in seen:
            seen.add(r["event"])
            own.append(r)
        if len(own) >= 2:
            break
    if not own:
        return None
    return {"company": own[0]["company"], "event": ix["events"][evs[0]]["name"], "passages": [_passage(r, 1.0, 1.0, i + 1) for i, r in enumerate(own)]}


def gap_answer(g: dict) -> str:
    ps = g["passages"]
    body = lambda p: _clean(p["text"].split(": ", 1)[1] if ": " in p["text"][:200] else p["text"])          # noqa: E731
    have = " ".join(f"On {p['event_name'] if p.get('event_name') else 'another event'} it reported: {body(p)} [{p['id']}]." for p in ps[:1])
    return (f"The sources provided do not cover what {g['company']} reported about {g['event']}. They only show what it reported about a different event. {have} "
            f"To answer I would need {g['company']}'s own filing for {g['event']}.")


def _bm25(qtoks: list, ix: dict, k1=1.4, b=0.75) -> dict:
    n = len(ix["items"])
    sc: dict = defaultdict(float)
    for t in set(qtoks):
        post = ix["inv"].get(t)
        if not post:
            continue
        idf = math.log(1 + (n - ix["df"][t] + 0.5) / (ix["df"][t] + 0.5))
        for i, tf in post:
            dl = ix["items"][i]["_len"]
            sc[i] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / ix["avgdl"]))
    return sc


def _passage(r: dict, score: float, cov: float, rank_id: int) -> dict:
    when = r.get("filed") or str(r.get("year") or "")
    title = r["title"]
    return {"id": rank_id, "title": title, "url": r["url"], "text": r["text"], "learned": when, "score": round(score, 2), "coverage": round(cov, 2),
            "kind": "case", "event": r["event"], "event_name": r.get("event_name"), "company": r.get("company"), "rec": r["kind"]}


def retrieve(q: str, k: int = 6, per_group: int = 2) -> list:
    """Best passages for a question about events and companies, diverse across companies, curated cards first when scores tie."""
    if not available():
        return []
    ix = _load()
    evs, cos = _match_entities(q, ix)
    qtoks = _tokens(q)
    # entity tokens are already handled by the boosts; the rest of the question picks the passages within the entity
    sc = _bm25(qtoks, ix)
    sem = {}
    if ix.get("sem") is not None and SEM_W > 0:
        cs = ix["sem"].scores(q)
        if cs is not None:
            for i in cs.argsort()[::-1][:60]:
                if cs[i] > 0.2:
                    sem[int(i)] = float(cs[i])
    if not sc and not sem and not evs and not cos:
        return []
    want_fig, want_resp = bool(_WANT_FIG.search(q)), bool(_WANT_RESP.search(q))
    ev_set, co_set = set(evs), set(cos)
    cand = set(sc) | set(sem)
    if ev_set or co_set:                                             # entity-matched items are candidates even when no other question word hits them
        for i, r in enumerate(ix["items"]):
            if r["event"] in ev_set or (r["kind"] in _CO_KINDS and _co_norm(r.get("company") or "") in co_set):
                cand.add(i)
    ranked = []
    top = max(sc.values()) if sc else 1.0
    stop = max(sem.values()) if sem else 1.0
    for i in cand:
        r = ix["items"][i]
        s = (1 - SEM_W) * sc.get(i, 0.0) / (top or 1.0) + SEM_W * sem.get(i, 0.0) / (stop or 1.0)          # 0..1 text relevance: BM25 blended with the offline semantic score
        if r["event"] in ev_set:
            s += 1.0
        if r["kind"] in _CO_KINDS and _co_norm(r.get("company") or "") in co_set:
            s += 1.0
            if ev_set and r["event"] not in ev_set:
                s -= 0.7                                               # the company on a different event is only a weak lead
        if want_fig and r.get("amounts"):
            s += 0.25
        if want_resp and (r.get("response") or r["kind"] == "resp"):
            s += 0.35
        if r["kind"] in ("fact", "resp", "frame"):
            s += 0.15                                                  # curated, source-checked text wins ties
        if r["kind"] == "frame":
            s -= 0.1
        ranked.append((s, i))
    ranked.sort(key=lambda x: -x[0])
    if co_set:                                                       # "what did X report ...": only X's own sentences answer that
        own = [(s, i) for s, i in ranked if ix["items"][i]["kind"] in _CO_KINDS and _co_norm(ix["items"][i].get("company") or "") in co_set
               and (not ev_set or ix["items"][i]["event"] in ev_set)]
        if own:
            ranked = own
    out, per = [], defaultdict(int)
    qset = set(qtoks)
    for s, i in ranked:
        r = ix["items"][i]
        key = (r["event"], r.get("company") or r["title"])
        if per[key] >= per_group:
            continue
        per[key] += 1
        cov = len(qset & set(_tokens(r["text"] + " " + r.get("event_name", "")))) / max(1, len(qset))
        if r["event"] in ev_set or (r["kind"] in _CO_KINDS and _co_norm(r.get("company") or "") in co_set):
            cov = max(cov, 0.6)
        out.append(_passage(r, s, cov, len(out) + 1))
        if len(out) >= k:
            break
    return out


# ── analogs for an exposure ─────────────────────────────────────────────────

_HAZ_WORDS = {
    "cyclone": "hurricane typhoon cyclone storm surge wind landfall", "tropical cyclone": "hurricane typhoon cyclone storm surge wind landfall",
    "flood": "flood rain river inundation monsoon", "coastal_flood": "storm surge coastal flood sea level", "wildfire": "wildfire fire ignition utility",
    "drought": "drought low water river levels hydropower rainfall", "heat": "heat heatwave temperature cooling", "cold": "freeze winter storm cold cold snap power outage",
    "hail": "hail convective storm", "sea_level": "sea level coastal storm surge", "wind": "wind storm windstorm", "earthquake": "earthquake seismic",
}


def analogs(hazard: str = "", assets: str = "", channels: Optional[list] = None, region: str = "", k: int = 3) -> list:
    """Closest curated cases for an exposure. Each result carries its frame, up to two facts, one response, the lessons and the channels they share."""
    if not available():
        return []
    ix = _load()
    cards = ix["cards"]
    if not cards:
        return []
    want = set(channels or [])
    qt = set(_tokens(_HAZ_WORDS.get(hazard.lower().replace(" ", "_"), hazard) + " " + assets + " " + hazard))
    by_ev = defaultdict(lambda: {"fact": [], "resp": []})
    for r in ix["items"]:
        if r["kind"] in ("fact", "resp"):
            by_ev[r["event"]][r["kind"]].append(r)
    scored = []
    for ev, fr in cards.items():
        ch = set(fr.get("channels") or [])
        jac = len(ch & want) / max(1, len(ch | want)) if want else 0.0
        text_t = set(_tokens(fr["text"]))
        kw = len(qt & text_t) / max(1, len(qt))
        reg = 0.3 if region and region.lower() in (fr.get("region") or "").lower() else 0.0
        s = 2.0 * jac + 1.5 * kw + reg
        if s > 0:
            scored.append((s, ev))
    scored.sort(key=lambda x: -x[0])
    out = []
    for s, ev in scored[:k]:
        fr = cards[ev]
        shared = sorted(set(fr.get("channels") or []) & want) if want else list(fr.get("channels") or [])
        facts = sorted(by_ev[ev]["fact"], key=lambda r: -len(r.get("amounts") or []))[:2]
        resp = by_ev[ev]["resp"][:1]
        out.append({"event": ev, "title": fr["title"], "year": fr.get("year"), "region": fr.get("region"), "score": round(s, 2), "shared_channels": shared,
                    "items": [fr] + facts + resp, "lessons": fr.get("lessons") or [], "url": fr["url"]})
    return out


def precedents(hazards: list, assets: str = "", channels: Optional[list] = None, region: str = "", k: int = 3) -> list:
    """JSON-ready historical precedents for the insurer view and reports: hazards = [(hazard, weight)] or [hazard]. Context for the reader, never a model input."""
    hz = [(h, 1.0) if isinstance(h, str) else (h[0], float(h[1])) for h in hazards] or [("", 1.0)]
    tot = sum(w for _, w in hz) or 1.0
    seen, out = set(), []
    for h, w in sorted(hz, key=lambda x: -x[1]):
        n = max(1, round(k * w / tot))
        for a in analogs(h, assets, channels, region, k=n + 2):
            if a["event"] in seen or len(out) >= k:
                continue
            seen.add(a["event"])
            facts = [{"text": r["text"].split("): ", 1)[1] if "): " in r["text"] else r["text"], "source": r["title"].split(" — ")[-1], "url": r["url"]} for r in a["items"] if r["kind"] == "fact"]
            resp = [{"text": r["text"].split("): ", 1)[1] if "): " in r["text"] else r["text"], "source": r["title"].split(" — ")[-1], "url": r["url"]} for r in a["items"] if r["kind"] == "resp"]
            out.append({"event": a["event"], "title": a["title"], "year": a["year"], "region": a["region"], "hazard_match": h or None, "shared_channels": a["shared_channels"],
                        "facts": facts, "responses": resp, "lessons": a["lessons"][:2], "url": a["url"]})
    return out


def analog_passages(an: list) -> list:
    """Flatten analogs() into numbered passages, in the layout the model saw during training (frame, facts, response per case)."""
    out = []
    for a in an:
        for r in a["items"]:
            out.append(_passage(r, a["score"], 1.0, len(out) + 1))
    return out


_CH_KEYS = (("insurance_availability", r"insur|reinsur|cover(?:age)?\b|premium"), ("financing_liquidity", r"rating|liquidity|capital|debt|credit|refinanc|equity|bond"),
            ("supply_chain", r"suppl|logistic|customer"), ("business_interruption", r"shut|interrupt|downtime|outage|closed|closure|lost revenue"),
            ("liability", r"liabilit|lawsuit|litigat|claims? against|compensation"), ("regulation_policy", r"regulat|polic|law|tax|levy|carbon price"),
            ("market_price", r"price|market|commodit"), ("asset_damage", r"damage|destroy|physical|repair|rebuild"), ("input_shortage", r"power|water|feedstock|transport|fuel|shortage|low river"),
            ("sovereign_backstop", r"government|state fund|public insur|sovereign|bailout|backstop"), ("demand_shift", r"demand|customers? (?:leave|switch)"))
_HAZ_KEYS = (("cyclone", r"hurricane|typhoon|cyclone|tropical storm|storm surge|landfall"), ("flood", r"flood|inundat|monsoon|heavy rain|river"), ("wildfire", r"wildfire|bushfire|forest fire|fire\b"),
             ("drought", r"drought|low water|dry spell|low river"), ("heat", r"heat ?wave|extreme heat|heat stress"), ("cold", r"freeze|winter storm|cold snap|deep freeze"), ("hail", r"hail"), ("earthquake", r"earthquake|seismic"))


def parse_exposure(text: str) -> dict:
    """Pull the hazard and the loss channels the user cares about out of a free-form request."""
    low = text.lower()
    haz = next((h for h, rx in _HAZ_KEYS if re.search(rx, low)), "")
    ch = [c for c, rx in _CH_KEYS if re.search(rx, low)]
    return {"hazard": haz, "channels": ch}


def compose_analog(an: list) -> tuple:
    """-> (passages, answer). The answer is built from the closest case and one second case, each sentence cited; the inference is labelled, as in training."""
    if not an:
        return [], ""
    top, second = an[0], (an[1] if len(an) > 1 else None)
    ps, ids = [], {}

    def add(r, key):
        ps.append(_passage(r, top["score"], 1.0, len(ps) + 1))
        ids[key] = len(ps)

    for j, r in enumerate(top["items"]):
        add(r, ("a", j))
    if second:
        fact = next((r for r in second["items"] if r["kind"] == "fact"), None)
        if fact:
            add(fact, ("b", 0))
    frame = top["items"][0]
    body = lambda r: _clean(r["text"].split("): ", 1)[1] if "): " in r["text"] else r["text"])          # noqa: E731
    s = f"The closest case is {top['title'].split(' (')[0]}, where the driver was {frame['driver']} [{ids[('a', 0)]}]. "
    for j, r in enumerate(top["items"][1:], 1):
        if r["kind"] == "fact":
            s += f"{body(r)} [{ids[('a', j)]}]. "
    resp = [(j, r) for j, r in enumerate(top["items"]) if r["kind"] == "resp"]
    if resp:
        s += f"Response used: {body(resp[0][1])} [{ids[('a', resp[0][0])]}]. "
    if ("b", 0) in ids:
        s += f"A second case is {second['title'].split(' (')[0]}: {body(ps[ids[('b', 0)] - 1])} [{ids[('b', 0)]}]. "
    from ..llmtrain.cases import CH_DESC
    sh = [CH_DESC[c] for c in top["shared_channels"] if c in CH_DESC]
    if sh:
        s += "Analysis (not stated in the sources): the first things to check are " + ", ".join(sh) + ". "
    if top["lessons"]:
        s += top["lessons"][0]
    return ps, s.strip()


# ── prompt, check, compose ──────────────────────────────────────────────────

SYS_CASE = ("You are ClimRisk's climate-risk analyst. Answer using ONLY the numbered sources. Cite every factual sentence with its source number in square brackets, like [2]. "
            "If the sources do not contain the answer, say exactly what is missing instead of guessing. Keep any inference clearly separate by starting it with 'Analysis (not stated in the sources):'. "
            "Never add a number that is not in the sources.")
SYS_FILING = ("You are ClimRisk's climate-risk analyst. Answer using ONLY the numbered sources, which are sentences copied from company filings. Cite every factual sentence with its source number in square brackets, like [2]. "
              "If the sources do not contain the answer, say exactly what is missing instead of guessing. Never add a number that is not in the sources.")


SYS_MACRO = ("You are ClimRisk's climate-risk analyst. Answer using ONLY the numbered sources, which are official national statistics from the World Bank. Cite every factual sentence with its source number in square brackets, like [2]. "
             "If the sources do not contain the answer, say exactly what is missing instead of guessing. Keep any inference clearly separate by starting it with 'Analysis (not stated in the sources):'. Never add a number that is not in the sources.")
SYS_XBRL = ("You are ClimRisk's climate-risk analyst. Answer using ONLY the numbered sources, which are structured annual figures from company filings with the SEC. Cite every factual sentence with its source number in square brackets, like [2]. "
            "If the sources do not contain the answer, say exactly what is missing instead of guessing. Keep any inference clearly separate by starting it with 'Analysis (not stated in the sources):'. Never add a number that is not in the sources.")


def system_prompt(ps: list) -> str:
    return SYS_FILING if ps and all(p.get("rec") == "filing" for p in ps) else SYS_XBRL if ps and all(p.get("rec") == "xbrl" for p in ps) else SYS_MACRO if ps and all(p.get("rec") == "macro" for p in ps) else SYS_CASE


_NUM = re.compile(r"\d[\d,]*\.?\d*")


def _nums(t: str) -> set:
    return {x.replace(",", "").rstrip(".").lstrip("0") or "0" for x in _NUM.findall(t)}


_MISSING = re.compile(r"do(?:es)? not (?:cover|report|say|contain|include|state)|not (?:covered|reported|stated|in the sources)|no (?:information|data|figure|source)|would need|missing", re.I)


_ABBR = re.compile(r"\b(Inc|Ltd|Corp|Co|Cos|LLC|LLP|L\.P|U\.S|U\.K|No|St|Mr|Mrs|Dr|vs|etc|approx|Jr|Sr)\.(?=\s)")


def _sentences(t: str) -> list:
    t = _ABBR.sub(lambda m: m.group(0)[:-1] + "\u2024", t)
    return [x.replace("\u2024", ".") for x in re.split(r"(?<=[.!?])\s+", t)]


def check(answer: str, ps: list) -> dict:
    """Mechanical honesty check. ok is False if a number is not in the sources, a citation does not exist, or no factual sentence is cited."""
    src = " ".join(p["text"] for p in ps)
    n_bad = sorted(_nums(answer) - _nums(src) - {str(i) for i in range(1, len(ps) + 1)})       # [1]..[n] are citation markers
    cites = {int(x) for x in re.findall(r"\[(\d+)\]", answer)}
    bad_cite = sorted(c for c in cites if c > len(ps) or c < 1)
    norm = re.sub(r"([.!?])\s*((?:\[\d+\]\s*)+)", r" \2\1", answer)
    sents = [x for x in _sentences(norm) if len(x.split()) > 4 and not x.lower().startswith("analysis") and not _MISSING.search(x)]
    uncited = [x[:80] for x in sents if not re.search(r"\[\d+\]", x)]
    return {"ok": not n_bad and not bad_cite and (bool(cites) or bool(_MISSING.search(answer))) and len(uncited) <= max(0, len(sents) // 4),
            "numbers_not_in_sources": n_bad, "bad_citations": bad_cite, "uncited_sentences": uncited, "cited": sorted(cites)}


def _clean(t: str) -> str:
    return t.rstrip(".").rstrip()


def compose(q: str, ps: list, max_items: int = 4) -> str:
    """An answer built only from the passages: each one quoted with its marker. Used when no model is running or the model's answer fails check()."""
    if not ps:
        return ""
    parts = []
    for p in ps[:max_items]:
        body = p["text"]
        body = body.split(": ", 1)[1] if ": " in body[:200] else body
        if body[:1] in "“\"" and body.count("”") + body.count('"') < 1 + (body[:1] == '"'):
            body = body[1:]
        who = p.get("company")
        lead = f"{who} reported" if who and p.get("rec") == "filing" else (f"{who}'s annual SEC financial data show" if who and p.get("rec") == "xbrl" else None)
        parts.append(f"{lead}: {_clean(body)} [{p['id']}]." if lead else f"{_clean(body)} [{p['id']}].")
    return " ".join(parts)


def gap_reply(q: str, ps: list) -> str:
    have = compose(q, ps[:2])
    return ("The sources I hold do not cover that specific event or company. " + (f"The closest material is this. {have} " if have else "") +
            "To answer I would need the company's own filing or an official statement on it.")


# ── building the index ──────────────────────────────────────────────────────

def build(vf_path: Path, edgar_path: Optional[Path], out: Path, allow_unverified: bool = False, xbrl_path: Optional[Path] = None, macro_path: Optional[Path] = None, reports_path: Optional[Path] = None) -> dict:
    from ..llmtrain import cases as C
    from ..llmtrain.cases_data import CASES
    vf = json.loads(Path(vf_path).read_text()) if vf_path and Path(vf_path).exists() else {}
    recs = []
    for c in C.usable(CASES, vf, allow_unverified):
        short = C.SHORT[c["id"]]
        base = {"event": c["id"], "event_name": short, "year": c["year"], "hazard": c["kind"], "region": c["region"], "channels": c["channels"], "company": None}
        recs.append({**base, "kind": "frame", "title": f"{c['title']} (case summary)", "url": c["sources"][0][0],
                     "text": f"{short} (case summary): the driver was {c['driver']}; the exposed assets were {c['assets']}; the event took place in {c['region']} in {c['year']}.",
                     "lessons": c["lessons"], "driver": c["driver"], "assets": c["assets"]})
        for text, si in c["facts"]:
            outlet = c["sources"][si][1]
            recs.append({**base, "kind": "fact", "title": f"{c['title']} — {outlet}", "url": c["sources"][si][0], "text": f"{short} ({outlet}): {text}", "amounts": re.findall(r"[\$€£]\s?\d[\d,\.]*|\d[\d,\.]*\s?(?:million|billion)", text)})
        for typ, text, si in c["responses"]:
            outlet = c["sources"][si][1]
            recs.append({**base, "kind": "resp", "title": f"{c['title']} — {outlet}", "url": c["sources"][si][0], "text": f"{short} ({outlet}), response ({typ}): {text}", "response": [typ]})
    n_curated = len(recs)
    n_files = 0
    if reports_path is None and edgar_path:                           # L/edgar/edgar_cases.jsonl -> L/reports/report_cases.jsonl, when the report collector has been run
        reports_path = Path(edgar_path).parent.parent / "reports" / "report_cases.jsonl"
    srcs = [Path(x) for x in (edgar_path, reports_path) if x and Path(x).exists()]
    if srcs:
        from ..llmtrain.edgar_cases import clean_quote
        for sp in srcs:
          for line in sp.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rep = str(r.get("form", "")).lower() == "climate report"
            for q in r["quotes"]:
                cq = clean_quote(q["quote"])
                if not cq:
                    continue
                label = f"climate report {r['filed']}" if rep else f"Form {r['form']} filed {r['filed']}"
                recs.append({"event": r["event"], "event_name": r["event_name"], "year": None, "hazard": r["hazard"], "region": r["region"], "channels": [], "company": r["company"],
                             "kind": "filing", "title": f"{r['company']} — {label} ({r['event_name']})", "url": r["url"], "filed": r["filed"],
                             "text": f"{r['company']}, {label}: {cq}", "amounts": q.get("amounts") or [], "response": q.get("response") or []})
            n_files += 1
    n_xbrl = 0
    if xbrl_path and Path(xbrl_path).exists():
        for line in Path(xbrl_path).read_text().splitlines():
            if not line.strip():
                continue
            x = json.loads(line)
            recs.append({"event": x["event"], "event_name": x["event_name"], "year": None, "hazard": None, "region": None, "channels": [], "company": x["company"], "kind": "xbrl",
                         "title": f"{x['company']} — annual financial data (SEC XBRL) around {x['event_name']}", "url": x["url"], "filed": None, "text": x["text"], "amounts": x.get("amounts") or []})
            n_xbrl += 1
    n_macro = 0
    if macro_path and Path(macro_path).exists():
        for line in Path(macro_path).read_text().splitlines():
            if not line.strip():
                continue
            x = json.loads(line)
            recs.append({"event": x["event"], "event_name": x["event_name"], "year": None, "hazard": None, "region": x.get("country"), "channels": [], "company": None, "kind": "macro",
                         "title": f"{x['country']} — World Bank national statistics around {x['event_name']}", "url": x["url"], "filed": None, "text": x["text"], "amounts": x.get("amounts") or []})
            n_macro += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(out)
    return {"records": len(recs), "curated_records": n_curated, "filings": n_files, "xbrl": n_xbrl, "macro": n_macro, "path": str(out)}


def main():
    a = sys.argv[1:]
    if not a or a[0] != "build":
        print(__doc__)
        return
    out = Path(a[3]) if len(a) > 3 and not a[3].startswith("--") else PACKAGED
    xb = Path(a[a.index("--xbrl") + 1]) if "--xbrl" in a else None
    mc = Path(a[a.index("--macro") + 1]) if "--macro" in a else None
    rp = Path(a[a.index("--reports") + 1]) if "--reports" in a else None
    r = build(Path(a[1]), Path(a[2]) if len(a) > 2 else None, out, "--allow-unverified" in a, xb, mc, rp)
    print(json.dumps({**r, **stats()} if Path(r["path"]) == index_path() else r, indent=1))


if __name__ == "__main__":
    main()
