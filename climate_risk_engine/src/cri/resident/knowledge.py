"""
cri/resident/knowledge.py — what the resident agent has learned from public sources, searchable and cited.

The agent reads public pages (Wikipedia, NOAA, the ClimRisk research pieces, any URL you tell it to learn), cuts
them into passages, and indexes them in SQLite FTS5 (BM25 ranking). Answers quote these passages with their source.
Nothing is invented: if the index has nothing relevant the agent says so and goes and learns the topic.

  learn_seed()          first-run and weekly refresh of the seed sources (fail-soft, one thread per source)
  learn_url(url)        learn any web page the user names
  learn_topic(q)        find and learn the best Wikipedia pages for a topic the index lacks
  search(q, k)          best passages for a question, with title / url / learned-on date
  stats()               how much it knows and from where
"""
from __future__ import annotations

import html as _html
import json
import os
import re
import sqlite3
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from . import memory as M

_UA = {"User-Agent": "ClimRisk-resident/1.0 (climate risk research agent; contact shrinivashdkannan@gmail.com)"}
_LOCK = threading.RLock()
_WORDS = (90, 170)               # passage length range, words

WIKI = "https://en.wikipedia.org/w/api.php"
WIKI_TITLES = [
    # hazards
    "Tropical cyclone", "Storm surge", "Saffir–Simpson scale", "Holland model", "Flood", "Coastal flooding", "Flash flood", "Sea level rise", "Heat wave", "Urban heat island",
    "Wildfire", "Landslide", "Drought", "El Niño–Southern Oscillation", "Cyclone Amphan", "Hurricane Katrina", "Hurricane Harvey", "Typhoon Haiyan", "2022 Pakistan floods",
    "Bay of Bengal", "Tropical cyclone forecasting", "International Best Track Archive for Climate Stewardship",
    # physical-risk methods and data
    "Shallow water equations", "Digital elevation model", "Height Above Nearest Drainage", "Sentinel-2", "Sentinel-1", "Synthetic-aperture radar", "ERA5", "Fragility curve", "Flood risk assessment",
    # climate finance and policy
    "Climate risk", "Climate change mitigation", "Stranded asset", "Carbon price", "Carbon tax", "EU Emissions Trading System", "Carbon Border Adjustment Mechanism", "Corporate Sustainability Reporting Directive",
    "Task Force on Climate-related Financial Disclosures", "IFRS Sustainability Disclosure Standards", "Network for Greening the Financial System", "Paris Agreement", "Net zero emissions",
    "Social cost of carbon", "Marginal abatement cost", "Climate change adaptation", "Climate resilience", "Climate-related financial risk", "Value at risk", "Expected shortfall",
    # carbon accounting and industry
    "Greenhouse Gas Protocol", "Life-cycle assessment", "ISO 14040", "Carbon footprint", "Global warming potential", "Cement", "Portland cement", "Cement industry", "Steelmaking", "Green steel",
    "Aluminium smelting", "Ammonia production", "Oil refinery", "Carbon capture and storage", "Clinker (cement)", "Scope 3 emissions",
]
SEED_URLS = [
    ("https://climrisk.io/research/the-loud-risk-and-the-quiet-one/", "ClimRisk research: The loud risk and the quiet one", 7 * 86400),
    ("https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso_advisory/ensodisc.shtml", "NOAA CPC ENSO diagnostic discussion", 12 * 3600),
]
_REFRESH_WIKI = 30 * 86400


def _db() -> sqlite3.Connection:
    c = sqlite3.connect(str(M.db_path()), timeout=30)
    c.execute("CREATE TABLE IF NOT EXISTS kn_sources (id INTEGER PRIMARY KEY, url TEXT UNIQUE, title TEXT, kind TEXT, fetched REAL, n INTEGER, status TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS kn_passages (id INTEGER PRIMARY KEY, source_id INTEGER, title TEXT, url TEXT, text TEXT, ts REAL)")
    try:
        c.execute("CREATE VIRTUAL TABLE IF NOT EXISTS kn_fts USING fts5(text, title, content='kn_passages', content_rowid='id', tokenize='porter unicode61')")
    except sqlite3.OperationalError:
        pass                                                      # FTS5 unavailable: search() falls back to LIKE scoring
    return c


def _fts_ok(c) -> bool:
    return bool(c.execute("SELECT 1 FROM sqlite_master WHERE name='kn_fts'").fetchone())


# ── fetching and cleaning ────────────────────────────────────────────────────

def _get(url: str, timeout: float = 25.0) -> str:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read(3_000_000)
        enc = r.headers.get_content_charset() or "utf-8"
    return raw.decode(enc, errors="replace")


def _strip_html(s: str) -> tuple:
    title = ""
    m = re.search(r"<title[^>]*>(.*?)</title>", s, re.S | re.I)
    if m:
        title = _html.unescape(re.sub(r"\s+", " ", m.group(1))).strip()
    s = re.sub(r"(?is)<(script|style|nav|header|footer|form|noscript|svg|aside)[^>]*>.*?</\1>", " ", s)
    art = re.search(r"(?is)<(article|main)[^>]*>(.*?)</\1>", s)
    if art and len(art.group(2)) > 2000:
        s = art.group(2)
    s = re.sub(r"(?i)</(p|div|h[1-6]|li|tr|br)>", "\n", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    s = _html.unescape(s)
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n\n", s)
    return title, s.strip()


def _chunk(text: str) -> list:
    """Paragraph-aware passages of ~90-170 words; headings stay attached to the paragraph that follows."""
    # Wikipedia headings arrive as '== Name ==' lines; keep them as a plain label in front of the passage they introduce.
    lines = re.split(r"\n+(?==+ )|\n\s*\n", text)
    paras = []
    for p in lines:
        p = p.strip()
        if not p:
            continue
        m = re.match(r"^=+\s*(.+?)\s*=+\s*(.*)$", p, re.S)
        if m:
            paras.append(("H", m.group(1)))
            if m.group(2).strip():
                paras.append(("P", m.group(2).strip()))
        else:
            paras.append(("P", p))
    out, cur, n, head = [], [], 0, ""

    def flush():
        nonlocal cur, n
        if cur and n >= 30:
            out.append((head + ". " if head else "") + " ".join(cur))
        cur, n = [], 0

    for kind, p in paras:
        if kind == "H":
            flush(); head = p; continue
        w = len(p.split())
        if w < 6:
            continue                                               # navigation fragments and one-word lines
        if n and n + w > _WORDS[1]:
            flush()
        cur.append(p); n += w
        if n >= _WORDS[0]:
            flush()
    flush()
    return [re.sub(r"\s+", " ", x).strip() for x in out if len(x.split()) >= 30]


_CUT = re.compile(r"\n==+\s*(References|External links|See also|Notes|Further reading|Sources|Bibliography|Citations)\s*==+.*", re.S | re.I)


def _store(url: str, title: str, kind: str, passages: list) -> int:
    with _LOCK:
        c = _db()
        try:
            row = c.execute("SELECT id FROM kn_sources WHERE url=?", (url,)).fetchone()
            if row:
                sid = row[0]
                if _fts_ok(c):
                    for (pid, t, ti) in c.execute("SELECT id, text, title FROM kn_passages WHERE source_id=?", (sid,)).fetchall():
                        c.execute("INSERT INTO kn_fts(kn_fts, rowid, text, title) VALUES('delete', ?, ?, ?)", (pid, t, ti))
                c.execute("DELETE FROM kn_passages WHERE source_id=?", (sid,))
            else:
                sid = c.execute("INSERT INTO kn_sources(url,title,kind,fetched,n,status) VALUES(?,?,?,?,?,?)", (url, title, kind, time.time(), 0, "ok")).lastrowid
            now = time.time()
            for t in passages:
                pid = c.execute("INSERT INTO kn_passages(source_id,title,url,text,ts) VALUES(?,?,?,?,?)", (sid, title, url, t, now)).lastrowid
                if _fts_ok(c):
                    c.execute("INSERT INTO kn_fts(rowid, text, title) VALUES(?,?,?)", (pid, t, title))
            c.execute("UPDATE kn_sources SET title=?, kind=?, fetched=?, n=?, status='ok' WHERE id=?", (title, kind, now, len(passages), sid))
            c.commit()
        finally:
            c.close()
    return len(passages)


def _mark_failed(url: str, title: str, kind: str, why: str) -> None:
    with _LOCK:
        c = _db()
        try:
            c.execute("INSERT INTO kn_sources(url,title,kind,fetched,n,status) VALUES(?,?,?,?,0,?) ON CONFLICT(url) DO UPDATE SET fetched=excluded.fetched, status=excluded.status",
                      (url, title, kind, time.time(), "failed: " + why[:120]))
            c.commit()
        finally:
            c.close()


# ── learning ─────────────────────────────────────────────────────────────────

def _wiki_extracts(titles: list) -> list:
    """[(title, url, plaintext)] for up to 20 titles in one request."""
    q = {"action": "query", "prop": "extracts", "explaintext": 1, "exsectionformat": "wiki", "titles": "|".join(titles), "format": "json", "redirects": 1, "exlimit": "max", "formatversion": 2}
    j = json.loads(_get(WIKI + "?" + urllib.parse.urlencode(q)))
    out = []
    for p in (j.get("query", {}).get("pages") or []):
        if p.get("missing") or not p.get("extract"):
            continue
        t = p["title"]
        out.append((t, "https://en.wikipedia.org/wiki/" + urllib.parse.quote(t.replace(" ", "_")), p["extract"]))
    return out


def _learn_wiki(titles: list) -> int:
    got = 0
    for i in range(0, len(titles), 1):                              # extracts with exlimit=max returns the intro only for >1 title; fetch singly for the full article
        try:
            for (t, url, text) in _wiki_extracts(titles[i:i + 1]):
                text = _CUT.sub("", "\n" + text)
                ps = _chunk(text)[:40]
                if ps:
                    got += _store(url, t + " (Wikipedia)", "wikipedia", ps)
        except Exception as e:                                       # noqa: BLE001
            _mark_failed("https://en.wikipedia.org/wiki/" + urllib.parse.quote(titles[i].replace(" ", "_")), titles[i], "wikipedia", f"{type(e).__name__}: {e}")
    return got


def learn_text(url: str, title: str, text: str, kind: str = "engine") -> int:
    """Index text the engine itself supplies (method notes). Re-learning replaces the old passages."""
    ps = _chunk(text) or [re.sub(r"\s+", " ", text).strip()]
    return _store(url, title, kind, ps)


def learn_url(url: str, title: Optional[str] = None, kind: str = "web") -> dict:
    if not re.match(r"^https?://", url or ""):
        return {"ok": False, "error": "that is not a web address"}
    try:
        raw = _get(url)
        t, text = _strip_html(raw) if "<" in raw[:2000] else ("", raw)
        ps = _chunk(text)[:60]
        if not ps:
            _mark_failed(url, title or t or url, kind, "no readable text")
            return {"ok": False, "error": "no readable text found at that address"}
        n = _store(url, title or t or url, kind, ps)
        return {"ok": True, "passages": n, "title": title or t or url}
    except Exception as e:                                           # noqa: BLE001
        _mark_failed(url, title or url, kind, f"{type(e).__name__}: {e}")
        return {"ok": False, "error": f"could not read it: {type(e).__name__}"}


def _due(url: str, max_age: float) -> bool:
    c = _db()
    try:
        r = c.execute("SELECT fetched, status FROM kn_sources WHERE url=?", (url,)).fetchone()
    finally:
        c.close()
    if not r:
        return True
    if str(r[1]).startswith("failed"):
        return time.time() - r[0] > 3600                             # retry failures hourly
    return time.time() - r[0] > max_age


def learn_seed(force: bool = False, max_wiki: int = 80) -> dict:
    """Read the seed sources that are new or stale. Safe to call every autopilot cycle: it does nothing when everything is fresh."""
    t0 = time.time()
    if os.environ.get("CRI_STORY_OFFLINE", "").strip() not in ("", "0"):      # tests and air-gapped runs: never touch the network
        s = stats()
        return {"new_passages": 0, "sources_checked": 0, "elapsed_s": 0.0, "total_passages": s["passages"], "total_sources": s["sources"]}
    todo_urls = [(u, t, a) for (u, t, a) in SEED_URLS if force or _due(u, a)]
    wiki_due = [t for t in WIKI_TITLES if force or _due("https://en.wikipedia.org/wiki/" + urllib.parse.quote(t.replace(" ", "_")), _REFRESH_WIKI)][:max_wiki]
    new = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(learn_url, u, t, "research" if "climrisk" in u else "noaa") for (u, t, _) in todo_urls]
        chunks = [wiki_due[i::4] for i in range(4) if wiki_due[i::4]]
        futs += [pool.submit(lambda ch=ch: {"ok": True, "passages": _learn_wiki(ch)}) for ch in chunks]
        for f in futs:
            try:
                r = f.result()
                new += int(r.get("passages", 0)) if r.get("ok") else 0
            except Exception:                                        # noqa: BLE001
                pass
    s = stats()
    return {"new_passages": new, "sources_checked": len(todo_urls) + len(wiki_due), "elapsed_s": round(time.time() - t0, 1), "total_passages": s["passages"], "total_sources": s["sources"]}


def learn_topic(q: str, n_pages: int = 2) -> dict:
    """The index does not cover this: find the best Wikipedia pages for it and learn them now."""
    if os.environ.get("CRI_STORY_OFFLINE", "").strip() not in ("", "0"):
        return {"ok": False, "error": "offline mode", "learned": []}
    try:
        sq = {"action": "query", "list": "search", "srsearch": q, "srlimit": n_pages + 2, "format": "json", "formatversion": 2}
        hits = json.loads(_get(WIKI + "?" + urllib.parse.urlencode(sq))).get("query", {}).get("search", [])
    except Exception as e:                                           # noqa: BLE001
        return {"ok": False, "error": f"could not search: {type(e).__name__}", "learned": []}
    titles = [h["title"] for h in hits if not re.search(r"\(disambiguation\)|^List of ", h["title"])][:n_pages]
    got = _learn_wiki(titles) if titles else 0
    return {"ok": got > 0, "learned": titles, "passages": got}


# ── search ───────────────────────────────────────────────────────────────────

_STOP = set("a an the and or of to in on at for from by with without is are was were be been being do does did how why what when where which who whom whose can could should would will "
            "it its this that these those as about into than then so such not no yes me my we our you your i tell explain describe give show please more most any some there their they them his her he she".split())


_STOP |= set("between difference differences differ differs work works working mean means meaning exactly really actually simple simply right now currently today kind "
             "main key way ways much many very also just".split())


def _terms(q: str) -> list:
    q = re.sub(r"[\-‐-―]", " ", q.lower())                  # saffir-simpson -> saffir simpson (the pages use en dashes)
    return [w for w in re.findall(r"[a-z0-9][a-z0-9]{1,}", q) if w not in _STOP]


def stem(t: str) -> str:
    for suf in ("ing", "ed", "es", "ly", "s"):
        if t.endswith(suf) and len(t) - len(suf) >= 4:
            return t[:-len(suf)]
    return t


def has(term: str, text_lower: str) -> bool:
    """Is the term (or its stem) in the text? Dashes in the text count as spaces."""
    return stem(term) in re.sub(r"[\-‐-―]", " ", text_lower)


def search(q: str, k: int = 6) -> list:
    ts = _terms(q)
    if not ts:
        return []
    c = _db()
    try:
        rows = []
        if _fts_ok(c):
            match = " OR ".join(f'"{t}"' for t in dict.fromkeys(ts))
            try:
                rows = c.execute("SELECT p.id, p.title, p.url, p.text, p.ts, bm25(kn_fts, 1.0, 2.5) AS s, "
                                 "(SELECT MIN(id) FROM kn_passages q WHERE q.source_id = p.source_id) = p.id AS first "
                                 "FROM kn_fts JOIN kn_passages p ON p.id = kn_fts.rowid WHERE kn_fts MATCH ? ORDER BY s LIMIT ?", (match, k * 8)).fetchall()
                # A page whose title names the subject, and a page's opening passage (where definitions live), outrank a passing mention elsewhere.
                uniq = set(ts)

                def adj(r):
                    th = sum(1 for t in uniq if t in r[1].lower())
                    return r[5] - 2.5 * th - (1.5 if r[6] else 0.0)             # bm25 is lower-is-better
                rows = [r[:6] if False else (r[0], r[1], r[2], r[3], r[4], adj(r)) for r in sorted(rows, key=adj)][:k * 4]
            except sqlite3.OperationalError:
                rows = []
        if not rows:                                                 # no FTS5: crude term-overlap scoring
            like = " OR ".join("lower(text) LIKE ?" for _ in ts[:6])
            cand = c.execute(f"SELECT id, title, url, text, ts, 0 FROM kn_passages WHERE {like} LIMIT 400", [f"%{t}%" for t in ts[:6]]).fetchall()
            rows = sorted(cand, key=lambda r: -sum(r[3].lower().count(t) for t in ts))[:k * 4]
            rows = [(r[0], r[1], r[2], r[3], r[4], -sum(r[3].lower().count(t) for t in ts)) for r in rows]
    finally:
        c.close()
    # keep the best passages but no more than two per source, so the answer draws on more than one page
    out, per = [], {}
    for (pid, title, url, text, ts_, score) in rows:
        if per.get(url, 0) >= 2:
            continue
        per[url] = per.get(url, 0) + 1
        covered = sum(1 for t in set(ts) if has(t, text.lower()))
        out.append({"id": pid, "title": title, "url": url, "text": text, "learned": time.strftime("%d %b %Y", time.localtime(ts_)), "score": round(-score, 2), "coverage": round(covered / max(1, len(set(ts))), 2)})
        if len(out) >= k:
            break
    return out


def stats() -> dict:
    c = _db()
    try:
        n = c.execute("SELECT COUNT(*) FROM kn_passages").fetchone()[0]
        s = c.execute("SELECT COUNT(*) FROM kn_sources WHERE status='ok'").fetchone()[0]
        bad = c.execute("SELECT COUNT(*) FROM kn_sources WHERE status!='ok'").fetchone()[0]
        kinds = dict(c.execute("SELECT kind, COUNT(*) FROM kn_sources WHERE status='ok' GROUP BY kind").fetchall())
        last = c.execute("SELECT MAX(fetched) FROM kn_sources").fetchone()[0]
    finally:
        c.close()
    return {"passages": n, "sources": s, "failed": bad, "kinds": kinds, "last_learned": last}


def forget(url: str) -> int:
    with _LOCK:
        c = _db()
        try:
            row = c.execute("SELECT id FROM kn_sources WHERE url=?", (url,)).fetchone()
            if not row:
                return 0
            if _fts_ok(c):
                for (pid, t, ti) in c.execute("SELECT id, text, title FROM kn_passages WHERE source_id=?", (row[0],)).fetchall():
                    c.execute("INSERT INTO kn_fts(kn_fts, rowid, text, title) VALUES('delete', ?, ?, ?)", (pid, t, ti))
            n = c.execute("DELETE FROM kn_passages WHERE source_id=?", (row[0],)).rowcount
            c.execute("DELETE FROM kn_sources WHERE id=?", (row[0],))
            c.commit()
            return n
        finally:
            c.close()
