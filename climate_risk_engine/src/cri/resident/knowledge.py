"""
cri/resident/knowledge.py — what the resident agent has learned from public sources, searchable and cited.

The agent reads public pages from primary sources (NOAA, NWS, NHC, NASA, USGS, FEMA, the EU Commission, IFRS/ISSB, the GHG Protocol,
the ClimRisk research pieces, any URL you tell it to learn), cuts
them into passages, and indexes them in SQLite FTS5 (BM25 ranking). Answers quote these passages with their source.
Nothing is invented: if the index has nothing relevant the agent says so and goes and learns the topic.

  learn_seed()          first-run and weekly refresh of the seed sources (fail-soft, one thread per source)
  learn_url(url)        learn any web page the user names
  learn_topic(q)        find and learn the best primary-source pages for a topic the index lacks (curated list; no wikis, no forums)
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

# Primary sources only: government agencies, standard setters and regulators. Wikipedia and other user-edited sites are never read:
# they are not citable in a risk report. (title, url, keywords that route a question to it)
PRIMARY = [
    ("NOAA NHC: Saffir-Simpson Hurricane Wind Scale", "https://www.nhc.noaa.gov/aboutsshws.php", "hurricane typhoon cyclone saffir simpson category wind scale damage"),
    ("NOAA NHC: Storm surge overview", "https://www.nhc.noaa.gov/surge/", "storm surge tide coastal flood cyclone inundation"),
    ("NOAA: Is sea level rising?", "https://oceanservice.noaa.gov/facts/sealevel.html", "sea level rise coastal subsidence tide"),
    ("NASA: Sea level overview", "https://sealevel.nasa.gov/understanding-sea-level/overview", "sea level rise ice melt thermal expansion"),
    ("NWS: Heat safety", "https://www.weather.gov/safety/heat", "heat wave extreme heat illness"),
    ("NWS: Flood safety", "https://www.weather.gov/safety/flood", "flood flash river rainfall"),
    ("NWS: Wildfire safety", "https://www.weather.gov/safety/wildfire", "wildfire fire smoke"),
    ("NOAA CPC: ENSO diagnostic discussion", "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso_advisory/ensodisc.shtml", "el nino la nina enso oscillation"),
    ("Climate.gov: ENSO", "https://www.climate.gov/enso", "el nino la nina enso oscillation"),
    ("USGS: Earthquake hazards", "https://www.usgs.gov/programs/earthquake-hazards", "earthquake seismic fault ground motion"),
    ("FEMA: Flood maps", "https://www.fema.gov/flood-maps", "flood zone map floodplain sfha"),
    ("FEMA: Flood insurance (NFIP)", "https://www.fema.gov/flood-insurance", "nfip flood insurance claims premium"),
    ("FEMA: Building codes", "https://www.fema.gov/emergency-managers/risk-management/building-science/building-codes", "building code construction design elevation foundation wind resistant"),
    ("GOV.UK: Check your long term flood risk", "https://www.gov.uk/check-long-term-flood-risk", "flood risk england river surface water environment agency"),
    ("EPA: Heat islands", "https://www.epa.gov/heatislands", "urban heat island pavement cooling"),
    ("AirNow: AQI basics", "https://www.airnow.gov/aqi/aqi-basics/", "air quality index pm2.5 ozone particulate"),
    ("EU Commission: Carbon Border Adjustment Mechanism", "https://taxation-customs.ec.europa.eu/carbon-border-adjustment-mechanism_en", "cbam carbon border adjustment cement steel aluminium"),
    ("EU Commission: EU Emissions Trading System", "https://climate.ec.europa.eu/eu-action/carbon-markets/eu-emissions-trading-system-eu-ets_en", "ets emissions trading carbon price allowance"),
    ("IFRS: S2 Climate-related Disclosures", "https://www.ifrs.org/issued-standards/ifrs-sustainability-standards-navigator/ifrs-s2-climate-related-disclosures/", "ifrs s2 issb climate disclosure physical transition risk"),
    ("IFRS: TCFD recommendations", "https://www.ifrs.org/sustainability/tcfd/", "tcfd task force climate disclosure"),
    ("GHG Protocol: Corporate Standard", "https://ghgprotocol.org/corporate-standard", "greenhouse gas protocol scope 1 2 3 inventory accounting"),
]
SEED_URLS = [
    ("https://climrisk.io/research/the-loud-risk-and-the-quiet-one/", "ClimRisk research: The loud risk and the quiet one", 7 * 86400),
] + [(u, t, 30 * 86400 if "cpc.ncep" not in u else 12 * 3600) for (t, u, _k) in PRIMARY]
_UNTRUSTED = re.compile(r"(^|\.)(wikipedia|wikimedia|wikidata|wiktionary|fandom|quora|reddit|answers\.com|medium)\.", re.I)


def trusted(url: str) -> bool:
    host = urllib.parse.urlparse(url or "").netloc.lower()
    return bool(host) and not _UNTRUSTED.search(host)


_PURGED = {"done": False}


def purge_untrusted() -> int:
    """Remove anything learned from a user-edited site (earlier versions read Wikipedia). Runs once per process."""
    if _PURGED["done"]:
        return 0
    _PURGED["done"] = True
    n = 0
    with _LOCK:
        c = _db()
        try:
            for (sid, url) in c.execute("SELECT id, url FROM kn_sources").fetchall():
                if not trusted(url):
                    if _fts_ok(c):
                        for (pid, t, ti) in c.execute("SELECT id, text, title FROM kn_passages WHERE source_id=?", (sid,)).fetchall():
                            c.execute("INSERT INTO kn_fts(kn_fts, rowid, text, title) VALUES('delete', ?, ?, ?)", (pid, t, ti))
                    n += c.execute("DELETE FROM kn_passages WHERE source_id=?", (sid,)).rowcount
                    c.execute("DELETE FROM kn_sources WHERE id=?", (sid,))
            c.commit()
        finally:
            c.close()
    return n


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
    # '== Name ==' heading lines (if present) are kept as a plain label in front of the passage they introduce.
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



def learn_text(url: str, title: str, text: str, kind: str = "engine") -> int:
    """Index text the engine itself supplies (method notes). Re-learning replaces the old passages."""
    ps = _chunk(text) or [re.sub(r"\s+", " ", text).strip()]
    return _store(url, title, kind, ps)


def learn_url(url: str, title: Optional[str] = None, kind: str = "web") -> dict:
    if not re.match(r"^https?://", url or ""):
        return {"ok": False, "error": "that is not a web address"}
    if not trusted(url):
        return {"ok": False, "error": "I do not learn from wikis or user-edited sites: they are not citable. Point me to a regulator, agency, standard-setter or paper."}
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
    purge_untrusted()
    todo_urls = [(u, t, a) for (u, t, a) in SEED_URLS if force or _due(u, a)][:max_wiki]
    new = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(learn_url, u, t, "research" if "climrisk" in u else "primary") for (u, t, _) in todo_urls]
        for f in futs:
            try:
                r = f.result()
                new += int(r.get("passages", 0)) if r.get("ok") else 0
            except Exception:                                        # noqa: BLE001
                pass
    s = stats()
    return {"new_passages": new, "sources_checked": len(todo_urls), "elapsed_s": round(time.time() - t0, 1), "total_passages": s["passages"], "total_sources": s["sources"]}


def learn_topic(q: str, n_pages: int = 2) -> dict:
    """The index does not cover this: learn the best-matching primary-source pages from the curated list. If none match, say so; never fall back to a wiki."""
    if os.environ.get("CRI_STORY_OFFLINE", "").strip() not in ("", "0"):
        return {"ok": False, "error": "offline mode", "learned": []}
    ts = {stem(t) for t in _terms(q)}
    scored = []
    for (title, url, kw) in PRIMARY:
        k = {stem(w) for w in kw.split()}
        sc = len(ts & k)
        if sc:
            scored.append((sc, title, url))
    scored.sort(key=lambda x: -x[0])
    learned, got = [], 0
    for (_sc, title, url) in scored[:n_pages]:
        r = learn_url(url, title, "primary")
        if r.get("ok"):
            learned.append(title); got += r["passages"]
    return {"ok": got > 0, "learned": learned, "passages": got,
            "error": None if got else "no primary source on my list covers this; I will not answer it from a wiki"}


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
    purge_untrusted()
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
