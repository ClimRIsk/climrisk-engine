"""
cri/resident/userdocs.py — the user's own documents (reports, filings, registers, notes), searchable and cited, kept on this computer.

Add a file and its text is cut into passages, stored in a private SQLite file (~/.climrisk/userdocs.db, never uploaded, never mixed into the public knowledge index) and
searched with BM25 blended with the offline dense embeddings (cri.resident.semantic) when they are available. Answers quote the user's passages with the file name and the
page / sheet / section they came from; if the documents do not contain the answer, the agent says so.

  add(path)               read a file (txt, md, csv, pdf, docx, xlsx) and index it                -> {ok, name, passages, pages}
  add_text(name, text)    index text the user pasted
  list_docs()             what is indexed                                                           -> [{id, name, passages, added}]
  remove(name_or_id)      forget a document                                                         -> passages removed
  search(q, k)            best passages, in the shape knowledge.search() returns, so the same answer path can use them
  ask(q)                  answer from the user's documents only (cited), or say they do not cover it
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import sqlite3
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

_LOCK = threading.RLock()
_WORDS = (90, 170)
SEM_W = float(os.environ.get("CRI_RAG_SEM_W", "0.35"))
MAX_BYTES = 60 * 1024 * 1024
_CACHE: dict = {"stamp": None, "rows": [], "inv": {}, "df": {}, "avgdl": 1.0, "mat": None}


def _path() -> Path:
    home = Path(os.environ.get("CRI_HOME") or (Path.home() / ".climrisk"))
    home.mkdir(parents=True, exist_ok=True)
    return home / "userdocs.db"


def _db() -> sqlite3.Connection:
    c = sqlite3.connect(str(_path()))
    c.execute("CREATE TABLE IF NOT EXISTS ud_docs (id INTEGER PRIMARY KEY, name TEXT UNIQUE, sha TEXT, added REAL, n INTEGER)")
    c.execute("CREATE TABLE IF NOT EXISTS ud_passages (id INTEGER PRIMARY KEY, doc_id INTEGER, loc TEXT, text TEXT, vec BLOB)")
    return c


# ── tokens (same light stemming as the case library) ─────────────────────────────────────────────

_STOP = set("a an the and or of to in on at for from by with without is are was were be been being do does did how why what when where which who whom whose can could should would will it its this that "
            "these those as about into than then so such not no yes me my we our you your i tell explain describe give show please more most any some there their they them his her he she say said".split())


def _stem(t: str) -> str:
    for suf in ("ing", "ed", "es", "ly", "s"):
        if t.endswith(suf) and len(t) - len(suf) >= 4:
            return t[:-len(suf)]
    return t


def _tokens(s: str) -> list:
    s = re.sub(r"[\-‐-―]", " ", s.lower())
    return [_stem(w) for w in re.findall(r"[a-z0-9][a-z0-9]+", s) if w not in _STOP]


# ── reading files ─────────────────────────────────────────────────────────────────────────────────

def _read_pdf(p: Path) -> list:
    try:
        from pypdf import PdfReader
    except Exception as e:                                           # noqa: BLE001
        raise RuntimeError("reading PDFs needs the pypdf package, which is not installed") from e
    out = []
    for i, pg in enumerate(PdfReader(str(p)).pages, 1):
        try:
            out.append((f"p.{i}", pg.extract_text() or ""))
        except Exception:                                            # noqa: BLE001
            out.append((f"p.{i}", ""))
    return out


def _read_docx(p: Path) -> list:
    import zipfile
    from xml.etree import ElementTree as ET
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(p) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    paras, head = [], ""
    sections: list = []
    for para in root.iter(ns + "p"):
        txt = "".join(t.text or "" for t in para.iter(ns + "t")).strip()
        if not txt:
            continue
        st = para.find(f"{ns}pPr/{ns}pStyle")
        style = (st.get(ns + "val") if st is not None else "") or ""
        if style.lower().startswith("heading") or style.lower() == "title":
            if paras:
                sections.append((head or "document", "\n\n".join(paras)))
            head, paras = txt, []
        else:
            paras.append(txt)
    if paras:
        sections.append((head or "document", "\n\n".join(paras)))
    return sections


def _read_xlsx(p: Path) -> list:
    from openpyxl import load_workbook
    wb = load_workbook(str(p), read_only=True, data_only=True)
    out = []
    for ws in wb.worksheets:
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            continue
        head = [str(h) if h is not None else "" for h in rows[0]]
        lines = []
        for r in rows[1:]:
            cells = [f"{head[i] or 'col' + str(i + 1)}: {v}" for i, v in enumerate(r) if v not in (None, "")]
            if cells:
                lines.append("; ".join(cells) + ".")
        out.append((f"sheet {ws.title}", "\n\n".join(lines)))
    return out


def _read_csv(p: Path) -> list:
    import csv
    with open(p, newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.reader(f))
    if not rows:
        return []
    head = rows[0]
    lines = ["; ".join(f"{head[i] or 'col' + str(i + 1)}: {v}" for i, v in enumerate(r) if v.strip()) + "." for r in rows[1:] if any(v.strip() for v in r)]
    return [("table", "\n\n".join(lines))]


def read_file(p: Path) -> list:
    """[(location, text)] for a file; raises RuntimeError with a plain message when it cannot be read."""
    ext = p.suffix.lower()
    if p.stat().st_size > MAX_BYTES:
        raise RuntimeError("that file is larger than 60 MB")
    if ext == ".pdf":
        return _read_pdf(p)
    if ext == ".docx":
        return _read_docx(p)
    if ext in (".xlsx", ".xlsm"):
        return _read_xlsx(p)
    if ext in (".csv", ".tsv"):
        return _read_csv(p)
    if ext in (".txt", ".md", ".markdown", ".json", ".html", ".htm", ".log", ""):
        t = p.read_text(encoding="utf-8", errors="replace")
        if ext in (".html", ".htm"):
            t = re.sub(r"<(script|style)[\s\S]*?</\1>", " ", t, flags=re.I)
            t = re.sub(r"<[^>]+>", " ", t)
        return [("text", t)]
    raise RuntimeError(f"I cannot read {ext} files yet (txt, md, csv, pdf, docx, xlsx work)")


def _chunk(text: str) -> list:
    paras = [re.sub(r"\s+", " ", x).strip() for x in re.split(r"\n\s*\n|\n(?=[A-Z0-9][^\n]{0,80}\n)", text) if x.strip()]
    if len(paras) <= 1:
        paras = [re.sub(r"\s+", " ", x).strip() for x in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)]
    out, cur, n = [], [], 0
    for p in paras:
        w = len(p.split())
        if w < 4:
            continue
        if n and n + w > _WORDS[1]:
            out.append(" ".join(cur)); cur, n = [], 0
        cur.append(p); n += w
        if n >= _WORDS[0]:
            out.append(" ".join(cur)); cur, n = [], 0
    if cur and (n >= 20 or not out):
        out.append(" ".join(cur))
    return [x for x in out if len(x.split()) >= 8]


# ── storing ───────────────────────────────────────────────────────────────────────────────────────

def _store(name: str, sha: str, parts: list) -> dict:
    passages = []
    for loc, text in parts:
        for ch in _chunk(text):
            passages.append((loc, ch))
    if not passages:
        return {"ok": False, "error": "no readable text in that file (a scanned PDF needs OCR first)"}
    vecs = None
    try:
        from . import semantic as S
        vecs = S.embed_texts([f"{name}. {t}" for _, t in passages])
    except Exception:                                                # noqa: BLE001
        vecs = None
    with _LOCK:
        c = _db()
        try:
            old = c.execute("SELECT id FROM ud_docs WHERE name=?", (name,)).fetchone()
            if old:
                c.execute("DELETE FROM ud_passages WHERE doc_id=?", (old[0],))
                c.execute("DELETE FROM ud_docs WHERE id=?", (old[0],))
            cur = c.execute("INSERT INTO ud_docs(name, sha, added, n) VALUES (?,?,?,?)", (name, sha, time.time(), len(passages)))
            did = cur.lastrowid
            for i, (loc, t) in enumerate(passages):
                blob = vecs[i].astype("float16").tobytes() if vecs is not None else None
                c.execute("INSERT INTO ud_passages(doc_id, loc, text, vec) VALUES (?,?,?,?)", (did, loc, t, blob))
            c.commit()
        finally:
            c.close()
    _CACHE["stamp"] = None
    return {"ok": True, "name": name, "passages": len(passages), "locations": len({l for l, _ in passages}), "semantic": vecs is not None}


def add(path: str) -> dict:
    p = Path(os.path.expanduser(str(path).strip().strip('"\'')))
    if not p.is_file():
        return {"ok": False, "error": f"I cannot find a file at {p}"}
    try:
        parts = read_file(p)
    except RuntimeError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:                                           # noqa: BLE001
        return {"ok": False, "error": f"I could not read that file ({type(e).__name__})"}
    return _store(p.name, hashlib.sha256(p.read_bytes()).hexdigest()[:16], parts)


def add_text(name: str, text: str) -> dict:
    return _store(name, hashlib.sha256(text.encode()).hexdigest()[:16], [("text", text)])


def list_docs() -> list:
    c = _db()
    try:
        return [{"id": r[0], "name": r[1], "passages": r[2], "added": time.strftime("%d %b %Y", time.localtime(r[3]))} for r in c.execute("SELECT id, name, n, added FROM ud_docs ORDER BY added DESC")]
    finally:
        c.close()


def remove(which) -> int:
    with _LOCK:
        c = _db()
        try:
            row = c.execute("SELECT id FROM ud_docs WHERE id=? OR lower(name)=lower(?)", (which if str(which).isdigit() else -1, str(which))).fetchone()
            if not row:
                row = c.execute("SELECT id FROM ud_docs WHERE lower(name) LIKE ?", (f"%{str(which).lower()}%",)).fetchone()
            if not row:
                return 0
            n = c.execute("DELETE FROM ud_passages WHERE doc_id=?", (row[0],)).rowcount
            c.execute("DELETE FROM ud_docs WHERE id=?", (row[0],))
            c.commit()
        finally:
            c.close()
    _CACHE["stamp"] = None
    return n


# ── searching ─────────────────────────────────────────────────────────────────────────────────────

def _load() -> dict:
    p = _path()
    stamp = (p.stat().st_mtime if p.exists() else 0, p.stat().st_size if p.exists() else 0)
    with _LOCK:
        if _CACHE["stamp"] == stamp:
            return _CACHE
        c = _db()
        try:
            rows = c.execute("SELECT p.id, d.name, p.loc, p.text, p.vec, d.added FROM ud_passages p JOIN ud_docs d ON d.id = p.doc_id ORDER BY p.id").fetchall()
        finally:
            c.close()
        inv, df, total = defaultdict(list), defaultdict(int), 0
        items = []
        for i, (pid, name, loc, text, vec, added) in enumerate(rows):
            toks = _tokens(text + " " + name)
            total += len(toks) or 1
            tf: dict = defaultdict(int)
            for t in toks:
                tf[t] += 1
            for t, n in tf.items():
                inv[t].append((i, n)); df[t] += 1
            items.append({"id": pid, "name": name, "loc": loc, "text": text, "len": len(toks) or 1, "added": added})
        mat = None
        if rows and all(r[4] is not None for r in rows):
            import numpy as np
            mat = np.stack([np.frombuffer(r[4], dtype=np.float16).astype(np.float32) for r in rows])
            nn = np.linalg.norm(mat, axis=1, keepdims=True); nn[nn == 0] = 1.0
            mat = mat / nn
        _CACHE.update(stamp=stamp, rows=items, inv=dict(inv), df=dict(df), avgdl=total / max(1, len(items)), mat=mat)
        return _CACHE


def has_docs() -> bool:
    try:
        return bool(_load()["rows"])
    except Exception:                                                # noqa: BLE001
        return False


def search(q: str, k: int = 6) -> list:
    ix = _load()
    items = ix["rows"]
    if not items:
        return []
    qt = set(_tokens(q))
    n = len(items)
    sc: dict = defaultdict(float)
    for t in qt:
        post = ix["inv"].get(t)
        if not post:
            continue
        idf = math.log(1 + (n - ix["df"][t] + 0.5) / (ix["df"][t] + 0.5))
        for i, tf in post:
            sc[i] += idf * tf * 2.4 / (tf + 1.4 * (0.25 + 0.75 * items[i]["len"] / ix["avgdl"]))
    top = max(sc.values()) if sc else 1.0
    sem: dict = {}
    if ix["mat"] is not None and SEM_W > 0:
        try:
            from . import semantic as S
            v = S.embed_texts([q], query=True)
            if v is not None:
                import numpy as np
                v = v[0] / (np.linalg.norm(v[0]) or 1.0)
                cs = ix["mat"] @ v
                for i in cs.argsort()[::-1][:40]:
                    if cs[i] > 0.2:
                        sem[int(i)] = float(cs[i])
        except Exception:                                            # noqa: BLE001
            sem = {}
    stop = max(sem.values()) if sem else 1.0
    w = SEM_W if sem else 0.0
    ranked = sorted(((((1 - w) * sc.get(i, 0.0) / (top or 1.0)) + w * sem.get(i, 0.0) / (stop or 1.0), i) for i in set(sc) | set(sem)), reverse=True)
    out, per = [], defaultdict(int)
    for s, i in ranked:
        r = items[i]
        if per[r["name"]] >= 3:
            continue
        per[r["name"]] += 1
        cov = len(qt & set(_tokens(r["text"]))) / max(1, len(qt))
        out.append({"id": len(out) + 1, "title": f'{r["name"]} ({r["loc"]})', "url": f'file://{r["name"]}#{r["loc"]}', "text": r["text"], "learned": time.strftime("%d %b %Y", time.localtime(r["added"])),
                    "score": round(s, 3), "coverage": round(cov, 2), "kind": "userdoc", "doc": r["name"], "loc": r["loc"]})
        if len(out) >= k:
            break
    return out


def ask(q: str, k: int = 6) -> dict:
    """Answer from the user's documents only. Every sentence is quoted from a passage and cited as [n]; no passage that covers the question means no answer."""
    from . import answer as AN, knowledge as K
    ps = search(q, k)
    if not ps:
        if not has_docs():
            return {"ok": False, "reply": "You have not added any documents yet. Say “add document <path>” and I will read it.", "sources": [], "mode": "none"}
        return {"ok": False, "reply": "Your documents do not seem to cover that, so I will not guess.", "sources": [], "mode": "none"}
    best = max(p["coverage"] for p in ps[:3])
    dense_hit = ps[0]["score"] >= 0.55
    if best < 0.34 and not dense_hit:
        return {"ok": False, "reply": "Your documents do not seem to cover that, so I will not guess. The closest passages are listed below.", "sources": AN._src(ps[:3]), "mode": "none"}
    ls = AN.llm_status()
    text, mode = None, "extractive"
    if ls["available"]:
        text = AN._llm(q, ps, ls["model"], None)
        if text:
            mode = f"language model ({ls['model']}, running on this computer)"
    if not text:
        text = AN._extractive(q, ps)
    if not text:
        return {"ok": False, "reply": "I found related passages in your documents but no sentence that answers the question directly; they are listed below.", "sources": AN._src(ps[:4]), "mode": "none"}
    used = sorted({int(x) for x in re.findall(r"\[(\d+)\]", text) if int(x) <= len(ps)})
    return {"ok": True, "reply": text, "sources": AN._src(ps, used), "mode": mode}
