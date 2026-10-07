"""
cri/resident/memory.py — the resident agent's persistent memory (SQLite, stdlib only).

Tables: messages (conversation), facts (things the user told the agent about their sites), feedback (corrections that
retrain the intent model), runs (every tool chain the agent executed), posts (LinkedIn drafts and their status), state
(key/value: last result, pending approval). Lives in $CRI_DATA_DIR/resident.db (default ~/.climrisk/resident.db).
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

_LOCK = threading.RLock()


def db_path() -> Path:
    base = Path(os.environ.get("CRI_DATA_DIR") or Path.home() / ".climrisk")
    base.mkdir(parents=True, exist_ok=True)
    return base / "resident.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY, ts REAL, session TEXT, role TEXT, text TEXT, intent TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS facts(id INTEGER PRIMARY KEY, ts REAL, subject TEXT, text TEXT, kv TEXT);
CREATE TABLE IF NOT EXISTS feedback(id INTEGER PRIMARY KEY, ts REAL, text TEXT, predicted TEXT, intent TEXT, used INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS runs(id INTEGER PRIMARY KEY, ts REAL, intent TEXT, args TEXT, steps TEXT, ok INTEGER, summary TEXT);
CREATE TABLE IF NOT EXISTS posts(id INTEGER PRIMARY KEY, ts REAL, title TEXT, text TEXT, image BLOB, status TEXT, meta TEXT, published_ts REAL, urn TEXT);
CREATE TABLE IF NOT EXISTS state(k TEXT PRIMARY KEY, v TEXT);
"""


def _c() -> sqlite3.Connection:
    c = sqlite3.connect(str(db_path()), timeout=15)
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    return c


def _rows(q: str, a: tuple = ()) -> list:
    with _LOCK, _c() as c:
        return [dict(r) for r in c.execute(q, a).fetchall()]


def _exec(q: str, a: tuple = ()) -> int:
    with _LOCK, _c() as c:
        cur = c.execute(q, a)
        return cur.lastrowid


# messages
def add_message(session: str, role: str, text: str, intent: str = "", data: Optional[dict] = None) -> int:
    return _exec("INSERT INTO messages(ts,session,role,text,intent,data) VALUES(?,?,?,?,?,?)",
                 (time.time(), session, role, text, intent, json.dumps(data or {}, default=str)[:20000]))


def recent_messages(session: str, n: int = 20) -> list:
    r = _rows("SELECT * FROM messages WHERE session=? ORDER BY id DESC LIMIT ?", (session, n))
    return list(reversed(r))


# facts
def add_fact(subject: str, text: str, kv: Optional[dict] = None) -> int:
    return _exec("INSERT INTO facts(ts,subject,text,kv) VALUES(?,?,?,?)", (time.time(), subject.strip().lower(), text.strip(), json.dumps(kv or {})))


def facts(subject: Optional[str] = None, limit: int = 50) -> list:
    if subject:
        s = f"%{subject.strip().lower()}%"
        r = _rows("SELECT * FROM facts WHERE subject LIKE ? OR lower(text) LIKE ? ORDER BY id DESC LIMIT ?", (s, s, limit))
    else:
        r = _rows("SELECT * FROM facts ORDER BY id DESC LIMIT ?", (limit,))
    for x in r:
        x["kv"] = json.loads(x["kv"] or "{}")
    return r


def forget_fact(fid: int) -> None:
    _exec("DELETE FROM facts WHERE id=?", (fid,))


# feedback
def add_feedback(text: str, predicted: str, intent: str) -> int:
    return _exec("INSERT INTO feedback(ts,text,predicted,intent) VALUES(?,?,?,?)", (time.time(), text, predicted, intent))


def feedback_rows(only_unused: bool = False) -> list:
    return _rows("SELECT * FROM feedback" + (" WHERE used=0" if only_unused else "") + " ORDER BY id")


def mark_feedback_used() -> None:
    _exec("UPDATE feedback SET used=1")


# runs
def add_run(intent: str, args: dict, steps: list, ok: bool, summary: str) -> int:
    return _exec("INSERT INTO runs(ts,intent,args,steps,ok,summary) VALUES(?,?,?,?,?,?)",
                 (time.time(), intent, json.dumps(args, default=str)[:4000], json.dumps(steps, default=str)[:30000], int(ok), summary[:500]))


def runs(limit: int = 30) -> list:
    r = _rows("SELECT id,ts,intent,args,ok,summary FROM runs ORDER BY id DESC LIMIT ?", (limit,))
    for x in r:
        x["args"] = json.loads(x["args"] or "{}")
    return r


# posts
def add_post(title: str, text: str, image: Optional[bytes] = None, meta: Optional[dict] = None) -> int:
    return _exec("INSERT INTO posts(ts,title,text,image,status,meta) VALUES(?,?,?,?,?,?)",
                 (time.time(), title, text, image, "draft", json.dumps(meta or {}, default=str)))


def get_post(pid: int) -> Optional[dict]:
    r = _rows("SELECT * FROM posts WHERE id=?", (pid,))
    if not r:
        return None
    p = r[0]; p["meta"] = json.loads(p["meta"] or "{}"); return p


def update_post(pid: int, **kw: Any) -> None:
    sets, vals = [], []
    for k, v in kw.items():
        if k in ("title", "text", "image", "status", "published_ts", "urn"):
            sets.append(f"{k}=?"); vals.append(v)
        elif k == "meta":
            sets.append("meta=?"); vals.append(json.dumps(v, default=str))
    if sets:
        _exec(f"UPDATE posts SET {','.join(sets)} WHERE id=?", tuple(vals) + (pid,))


def posts(limit: int = 30) -> list:
    r = _rows("SELECT id,ts,title,text,status,published_ts,urn FROM posts ORDER BY id DESC LIMIT ?", (limit,))
    return r


# state
def set_state(k: str, v: Any) -> None:
    _exec("INSERT INTO state(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, json.dumps(v, default=str)[:200000]))


def get_state(k: str, default: Any = None) -> Any:
    r = _rows("SELECT v FROM state WHERE k=?", (k,))
    return json.loads(r[0]["v"]) if r else default
