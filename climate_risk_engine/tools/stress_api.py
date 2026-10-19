"""API stress harness: sends every route baseline + hostile variants and classifies the outcome.

  python tools/stress_api.py --mode offline|online --start N --count M --out results.jsonl [--only substr]

Outcomes per request:  ok (2xx) | rejected (4xx, expected for bad input) | upstream (502/503/504 with a message) | CRASH (500 or unhandled) | TIMEOUT | BADJSON (NaN/Infinity or not JSON) | LEAK (traceback text in a response)
Routes with external side effects (publishing, deleting, licence changes, LLM chat) are skipped. Runs against a throwaway HOME so nothing real is touched.
"""
import argparse, copy, json, os, signal, socket, sys, tempfile, time, math, re, traceback
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="offline"); ap.add_argument("--start", type=int, default=0); ap.add_argument("--count", type=int, default=10**6)
ap.add_argument("--out", default="/tmp/stress_api.jsonl"); ap.add_argument("--only", default=""); ap.add_argument("--timeout", type=int, default=25)
ap.add_argument("--max-variants", type=int, default=30); ap.add_argument("--skip", default=""); ap.add_argument("--resume", action="store_true")
a = ap.parse_args()

HOME = tempfile.mkdtemp(prefix="cri_stress_")
os.environ.update(HOME=HOME, CRI_HOME=HOME, CRI_CACHE_DIR=HOME + "/cache", CRI_RAG_SEMANTIC="0")
os.environ.setdefault("CRI_LICENCE_MODE", "off")
sys.path[:0] = [str(Path(__file__).resolve().parent.parent / "src")]

if a.mode == "offline":                                  # nothing may reach the network: tests how routes fail
    def _no(*x, **k): raise OSError("network disabled by stress harness")
    socket.socket.connect = _no
    socket.create_connection = _no
    _gai = socket.getaddrinfo
    socket.getaddrinfo = lambda *x, **k: (_ for _ in ()).throw(socket.gaierror("network disabled by stress harness"))

from fastapi.testclient import TestClient
from cri.api.main import app
client = TestClient(app, raise_server_exceptions=False)
spec = app.openapi()
SK = re.compile(r"publish|delete|linkedin|licen[sc]e|/chat|/ask|agent/|/connect|oauth|webhook|retrain|train|upload|shutdown|maintainer|/docs/|feedback|autopilot|scan|post_draft|draft", re.I)
SCH = spec.get("components", {}).get("schemas", {})


def deref(s, depth=0):
    if depth > 6 or not isinstance(s, dict): return s or {}
    if "$ref" in s: return deref(SCH.get(s["$ref"].split("/")[-1], {}), depth + 1)
    for k in ("anyOf", "oneOf"):
        if k in s:
            opts = [x for x in s[k] if x.get("type") != "null"]; return deref(opts[0] if opts else {}, depth + 1)
    if "allOf" in s: return deref(s["allOf"][0], depth + 1)
    return s


def synth(s, name="", depth=0):
    s = deref(s)
    if "default" in s and s["default"] is not None: return s["default"]
    if "enum" in s: return s["enum"][0]
    t, n = s.get("type"), name.lower()
    if t in ("integer", "number"):
        if "lat" in n: return 51.5
        if "lon" in n or "lng" in n: return 4.4
        if "year" in n: return 2030
        v = s.get("minimum", s.get("exclusiveMinimum", 1)); lo = v if v is not None else 1
        return max(lo, 1) if s.get("maximum", 1e9) >= max(lo, 1) else lo
    if t == "boolean": return False
    if t == "array":
        return [synth(s.get("items", {}), name, depth + 1)] if depth < 4 else []
    if t == "object" or "properties" in s:
        out = {}
        for k, v in (s.get("properties") or {}).items():
            if k in (s.get("required") or []) or depth < 1: out[k] = synth(v, k, depth + 1)
        return out
    if t == "string":
        if "company" in n or n in ("name", "query", "q"): return "Test Company"
        if "country" in n: return "NL"
        if "sector" in n: return "Utilities"
        if "email" in n: return "a@b.co"
        if "date" in n: return "2026-01-01"
        return "test"
    return None


EXTREME_NUM = [0, -1, 1e308, -1e308, 1e-308, 2**63, 91, -91, 181, -181, 90, -90, 180, -180]
WRONG = [None, "", " ", "x" * 20000, [], {}, [None], {"a": {"b": {"c": [1, 2, 3]}}}, "<script>alert(1)</script>", "'; DROP TABLE x;--", "../../etc/passwd", "\u0000", "🌍" * 50, True, 3.14, -7]


def variants(base, schema):
    out = [("baseline", base)]
    if not isinstance(base, dict):
        return out + [("empty", None)]
    props = deref(schema).get("properties", {}) or {}
    req = deref(schema).get("required", []) or []
    out.append(("empty_obj", {}))
    for r in req[:6]:
        b = copy.deepcopy(base); b.pop(r, None); out.append((f"missing:{r}", b))
    for k in list(base)[:10]:
        ps = deref(props.get(k, {})); isnum = ps.get("type") in ("integer", "number")
        vals = (EXTREME_NUM if isnum else []) + WRONG[:9]
        for i, w in enumerate(vals):
            b = copy.deepcopy(base); b[k] = w; out.append((f"{k}={repr(w)[:24]}", b))
    return out[:a.max_variants]


NANPAT = re.compile(r"\bNaN\b|\bInfinity\b")
LEAK = re.compile(r"Traceback \(most recent call last\)|File \"/.*\.py\", line \d+")


class TO(BaseException): pass
def _al(sig, frm): raise TO()
signal.signal(signal.SIGALRM, _al)

ops = []
for path, item in spec["paths"].items():
    for m, o in item.items():
        if m.upper() not in ("GET", "POST", "PUT"): continue
        if SK.search(path) or (a.only and a.only not in path): continue
        ops.append((m.upper(), path, o))
ops = ops[a.start:a.start + a.count]
if a.skip: ops = [o for o in ops if not any(k and k in o[1] for k in a.skip.split(","))]
if a.resume and os.path.exists(a.out):
    done = {json.loads(l)["route"] for l in open(a.out)}
    ops = [o for o in ops if f"{o[0]} {o[1]}" not in done]
f = open(a.out, "a")


def fill_path(path, o):
    for prm in o.get("parameters", []):
        if prm["in"] == "path":
            path = path.replace("{" + prm["name"] + "}", "1" if prm.get("schema", {}).get("type") in ("integer", "number") else "test")
    return path


for m, path, o in ops:
    url = fill_path(path, o); print("ROUTE", m, path, flush=True)
    qp = {}
    for prm in o.get("parameters", []):
        if prm["in"] == "query" and prm.get("required"): qp[prm["name"]] = synth(prm.get("schema", {}), prm["name"])
    body_schema = None
    if "requestBody" in o:
        body_schema = o["requestBody"]["content"].get("application/json", {}).get("schema")
    if "requestBody" in o and body_schema is None:
        continue                                         # multipart/other: covered separately
    base = synth(body_schema) if body_schema else None
    vs = variants(base, body_schema) if body_schema else [("baseline", None)]
    for label, payload in vs:
        t0 = time.time(); rec = {"route": f"{m} {path}", "variant": label, "mode": a.mode}
        try:
            signal.alarm(a.timeout)
            kw = {"params": qp} if qp else {}
            if payload is not None or body_schema: kw["json"] = payload
            r = client.request(m, url, **kw)
            signal.alarm(0)
            txt = r.text; rec["status"] = r.status_code
            if r.status_code >= 500 and r.status_code not in (502, 503, 504): rec["outcome"] = "CRASH"
            elif r.status_code in (502, 503, 504): rec["outcome"] = "upstream"
            elif r.status_code >= 400: rec["outcome"] = "rejected"
            else: rec["outcome"] = "ok"
            if LEAK.search(txt): rec["outcome"] = "LEAK"
            if rec["outcome"] == "ok" and "json" in r.headers.get("content-type", ""):
                if NANPAT.search(txt):
                    rec["outcome"] = "BADJSON"; rec["note"] = "NaN/Infinity in JSON"
                else:
                    try: json.loads(txt)
                    except Exception: rec["outcome"] = "BADJSON"
            if rec["outcome"] in ("CRASH", "LEAK", "BADJSON"): rec["body"] = txt[:300]
        except TO:
            rec["outcome"] = "TIMEOUT"
        except BaseException as e:                         # noqa: BLE001
            signal.alarm(0); rec["outcome"] = "CRASH"; rec["body"] = f"{type(e).__name__}: {str(e)[:200]}"
        rec["s"] = round(time.time() - t0, 2)
        f.write(json.dumps(rec) + "\n"); f.flush()
print("done", len(ops), "routes; ops index", a.start, "->", a.start + len(ops))
