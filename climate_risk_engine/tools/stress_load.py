"""Concurrency / determinism: identical requests fired in parallel must all succeed and return identical bodies."""
import os, sys, tempfile, json, time, hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
h = tempfile.mkdtemp(); os.environ.update(HOME=h, CRI_HOME=h, CRI_RAG_SEMANTIC="0")
sys.path[:0] = [str(Path(__file__).resolve().parent.parent / "src")]
from fastapi.testclient import TestClient
from cri.api.main import app
c = TestClient(app, raise_server_exceptions=False)
CASES = [("POST", "/agent/assess", {"company_name": "Acme Cement", "sector_hint": "cement", "revenue_usd_m": 900}),
         ("POST", "/agent/assess", {"company_name": "Acme Utility", "sector_hint": "utilities", "revenue_usd_m": 4000}),
         ("GET", "/physical/water-risk?lat=51.9&lon=4.5&sector=cement", None),
         ("POST", "/analyze/location-hazard", {"lat": 10, "lon": 10})]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
def call(i):
    m, p, b = CASES[i % len(CASES)]
    t = time.time(); r = c.request(m, p, json=b) if b else c.request(m, p)
    body = r.text
    try:
        j = r.json()
        for k in ("elapsed_s", "generated", "timestamp", "run_id", "id", "created"): j.pop(k, None) if isinstance(j, dict) else None
        body = json.dumps(j, sort_keys=True, default=str)
    except Exception: pass
    return i % len(CASES), r.status_code, hashlib.md5(body.encode()).hexdigest(), time.time() - t
t0 = time.time()
with ThreadPoolExecutor(16) as ex: res = list(ex.map(call, range(N)))
bad = [r for r in res if r[1] >= 500]
for k in range(len(CASES)):
    hs = {r[2] for r in res if r[0] == k}
    print(CASES[k][1], "n=%d" % sum(1 for r in res if r[0] == k), "distinct bodies:", len(hs), "statuses:", sorted({r[1] for r in res if r[0] == k}))
print("5xx:", len(bad), "wall %.1fs" % (time.time() - t0), "slowest %.1fs" % max(r[3] for r in res))
