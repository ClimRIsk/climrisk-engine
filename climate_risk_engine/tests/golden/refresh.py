"""Refresh the golden snapshot (needs network). Review the diff before committing.

  python tests/golden/refresh.py                  # all sites
  python tests/golden/refresh.py --budget 140     # resumable: stop starting new batches after 140 s, rerun to continue
Partial progress is kept in tests/golden/_partial.json and turned into snapshot_<date>.json when every site is done.
"""
import json, os, sys, tempfile, time, datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
here = Path(__file__).resolve().parent
sys.path[:0] = [str(here), str(here.parent.parent / "src")]
h = tempfile.mkdtemp(); os.environ.update(HOME=h, CRI_HOME=h, CRI_RAG_SEMANTIC="0")
from sites import SITES
from cri.physical_report import build_asset_report

def one(n):
    la, lo, _, _ = SITES[n]
    try:
        a = build_asset_report(la, lo, n, "cement", None, None, include_image=False)
        return n, {z["key"]: z["rating"] for z in a["zones"]}
    except Exception as e:
        return n, {"error": str(e)[:100]}

budget = float(sys.argv[sys.argv.index("--budget") + 1]) if "--budget" in sys.argv else 1e9
part = here / "_partial.json"
res = json.loads(part.read_text()) if part.exists() and "--fresh" not in sys.argv else {}
t0 = time.time()
todo = [n for n in SITES if n not in res]
while todo and time.time() - t0 < budget:
    batch, todo = todo[:7], todo[7:]
    with ThreadPoolExecutor(7) as ex:
        for n, r in ex.map(one, batch):
            res[n] = r
    part.write_text(json.dumps(res))
left = [n for n in SITES if n not in res]
if left:
    print(f"{len(res)}/{len(SITES)} done; rerun to continue ({len(left)} left)")
else:
    d = datetime.date.today().isoformat()
    (here / f"snapshot_{d}.json").write_text(json.dumps({"generated": d, "mode": "online", "sites": res}, indent=1))
    part.unlink()
    print("written; remove the older snapshot file and review the diff")
