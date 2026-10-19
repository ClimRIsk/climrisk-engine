"""Refresh the golden snapshot (needs network). Reviews the diff before committing.  python tests/golden/refresh.py"""
import json, os, sys, tempfile
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
with ThreadPoolExecutor(7) as ex: res = dict(ex.map(one, SITES))
import datetime
(here / f"snapshot_{datetime.date.today().isoformat()}.json").write_text(json.dumps({"generated": datetime.date.today().isoformat(), "mode": "online", "sites": res}, indent=1))
print("written; remove the older snapshot file and review the diff")
