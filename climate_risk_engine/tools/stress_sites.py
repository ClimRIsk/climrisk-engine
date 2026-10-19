"""Global extreme-site sweep: send awkward coordinates to the site-level routes and check for crashes, NaN/Infinity, and
silent-zero results (null must not read as zero risk).

  python tools/stress_sites.py --route location-hazard|water-risk|asset|report [--only name] --out file.jsonl
"""
import argparse, json, os, re, sys, tempfile, time
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--route", required=True); ap.add_argument("--out", default="/tmp/sites.jsonl")
ap.add_argument("--only", default=""); ap.add_argument("--start", type=int, default=0); ap.add_argument("--count", type=int, default=999)
a = ap.parse_args()
h = tempfile.mkdtemp(prefix="cri_sites_"); os.environ.update(HOME=h, CRI_HOME=h, CRI_RAG_SEMANTIC="0")
sys.path[:0] = [str(Path(__file__).resolve().parent.parent / "src")]
from fastapi.testclient import TestClient
from cri.api.main import app
c = TestClient(app, raise_server_exceptions=False)

SITES = {  # name: (lat, lon, why)
 "north_pole": (90, 0, "pole"), "south_pole": (-90, 0, "pole"), "near_pole_n": (89.9, 45, "pole"), "antimeridian_e": (0, 180, "antimeridian"),
 "antimeridian_w": (0, -180, "antimeridian"), "fiji_dateline": (-17.7, 179.99, "antimeridian"), "null_island": (0, 0, "gulf of guinea ocean"),
 "mid_pacific": (-40, -140, "open ocean"), "mid_atlantic": (30, -40, "open ocean"), "dead_sea": (31.5, 35.5, "-430 m"), "death_valley": (36.2, -116.8, "-86 m"),
 "rotterdam": (51.92, 4.48, "below sea level"), "new_orleans": (29.95, -90.07, "below sea level"), "everest": (27.99, 86.93, "8 km"), "la_paz": (-16.5, -68.15, "high plateau"),
 "atoll_male": (4.18, 73.51, "atoll 1.5 m"), "tuvalu": (-8.52, 179.2, "atoll"), "kiribati": (1.87, -157.4, "atoll"), "sahara": (23, 12, "hyper-arid"),
 "amazon": (-3.1, -60, "rainforest"), "greenland_ice": (72, -40, "ice sheet"), "antarctica_coast": (-77.85, 166.67, "ice"), "siberia_yakutsk": (62, 129.7, "permafrost"),
 "dhaka": (23.8, 90.4, "delta"), "jakarta": (-6.2, 106.8, "sinking coast"), "tokyo": (35.7, 139.7, "seismic+coastal"), "kathmandu": (27.7, 85.3, "seismic"),
 "miami": (25.76, -80.19, "hurricane+SLR"), "lagos": (6.5, 3.4, "coastal"), "reykjavik": (64.1, -21.9, "high lat volcanic"), "svalbard": (78.2, 15.6, "arctic"),
 "gibraltar": (36.14, -5.35, "strait"), "lake_victoria": (-1, 33, "inland water"), "caspian": (41, 51, "below sea level lake"), "mariana_trench": (11.35, 142.2, "deep ocean"),
 "equator_atlantic": (0, -25, "ocean"), "int_float_prec": (51.5000000001, 4.4000000001, "float precision"), "neg_zero": (-0.0, -0.0, "negative zero"),
 "tiny": (1e-9, 1e-9, "epsilon"),
}
BADNUM = re.compile(r"\bNaN\b|\bInfinity\b")
names = [n for n in SITES if a.only in n][a.start:a.start + a.count]
f = open(a.out, "a")
for n in names:
    lat, lon, why = SITES[n]; t0 = time.time(); rec = {"route": a.route, "site": n, "why": why}
    try:
        if a.route == "location-hazard": r = c.post("/analyze/location-hazard", json={"lat": lat, "lon": lon, "location": n})
        elif a.route == "water-risk": r = c.get("/physical/water-risk", params={"lat": lat, "lon": lon, "sector": "cement"})
        elif a.route == "asset": r = c.post("/asset/assess", json={"lat": lat, "lon": lon, "asset_name": n, "include_visuals": False, "radius_km": 1.0})
        elif a.route == "report": r = c.post("/reports/physical-risk", json={"assets": [{"lat": lat, "lon": lon, "name": n, "sector": "cement"}], "include_image": False})
        else: raise SystemExit("route?")
        rec["status"] = r.status_code; t = r.text
        rec["outcome"] = "ok" if r.status_code < 400 else ("rejected" if r.status_code < 500 else "CRASH")
        if re.search(r"Traceback \(most recent", t): rec["outcome"] = "LEAK"
        if r.status_code < 400 and BADNUM.search(t): rec["outcome"] = "BADJSON"
        rec["len"] = len(t); rec["head"] = t[:260] if rec["outcome"] != "ok" else ""
        if r.status_code < 400:
            try: rec["json"] = r.json()
            except Exception: rec["outcome"] = "BADJSON"
    except BaseException as e:
        rec["outcome"] = "CRASH"; rec["head"] = f"{type(e).__name__}: {e}"[:200]
    rec["s"] = round(time.time() - t0, 1)
    j = rec.pop("json", None)
    if j is not None:
        if a.route == "report" and isinstance(j, dict) and j.get("assets"):
            x0 = j["assets"][0]; rec["overall"] = x0.get("overall_rating"); rec["site_summary"] = (x0.get("site") or {}).get("summary", "")[:160]
            rec["zones"] = {z["key"]: z.get("rating") for z in x0.get("zones", [])}; rec["gaps"] = [g[:80] for g in x0.get("data_gaps", [])]
        rec["keys"] = list(j.keys())[:12] if isinstance(j, dict) else type(j).__name__
        rec["blob"] = json.dumps(j)[:1500]
    f.write(json.dumps(rec) + "\n"); f.flush()
print("done", len(names))
