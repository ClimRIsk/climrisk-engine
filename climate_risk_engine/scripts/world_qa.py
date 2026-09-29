"""
Worldwide quality test of the app's two main features.

  A. Company Asset Map — companies on every continent: found? speed? do the
     cyclone / flood screens make geographic sense?
  B. Site analysis (the 3D view's backend, visuals included) — stress sites:
     below sea level, equator, desert, plateau, Arctic, dateline, southern
     hemisphere, monsoon and typhoon coasts: completes? errors? speed? sensible?

Results: data/qa/world_qa_companies.json, data/qa/world_qa_sites/<id>.json
Usage:
  python scripts/world_qa.py --part companies
  QA_BATCH=2 python scripts/world_qa.py --part sites     (repeat until done)
  python scripts/world_qa.py --part report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "data" / "qa"
SITES_DIR = "world_qa_sites" + (("_" + os.environ["QA_RUN"]) if os.environ.get("QA_RUN") else "")
(OUT / SITES_DIR).mkdir(parents=True, exist_ok=True)

# (query, continent, expectation)  expectation: "cyclone" = expect some cyclone-exposed assets,
# "no_cyclone" = expect none (Europe / interior), None = no strong expectation
COMPANIES = [
    ("NTPC", "Asia", "cyclone"), ("Adani Power", "Asia", None), ("Tata Steel", "Asia", None),
    ("China Huaneng", "Asia", "cyclone"), ("Anhui Conch", "Asia", "cyclone"), ("POSCO", "Asia", "cyclone"),
    ("JERA", "Asia", "cyclone"), ("Nippon Steel", "Asia", "cyclone"), ("Semen Indonesia", "Asia", None),
    ("Petronas", "Asia", None), ("San Miguel", "Asia", "cyclone"),
    ("RWE", "Europe", "no_cyclone"), ("Enel", "Europe", None), ("Heidelberg", "Europe", None),
    ("ArcelorMittal", "Europe", None), ("Uniper", "Europe", "no_cyclone"), ("Equinor", "Europe", None),
    ("Duke Energy", "North America", "cyclone"), ("Exxon Mobil", "North America", "cyclone"),
    ("Cemex", "North America", None), ("Comision Federal de Electricidad", "North America", "cyclone"),
    ("Petrobras", "South America", None), ("Vale", "South America", None), ("Codelco", "South America", "no_cyclone"),
    ("Eskom", "Africa", "no_cyclone"), ("Sasol", "Africa", None), ("Dangote", "Africa", "no_cyclone"),
    ("OCP", "Africa", None),
    ("AGL Energy", "Oceania", None), ("BHP", "Oceania", "cyclone"), ("Rio Tinto", "Oceania", None),
    ("Fortescue", "Oceania", "cyclone"),
    ("Saudi Arabian Oil", "Middle East", None), ("Abu Dhabi National Oil", "Middle East", None), ("SABIC", "Middle East", None),
]

SITES = [  # id, name, lat, lon, type, what it stresses, sanity check
    ("rotterdam", "Rotterdam Botlek (below sea level)", 51.8740, 4.3310, "port", "polder below sea level", "flood_not_negligible"),
    ("singapore", "Singapore Jurong Island", 1.2660, 103.6960, "industrial", "equator, no cyclones", "no_cyclone"),
    ("dubai", "Dubai Jebel Ali", 25.0110, 55.0610, "port", "desert heat", "no_cyclone"),
    ("johannesburg", "Johannesburg (1,750 m plateau)", -26.2040, 28.0470, "office", "high plateau", "no_cyclone"),
    ("norilsk", "Norilsk smelter (Arctic)", 69.3290, 88.1790, "industrial", "Arctic, snow, long winter", "no_cyclone"),
    ("fiji", "Suva, Fiji (dateline region)", -18.1420, 178.4410, "port", "near 180°, S. Pacific cyclones", "cyclone"),
    ("perth", "Kwinana, Perth", -32.2050, 115.7700, "industrial", "southern hemisphere, mediterranean", None),
    ("mumbai", "Mumbai Trombay", 19.0120, 72.9230, "industrial", "monsoon coast", None),
    ("tokyo", "Kawasaki, Tokyo Bay", 35.5100, 139.7500, "industrial", "typhoon coast, dense city", "cyclone"),
    ("shanghai", "Shanghai Waigaoqiao", 31.3570, 121.5870, "port", "typhoon coast, delta", "cyclone"),
    ("lagos", "Lagos Apapa", 6.4410, 3.3720, "port", "tropical coast, Africa", "no_cyclone"),
    ("saopaulo", "Cubatão, São Paulo", -23.8900, -46.4250, "industrial", "S. America escarpment", "no_cyclone"),
    ("lima", "Callao, Lima (desert coast)", -12.0500, -77.1400, "port", "hyper-arid coast", "no_cyclone"),
    ("brisbane", "Brisbane port", -27.3840, 153.1720, "port", "subtropical, cyclone margin", None),
    # regression sites — the fixes must not hide real risk
    ("jamuna", "Jamuna char (radar-observed floods)", 24.4200, 89.7600, "warehouse", "monsoon floodplain", "flood_not_negligible"),
    ("charleston", "Charleston peninsula (2.6 m)", 32.7765, -79.9311, "office", "validated low-lying coast", "flood_not_negligible"),
    ("kedarnath", "Kedarnath valley", 30.7346, 79.0669, "hospital", "debris-flow valley", "debris_high"),
]


def run_companies():
    from cri.data.asset_register import company_exposure, search, stats
    t = time.time(); stats(); load_s = time.time() - t
    rows = []
    for q, cont, expect in COMPANIES:
        t = time.time(); res = search(q, limit=20); t_search = (time.time() - t) * 1000
        grp = [r["key"] for r in res if r["match"] == "group"]
        keys = grp or [r["key"] for r in res if r["match"] in ("exact", "prefix")] or [r["key"] for r in res[:1]]
        row = {"query": q, "continent": cont, "expect": expect, "found": bool(res),
               "search_ms": round(t_search, 1), "entities": [r["name"] for r in res[:3]]}
        if keys:
            t = time.time(); e = company_exposure(keys); row["exposure_ms"] = round((time.time() - t) * 1000, 1)
            s = e["summary"]
            row.update(n_assets=e["n_assets"], n_countries=len(s["countries"]),
                       top_countries=list(s["countries"])[:4],
                       cyclone_cat1_share=s["share_assets_cyclone_cat1plus_100yr"],
                       any_cyclone=any(a["cyclone"]["rate_ts"] >= 0.05 for a in e["assets"]),
                       low_lying_share=s["share_assets_low_lying_lt3m"], by_level=s["by_level"],
                       worst=e["assets"][0]["name"] + " — " + e["assets"][0]["screen_level"])
            row["sanity"] = ("ok" if expect is None else
                             "ok" if (expect == "cyclone") == row["any_cyclone"] else "CHECK")
        else:
            row["sanity"] = "NOT FOUND"
        rows.append(row)
        print(f"{q:32s} {cont:14s} found={row['found']!s:5s} assets={row.get('n_assets', 0):4d} "
              f"search={row['search_ms']:5.1f}ms exposure={row.get('exposure_ms', 0):5.1f}ms "
              f"cyclone={row.get('any_cyclone')} sanity={row['sanity']}", flush=True)
    (OUT / "world_qa_companies.json").write_text(json.dumps({"register_load_s": round(load_s, 2), "rows": rows}, indent=1))


def _site(s):
    sid, name, lat, lon, typ, stress, check = s
    f = OUT / SITES_DIR / f"{sid}.json"
    if f.exists():
        return f"cached {sid}"
    from cri.analysis.asset_intelligence import assess_asset
    t = time.time()
    try:
        r = assess_asset(lat, lon, name, typ, include_visuals=True, sar_timeout_s=60)
        el = time.time() - t
        fnd = {x["hazard"]: x["severity"] for x in r["findings"]}
        v = r.get("visual") or {}
        out = {"id": sid, "name": name, "stress": stress, "check": check, "ok": True, "elapsed_s": round(el, 1),
               "risk_signal": r["risk_signal"], "headline": r["headline"], "findings": fnd,
               "errors": r.get("errors"), "scene_date": (r.get("satellite") or {}).get("scene_date"),
               "radar_dates": (r.get("radar") or {}).get("scenes_used"),
               "has_dem": bool(v.get("dem")), "has_texture": bool(v.get("texture_jpeg_b64")),
               "layers": sorted((v.get("layers") or {}).keys()),
               "payload_kb": round(len(json.dumps(r)) / 1024)}
        cyc = fnd.get("tropical_cyclone", "NEGLIGIBLE")
        if check == "no_cyclone":
            out["sanity"] = "ok" if cyc in ("NEGLIGIBLE", "LOW") else "CHECK"
        elif check == "cyclone":
            out["sanity"] = "ok" if cyc not in ("NEGLIGIBLE",) else "CHECK"
        elif check == "debris_high":
            out["sanity"] = "ok" if fnd.get("debris_flow") == "HIGH" else "CHECK"
        elif check == "flood_not_negligible":
            out["sanity"] = "ok" if fnd.get("flood") not in ("NEGLIGIBLE", "LOW") else "CHECK"
        else:
            out["sanity"] = "ok"
    except Exception as exc:                              # noqa: BLE001
        import traceback
        out = {"id": sid, "name": name, "ok": False, "error": repr(exc), "trace": traceback.format_exc()[-1500:],
               "elapsed_s": round(time.time() - t, 1)}
    f.write_text(json.dumps(out, indent=1))
    return f"{sid}: ok={out['ok']} {out.get('elapsed_s')}s {out.get('risk_signal', out.get('error', ''))}"


def run_sites():
    todo = [s for s in SITES if not (OUT / SITES_DIR / f"{s[0]}.json").exists()]
    batch = todo[:int(os.environ.get("QA_BATCH", "2"))]
    with ThreadPoolExecutor(max_workers=max(1, len(batch))) as pool:
        for m in pool.map(_site, batch):
            print(m, flush=True)
    print(f"sites done {len(SITES) - len(todo) + len(batch)}/{len(SITES)}")


def report():
    c = json.loads((OUT / "world_qa_companies.json").read_text())
    sites = [json.loads((OUT / SITES_DIR / f"{s[0]}.json").read_text())
             for s in SITES if (OUT / SITES_DIR / f"{s[0]}.json").exists()]
    print(json.dumps({"companies": c, "sites": sites}, indent=1)[:200])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["companies", "sites", "report"], required=True)
    a = ap.parse_args()
    {"companies": run_companies, "sites": run_sites, "report": report}[a.part]()
