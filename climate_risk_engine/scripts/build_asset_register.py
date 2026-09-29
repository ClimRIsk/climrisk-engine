"""
Build the global company → asset register shipped with the engine.

Sources (free, public)
  Climate TRACE v6 API     global heavy-industry assets with owners, location,
                           capacity and emissions — power, manufacturing (steel,
                           cement, aluminium, chemicals, pulp & paper…),
                           fossil-fuel operations, mineral extraction
  EPA GHGRP 2023           11k US facilities with parent companies and
                           ownership shares (cached by build_emissions_dataset.py)

Output (loaded into memory by cri.data.asset_register — lookups are instant)
  src/cri/data/asset_register.npz        per-asset arrays (lat, lon, sector, …)
  src/cri/data/asset_register_companies.json   company → asset indices + aliases

Stages (cached under data/asset_register/raw/)
  fetch   Climate TRACE pages, trimmed to the fields we use (repeat until done)
  build   merge + normalise owners + write the runtime files

Usage
  python scripts/build_asset_register.py --stage fetch   (repeat until "all sectors complete")
  python scripts/build_asset_register.py --stage build
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "asset_register" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
OUT_NPZ = ROOT / "src" / "cri" / "data" / "asset_register.npz"
OUT_JSON = ROOT / "src" / "cri" / "data" / "asset_register_companies.json"

CT = "https://api.climatetrace.org/v6/assets"
SECTORS = ["power", "fossil-fuel-operations", "mineral-extraction", "manufacturing"]
PAGE = 1000
WAVE = int(os.environ.get("CT_WAVE", "10"))

_SUFFIX = re.compile(
    r"\b(the|inc|incorporated|corp|corporation|co|company|companies|llc|l\.l\.c|lp|l\.p|ltd|limited|plc|"
    r"holdings?|group|sa|nv|ag|se|na|usa|us|trust|partners|pvt|private|pte|bhd|tbk|spa|s\.a|gmbh|kk|oyj|asa|ab)\b")


def norm(name: str) -> str:
    s = str(name).lower().replace("&", " and ")
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = _SUFFIX.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


# ── fetch ─────────────────────────────────────────────────────────────────────

def _trim(a: dict) -> dict:
    em = (a.get("EmissionsSummary") or [{}])
    e = next((x for x in em if x.get("Gas") == "co2e_100yr"), em[0] if em else {})
    c = a.get("Centroid") or {}
    geom = c.get("Geometry") if isinstance(c, dict) else None
    lon, lat = (geom or [None, None])[:2] if isinstance(geom, list) else (None, None)
    owners = []
    for o in a.get("Owners") or []:
        n = o.get("CompanyName")
        if n and n not in [x[1] for x in owners]:
            owners.append((o.get("CompanyId"), n))
    return {"id": a.get("Id"), "name": a.get("Name"), "country": a.get("Country"), "sector": a.get("Sector"),
            "type": a.get("AssetType"), "lat": lat, "lon": lon, "owners": owners,
            "capacity": e.get("Capacity"), "capacity_units": e.get("CapacityUnits"),
            "emissions_t": e.get("EmissionsQuantity")}


def _page(sector: str, i: int) -> int:
    f = RAW / f"ct_{sector}_{i:04d}.json"
    if f.exists():
        return len(json.loads(f.read_text()))
    url = f"{CT}?limit={PAGE}&offset={i * PAGE}&sectors={sector}"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "ClimRisk-Engine/1.0"}),
                                timeout=120) as r:
        assets = json.loads(r.read()).get("assets", [])
    f.write_text(json.dumps([_trim(a) for a in assets]))
    return len(assets)


def stage_fetch() -> None:
    done_file = RAW / "complete.json"
    done = json.loads(done_file.read_text()) if done_file.exists() else {}
    for sec in SECTORS:
        if done.get(sec):
            continue
        have = sorted(int(p.split("_")[-1][:4]) for p in glob.glob(str(RAW / f"ct_{sec}_*.json")))
        start = (max(have) + 1) if have else 0
        idx = list(range(start, start + WAVE))
        with ThreadPoolExecutor(WAVE) as pool:
            counts = list(pool.map(lambda i: _page(sec, i), idx))
        print(f"{sec}: pages {idx[0]}–{idx[-1]} sizes {counts}", flush=True)
        if any(c < PAGE for c in counts):
            done[sec] = True
            done_file.write_text(json.dumps(done))
        return                      # one wave per run keeps each call short
    print("all sectors complete")


# ── build ─────────────────────────────────────────────────────────────────────

def _ghgrp_assets() -> list[dict]:
    raw = ROOT / "data" / "training" / "raw"
    facs = []
    for f in glob.glob(str(raw / "epa_PUB_DIM_FACILITY_2023_*.json")):
        facs += json.loads(Path(f).read_text())
    em = defaultdict(float)
    for f in glob.glob(str(raw / "epa_PUB_FACTS_SECTOR_GHG_EMISSION_2023_*.json")):
        for e in json.loads(Path(f).read_text()):
            if e.get("gas_id") != 8:
                em[e["facility_id"]] += float(e.get("co2e_emission") or 0)
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_emissions_dataset import parse_parents
    out = []
    for f in facs:
        if f.get("latitude") is None:
            continue
        owners = [(None, n) for n, _ in parse_parents(f.get("parent_company"))]
        out.append({"id": f"ghgrp-{f['facility_id']}", "name": f.get("facility_name"), "country": "USA",
                    "sector": "ghgrp-" + str(f.get("naics_code") or "")[:3], "type": None,
                    "lat": f["latitude"], "lon": f["longitude"], "owners": owners,
                    "capacity": None, "capacity_units": None, "emissions_t": em.get(f["facility_id"])})
    return out


def stage_build() -> None:
    assets = []
    for f in sorted(glob.glob(str(RAW / "ct_*.json"))):
        assets += json.loads(Path(f).read_text())
    n_ct = len(assets)
    assets += _ghgrp_assets()
    assets = [a for a in assets if a["lat"] is not None and a["lon"] is not None and a["owners"]]

    # de-duplicate GHGRP vs Climate TRACE: same owner within ~2 km → keep Climate TRACE record
    ct_keys = {(norm(o[1]), round(a["lat"], 2), round(a["lon"], 2))
               for a in assets if not str(a["id"]).startswith("ghgrp") for o in a["owners"]}
    kept = [a for a in assets if not (str(a["id"]).startswith("ghgrp") and
            any((norm(o[1]), round(a["lat"], 2), round(a["lon"], 2)) in ct_keys for o in a["owners"]))]

    sectors = sorted({a["sector"] or "" for a in kept})
    countries = sorted({a["country"] or "" for a in kept})
    companies: dict[str, dict] = {}
    for i, a in enumerate(kept):
        for cid, name in a["owners"]:
            k = norm(name)
            if not k:
                continue
            c = companies.setdefault(k, {"name": name, "ids": set(), "aliases": set(), "assets": []})
            c["aliases"].add(name)
            if cid:
                c["ids"].add(cid)
            c["assets"].append(i)

    np.savez_compressed(
        OUT_NPZ,
        lat=np.array([a["lat"] for a in kept], np.float32),
        lon=np.array([a["lon"] for a in kept], np.float32),
        sector=np.array([sectors.index(a["sector"] or "") for a in kept], np.int16),
        country=np.array([countries.index(a["country"] or "") for a in kept], np.int16),
        emissions_t=np.array([a["emissions_t"] if a["emissions_t"] is not None else np.nan for a in kept], np.float32),
        capacity=np.array([a["capacity"] if a["capacity"] is not None else np.nan for a in kept], np.float32),
        name=np.array([str(a["name"] or "")[:80] for a in kept]),
        atype=np.array([str(a["type"] or "")[:40] for a in kept]),
        source=np.array(["EPA GHGRP 2023" if str(a["id"]).startswith("ghgrp") else "Climate TRACE v6" for a in kept]),
        sectors=np.array(sectors), countries=np.array(countries),
    )
    comp_out = {k: {"name": v["name"], "ids": sorted(v["ids"]), "aliases": sorted(v["aliases"])[:8],
                    "assets": v["assets"]} for k, v in companies.items()}
    OUT_JSON.write_text(json.dumps({"built": time.strftime("%Y-%m-%d"), "n_assets": len(kept),
                                    "n_companies": len(comp_out), "companies": comp_out}, separators=(",", ":")))
    print(f"Climate TRACE assets fetched: {n_ct}; register: {len(kept)} assets with owners, "
          f"{len(comp_out)} companies, {len(countries)} countries → "
          f"{OUT_NPZ.stat().st_size/1e6:.1f} MB + {OUT_JSON.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["fetch", "build"], required=True)
    a = ap.parse_args()
    t = time.time()
    {"fetch": stage_fetch, "build": stage_build}[a.stage]()
    print(f"({time.time() - t:.0f}s)")
