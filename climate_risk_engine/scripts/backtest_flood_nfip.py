"""
Backtest the asset-intelligence FLOOD score against real insured flood losses.

Ground truth
────────────
FEMA NFIP redacted claims (OpenFEMA v3 `NfipClaims`). Claim coordinates are
rounded to 0.1° (~11 km) for privacy, so the test runs at census-tract level
(`censusGeoid`), not at individual buildings.

Design
──────
1. Count NFIP claims per tract for a county since YEAR_FROM.
2. Tract centroid + land area from the Census 2020 Gazetteer.
3. Exposure normalisation: claims per km² of BUILT-UP land
   (tract land area × ESA WorldCover built-up share around the centroid) —
   the Census population API now requires a personal key.
4. Sample the top and bottom quintiles of claim density, score each tract
   centroid with analysis.asset_intelligence.assess_asset (flood finding),
   and measure discrimination: ROC AUC (high vs low group) and Spearman rank
   correlation with normalised claim rate.
5. Store every component metric so a data-fitted flood score can be learned
   and compared out-of-sample.

Caveats recorded in the output
──────────────────────────────
• Tract-level: a tract score at the centroid stands in for all its buildings.
• NFIP claims reflect insurance take-up (mandatory inside FEMA SFHA zones)
  as well as hazard, so they are a biased — but real — loss signal.

Usage
─────
  python scripts/backtest_flood_nfip.py --county 48201 --stage claims|tracts|sample|score|evaluate
  SCORE_BATCH=4 limits how many new tracts are scored per run (cached per tract).
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "data" / "backtest"
RAW = OUT / "raw"
RAW.mkdir(parents=True, exist_ok=True)

FEMA = "https://www.fema.gov/api/open/v3/NfipClaims"
GAZ = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2020_Gazetteer/2020_Gaz_tracts_national.zip"
YEAR_FROM = 2010
N_PER_GROUP = int(os.environ.get("N_PER_GROUP", "30"))


def _get(url: str, timeout: int = 90) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "ClimRisk-Engine/1.0 research"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


# ── 1. claims per tract ───────────────────────────────────────────────────────

def stage_claims(county: str) -> Path:
    out = RAW / f"claims_{county}.json"
    if out.exists():
        return out
    counts, paid = Counter(), Counter()
    skip, page = 0, 10000
    flt = f"countyCode eq '{county}' and yearOfLoss ge {YEAR_FROM}"
    while True:
        q = urllib.parse.urlencode({"$filter": flt, "$top": page, "$skip": skip,
                                    "$select": "censusGeoid,netBuildingPaymentAmount,netContentsPaymentAmount"})
        rows = json.loads(_get(f"{FEMA}?{q}")).get("NfipClaims", [])
        for r in rows:
            g = str(r.get("censusGeoid") or "")[:11]          # tract GEOID = first 11 digits
            if len(g) == 11:
                counts[g] += 1
                paid[g] += float(r.get("netBuildingPaymentAmount") or 0) + float(r.get("netContentsPaymentAmount") or 0)
        if len(rows) < page:
            break
        skip += page
    out.write_text(json.dumps({"county": county, "year_from": YEAR_FROM, "n_claims": sum(counts.values()),
                               "claims": counts, "paid_usd": paid}))
    return out


# ── 2. tract centroids / land area ───────────────────────────────────────────

def stage_tracts(county: str) -> Path:
    out = RAW / f"tracts_{county}.json"
    if out.exists():
        return out
    gz = RAW / "gazetteer_tracts_2020.zip"
    if not gz.exists():
        gz.write_bytes(_get(GAZ, timeout=120))
    with zipfile.ZipFile(gz) as z:
        name = [n for n in z.namelist() if n.endswith(".txt")][0]
        lines = io.TextIOWrapper(z.open(name), encoding="latin-1").read().splitlines()
    hdr = [h.strip() for h in lines[0].split("\t")]
    tracts = []
    for ln in lines[1:]:
        f = dict(zip(hdr, [x.strip() for x in ln.split("\t")]))
        if f["GEOID"].startswith(county):
            tracts.append({"geoid": f["GEOID"], "lat": float(f["INTPTLAT"]), "lon": float(f["INTPTLONG"]),
                           "aland_km2": float(f["ALAND"]) / 1e6})
    out.write_text(json.dumps(tracts))
    return out


# ── 3. stratified sample ──────────────────────────────────────────────────────

def stage_sample(county: str) -> Path:
    out = RAW / f"sample_{county}.json"
    if out.exists():
        return out
    claims = json.loads(stage_claims(county).read_text())
    tracts = json.loads(stage_tracts(county).read_text())

    # Claims carry a mix of 2010- and 2020-vintage tract IDs. 2010 tracts that were
    # split for 2020 keep their first 4 tract digits (e.g. 432100 → 432101, 432102),
    # so an unmatched 2010 ID's claims are shared equally across its 2020 children.
    ids20 = {t["geoid"] for t in tracts}
    n_cl, paid = Counter(), Counter()
    reassigned = unassigned = 0
    for g, n in claims["claims"].items():
        p = claims["paid_usd"].get(g, 0.0)
        if g in ids20:
            n_cl[g] += n; paid[g] += p
            continue
        kids = [k for k in ids20 if k[:9] == g[:9]]
        if kids:
            for k in kids:
                n_cl[k] += n / len(kids); paid[k] += p / len(kids)
            reassigned += n
        else:
            unassigned += n

    rows = []
    for t in tracts:
        if t["aland_km2"] < 0.2:
            continue
        n = n_cl.get(t["geoid"], 0)
        rows.append({**t, "claims": round(n, 2), "paid_usd": round(paid.get(t["geoid"], 0.0)),
                     "claims_per_km2": n / t["aland_km2"]})
    matched = sum(1 for r in rows if r["claims"] > 0)
    print(f"claims on 2020 IDs + split-tract reassignment: {reassigned} reassigned, {unassigned} unassigned")
    rows.sort(key=lambda r: r["claims_per_km2"])
    q = max(1, len(rows) // 5)
    low, high = rows[:q], rows[-q:]
    step_l, step_h = max(1, len(low) // N_PER_GROUP), max(1, len(high) // N_PER_GROUP)
    sample = ([{**r, "group": "low"} for r in low[::step_l][:N_PER_GROUP]] +
              [{**r, "group": "high"} for r in high[::step_h][:N_PER_GROUP]])
    out.write_text(json.dumps({"county": county, "n_tracts": len(rows), "tracts_with_claims": matched,
                               "n_claims_total": claims["n_claims"], "sample": sample}))
    return out


# ── 4. score tract centroids ─────────────────────────────────────────────────

def _score_one(t: dict) -> dict:
    from cri.analysis.asset_intelligence import assess_asset
    r = assess_asset(t["lat"], t["lon"], asset_name=t["geoid"], asset_type="residential",
                     radius_km=2.0, include_visuals=False, sar_timeout_s=60)
    fl = next(f for f in r["findings"] if f["hazard"] == "flood")
    ter, sat, lc, sar = r.get("terrain") or {}, r.get("satellite") or {}, r.get("land_cover") or {}, r.get("radar") or {}
    return {
        "geoid": t["geoid"], "flood_score": fl["score"], "flood_severity": fl["severity"],
        "mechanisms": fl["mechanisms"],
        "elevation_m": ter.get("elevation_m"), "tpi_m": ter.get("tpi_m"),
        "terrain_flood": ter.get("flood_exposure"), "upslope_km2": ter.get("upslope_area_km2"),
        "nearest_water_m": sat.get("nearest_water_m"),
        "opt_ephemeral_500m": sat.get("ephemeral_water_within_500m"),
        "sar_available": sar.get("available", False),
        "sar_intermittent_500m": sar.get("intermittent_water_within_500m"),
        "sar_ever_wet_500m": sar.get("ever_wet_within_500m"),
        "builtup_1km": (lc.get("within_1km") or {}).get("built_up"),
        "wetland_1km": lc.get("wetland_within_1km"),
        "errors": r.get("errors"), "elapsed_s": r.get("elapsed_s"),
    }


def stage_score(county: str) -> Path:
    sdir = RAW / f"scores_{county}"
    sdir.mkdir(exist_ok=True)
    sample = json.loads(stage_sample(county).read_text())["sample"]
    todo = [t for t in sample if not (sdir / f"{t['geoid']}.json").exists()]
    batch = int(os.environ.get("SCORE_BATCH", "4"))
    work = todo[:batch]

    def run(t):
        try:
            res = _score_one(t)
            (sdir / f"{t['geoid']}.json").write_text(json.dumps(res))
            return f"ok {t['geoid']} {res['flood_severity']} ({res['elapsed_s']}s)"
        except Exception as exc:        # noqa: BLE001
            return f"ERR {t['geoid']}: {exc}"

    with ThreadPoolExecutor(max_workers=min(6, max(1, len(work)))) as pool:
        for msg in pool.map(run, work):
            print(msg, flush=True)
    print(f"scored {len(sample) - len(todo) + len(work)}/{len(sample)}")
    return sdir


# ── 5. evaluate ───────────────────────────────────────────────────────────────

def _auc(pos: list[float], neg: list[float]) -> float:
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg)) if pos and neg else float("nan")


def _spearman(x: list[float], y: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return cov / math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry) or 1)


def stage_evaluate(county: str) -> Path:
    meta = json.loads(stage_sample(county).read_text())
    sdir = RAW / f"scores_{county}"
    rows = []
    for t in meta["sample"]:
        f = sdir / f"{t['geoid']}.json"
        if f.exists():
            s = json.loads(f.read_text())
            bu = s.get("builtup_1km")
            rate = t["claims"] / (t["aland_km2"] * bu) if bu and bu > 0.02 else None
            rows.append({**t, **s, "claims_per_builtup_km2": rate})
    hi = [r for r in rows if r["group"] == "high"]
    lo = [r for r in rows if r["group"] == "low"]

    def comp(key, sign=1):
        vals = [(r[key], r) for r in rows if r.get(key) is not None]
        p = [sign * v for v, r in vals if r["group"] == "high"]
        n = [sign * v for v, r in vals if r["group"] == "low"]
        rr = [(sign * v, r["claims_per_builtup_km2"]) for v, r in vals if r.get("claims_per_builtup_km2") is not None]
        return {"auc_high_vs_low": round(_auc(p, n), 3),
                "spearman_vs_claims_per_builtup_km2": round(_spearman([a for a, _ in rr], [b for _, b in rr]), 3) if len(rr) > 5 else None,
                "n": len(vals)}

    res = {
        "county": county, "n_scored": len(rows), "n_high": len(hi), "n_low": len(lo),
        "n_tracts_county": meta["n_tracts"], "n_claims_county": meta["n_claims_total"],
        "flood_score": comp("flood_score"),
        "components": {
            "low_elevation (−elev)": comp("elevation_m", -1),
            "low_TPI (−tpi)": comp("tpi_m", -1),
            "upslope_area": comp("upslope_km2"),
            "near_water (−dist)": comp("nearest_water_m", -1),
            "optical_ephemeral_500m": comp("opt_ephemeral_500m"),
            "radar_intermittent_500m": comp("sar_intermittent_500m"),
            "radar_ever_wet_500m": comp("sar_ever_wet_500m"),
            "wetland_1km": comp("wetland_1km"),
        },
        "severity_by_group": {g: Counter(r["flood_severity"] for r in rows if r["group"] == g) for g in ("high", "low")},
        "caveats": ["Tract-level test: centroid score stands in for all buildings in the tract.",
                    "NFIP claims also reflect insurance take-up (mandatory in FEMA SFHA zones).",
                    "Single county — results are not yet evidence of global skill."],
    }
    out = OUT / f"flood_backtest_{county}.json"
    out.write_text(json.dumps(res, indent=1, default=dict))
    print(json.dumps(res, indent=1, default=dict))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--county", default="48201")
    ap.add_argument("--stage", choices=["claims", "tracts", "sample", "score", "evaluate"], required=True)
    a = ap.parse_args()
    fn = {"claims": stage_claims, "tracts": stage_tracts, "sample": stage_sample,
          "score": stage_score, "evaluate": stage_evaluate}[a.stage]
    t0 = time.time()
    p = fn(a.county)
    print(f"{a.stage}: {p} ({time.time() - t0:.0f}s)")
