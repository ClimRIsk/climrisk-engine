"""
Area-based flood model backtest: share of BUILT-UP land near drainage (HAND)
per census tract vs FEMA NFIP claims per built-up km², across several counties.

Why: scripts/backtest_flood_nfip.py showed the original flood score and HAND at
the tract centroid have no skill (AUC ≈ 0.44–0.50) — one point cannot stand in
for a 2–50 km² tract. Here every 10 m cell of built-up land inside the tract
polygon is used.

Counties (different flood regimes)
  48201 Harris County, TX         pluvial / bayou (Harvey 2017, Imelda 2019)
  22033 East Baton Rouge, LA      riverine + pluvial (Aug 2016)
  45019 Charleston County, SC     coastal / tidal / surge

Stages (cached per tract under data/backtest/raw/area/)
  sample   stratified: 12 tracts from each claims-per-km² quintile per county
  features HAND grids + WorldCover built-up + tract polygon → area features
  evaluate within-county Spearman, and leave-one-county-out model transfer

Usage
  python scripts/backtest_flood_area.py --stage sample
  FEAT_BATCH=3 python scripts/backtest_flood_area.py --stage features   (repeat)
  python scripts/backtest_flood_area.py --stage evaluate
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import sys
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import backtest_flood_nfip as nf    # noqa: E402  (claims + gazetteer helpers)

OUT = ROOT / "data" / "backtest"
RAW = OUT / "raw" / "area"
RAW.mkdir(parents=True, exist_ok=True)
COUNTIES = {"48201": "Harris TX (pluvial)", "22033": "East Baton Rouge LA (riverine)",
            "45019": "Charleston SC (coastal)"}
PER_QUINTILE = int(os.environ.get("PER_QUINTILE", "12"))
CB = "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_{st}_tract_500k.zip"


# ── claims → 2020 tracts (incl. 2010 split-tract reassignment) ────────────────

def tract_claims(county: str) -> dict[str, float]:
    claims = json.loads(nf.stage_claims(county).read_text())
    ids20 = {t["geoid"] for t in json.loads(nf.stage_tracts(county).read_text())}
    n = Counter()
    for g, c in claims["claims"].items():
        if g in ids20:
            n[g] += c
            continue
        kids = [k for k in ids20 if k[:9] == g[:9]]
        for k in kids:
            n[k] += c / len(kids)
    return dict(n)


# ── polygons ─────────────────────────────────────────────────────────────────

def polygons(state: str) -> dict[str, dict]:
    cache = RAW / f"tracts_poly_{state}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    import shapefile
    zf = RAW / f"cb_2020_{state}_tract_500k.zip"
    if not zf.exists():
        zf.write_bytes(nf._get(CB.format(st=state), timeout=120))
    with zipfile.ZipFile(zf) as z:
        base = [n[:-4] for n in z.namelist() if n.endswith(".shp")][0]
        r = shapefile.Reader(shp=io.BytesIO(z.read(base + ".shp")), dbf=io.BytesIO(z.read(base + ".dbf")),
                             shx=io.BytesIO(z.read(base + ".shx")))
        fields = [f[0] for f in r.fields[1:]]
        out = {}
        for sr in r.iterShapeRecords():
            rec = dict(zip(fields, sr.record))
            out[rec["GEOID"]] = {"bbox": list(sr.shape.bbox), "geom": sr.shape.__geo_interface__}
    cache.write_text(json.dumps(out))
    return out


# ── stage: stratified sample ─────────────────────────────────────────────────

def stage_sample() -> Path:
    out = RAW / "sample.json"
    if out.exists():
        return out
    rng = random.Random(42)
    sample = []
    for county in COUNTIES:
        tracts = [t for t in json.loads(nf.stage_tracts(county).read_text()) if t["aland_km2"] >= 0.2]
        cl = tract_claims(county)
        for t in tracts:
            t["claims"] = round(cl.get(t["geoid"], 0.0), 2)
            t["claims_per_km2"] = t["claims"] / t["aland_km2"]
        tracts.sort(key=lambda t: t["claims_per_km2"])
        q = len(tracts) / 5
        for i in range(5):
            chunk = tracts[int(i * q):int((i + 1) * q)]
            for t in rng.sample(chunk, min(PER_QUINTILE, len(chunk))):
                sample.append({**t, "county": county, "quintile": i + 1})
        print(f"{county} {COUNTIES[county]}: {len(tracts)} tracts, {sum(cl.values()):.0f} claims")
    out.write_text(json.dumps(sample))
    return out


# ── stage: features ──────────────────────────────────────────────────────────

def _features_one(t: dict, polys: dict) -> dict:
    from rasterio.features import rasterize
    from cri.terrain.hand import area_flood_features, builtup_on_grid, hand_grid
    p = polys[t["geoid"]]
    minx, miny, maxx, maxy = p["bbox"]
    clat, clon = (miny + maxy) / 2, (minx + maxx) / 2
    half_km = max((maxy - miny) * 111.32, (maxx - minx) * 111.32 * math.cos(math.radians(clat))) / 2
    radius = min(max(half_km + 1.5, 2.5), 10.0)       # 1.5 km buffer for upstream drainage
    hg = hand_grid(clat, clon, radius_km=radius)
    region = rasterize([(p["geom"], 1)], out_shape=hg.dem.shape, transform=hg.transform,
                       fill=0, dtype="uint8").astype(bool)
    built = builtup_on_grid(hg)
    f = area_flood_features(hg, region, built)
    f.update({"geoid": t["geoid"], "window_radius_km": round(radius, 2),
              "truncated": half_km + 1.5 > 10.0,
              "median_elev_built_m": round(float(np.nanmedian(hg.dem[region & built])), 2)
              if built is not None and (region & built).any() else None})
    return f


def stage_features() -> None:
    sample = json.loads(stage_sample().read_text())
    fdir = RAW / "features"
    fdir.mkdir(exist_ok=True)
    polys = {}
    for st in {t["county"][:2] for t in sample}:
        polys.update(polygons(st))
    todo = [t for t in sample if not (fdir / f"{t['geoid']}.json").exists()]
    work = todo[:int(os.environ.get("FEAT_BATCH", "3"))]

    def run(t):
        try:
            f = _features_one(t, polys)
            (fdir / f"{t['geoid']}.json").write_text(json.dumps(f))
            return f"ok {t['geoid']} built={f.get('builtup_km2')}km2 r={f['window_radius_km']}"
        except Exception as exc:     # noqa: BLE001
            return f"ERR {t['geoid']}: {exc}"

    with ThreadPoolExecutor(max_workers=min(3, max(1, len(work)))) as pool:
        for m in pool.map(run, work):
            print(m, flush=True)
    print(f"features {len(sample) - len(todo) + len(work)}/{len(sample)}")


# ── stage: evaluate ──────────────────────────────────────────────────────────

FEATURES = [
    "share_built_hand_lt_0.5m_at_0.1km2", "share_built_hand_lt_1.0m_at_0.1km2",
    "share_built_hand_lt_2.0m_at_0.1km2", "share_built_hand_lt_5.0m_at_0.1km2",
    "share_built_hand_lt_0.5m_at_1.0km2", "share_built_hand_lt_1.0m_at_1.0km2",
    "share_built_hand_lt_2.0m_at_1.0km2", "share_built_hand_lt_5.0m_at_1.0km2",
    "neg_median_built_hand_at_0.1km2", "neg_median_built_hand_at_1.0km2",
    "neg_median_elev_built_m",
]


def load_rows() -> list[dict]:
    sample = json.loads(stage_sample().read_text())
    rows = []
    for t in sample:
        f = RAW / "features" / f"{t['geoid']}.json"
        if not f.exists():
            continue
        r = {**t, **json.loads(f.read_text())}
        for k in ("median_built_hand_at_0.1km2", "median_built_hand_at_1.0km2", "median_elev_built_m"):
            r["neg_" + k] = -r[k] if r.get(k) is not None else None
        bu = r.get("builtup_km2") or 0
        r["rate"] = r["claims"] / bu if bu >= 0.1 else None          # claims per built-up km²
        rows.append(r)
    return rows


def stage_evaluate() -> Path:
    rows = [r for r in load_rows() if r["rate"] is not None]
    res = {"n_tracts": len(rows), "by_county": {}, "loco": {}}
    for c, name in COUNTIES.items():
        rc = [r for r in rows if r["county"] == c]
        top = [r for r in rc if r["quintile"] == 5]
        bot = [r for r in rc if r["quintile"] == 1]
        feats = {}
        for k in FEATURES:
            v = [(r[k], math.log1p(r["rate"])) for r in rc if r.get(k) is not None]
            if len(v) < 8:
                continue
            feats[k] = {"spearman": round(nf._spearman([a for a, _ in v], [b for _, b in v]), 3),
                        "auc_q5_vs_q1": round(nf._auc([r[k] for r in top if r.get(k) is not None],
                                                      [r[k] for r in bot if r.get(k) is not None]), 3),
                        "n": len(v)}
        res["by_county"][c] = {"name": name, "n": len(rc), "features": feats}

    # Leave-one-county-out: fit on two counties, rank the third.
    # Target standardised within county (claims also reflect county insurance take-up).
    use = [k for k in FEATURES if all(r.get(k) is not None for r in rows)]
    for held in COUNTIES:
        tr = [r for r in rows if r["county"] != held]
        te = [r for r in rows if r["county"] == held]
        if len(te) < 8 or len(tr) < 16:
            continue

        def zt(rs):
            out = []
            for c in {r["county"] for r in rs}:
                v = np.array([math.log1p(r["rate"]) for r in rs if r["county"] == c])
                mu, sd = v.mean(), v.std() or 1
                out += [(r, (math.log1p(r["rate"]) - mu) / sd) for r in rs if r["county"] == c]
            return out
        trz = zt(tr)
        X = np.array([[r[k] for k in use] for r, _ in trz])
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Xs = (X - mu) / sd
        y = np.array([z for _, z in trz])
        lam = 1.0                                          # ridge — few samples, correlated features
        w = np.linalg.solve(Xs.T @ Xs + lam * np.eye(len(use)), Xs.T @ y)
        Xt = (np.array([[r[k] for k in use] for r in te]) - mu) / sd
        pred = Xt @ w
        yt = [math.log1p(r["rate"]) for r in te]
        top = [p for p, r in zip(pred, te) if r["quintile"] == 5]
        bot = [p for p, r in zip(pred, te) if r["quintile"] == 1]
        res["loco"][held] = {"spearman_heldout": round(nf._spearman(list(pred), yt), 3),
                             "auc_q5_vs_q1_heldout": round(nf._auc(top, bot), 3),
                             "weights": {k: round(float(v), 3) for k, v in zip(use, w)}}
    out = OUT / "flood_area_backtest.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sample", "features", "evaluate"], required=True)
    a = ap.parse_args()
    t0 = time.time()
    {"sample": stage_sample, "features": stage_features, "evaluate": stage_evaluate}[a.stage]()
    print(f"{a.stage} done in {time.time() - t0:.0f}s")
