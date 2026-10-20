"""Named-event hindcast: Carlisle, Storm Desmond (5-7 December 2015), and the January 2005 flood, scored building by building.

Question a customer asks: before the event, would the engine have pointed at the buildings that flooded? Truth is the Environment Agency Recorded Flood Outline for the event
(buildings whose footprint touches it); buildings are every OpenStreetMap footprint in the 5.5 x 5.5 km core (12,855). Scores computed from terrain only, before looking at the outline:
  lidar_hand   height above nearest drainage from the EA 1 m LiDAR bare earth (the engine's flood-floor input), minimum within 12 m of the footprint centroid
  global_hand  the same method on the global 10-30 m tiles (what the engine used before LiDAR)
  global_rel   height above the lowest ground within 1 km, global tiles
Reported: ROC area; gain (share of flooded buildings found when the top 10/20/30/50 % of buildings by score are flagged); the engine's fixed bands (HAND < 1 m HIGH, < 5 m MODERATE,
< 10 m LOW) with hit and false-alarm rates and how many buildings each flags against how many actually flooded.
Not tested: depth or loss (the outline gives extent, not depth; no per-building claims are public). Terrain is the 2022 LiDAR, after the 2015 event and after the 2022 Carlisle flood-defence
scheme, so protected buildings that flooded in 2015 can read safe: that costs hits and is stated, not corrected.

  PYTHONPATH=.:src python scripts/hindcast_carlisle.py --town carlisle|cockermouth|york --stage buildings|terrain|score     (writes data/backtest/<town>_hindcast.json)
"""
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src")]
RAW = ROOT / "data" / "backtest" / "raw"
TOWNS = {   # centre lat, lon, half-width km (box), event name keys in the EA outline names
    "carlisle": (54.895, -2.9375, 3.1, {"Desmond_2015": "05_12_2015", "Jan_2005": "2005 Carlisle Event"}),
    "cockermouth": (54.664, -3.363, 2.6, {"Nov_2009": "November 2009"}),
    "york": (53.958, -1.083, 3.1, {"Dec_2015": "December 2015 Flood Event", "Storm_Dennis_2020": "Storm Dennis"}),
}
TOWN = sys.argv[sys.argv.index("--town") + 1] if "--town" in sys.argv else "carlisle"
LAT0, LON0, HALF_KM, EVENTS = TOWNS[TOWN]
OUT = ROOT / "data" / "backtest" / f"{TOWN}_hindcast.json"
WFS = "https://environment.data.gov.uk/spatialdata/recorded-flood-outlines/wfs"
TYPE = "dataset-8c75e700-d465-11e4-8b5b-f0def148f590:Recorded_Flood_Outlines"


def stage_buildings():
    import time
    from cri import hazardsurface as H
    f = RAW / f"{TOWN}_buildings.json"
    part = RAW / f"{TOWN}_buildings_partial.json"
    got = json.loads(part.read_text()) if part.exists() else {}
    dlat = HALF_KM / 111.32
    dlon = HALF_KM / (111.32 * np.cos(np.radians(LAT0)))
    t0 = time.time()
    for k in range(4):
        if str(k) in got or time.time() - t0 > 100:
            continue
        i, j = divmod(k, 2)
        s_, n_ = LAT0 - dlat + i * dlat, LAT0 - dlat + (i + 1) * dlat
        w_, e_ = LON0 - dlon + j * dlon, LON0 - dlon + (j + 1) * dlon
        q = f'[out:json][timeout:90];way["building"]({s_},{w_},{n_},{e_});out geom;'
        for url in H.OVERPASS:
            try:
                req = urllib.request.Request(url, data=urllib.parse.urlencode({"data": q}).encode(), headers={"User-Agent": "ClimRisk/1.0"})
                got[str(k)] = json.loads(urllib.request.urlopen(req, timeout=60).read())["elements"]
                print(TOWN, "box", k, len(got[str(k)]), flush=True)
                break
            except Exception as ex:                                  # noqa: BLE001
                print("fail", url[:30], str(ex)[:60], flush=True)
        part.write_text(json.dumps(got))
    if len(got) == 4:
        f.write_text(json.dumps(got))
        print("buildings complete", {k: len(v) for k, v in got.items()})


def stage_terrain():
    from cri.connectors import terrain_tiles as T
    from cri.terrain.lidar_flood import hand_and_rel
    for label, env in (("lidar", "1"), ("global", "0")):
        os.environ["CRI_LIDAR"] = env
        f = RAW / f"{TOWN}_hand_{label}.npz"
        if f.exists():
            continue
        g = T.get_elevation_grid(LAT0, LON0, radius_km=HALF_KM)
        if label == "lidar" and "Environment Agency" not in g.source:
            raise SystemExit("LiDAR not returned; try again later")
        rel, hand = hand_and_rel(g.elevations, g.cell_size_m)
        np.savez_compressed(f, hand=hand, rel=rel, res=g.cell_size_m, row=g.asset_row, col=g.asset_col, source=g.source)
        print(label, g.source[:60], g.elevations.shape, round(g.cell_size_m, 1), flush=True)


def outlines():
    from pyproj import Transformer
    from shapely.geometry import shape
    from shapely.ops import unary_union
    f = RAW / f"{TOWN}_outlines.json"
    if not f.exists():
        e, n = Transformer.from_crs(4326, 27700, always_xy=True).transform(LON0, LAT0)
        h = HALF_KM * 1000 + 1100
        q = {"service": "WFS", "version": "2.0.0", "request": "GetFeature", "typeNames": TYPE, "outputFormat": "application/json", "srsName": "EPSG:27700",
             "bbox": f"{e - h},{n - h},{e + h},{n + h},urn:ogc:def:crs:EPSG::27700", "count": 3000}
        f.write_bytes(urllib.request.urlopen(urllib.request.Request(WFS + "?" + urllib.parse.urlencode(q), headers={"User-Agent": "ClimRisk/1.0"}), timeout=90).read())
    d = json.loads(f.read_text())
    out = {}
    for label, key in EVENTS.items():
        gs = [shape(x["geometry"]) for x in d["features"] if x.get("geometry") and key in str(x["properties"].get("name"))]
        out[label] = unary_union(gs) if gs else None
    return out


def auc(score, y):
    pos, neg = score[y == 1], score[y == 0]
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1.0
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def gain(score, y, fracs=(0.1, 0.2, 0.3, 0.5)):
    order = np.argsort(score)                                  # lowest score = most exposed first
    cum = np.cumsum(y[order])
    return {f"top_{int(f * 100)}pct": round(float(cum[int(f * len(y)) - 1] / max(1, y.sum())), 3) for f in fracs}


def rule(flag, y):
    tp, fp, fn = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum()), int((~flag & (y == 1)).sum())
    return {"flagged": int(flag.sum()), "hit_rate": round(tp / max(1, tp + fn), 3), "false_alarm_rate": round(fp / max(1, int((y == 0).sum())), 3), "precision": round(tp / max(1, tp + fp), 3), "csi": round(tp / max(1, tp + fp + fn), 3)}


def stage_score():
    from pyproj import Transformer
    from shapely.geometry import Polygon
    from shapely.strtree import STRtree
    bl = json.loads((RAW / f"{TOWN}_buildings.json").read_text())
    els = [e for v in bl.values() for e in v if e.get("geometry") and len(e["geometry"]) >= 4]
    seen, rings, ids = set(), [], []
    for e in els:
        if e["id"] in seen:
            continue
        seen.add(e["id"])
        rings.append([(p["lon"], p["lat"]) for p in e["geometry"]])
        ids.append(e["id"])
    tf = Transformer.from_crs(4326, 27700, always_xy=True)
    polys = []
    for r in rings:
        xs, ys = tf.transform([p[0] for p in r], [p[1] for p in r])
        p = Polygon(zip(xs, ys))
        polys.append(p if p.is_valid else p.buffer(0))
    cen = np.array([[p.centroid.x, p.centroid.y] for p in polys])
    ev = outlines()
    e0, n0 = tf.transform(LON0, LAT0)
    res = {"n_buildings": len(polys), "events": {}, "scores": {}}
    scores = {}
    for label in ("lidar", "global"):
        z = np.load(RAW / f"{TOWN}_hand_{label}.npz", allow_pickle=True)
        hand, rel, cell = z["hand"], z["rel"], float(z["res"])
        row0, col0 = int(z["row"]), int(z["col"])
        if label == "lidar":                                   # projected grid, north-up in British National Grid
            c = (cen[:, 0] - e0) / cell + col0
            r = -(cen[:, 1] - n0) / cell + row0
        else:                                                  # Mercator ground metres from the pin
            lo, la = Transformer.from_crs(27700, 4326, always_xy=True).transform(cen[:, 0], cen[:, 1])
            c = (lo - LON0) * 111320.0 * np.cos(np.radians(LAT0)) / cell + col0
            r = -(la - LAT0) * 111320.0 / cell + row0
        ri, ci = np.round(r).astype(int), np.round(c).astype(int)
        k = max(1, int(12 / cell))
        vals = {"hand": np.full(len(polys), np.nan), "rel": np.full(len(polys), np.nan)}
        for i in range(len(polys)):
            if 0 <= ri[i] < hand.shape[0] and 0 <= ci[i] < hand.shape[1]:
                for nm, arr in (("hand", hand), ("rel", rel)):
                    w = arr[max(0, ri[i] - k): ri[i] + k + 1, max(0, ci[i] - k): ci[i] + k + 1]
                    if np.isfinite(w).any():
                        vals[nm][i] = np.nanmin(w)
        scores[f"{label}_hand"] = vals["hand"]
        if label == "global":
            scores["global_rel"] = vals["rel"]
    res["scores_covered"] = {k: int(np.isfinite(v).sum()) for k, v in scores.items()}
    tree = STRtree(polys)
    for label, geom in ev.items():
        if geom is None:
            continue
        y = np.zeros(len(polys), int)
        for i in tree.query(geom, predicate="intersects"):
            y[i] = 1
        ok = np.ones(len(polys), bool)
        for v in scores.values():
            ok &= np.isfinite(v)
        res["events"][label] = {"flooded_buildings": int(y[ok].sum()), "buildings_scored": int(ok.sum()), "share_flooded": round(float(y[ok].mean()), 3), "metrics": {}}
        for nm, v in scores.items():
            vv, yy = v[ok], y[ok]
            m = {"auc": round(auc(-vv, yy), 3), "gain": gain(vv, yy)}
            if nm.endswith("hand"):
                m["rules"] = {f"HAND<{t:g}m": rule(vv < t, yy) for t in (1.0, 5.0, 10.0)}
            res["events"][label]["metrics"][nm] = m
        # spatial block bootstrap (500 m blocks, 1000 resamples): neighbouring buildings flood together, so a plain bootstrap would be too confident
        blk = (np.floor(cen[ok, 0] / 500).astype(int) * 100000 + np.floor(cen[ok, 1] / 500).astype(int))
        ub = np.unique(blk)
        idx_by = {b: np.nonzero(blk == b)[0] for b in ub}
        rng = np.random.default_rng(7)
        yy = y[ok]
        sc = {nm: v[ok] for nm, v in scores.items()}
        boots = {"auc_lidar": [], "auc_global": [], "auc_diff": [], "top20_lidar": []}
        for _ in range(1000):
            pick = rng.choice(ub, len(ub))
            ii = np.concatenate([idx_by[b] for b in pick])
            if yy[ii].sum() < 10:
                continue
            a1, a2 = auc(-sc["lidar_hand"][ii], yy[ii]), auc(-sc["global_hand"][ii], yy[ii])
            boots["auc_lidar"].append(a1); boots["auc_global"].append(a2); boots["auc_diff"].append(a1 - a2)
            boots["top20_lidar"].append(gain(sc["lidar_hand"][ii], yy[ii], (0.2,))["top_20pct"])
        res["events"][label]["block_bootstrap_95ci"] = {k: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)] for k, v in boots.items()}
        res["events"][label]["random_flagging_gain"] = "top 10/20/30/50 % flagged finds 10/20/30/50 % of flooded buildings"
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ('block_bootstrap_95ci',)} for k, v in res['events'].items()}, indent=1))


if __name__ == "__main__":
    st = sys.argv[sys.argv.index("--stage") + 1]
    {"buildings": stage_buildings, "terrain": stage_terrain, "score": stage_score}[st]()
