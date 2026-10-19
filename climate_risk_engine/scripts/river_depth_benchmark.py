"""Does the HAND + rating-curve river depth (finance/river_depth.py) place the flood edge where real floods went? Scored on the 12 flooded English towns.

Same sampling as scripts/lidar_flood_benchmark.py: points inside Environment Agency Recorded Flood Outlines (fluvial) versus points >= 100 m outside any outline in
the same 5 km box, on the 1 m LiDAR DEM. Because the model stage is one number per reach, its RANK is the HAND rank already benchmarked (AUC 0.96); what this adds
is the question the rank cannot answer: is the wet/dry EDGE in the right place? Reported per town and pooled:
  hit rate (share of in-outline points the rule calls wet), false-alarm rate (share of outside points called wet), critical success index CSI = hits / (hits+misses+false alarms)
for: the model's 1-in-100 depth > 0 ; fixed rules HAND < 1 m, HAND < 5 m ; and the model with stage x 2 (what the flow would need to be to reach the edge).
Pins are the town centre point listed; recorded outlines mix events of different size and age.

  PYTHONPATH=.:src python scripts/river_depth_benchmark.py --budget 120   (resumable; data/backtest/river_depth_benchmark.json)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import lidar_flood_benchmark as L          # noqa: E402

OUT = ROOT / "data" / "backtest" / "river_depth_benchmark.json"


def rates(wet, y):
    hit, fa = float(wet[y == 1].mean()), float(wet[y == 0].mean())
    tp, fp, fn = int(wet[y == 1].sum()), int(wet[y == 0].sum()), int((~wet[y == 1]).sum())
    return {"hit": round(hit, 3), "false_alarm": round(fa, 3), "csi": round(tp / max(1, tp + fp + fn), 3), "tp": tp, "fp": fp, "fn": fn}


def one(name, lat, lon, rng, n_pts=150):
    from pyproj import Transformer
    from shapely.geometry import Point
    from cri.connectors import terrain_tiles as T
    from cri.finance.river_depth import site_river_depth
    to_en = Transformer.from_crs(4326, 27700, always_xy=True)
    e0, n0 = to_en.transform(lon, lat)
    poly, nfeat = L.outlines(e0, n0)
    if poly is None or poly.area < 5e4:
        return {"skipped": "no recorded fluvial outline of useful size"}
    box = Point(e0, n0).buffer(L.HALF * 0.85, cap_style=3)
    pos, neg = poly.intersection(box), box.difference(poly.simplify(3.0).buffer(100))
    if pos.is_empty or neg.is_empty:
        return {"skipped": "empty sample zone"}

    def sample(zone, k):
        out, (x0, y0, x1, y1), tries = [], zone.bounds, 0
        while len(out) < k and tries < 40000:
            tries += 1
            p = Point(rng.uniform(x0, x1), rng.uniform(y0, y1))
            if zone.contains(p):
                out.append((p.x, p.y))
        return out
    P, N = sample(pos, n_pts), sample(neg, n_pts)
    pts, y = np.array(P + N), np.array([1] * len(P) + [0] * len(N))
    g = T.get_elevation_grid(lat, lon, radius_km=L.HALF / 1000.0)
    if "Environment Agency" not in g.source:
        return {"skipped": "LiDAR not returned"}
    r = site_river_depth(lat, lon, rp=100, radius_km=L.HALF / 1000.0, dem=g, keep_grids=True)
    if not r.get("available"):
        return {"skipped": f"model unavailable: {r.get('reason')}"}
    hand, stage = r["_hand"], r["_stage"]
    res = g.cell_size_m
    cc, rr = (pts[:, 0] - e0) / res + g.asset_col, -(pts[:, 1] - n0) / res + g.asset_row
    ri, ci = np.round(rr).astype(int), np.round(cc).astype(int)
    ok = (ri >= 0) & (ri < hand.shape[0]) & (ci >= 0) & (ci < hand.shape[1])
    v = np.full(len(pts), np.nan)
    v[ok] = hand[ri[ok], ci[ok]]
    m = np.isfinite(v)
    h, yy = v[m], y[m]
    out = {"n_pos": int(yy.sum()), "n_neg": int((1 - yy).sum()), "q100_m3s": r["q_rp_m3s"], "q_bankfull_m3s": r["q_bankfull_m3s"], "stage_m": r["stage_m"],
           "glofas_snap_km": r["glofas_snap_km"], "model_rp100": rates(h < stage, yy), "stage_x2": rates(h < 2 * stage, yy), "hand_lt_1m": rates(h < 1.0, yy),
           "hand_lt_5m": rates(h < 5.0, yy)}
    out["_points"] = {"hand": [round(float(a), 2) for a in h], "y": [int(b) for b in yy]}
    return out


def main():
    budget = float(sys.argv[sys.argv.index("--budget") + 1]) if "--budget" in sys.argv else 1e9
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(OUT.read_text()) if OUT.exists() and "--fresh" not in sys.argv else {}
    t0 = time.time()
    for i, (name, (la, lo)) in enumerate(L.TOWNS.items()):
        if name in done or time.time() - t0 > budget:
            continue
        try:
            done[name] = one(name, la, lo, np.random.default_rng(300 + i))
        except Exception as e:                                  # noqa: BLE001
            done[name] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
        OUT.write_text(json.dumps(done, indent=1))
        print(name, {k: v for k, v in done[name].items() if k != "_points"}, round(time.time() - t0), flush=True)
    print("remaining:", [n for n in L.TOWNS if n not in done])


if __name__ == "__main__":
    main()
