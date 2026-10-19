"""Does the surface-water (rain-on-grid) model rank real surface-water flooding better than simple terrain scores? Scored on England's recorded outlines.

Positives: random points inside Environment Agency Recorded Flood Outlines whose cause is 'local drainage/surface water' (small polygons, < 1 km2; the very large
regional event polygons are excluded). Negatives: random points 100 m to 1 km outside any recorded outline of the same box, so both classes sit in the same
neighbourhood (a rank test must not be won just by telling town from open country). Scores computed on the SAME points:
  model_lidar   - maximum depth from `finance.pluvial.site_pluvial` on the 1 m EA LiDAR (resampled ~12 m), 1-in-50 design storm
  model_global  - the same model on the global 10-30 m tiles
  pit_depth     - blue-spot baseline: depth of the depression (fill-sinks minus ground) on the LiDAR grid
  low_position  - height above the lowest ground within about 500 m (LiDAR grid)
Metric: ROC area under curve per town and pooled (0.5 = no skill). Nothing fitted. Recorded outlines are where floods were mapped, not everywhere; they are
sparse, so the sample is small; terrain is modern while floods are historic; drainage is not in the terrain.

  PYTHONPATH=.:src python scripts/pluvial_benchmark.py --budget 140   (resumable; data/backtest/pluvial_benchmark.json)
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "backtest" / "pluvial_benchmark.json"
TOWNS = {"Oxford": (51.752, -1.258), "Reading": (51.455, -0.97), "Bristol": (51.45, -2.59), "Leeds": (53.80, -1.55), "Sheffield": (53.381, -1.470),
         "Doncaster": (53.52, -1.13), "Bradford": (53.79, -1.75), "Hull": (53.745, -0.336)}
WFS = "https://environment.data.gov.uk/spatialdata/recorded-flood-outlines/wfs"
TYPE = "dataset-8c75e700-d465-11e4-8b5b-f0def148f590:Recorded_Flood_Outlines"
HALF = 2500.0
RADIUS_KM = 1.8
RP = 50


def outlines(e, n):
    from shapely.geometry import shape
    from shapely.ops import unary_union
    q = {"service": "WFS", "version": "2.0.0", "request": "GetFeature", "typeNames": TYPE, "outputFormat": "application/json", "srsName": "EPSG:27700",
         "bbox": f"{e - HALF},{n - HALF},{e + HALF},{n + HALF},urn:ogc:def:crs:EPSG::27700", "count": 3000}
    with urllib.request.urlopen(urllib.request.Request(WFS + "?" + urllib.parse.urlencode(q), headers={"User-Agent": "ClimRisk/1.0"}), timeout=90) as r:
        d = json.loads(r.read())
    allg, sw = [], []
    for f in d["features"]:
        if not f.get("geometry"):
            continue
        g = shape(f["geometry"])
        allg.append(g)
        if "surface" in str(f["properties"].get("flood_caus")).lower() and g.area < 1e6:
            sw.append(g)
    return (unary_union(sw) if sw else None), unary_union(allg), len(sw)


def pit_and_low(z, dx):
    from skimage.morphology import reconstruction
    from scipy.ndimage import minimum_filter
    seed = z.copy()
    seed[1:-1, 1:-1] = z.max()
    filled = reconstruction(seed, z, method="erosion")
    k = max(3, int(500 / dx) | 1)
    return filled - z, z - minimum_filter(z, size=k, mode="nearest")


def one_town(name, lat0, lon0, rng, n_pts=150):
    from pyproj import Transformer
    from shapely.geometry import Point
    from sklearn.metrics import roc_auc_score
    from cri.finance.pluvial import site_pluvial
    to_en = Transformer.from_crs(4326, 27700, always_xy=True)
    to_ll = Transformer.from_crs(27700, 4326, always_xy=True)
    e0, n0 = to_en.transform(lon0, lat0)
    sw, allp, nfeat = outlines(e0, n0)
    if sw is None or sw.area < 2e3:
        return {"skipped": "no small surface-water outline of useful size", "outlines": nfeat}
    c = sw.centroid
    lon, lat = to_ll.transform(c.x, c.y)
    e0, n0 = c.x, c.y
    box = Point(e0, n0).buffer(RADIUS_KM * 1000 * 0.85, cap_style=3)
    pos_zone = sw.intersection(box)
    local = allp.intersection(box.buffer(300)).simplify(3.0)          # the regional event polygons carry huge vertex counts; clip and simplify first
    neg_zone = box.intersection(sw.simplify(3.0).buffer(1000)).difference(local.buffer(100))
    if pos_zone.is_empty or neg_zone.is_empty:
        return {"skipped": "empty sample zone"}

    def sample(zone, k):
        out, (x0, y0, x1, y1), tries = [], zone.bounds, 0
        while len(out) < k and tries < 40000:
            tries += 1
            p = Point(rng.uniform(x0, x1), rng.uniform(y0, y1))
            if zone.contains(p):
                out.append((p.x, p.y))
        return out
    P, N = sample(pos_zone, n_pts), sample(neg_zone, n_pts)
    pts = np.array(P + N)
    y = np.array([1] * len(P) + [0] * len(N))
    res = {"n_pos": len(P), "n_neg": len(N), "outline_km2": round(sw.area / 1e6, 3), "outlines": nfeat}
    for label, env in (("lidar", "1"), ("global", "0")):
        os.environ["CRI_LIDAR"] = env
        r = site_pluvial(lat, lon, rp=RP, radius_km=RADIUS_KM, cell_cap=300, keep_grids=True)
        if not r.get("available"):
            return {"skipped": f"model unavailable: {r.get('reason')}"}
        if label == "lidar" and "LIDAR" not in str(r["terrain_source"]).upper():
            return {"skipped": "LiDAR not returned for this box"}
        h, z, (ai, aj), dx = r["_h_max"], r["_z"], r["_asset"], r["cell_m"]
        if label == "lidar":
            cc = (pts[:, 0] - e0) / dx + aj
            rr = -(pts[:, 1] - n0) / dx + ai
        else:
            lo, la = to_ll.transform(pts[:, 0], pts[:, 1])
            cc = (lo - lon) * 111320.0 * np.cos(np.radians(lat)) / dx + aj
            rr = -(la - lat) * 111320.0 / dx + ai
        ri, ci = np.round(rr).astype(int), np.round(cc).astype(int)
        ok = (ri >= 1) & (ri < h.shape[0] - 1) & (ci >= 1) & (ci < h.shape[1] - 1)
        from scipy.ndimage import maximum_filter
        scores = {f"model_{label}": maximum_filter(h, size=3)}
        if label == "lidar":
            pit, low = pit_and_low(z, dx)
            scores["pit_depth"], scores["low_position"] = maximum_filter(pit, size=3), -low          # larger score = more flood-prone for every entry
        for fname, arr in scores.items():
            v = np.full(len(pts), np.nan)
            v[ok] = arr[ri[ok], ci[ok]]
            m = np.isfinite(v) & ok
            if m.sum() > 20 and len(set(y[m])) == 2:
                res[f"{fname}_auc"] = round(float(roc_auc_score(y[m], v[m])), 3)
                res[f"{fname}_n"] = int(m.sum())
                res.setdefault("_points", {})[fname] = {"v": [round(float(a), 3) for a in v[m]], "y": [int(b) for b in y[m]]}
        res[f"{label}_cell_m"] = dx
        res["daily_rp_mm"] = r["daily_rp_mm"]
    os.environ["CRI_LIDAR"] = "1"
    return res


def main():
    budget = float(sys.argv[sys.argv.index("--budget") + 1]) if "--budget" in sys.argv else 1e9
    sys.path[:0] = [str(ROOT / "src")]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(OUT.read_text()) if OUT.exists() and "--fresh" not in sys.argv else {}
    t0 = time.time()
    for i, (name, (la, lo)) in enumerate(TOWNS.items()):
        if name in done or time.time() - t0 > budget:
            continue
        try:
            done[name] = one_town(name, la, lo, np.random.default_rng(200 + i))
        except Exception as e:                                  # noqa: BLE001
            done[name] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
        OUT.write_text(json.dumps(done, indent=1))
        print(name, {k: v for k, v in done[name].items() if k != "_points"}, round(time.time() - t0), flush=True)
    print("remaining:", [n for n in TOWNS if n not in done])


if __name__ == "__main__":
    main()
