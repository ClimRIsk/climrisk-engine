"""Does open LiDAR terrain rank real flooding better than the global tiles? Scored on England's recorded flood outlines.

For each flooded town: positives are random points inside the Environment Agency Recorded Flood Outlines (fluvial), negatives are random points in the
same 5 km box that are at least 100 m outside any recorded outline. Two terrain scores are computed on the SAME points from two DEMs of the same place:
  global  - AWS Terrarium tiles (about 10 m, the engine's previous terrain)
  lidar   - Environment Agency 1 m bare-earth composite DTM, resampled to about 6 m
Scores: height above local low (elevation minus the lowest ground within 1 km) and HAND (height above nearest drainage, 1 km2 channel threshold).
Metric: ROC area under the curve per town and pooled (0.5 = no skill). Nothing is fitted: the scores are used as they are, lower = more flood-prone.
Caveats: modern terrain for historic floods; defences built after an event are in the LiDAR; recorded outlines are where floods were mapped, not everywhere.

  PYTHONPATH=.:src python scripts/lidar_flood_benchmark.py --budget 140      (resumable; results in data/backtest/lidar_flood_benchmark.json)
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
OUT = ROOT / "data" / "backtest" / "lidar_flood_benchmark.json"
TOWNS = {"Carlisle": (54.8925, -2.9329), "Cockermouth": (54.664, -3.363), "York": (53.958, -1.083), "Tadcaster": (53.883, -1.263),
         "Hebden Bridge": (53.742, -2.014), "Shrewsbury": (52.707, -2.754), "Appleby": (54.577, -2.489), "Lewes": (50.874, 0.009),
         "Kendal": (54.328, -2.746), "Morpeth": (55.168, -1.690), "Upton upon Severn": (52.065, -2.210), "Tewkesbury": (51.995, -2.160)}
WFS = "https://environment.data.gov.uk/spatialdata/recorded-flood-outlines/wfs"
TYPE = "dataset-8c75e700-d465-11e4-8b5b-f0def148f590:Recorded_Flood_Outlines"
HALF = 2500.0


def outlines(e, n):
    from shapely.geometry import shape
    from shapely.ops import unary_union
    q = {"service": "WFS", "version": "2.0.0", "request": "GetFeature", "typeNames": TYPE, "outputFormat": "application/json", "srsName": "EPSG:27700",
         "bbox": f"{e - HALF},{n - HALF},{e + HALF},{n + HALF},urn:ogc:def:crs:EPSG::27700", "count": 3000}
    with urllib.request.urlopen(urllib.request.Request(WFS + "?" + urllib.parse.urlencode(q), headers={"User-Agent": "ClimRisk/1.0"}), timeout=90) as r:
        d = json.loads(r.read())
    gs = [shape(f["geometry"]) for f in d["features"] if str(f["properties"].get("fluvial_f")).lower() == "true" and f.get("geometry")]
    return unary_union(gs) if gs else None, len(gs)


def hand_and_rel(dem, res):
    """Return (height above local low within 1 km, HAND at 1 km2) arrays for a DEM in metres."""
    from scipy.ndimage import minimum_filter
    from pysheds.view import Raster
    from cri.terrain.pysheds_analyzer import _NODATA, _build_pysheds_grid
    valid = np.isfinite(dem)
    filled = np.where(valid, dem, np.nanmax(dem))
    k = max(3, int(1000 / res) | 1)
    rel = dem - minimum_filter(np.where(valid, dem, np.inf), size=k, mode="nearest")
    grd, raster = _build_pysheds_grid(np.where(valid, dem, _NODATA), 0.0, 0.0, res, res, _NODATA)
    cond = grd.resolve_flats(grd.fill_depressions(grd.fill_pits(raster)))
    fdir = grd.flowdir(cond)
    acc = np.asarray(grd.accumulation(fdir), float)
    ch = acc * (res / 1000.0) ** 2 >= 1.0
    hand = np.full(dem.shape, np.nan)
    if ch.any():
        h = np.asarray(grd.compute_hand(fdir, cond, Raster(ch, viewfinder=fdir.viewfinder)), float)
        h[(h < 0) | (h > 1e4) | ~valid] = np.nan
        hand = h
    return rel, hand


def one_town(name, lat, lon, rng, n_pts=120):
    from pyproj import Transformer
    from shapely.geometry import Point
    from sklearn.metrics import roc_auc_score
    from cri.connectors import terrain_tiles as T
    to_en = Transformer.from_crs(4326, 27700, always_xy=True)
    to_ll = Transformer.from_crs(27700, 4326, always_xy=True)
    e0, n0 = to_en.transform(lon, lat)
    poly, nfeat = outlines(e0, n0)
    if poly is None or poly.area < 5e4:
        return {"skipped": "no recorded fluvial outline of useful size in the box", "outlines": nfeat}
    box = Point(e0, n0).buffer(HALF * 0.85, cap_style=3)
    pos_zone, neg_zone = poly.intersection(box), box.difference(poly.buffer(100))
    if pos_zone.is_empty or neg_zone.is_empty:
        return {"skipped": "empty sample zone"}

    def sample(zone, k):
        out, (x0, y0, x1, y1) = [], zone.bounds
        tries = 0
        while len(out) < k and tries < 40000:
            tries += 1
            p = Point(rng.uniform(x0, x1), rng.uniform(y0, y1))
            if zone.contains(p):
                out.append((p.x, p.y))
        return out
    P, N = sample(pos_zone, n_pts), sample(neg_zone, n_pts)
    pts = np.array(P + N)
    y = np.array([1] * len(P) + [0] * len(N))
    res_out = {"n_pos": len(P), "n_neg": len(N), "outline_km2": round(poly.area / 1e6, 2), "outlines": nfeat}
    for label, env in (("global", "0"), ("lidar", "1")):
        os.environ["CRI_LIDAR"] = env
        g = T.get_elevation_grid(lat, lon, radius_km=HALF / 1000.0)
        dem, res = g.elevations, g.cell_size_m
        if label == "lidar" and "LIDAR" not in g.source.upper():
            return {"skipped": "LiDAR not returned for this box"}
        rel, hand = hand_and_rel(dem, res)
        if label == "lidar":                                   # exact projected grid: pixel offsets in metres from the pin's grid position
            cc = (pts[:, 0] - e0) / res + g.asset_col
            rr = -(pts[:, 1] - n0) / res + g.asset_row
        else:                                                  # Mercator ground metres from the pin; go through lat/lon so grid-north differences are handled
            lo, la = to_ll.transform(pts[:, 0], pts[:, 1])
            cc = (lo - lon) * 111320.0 * np.cos(np.radians(lat)) / res + g.asset_col
            rr = -(la - lat) * 111320.0 / res + g.asset_row
        ri, ci = np.round(rr).astype(int), np.round(cc).astype(int)
        ok = (ri >= 0) & (ri < dem.shape[0]) & (ci >= 0) & (ci < dem.shape[1])
        for fname, arr in (("height_above_local_low", rel), ("HAND", hand)):
            v = np.full(len(pts), np.nan)
            v[ok] = arr[ri[ok], ci[ok]]
            m = np.isfinite(v) & ok
            if m.sum() > 20 and len(set(y[m])) == 2:
                res_out[f"{label}_{fname}_auc"] = round(float(roc_auc_score(y[m], -v[m])), 3)     # lower value = more flood-prone
                res_out[f"{label}_{fname}_n"] = int(m.sum())
                res_out.setdefault("_points", {})[f"{label}_{fname}"] = {"v": [round(float(a), 2) for a in v[m]], "y": [int(b) for b in y[m]]}
        res_out[f"{label}_cell_m"] = round(float(res), 1)
    os.environ["CRI_LIDAR"] = "1"
    return res_out


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
            done[name] = one_town(name, la, lo, np.random.default_rng(100 + i))
        except Exception as e:                                  # noqa: BLE001
            done[name] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
        OUT.write_text(json.dumps(done, indent=1))
        print(name, done[name], round(time.time() - t0))
    left = [n for n in TOWNS if n not in done]
    print("remaining:", left)


if __name__ == "__main__":
    main()
