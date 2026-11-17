"""Re-score the PRODUCT surface-water model (`finance.pluvial.site_surface_water`: ~12 km window, 24 h storm, gauge-anchored design rain, drains overwhelmed) on England's
Environment Agency Recorded Flood Outlines (cause 'local drainage/surface water'), on exactly the same positive and negative points as `pluvial_benchmark.py` (same seeds, same
sampling zones), so the new setup can be compared with the old 600 m one and with plain terrain scores. The older table in docs/PLUVIAL_BENCHMARK.md must not be quoted for this model.

Scores on the same points:
  model_sw      - maximum depth, drains-overwhelmed case (the rating basis), from the ~60 m grid
  pit_depth     - blue-spot depth on that same grid
  low_position  - height above the lowest ground within about 500 m on that same grid
Metrics: ROC AUC with a 95 % bootstrap interval (points resampled; towns are not resampled, so the interval does not cover town-to-town spread), and hit rate / false-alarm rate
at a 5 cm ponding threshold (POD = share of flooded-outline points the model wets; FAR = share of nearby non-outline points it also wets). Nothing is fitted or tuned.
Limits: outlines are where floods were MAPPED (sparse, biased to where people reported or the EA surveyed), terrain is modern while floods are historic, sewers are not in the terrain,
the 60 m grid cannot resolve a street, and a negative point may have flooded without being mapped.

  PYTHONPATH=.:src python scripts/surface_water_benchmark.py --budget 140 [--rp 50] [--fresh]   (resumable; data/backtest/surface_water_benchmark.json)
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import pluvial_benchmark as PB                                                    # noqa: E402  (same outline query, same towns, same helpers)

OUT = ROOT / "data" / "backtest" / "surface_water_benchmark.json"
THRESH_M = 0.05
# Added after the first three towns had been scored, to widen a thin sample. Chosen for known pluvial flood history (2007, 2012, 2014), not for how the model did; every town that
# ran is reported, including skips. Seeds continue after the original eight so the original towns keep exactly the points the older benchmark used.
EXTRA = {"Gloucester": (51.865, -2.245), "Exeter": (50.72, -3.53), "Nottingham": (52.954, -1.15), "Newcastle": (54.978, -1.617), "Leicester": (52.636, -1.133), "Northampton": (52.24, -0.9)}
_RAW_OUTLINES = PB.outlines


def _cached_outlines(e, n):
    """The EA query returns ~25 MB per town and is slow some days; keep the geometries on disk so a retry does not repeat it."""
    import pickle
    p = Path(os.environ.get("CRI_BENCH_CACHE", "/tmp")) / f"ea_outlines_{int(e)}_{int(n)}.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    r = _RAW_OUTLINES(e, n)
    p.write_bytes(pickle.dumps(r))
    return r


PB.outlines = _cached_outlines


def prefetch_gauge(station, y0=1950, y1=2025, chunk=10):
    """NCEI's daily-summaries service can take minutes for 76 years in one request. Fetch decade chunks in parallel and store the joined CSV under the SAME cache key the product
    code (`rain_design.daily_gauge` -> `stationcheck._text`) reads, so the benchmark still runs the unchanged product path. Same data, only the transport differs."""
    import concurrent.futures as cf
    import urllib.request
    from cri.modellab import stationcheck as SC
    url = SC.daily_url(station, "PRCP", y0, y1)
    p = SC.D._cache_path(url).with_suffix(".csv")
    if p.exists():
        return "cached"
    spans = [(a, min(a + chunk - 1, y1)) for a in range(y0, y1 + 1, chunk)]

    def get(span):
        u = url.replace(f"startDate={y0}-01-01", f"startDate={span[0]}-01-01").replace(f"endDate={y1}-12-31", f"endDate={span[1]}-12-31")
        for _ in range(3):
            try:
                with urllib.request.urlopen(urllib.request.Request(u, headers=SC.UA), timeout=100) as r:
                    return r.read().decode("utf-8", "ignore")
            except Exception:                                              # noqa: BLE001
                time.sleep(2)
        raise RuntimeError(f"chunk {span} failed")
    with cf.ThreadPoolExecutor(len(spans)) as ex:
        parts = list(ex.map(get, spans))
    rows = [parts[0].splitlines()[0]] + [ln for t in parts for ln in t.splitlines()[1:] if ln.strip()]   # header once, then every chunk's data lines
    p.write_text("\n".join(rows) + "\n")
    return f"fetched {len(rows) - 1} rows"


def _auc_ci(y, v, rng, n=300):
    from sklearn.metrics import roc_auc_score
    a = float(roc_auc_score(y, v))
    idx = np.arange(len(y))
    bs = []
    for _ in range(n):
        s = rng.choice(idx, len(idx))
        if len(set(y[s])) == 2:
            bs.append(roc_auc_score(y[s], v[s]))
    return round(a, 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)


def one_town(name, lat0, lon0, rng, rp):
    from pyproj import Transformer
    from shapely.geometry import Point
    from scipy.ndimage import maximum_filter
    from cri.finance.pluvial import site_surface_water
    to_en = Transformer.from_crs(4326, 27700, always_xy=True)
    to_ll = Transformer.from_crs(27700, 4326, always_xy=True)
    e0, n0 = to_en.transform(lon0, lat0)
    sw, allp, nfeat = PB.outlines(e0, n0)
    if sw is None or sw.area < 2e3:
        return {"skipped": "no small surface-water outline of useful size", "outlines": nfeat}
    c = sw.centroid
    lon, lat = to_ll.transform(c.x, c.y)
    e0, n0 = c.x, c.y
    box = Point(e0, n0).buffer(PB.RADIUS_KM * 1000 * 0.85, cap_style=3)
    pos_zone = sw.intersection(box)
    local = allp.intersection(box.buffer(300)).simplify(3.0)
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
    P, N = sample(pos_zone, 150), sample(neg_zone, 150)
    pts = np.array(P + N)
    y = np.array([1] * len(P) + [0] * len(N))
    os.environ["CRI_LIDAR"] = "1"
    from cri.modellab import stationcheck as SC
    for st in SC.nearest(lat, lon, "PRCP", k=1, max_km=60.0, min_years=25):
        prefetch_gauge(st["id"])
    r = site_surface_water(lat, lon, rp=rp, keep_grids=True, bounds=False, use_cache=False)
    if not r.get("available"):
        return {"error": f"model unavailable: {r.get('reason')}"}
    if "LIDAR" not in str(r.get("terrain_source")).upper():
        return {"error": f"LiDAR not returned, the run fell back to {r.get('terrain_source')}; not scored (retry later)"}      # kept out of the table, retried on resume
    h, z, (ai, aj), dx = r["_h_max"], r["_z"], r["_asset"], r["_dx"]
    lo, la = to_ll.transform(pts[:, 0], pts[:, 1])
    cc = (lo - lon) * 111320.0 * np.cos(np.radians(lat)) / dx + aj
    rr = -(la - lat) * 111320.0 / dx + ai
    ri, ci = np.round(rr).astype(int), np.round(cc).astype(int)
    ok = (ri >= 1) & (ri < h.shape[0] - 1) & (ci >= 1) & (ci < h.shape[1] - 1)
    pit, low = PB.pit_and_low(z, dx)
    scores = {"model_sw": maximum_filter(h, size=3), "pit_depth": maximum_filter(pit, size=3), "low_position": -low}
    res = {"n_pos": len(P), "n_neg": len(N), "outline_km2": round(sw.area / 1e6, 3), "outlines": nfeat, "rp": rp, "cell_m": round(dx, 1), "terrain": r.get("terrain_source"),
           "design_rain_mm": r["rain"]["d1_mm"], "rain_source": r["rain"].get("source_used"), "gauge_km": r["rain"].get("gauge_km"), "storm_hours": r["storm_hours"]}
    for fname, arr in scores.items():
        v = np.full(len(pts), np.nan)
        v[ok] = arr[ri[ok], ci[ok]]
        m = np.isfinite(v) & ok
        if m.sum() > 20 and len(set(y[m])) == 2:
            a, lo_, hi_ = _auc_ci(y[m], v[m], rng)
            res[fname] = {"auc": a, "ci95": [lo_, hi_], "n": int(m.sum())}
            res.setdefault("_points", {})[fname] = {"v": [round(float(q), 3) for q in v[m]], "y": [int(q) for q in y[m]]}     # kept so a score can be audited
            if fname == "model_sw":
                wet = v[m] >= THRESH_M
                res[fname]["pod_5cm"] = round(float(wet[y[m] == 1].mean()), 3)
                res[fname]["far_5cm"] = round(float(wet[y[m] == 0].mean()), 3)
                res[fname]["share_cells_wet"] = round(float((h >= THRESH_M).mean()), 3)
    return res


def main():
    budget = float(sys.argv[sys.argv.index("--budget") + 1]) if "--budget" in sys.argv else 1e9
    rp = int(sys.argv[sys.argv.index("--rp") + 1]) if "--rp" in sys.argv else 50
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(OUT.read_text()) if OUT.exists() and "--fresh" not in sys.argv else {}
    done = {k: v for k, v in done.items() if "error" not in v}                  # a failed town is retried, a skipped one (no usable outline) is not
    t0 = time.time()
    TOWNS = {**PB.TOWNS, **EXTRA}
    for i, (name, (la, lo)) in enumerate(TOWNS.items()):
        if name in done or time.time() - t0 > budget:
            continue
        try:
            done[name] = one_town(name, la, lo, np.random.default_rng(200 + i), rp)
        except Exception as e:                                  # noqa: BLE001
            done[name] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}
        OUT.write_text(json.dumps(done, indent=1))
        print(name, done[name], round(time.time() - t0), flush=True)
    print("remaining:", [n for n in TOWNS if n not in done])


if __name__ == "__main__":
    main()
