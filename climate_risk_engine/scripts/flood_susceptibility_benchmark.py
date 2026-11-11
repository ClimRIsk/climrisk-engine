"""Which terrain index finds the ground that real floods covered, anywhere in the world?  (11 Nov 2026)

Truth: Global Flood Database v1.4 (Tellman et al., Nature 2021), observed flood extent from MODIS at 250 m for 913 events 2000-2018, public bucket
gs://gfd_v1_4, licence CC BY-NC 4.0 (used here only to MEASURE; no GFD pixel is stored in the engine or shipped). A pixel is a positive when the event flooded
it, a negative when it was observed clear (clear_views >= MIN_CLEAR) and not flooded, and permanent water is excluded.

Protocol (the same as scripts/pluvial_benchmark.py): 12 km windows centred on the densest flooded cells; positives and negatives come from the same window, so
the test is "which pixels of this neighbourhood flooded", not "which country floods".  Terrain: the engine's global elevation tiles (AWS Terrarium, zoom 12).
Each index is scored by ROC AUC per window, then averaged per event, then across events (every event counts once).  Nothing is fitted on the events here.

What it cannot tell: MODIS at 250 m does not see flooding inside dense cities (in Chennai in December 2015 the Velachery, Adyar and airport pixels are all "not
flooded" in this product), so these scores describe open-country, peri-urban and floodplain flooding; urban street flooding is outside the test.

  PYTHONPATH=src python scripts/flood_susceptibility_benchmark.py windows  GFD_DIR   -> windows.json
  PYTHONPATH=src python scripts/flood_susceptibility_benchmark.py score    GFD_DIR   -> results.json
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

MIN_CLEAR = 3
WIN_KM = 12.0
MARGIN_KM = 3.0
PX_M = 250.0


def event_dirs(gfd):
    return sorted(p for p in glob.glob(os.path.join(gfd, "ev", "DFO_*")) if os.path.exists(p + "/.ok"))


def read_event(d):
    import rasterio
    tif = glob.glob(d + "/*.tif")[0]
    props = json.load(open(glob.glob(d + "/*properties.json")[0]))
    with rasterio.open(tif) as r:
        fl, du, cv, cp, pw = (r.read(i + 1) for i in range(5))
        return props, r.transform, fl, cv, pw


def windows(gfd, per_event=4, min_pos=30, min_neg=60):
    out = []
    for d in event_dirs(gfd):
        props, tr, fl, cv, pw = read_event(d)
        pos = (fl == 1) & (pw < 1)
        neg = (fl == 0) & (cv >= MIN_CLEAR) & (pw < 1)
        w = int(WIN_KM * 1000 / PX_M)
        taken = []
        score = np.zeros((fl.shape[0] // w + 1, fl.shape[1] // w + 1))
        for i in range(0, fl.shape[0] - w + 1, w // 2):
            for j in range(0, fl.shape[1] - w + 1, w // 2):
                p, n = int(pos[i:i + w, j:j + w].sum()), int(neg[i:i + w, j:j + w].sum())
                if p >= min_pos and n >= min_neg:
                    taken.append((min(p, n), i, j, p, n))
        taken.sort(reverse=True)
        chosen = []
        for _, i, j, p, n in taken:
            if all(abs(i - a) >= w or abs(j - b) >= w for a, b, *_ in chosen):
                chosen.append((i, j, p, n))
            if len(chosen) >= per_event:
                break
        for i, j, p, n in chosen:
            lon0, lat0 = tr * (j, i)
            lon1, lat1 = tr * (j + w, i + w)
            out.append({"event": os.path.basename(d), "cause": props.get("dfo_main_cause"), "country": props.get("dfo_country"), "row": i, "col": j, "w": w, "n_pos": p, "n_neg": n,
                        "bbox": [min(lon0, lon1), min(lat0, lat1), max(lon0, lon1), max(lat0, lat1)]})
    return out


def auc(score, label):
    from scipy.stats import rankdata
    s, y = np.asarray(score, float), np.asarray(label, bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 < 10 or n0 < 10:
        return None
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def terrain_indices(bbox, zoom=12):
    """Indices on the engine's global terrain for a window, aggregated to 250 m. Returns dict of 2-D arrays (rows = north to south) or None."""
    from scipy.ndimage import minimum_filter, uniform_filter, maximum_filter
    from skimage.morphology import reconstruction
    from cri.connectors.terrain_tiles import get_elevation_grid
    from cri.terrain.lidar_flood import hand_and_rel
    lon0, lat0, lon1, lat1 = bbox
    lat, lon = (lat0 + lat1) / 2, (lon0 + lon1) / 2
    rad = (WIN_KM / 2 + MARGIN_KM) * 1.0
    g = get_elevation_grid(lat, lon, radius_km=rad, zoom=zoom)
    z = np.asarray(g.elevations, float)
    if np.isnan(z).all():
        return None
    z = np.where(np.isnan(z), np.nanmedian(z), z)
    dx = float(g.cell_size_m)
    out = {}
    out["elev"] = z
    rel1, hand1 = hand_and_rel(z, dx, channel_km2=1.0)
    _, hand_s = hand_and_rel(z, dx, channel_km2=0.25)
    _, hand_l = hand_and_rel(z, dx, channel_km2=4.0)
    out["rel_1km"] = rel1
    k3 = max(3, int(3000 / dx) | 1)
    out["rel_3km"] = z - minimum_filter(z, size=k3, mode="nearest")
    k8 = max(3, int(8000 / dx) | 1)
    out["rel_8km"] = z - minimum_filter(z, size=k8, mode="nearest")
    out["hand_1km2"], out["hand_025km2"], out["hand_4km2"] = hand1, hand_s, hand_l
    # the model the engine ships (fitted on Copernicus EMS 2023-26 events, analysis/flood_event_model.py), applied cell by cell: a test on events it never saw
    from scipy.stats import rankdata as _rk
    from cri.analysis import flood_event_model as FEM
    _, hand_01 = hand_and_rel(z, dx, channel_km2=0.1)
    ep = (_rk(z.ravel()) / z.size).reshape(z.shape)
    X = FEM.design(hand_01.ravel(), hand1.ravel(), ep.ravel(), (z - np.percentile(z, 10)).ravel())
    out["event_model"] = FEM.predict(FEM.load()["model"], X).reshape(z.shape)
    seed = z.copy(); seed[1:-1, 1:-1] = z.max()
    out["pit"] = reconstruction(seed, z, method="erosion") - z
    out["elev_pct"] = None
    return out, g


def to_window_grid(arr, g, bbox, n):
    """Resample a terrain-index array (cell grid of g) to the n x n 250 m window grid by block mean at each target cell."""
    from scipy.ndimage import map_coordinates
    lon0, lat0, lon1, lat1 = bbox
    lats = lat1 - (np.arange(n) + 0.5) * (lat1 - lat0) / n
    lons = lon0 + (np.arange(n) + 0.5) * (lon1 - lon0) / n
    # g has asset_row/asset_col at its centre and a square metric cell
    dxm = float(g.cell_size_m)
    clat, clon = float(g.lat), float(g.lon)
    ry = (clat - lats) * 111_320.0 / dxm
    rx = (lons - clon) * 111_320.0 * math.cos(math.radians(clat)) / dxm
    rows = g.asset_row + ry[:, None] * np.ones((1, n))
    cols = g.asset_col + rx[None, :] * np.ones((n, 1))
    a = np.where(np.isfinite(arr), arr, np.nanmedian(arr))
    # average the ~6x6 fine cells inside one 250 m pixel
    from scipy.ndimage import uniform_filter
    f = max(1, int(PX_M / dxm))
    am = uniform_filter(a, size=f, mode="nearest")
    return map_coordinates(am, [rows, cols], order=1, mode="nearest")


def score(gfd, wins, zoom=12, out_path="results.json", feats_dir=None):
    if feats_dir:
        os.makedirs(feats_dir, exist_ok=True)
    rows = []
    for k, w in enumerate(wins):
        try:
            props, tr, fl, cv, pw = read_event(os.path.join(gfd, "ev", w["event"]))
            n = w["w"]; sl = (slice(w["row"], w["row"] + n), slice(w["col"], w["col"] + n))
            pos = ((fl == 1) & (pw < 1))[sl]; neg = ((fl == 0) & (cv >= MIN_CLEAR) & (pw < 1))[sl]
            got = terrain_indices(w["bbox"], zoom)
            if got is None:
                continue
            idx, g = got
            lab, keep = pos.ravel(), (pos | neg).ravel()
            feats = {}
            r = {"event": w["event"], "cause": w["cause"], "country": w["country"], "n_pos": int(pos.sum()), "n_neg": int(neg.sum())}
            for name, a in idx.items():
                if a is None:
                    continue
                v = to_window_grid(a, g, w["bbox"], n).ravel()
                s = -v if name in ("elev", "rel_1km", "rel_3km", "rel_8km", "hand_1km2", "hand_025km2", "hand_4km2") else v        # event_model and pit: higher = more flood-like already        # low ground scores high; deeper pit scores high
                r[name] = auc(s[keep], lab[keep])
                from scipy.stats import rankdata
                feats[name] = (rankdata(np.where(np.isfinite(s), s, np.nanmedian(s))) / s.size).astype(np.float32)      # percentile of the cell within its 12 km window (1 = most susceptible)
            if feats_dir:
                np.savez_compressed(os.path.join(feats_dir, f"{w['event']}_{w['row']}_{w['col']}.npz"), keep=keep, lab=lab, **feats)
            rows.append(r)
            print(k + 1, "/", len(wins), w["event"], w["country"], {a: (None if b is None else round(b, 2)) for a, b in r.items() if a not in ("event", "cause", "country", "n_pos", "n_neg")}, flush=True)
        except Exception as e:                                                 # noqa: BLE001
            print("skip", w["event"], type(e).__name__, str(e)[:100], flush=True)
        json.dump(rows, open(out_path, "w"))
    return rows


def summarise(rows):
    names = [k for k in rows[0] if k not in ("event", "cause", "country", "n_pos", "n_neg")]
    ev = {}
    for r in rows:
        ev.setdefault(r["event"], []).append(r)
    res = {}
    for nme in names:
        per_event = []
        for e, rs in ev.items():
            v = [x[nme] for x in rs if x.get(nme) is not None]
            if v:
                per_event.append(float(np.mean(v)))
        res[nme] = {"events": len(per_event), "mean_auc": round(float(np.mean(per_event)), 3), "median_auc": round(float(np.median(per_event)), 3),
                    "share_events_above_0.6": round(float(np.mean([a > 0.6 for a in per_event])), 2)}
    return res


if __name__ == "__main__":
    mode, gfd = sys.argv[1], sys.argv[2]
    if mode == "windows":
        w = windows(gfd)
        json.dump(w, open(os.path.join(gfd, "windows.json"), "w"), indent=1)
        print(len(w), "windows from", len({x["event"] for x in w}), "events")
    elif mode == "score":
        wins = json.load(open(os.path.join(gfd, "windows.json")))
        lo, hi = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, len(wins))
        rows = score(gfd, wins[lo:hi], out_path=os.path.join(gfd, f"results_{lo}_{hi}.json"), feats_dir=os.path.join(gfd, "feats"))
        print(json.dumps(summarise(rows), indent=1))
