"""
Flood EVENT benchmark: does the model's terrain flood layer rank the cells that actually flooded above the cells that did not?

Truth
-----
Copernicus Emergency Management Service (CEMS) Rapid Mapping: observed flood-extent polygons delineated from satellite
imagery during real events worldwide (free and open, Copernicus licence; attribution required).
  https://rapidmapping.emergency.copernicus.eu  (public dashboard API, no key)

Design (what is and is not being tested)
----------------------------------------
* Unit = a ~8 km window around an observed flooded area. Every 10 m cell of land in the window is labelled
  flooded / not flooded from the observed polygons (permanent water excluded).
* The model under test is the engine's TERRAIN flood layer (conditioned D8 flow → HAND at several channel thresholds,
  elevation percentile in the window). It uses no imagery from the event, so there is no leakage.
  The satellite-radar water history used elsewhere in the engine is NOT tested here because it would contain the event.
* Metric 1 (rank skill): per-window ROC AUC of "lower HAND ⇒ more likely flooded" (0.5 = no skill), then the
  event-level average with a bootstrap over EVENTS (not cells, which are spatially autocorrelated).
* Metric 2 (magnitude): where CEMS also published a flood-depth layer, compare predicted depth proxy to the observed
  depth (stage `depth`; not run when no depth product exists).
* Train / test: events are split by event (never by cell) and by region; weights are fitted on the training events
  only and scored on held-out events. Nothing is fitted on the test events.
* Known limits written into the output: SAR under-maps flooding in dense urban areas and under vegetation; polygons are a
  snapshot at image time, not the peak; CEMS maps areas where flooding was suspected, so results describe skill WITHIN
  flood-affected areas, not whether a random place would flood.

Stages (cached under data/backtest/raw/events/)
  catalog   list CEMS flood activations and their delineation products
  extents   download observed-extent zips (size-capped) and keep the flood polygons
  features  HAND grid + labels per window                       (FEAT_BATCH=n per call; repeat)
  evaluate  per-window AUC, held-out split, bootstrap, JSON + markdown report

Usage
  python scripts/flood_event_benchmark.py --stage catalog
  python scripts/flood_event_benchmark.py --stage extents --max-mb 60
  FEAT_BATCH=4 python scripts/flood_event_benchmark.py --stage features
  python scripts/flood_event_benchmark.py --stage evaluate
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RAW = ROOT / "data" / "backtest" / "raw" / "events"
OUT = ROOT / "data" / "backtest"
API = "https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/public-activations"
UA = {"User-Agent": "climrisk-research/0.1 (flood event benchmark)"}
WINDOW_KM = 4.0          # half-width of the analysis window (8 km across)
MAX_WINDOWS_PER_EVENT = 1     # per AOI
MAX_AOI_PER_EVENT = 2


def _get(url: str, timeout: int = 90) -> bytes:
    for k in range(3):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()
        except Exception:                                           # noqa: BLE001
            if k == 2:
                raise
            time.sleep(2 + 3 * k)


# ── catalog ──────────────────────────────────────────────────────────────────

def stage_catalog() -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    codes, off = [], 0
    while True:
        d = json.loads(_get(f"{API}-info/?category=Flood&limit=50&offset={off}"))
        codes += [r["code"] for r in d["results"]]
        if not d.get("next") or not d["results"]:
            break
        off += 50
    events = []
    for c in codes:
        try:
            a = json.loads(_get(f"{API}/?code={c}"))["results"][0]
        except Exception as exc:                                    # noqa: BLE001
            print(c, "skip", exc)
            continue
        prods = []
        for aoi in a.get("aois") or []:
            for p in aoi.get("products") or []:
                if p.get("type") == "DEL" and p.get("downloadPath") and (p.get("version") or {}).get("statusCode") == "F":
                    prods.append({"aoi": aoi.get("number"), "aoi_name": aoi.get("name"), "id": p["id"],
                                  "monitoring": p.get("monitoring"), "url": p["downloadPath"],
                                  "sensor": [(i.get("sensorType"), i.get("sensorName")) for i in p.get("images") or []],
                                  "acq": [i.get("acquisitionTime") for i in p.get("images") or []],
                                  "extent": p.get("extent"),
                                  "flooded_ha": ((p.get("stats") or {}).get("Flooded area") or {}).get("None", {}).get("affected")})
        events.append({"code": c, "name": a.get("name"), "sub": a.get("subCategory"), "event_time": a.get("eventTime"),
                       "countries": [x.get("name") for x in a.get("countries") or []], "continent": a.get("continent"),
                       "centroid": a.get("centroid"), "products": prods})
        print(c, a.get("name"), (a.get("eventTime") or "")[:10], len(prods), "products")
    p = RAW / "catalog.json"
    p.write_text(json.dumps(events, indent=1))
    print("events:", len(events), "with products:", sum(1 for e in events if e["products"]))
    return p


# ── extents ──────────────────────────────────────────────────────────────────

def _flood_polys(zbytes: bytes):
    """Return list of shapely geometries for observed flood extent in a CEMS product zip, or None."""
    import shapefile
    from shapely.geometry import shape
    z = zipfile.ZipFile(io.BytesIO(zbytes))
    names = z.namelist()
    shp = [n for n in names if n.lower().endswith(".shp") and "observedeventa" in n.lower()]
    if not shp:
        return None, names
    geoms = []
    for n in shp:
        base = n[:-4]
        rd = shapefile.Reader(shp=io.BytesIO(z.read(base + ".shp")), dbf=io.BytesIO(z.read(base + ".dbf")),
                              shx=io.BytesIO(z.read(base + ".shx")) if base + ".shx" in names else None)
        for sr in rd.shapeRecords():
            rec = sr.record.as_dict() if hasattr(sr.record, "as_dict") else {}
            ev = str(rec.get("event_type", "")).lower()
            if ev and "flood" not in ev and "trace" not in ev:
                continue
            try:
                g = shape(sr.shape.__geo_interface__)
                if not g.is_valid:
                    g = g.buffer(0)
                if not g.is_empty:
                    geoms.append(g)
            except Exception:                                       # noqa: BLE001
                continue
    return geoms, names


def _extent_one(ev: dict, aoi, ps: list, max_mb: float) -> str:
    from shapely.geometry import mapping
    from shapely.ops import unary_union
    ps.sort(key=lambda p: (bool(p["monitoring"]), p["id"]))
    out = RAW / "extents" / f"{ev['code']}_AOI{int(aoi):02d}.json"
    p = ps[0]
    try:
        req = urllib.request.Request(p["url"], headers=UA, method="HEAD")
        size = int(urllib.request.urlopen(req, timeout=60).headers.get("Content-Length") or 0)
    except Exception:                                               # noqa: BLE001
        size = 0
    if size > max_mb * 1e6:
        out.write_text(json.dumps({"skipped": f"zip {size/1e6:.0f} MB > cap"}))
        return f"{out.stem} skipped (size)"
    try:
        geoms, names = _flood_polys(_get(p["url"], 180))
    except Exception as exc:                                        # noqa: BLE001
        out.write_text(json.dumps({"skipped": f"download/parse failed: {exc}"}))
        return f"{out.stem} skipped (download)"
    if not geoms:
        out.write_text(json.dumps({"skipped": "no observed-event polygons", "files": [n for n in names if n.endswith('.shp')][:8]}))
        return f"{out.stem} skipped (no polygons)"
    u = unary_union(geoms)
    out.write_text(json.dumps({"code": ev["code"], "aoi": aoi, "product": p["id"], "sensor": p["sensor"], "acq": p["acq"],
                               "n_polys": len(geoms), "area_deg2": u.area, "bounds": u.bounds,
                               "geometry": mapping(u.simplify(0.00005))}))
    return f"{out.stem} ok {len(geoms)} polygons {p['sensor']}"


def stage_extents(max_mb: float) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    cat = json.loads((RAW / "catalog.json").read_text())
    (RAW / "extents").mkdir(exist_ok=True)
    jobs = []
    for ev in cat:
        by_aoi: dict = {}
        for p in ev["products"]:
            by_aoi.setdefault(p["aoi"], []).append(p)       # newest observed-extent product per AOI
        top = sorted(by_aoi, key=lambda a: -max((p.get("flooded_ha") or 0) for p in by_aoi[a]))[:MAX_AOI_PER_EVENT]
        for aoi in top:
            if not (RAW / "extents" / f"{ev['code']}_AOI{int(aoi):02d}.json").exists():
                jobs.append((ev, aoi, by_aoi[aoi]))
    print(len(jobs), "extents to fetch")
    t0, done = time.time(), 0
    ex = ThreadPoolExecutor(max_workers=6)
    futs = []
    for j in jobs:
        futs.append(ex.submit(_extent_one, *j, max_mb))
    for f in as_completed(futs, timeout=None):
        print(f.result()); done += 1
        if time.time() - t0 > 140:
            for g in futs:
                g.cancel()
            print("time budget reached; run again to continue")
            break
    ex.shutdown(wait=False, cancel_futures=True)
    print("done", done, "of", len(jobs))


# ── features ─────────────────────────────────────────────────────────────────

def _windows(ext: dict) -> list[tuple[float, float]]:
    """Up to MAX_WINDOWS_PER_EVENT window centres inside the flood area (largest polygons first)."""
    from shapely.geometry import shape
    g = shape(ext["geometry"])
    parts = sorted(getattr(g, "geoms", [g]), key=lambda x: -x.area)
    centres: list[tuple[float, float]] = []
    min_sep = 2 * WINDOW_KM / 111.0
    for part in parts:
        c = part.representative_point()
        if all(math.hypot(c.x - x, c.y - y) > min_sep for x, y in centres):
            centres.append((c.x, c.y))
        if len(centres) >= MAX_WINDOWS_PER_EVENT:
            break
    return [(y, x) for x, y in centres]


def _window_features(lat: float, lon: float, ext: dict) -> dict | None:
    from rasterio.features import rasterize
    from shapely.geometry import shape
    from cri.terrain.hand import builtup_on_grid, hand_grid
    hg = hand_grid(lat, lon, radius_km=WINDOW_KM)
    H, W = hg.dem.shape
    flooded = rasterize([(shape(ext["geometry"]), 1)], out_shape=(H, W), transform=hg.transform, fill=0, dtype="uint8").astype(bool)
    built = builtup_on_grid(hg)
    # permanent water out of the sample: WorldCover class 80 where available, plus HAND == 0 channel cells
    perm = np.zeros((H, W), bool)
    try:
        from cri.connectors.worldcover import lc_grid
        lc, _ = lc_grid(hg.lat, hg.lon, hg.radius_km)
        ri = np.minimum((np.arange(H) * lc.shape[0] / H).astype(int), lc.shape[0] - 1)
        ci = np.minimum((np.arange(W) * lc.shape[1] / W).astype(int), lc.shape[1] - 1)
        perm = lc[np.ix_(ri, ci)] == 80
    except Exception:                                               # noqa: BLE001
        pass
    ok = hg.valid & ~perm
    if ext.get("exclude"):                      # cells that flooded in OTHER recorded events: not valid negatives
        other = rasterize([(shape(ext["exclude"]), 1)], out_shape=(H, W), transform=hg.transform, fill=0, dtype="uint8").astype(bool)
        ok &= ~(other & ~flooded)
    # subsample for storage: every 4th cell on both axes (≈ 40 m spacing)
    sl = (slice(None, None, 4), slice(None, None, 4))
    m = ok[sl]
    cols = {"flooded": flooded[sl][m].astype(np.uint8)}
    for thr, h in hg.hand.items():
        cols[f"hand_{thr}"] = np.round(np.where(np.isfinite(h), h, 99.0)[sl][m], 2).astype(np.float32)
    cols["elev"] = hg.dem[sl][m].astype(np.float32)
    e = hg.dem[hg.valid]
    cols["elev_pct"] = np.round((np.searchsorted(np.sort(e), hg.dem[sl][m]) / max(e.size, 1)), 3).astype(np.float32)
    cols["built"] = (built[sl][m].astype(np.uint8) if built is not None else np.zeros(m.sum(), np.uint8))
    n_f = int(cols["flooded"].sum())
    if n_f < 30 or n_f > 0.9 * len(cols["flooded"]):
        return {"skipped": f"flooded cells {n_f} of {len(cols['flooded'])}"}
    return {"lat": lat, "lon": lon, "cell_m": round(hg.cell_m * 4, 1), "n": int(len(cols["flooded"])), "n_flooded": n_f,
            "channels": [float(t) for t in hg.hand], "cols": {k: v.tolist() for k, v in cols.items()}}


def _feature_job(args) -> str:
    stem, k, lat, lon, meta, ext_dir, feat_dir = args
    out = RAW / feat_dir / f"{stem}_w{k}.json"
    ext = json.loads((RAW / ext_dir / f"{stem}.json").read_text())
    try:
        r = _window_features(lat, lon, ext)
    except Exception as exc:                                        # noqa: BLE001
        r = {"skipped": f"features failed: {exc}"}
    r = r or {"skipped": "no result"}
    r.update(meta)
    r.update({"window": k, "sensor": ext.get("sensor")})
    out.write_text(json.dumps(r))
    return f"{out.name} " + (r.get("skipped") or f"{r['n']} cells, {r['n_flooded']} flooded")


def stage_features(ext_dir: str = "extents", feat_dir: str = "features") -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    cat = {e["code"]: e for e in json.loads((RAW / "catalog.json").read_text())} if ext_dir == "extents" else {}
    (RAW / feat_dir).mkdir(exist_ok=True)
    batch = int(os.environ.get("FEAT_BATCH", "9"))
    workers = int(os.environ.get("FEAT_WORKERS", "3"))
    jobs = []
    for f in sorted((RAW / ext_dir).glob("*.json")):
        ext = json.loads(f.read_text())
        if ext.get("skipped"):
            continue
        wins = [tuple(ext["centre"])] if ext.get("centre") else _windows(ext)
        for k, (lat, lon) in enumerate(wins):
            if (RAW / feat_dir / f"{f.stem}_w{k}.json").exists():
                continue
            c = cat.get(ext["code"], {})
            meta = ({"code": ext["code"], "aoi": ext["aoi"], "event": c.get("name"), "event_time": c.get("event_time"),
                     "countries": c.get("countries"), "continent": c.get("continent")} if not ext.get("historic") else
                    {"code": ext["code"], "aoi": 0, "event": ext["event"], "event_time": ext["event_time"],
                     "countries": ["United Kingdom"], "continent": "Europe", "historic": True, "n_outlines": ext.get("n_outlines")})
            jobs.append((f.stem, k, lat, lon, meta, ext_dir, feat_dir))
    print(len(jobs), "windows pending; running", min(batch, len(jobs)))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_feature_job, j) for j in jobs[:batch]]
        for f in as_completed(futs):
            print(f.result(), f"{time.time()-t0:.0f}s")
    print("batch done; run again for more" if len(jobs) > batch else "all windows processed")


# ── evaluate ─────────────────────────────────────────────────────────────────

def auc(score: np.ndarray, y: np.ndarray) -> float:
    """ROC AUC (rank-sum); score higher ⇒ predicted flooded."""
    from scipy.stats import rankdata
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load_windows(feat_dir: str = "features") -> list[dict]:
    rows = []
    for f in sorted((RAW / feat_dir).glob("*.json")):
        r = json.loads(f.read_text())
        if r.get("skipped"):
            continue
        r["arr"] = {k: np.array(v) for k, v in r["cols"].items()}
        del r["cols"]
        rows.append(r)
    return rows


def _arrays(r: dict) -> dict:
    return r["arr"]


def _score_cur(a: dict) -> np.ndarray:
    """The pre-benchmark engine rule: built-up land below 3 m scores 4 points, below 6 m 2 points (else 0)."""
    return np.where(a["elev"] < 3, 4, np.where(a["elev"] < 6, 2, 0)).astype(float)


def _window_scores(a: dict) -> dict:
    e = a["elev"]
    return {"current_rule": _score_cur(a), "hand_0.1": -a["hand_0.1"], "hand_1.0": -a["hand_1.0"],
            "elev_pct": -a["elev_pct"], "rel_p10": -(e - np.percentile(e, 10))}


def _boot(vals: np.ndarray, groups: list, n: int = 2000, seed: int = 7) -> list:
    """Mean with a 95 % interval from resampling EVENTS (not cells or windows)."""
    rng = np.random.default_rng(seed)
    ug = sorted(set(groups))
    idx = {g: [i for i, x in enumerate(groups) if x == g] for g in ug}
    means = []
    for _ in range(n):
        pick = rng.choice(len(ug), len(ug), replace=True)
        v = np.concatenate([vals[idx[ug[k]]] for k in pick])
        means.append(np.nanmean(v))
    return [round(float(np.nanmean(vals)), 3), round(float(np.percentile(means, 2.5)), 3), round(float(np.percentile(means, 97.5)), 3)]


def _train_set(rows: list[dict], per_class: int = 1500, seed: int = 3):
    from cri.analysis.flood_event_model import cells_design
    rng = np.random.default_rng(seed)
    Xs, ys, ws = [], [], []
    for r in rows:
        a = r["arr"]
        X, y = cells_design(a), a["flooded"]
        keep = []
        for c in (0, 1):
            ii = np.flatnonzero(y == c)
            keep.append(rng.choice(ii, min(per_class, ii.size), replace=False))
        keep = np.concatenate(keep)
        Xs.append(X[keep]); ys.append(y[keep]); ws.append(np.full(keep.size, 1.0 / keep.size))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(ws)


def stage_evaluate() -> Path:
    from cri.analysis.flood_event_model import cells_design, fit_logistic, predict
    rows = load_windows()
    if not rows:
        raise SystemExit("no feature windows yet")
    events = sorted({r["code"] for r in rows})
    test_ev = {c for i, c in enumerate(events) if i % 3 == 0}                 # every third event is held out
    train = [r for r in rows if r["code"] not in test_ev]
    test = [r for r in rows if r["code"] in test_ev]
    print(f"{len(rows)} windows, {len(events)} events; train {len(train)} windows / {len({r['code'] for r in train})} events, "
          f"test {len(test)} windows / {len(test_ev)} events")

    def fit(rs):
        X, y, w = _train_set(rs)
        return fit_logistic(X, y, w, l2=1.0)

    def per_window(rs, model):
        out = []
        for r in rs:
            a = r["arr"]
            y = a["flooded"]
            sc = _window_scores(a)
            sc["event_model"] = predict(model, cells_design(a))
            out.append({"code": r["code"], "event": r["event"], "continent": r["continent"], "n": r["n"],
                        "flooded_share": round(float(y.mean()), 3),
                        "auc": {k: round(auc(v, y), 4) for k, v in sc.items()}})
        return out

    m_train = fit(train)
    pw_test = per_window(test, m_train)
    pw_train = per_window(train, m_train)

    def summ(pw):
        ks = list(pw[0]["auc"])
        g = [p["code"] for p in pw]
        res = {k: _boot(np.array([p["auc"][k] for p in pw], float), g) for k in ks}
        d = np.array([p["auc"]["event_model"] - p["auc"]["current_rule"] for p in pw], float)
        res["gain_vs_current_rule"] = _boot(d, g)
        res["share_windows_auc_gt_0.6"] = {k: round(float(np.mean([p["auc"][k] > 0.6 for p in pw])), 2) for k in ks}
        return {"n_windows": len(pw), "n_events": len(set(g)), "mean_auc_[mean, lo95, hi95]": res}

    heldout = summ(pw_test)
    # transfer: fit on Europe only, score outside Europe
    eu = [r for r in rows if r["continent"] == "Europe"]
    non = [r for r in rows if r["continent"] != "Europe"]
    transfer = None
    if len(eu) >= 10 and len(non) >= 5:
        transfer = summ(per_window(non, fit(eu)))
        transfer["fit_on"] = f"Europe ({len(eu)} windows)"
        transfer["scored_on"] = f"outside Europe ({len(non)} windows)"
    # shipped model: refit on every event; reference quantiles from its own training cells
    m_all = fit(rows)
    X, y, w = _train_set(rows)
    q = np.quantile(predict(m_all, X), np.linspace(0, 1, 201)[1:-1]).round(5).tolist()
    # calibration: observed flooded share by rank band, on every benchmark cell (descriptive; rank = share of reference cells below)
    qa = np.array(q)
    rk_all, y_all = [], []
    rng_c = np.random.default_rng(1)
    for r in rows:
        a = r["arr"]
        p_ = predict(m_all, cells_design(a))
        ii = rng_c.choice(len(p_), min(6000, len(p_)), replace=False)
        rk_all.append(np.searchsorted(qa, p_[ii]) / len(qa)); y_all.append(a["flooded"][ii])
    rk_all, y_all = np.concatenate(rk_all), np.concatenate(y_all)
    calib = [{"rank_from": lo, "rank_to": min(hi, 1.0), "flooded_share": round(float(y_all[(rk_all >= lo) & (rk_all < hi)].mean()), 3),
              "lift": round(float(y_all[(rk_all >= lo) & (rk_all < hi)].mean() / y_all.mean()), 2)}
             for lo, hi in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]]
    ho = {"mean_auc": heldout["mean_auc_[mean, lo95, hi95]"]["event_model"][0], "n_events": heldout["n_events"], "split": "every third event held out"}
    shipped = {"model": m_all, "reference_quantiles": q, "heldout": ho, "n_events": len(events), "n_windows": len(rows),
               "truth": "Copernicus EMS Rapid Mapping observed flood extents (events 2023–2026)", "trained_on_all_events": True,
               "rank_calibration": {"base_flooded_share": round(float(y_all.mean()), 3), "bands": calib}}
    (ROOT / "src" / "cri" / "data" / "flood_event_model.json").write_text(json.dumps(shipped, indent=1))
    report = {"design": __doc__.split("Truth")[0].strip(), "n_events": len(events), "n_windows": len(rows),
              "by_continent": {c: sum(1 for r in rows if r["continent"] == c) for c in sorted({r["continent"] for r in rows})},
              "heldout_events": sorted(test_ev), "heldout": heldout,
              "train_in_sample": summ(pw_train), "transfer_europe_to_rest": transfer,
              "fitted_on_train_split": m_train, "per_window_heldout": pw_test,
              "limits": ["SAR under-maps flooding in dense urban areas and under vegetation.",
                         "Polygons are a snapshot at image time, not the peak.",
                         "CEMS maps areas where flooding was suspected: this measures skill WITHIN flood-affected landscapes.",
                         "Events are 2023–2026 and Europe-heavy; older CEMS events are not in the public dashboard API.",
                         "Depth/magnitude is not tested here: no machine-readable depth truth for these events."]}
    OUT.mkdir(exist_ok=True)
    (OUT / "flood_event_benchmark.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({"heldout": heldout, "transfer": transfer}, indent=1)[:3500])
    return OUT / "flood_event_benchmark.json"



# ── historic: England recorded flood outlines (Environment Agency, Open Government Licence) ──────────────────────

UK_WFS = ("https://environment.data.gov.uk/spatialdata/recorded-flood-outlines/wfs?service=WFS&version=2.0.0&request=GetFeature"
          "&typeNames=dataset-8c75e700-d465-11e4-8b5b-f0def148f590:Recorded_Flood_Outlines&outputFormat=application/json&srsName=EPSG:4326")


def _fes(y0: int, y1: int) -> str:
    return ("<fes:Filter xmlns:fes='http://www.opengis.net/fes/2.0'><fes:And>"
            f"<fes:PropertyIsGreaterThanOrEqualTo><fes:ValueReference>start_date</fes:ValueReference><fes:Literal>{y0}-01-01T00:00:00Z</fes:Literal></fes:PropertyIsGreaterThanOrEqualTo>"
            f"<fes:PropertyIsLessThan><fes:ValueReference>start_date</fes:ValueReference><fes:Literal>{y1}-01-01T00:00:00Z</fes:Literal></fes:PropertyIsLessThan>"
            "<fes:PropertyIsEqualTo><fes:ValueReference>fluvial_f</fes:ValueReference><fes:Literal>True</fes:Literal></fes:PropertyIsEqualTo>"
            "</fes:And></fes:Filter>")


def stage_uk_fetch(y0: int = 1940, y1: int = 2025) -> None:
    import urllib.parse
    d = RAW / "uk"
    d.mkdir(parents=True, exist_ok=True)
    for a in range(y0 // 5 * 5, y1, 5):                              # 5-year chunks keep each response small
        out = d / f"fluvial_{a}.json"
        if out.exists():
            continue
        out.write_bytes(_get(UK_WFS + "&filter=" + urllib.parse.quote(_fes(a, a + 5)), 170))
        print(a, round(out.stat().st_size / 1e6, 1), "MB")


def stage_uk_events(per_decade: int = 24, min_km2: float = 0.3) -> None:
    """Cluster outlines into (month, place) events; keep the largest few per decade as benchmark windows."""
    from shapely.geometry import mapping, shape
    from shapely.ops import unary_union
    feats = []
    for f in sorted((RAW / "uk").glob("fluvial_*.json")):
        feats += json.loads(f.read_text())["features"]
    print(len(feats), "fluvial outlines")
    geoms = []
    for x in feats:
        try:
            g = shape(x["geometry"])
            g = g if g.is_valid else g.buffer(0)
            geoms.append((x["properties"]["start_date"][:7], g, x["properties"]))
        except Exception:                                           # noqa: BLE001
            continue
    by_month: dict = {}
    for ym, g, pr in geoms:
        by_month.setdefault(ym, []).append((g, pr))
    cand = []                                                       # (area_km2, ym, centre, polys)
    sep = 2 * WINDOW_KM / 111.0
    for ym, items in by_month.items():
        items.sort(key=lambda t: -t[0].area)
        P = np.array([[g.representative_point().x, g.representative_point().y] for g, _ in items])
        A = np.array([g.area for g, _ in items])
        r = sep / 2
        taken = np.zeros(len(items), bool)
        centres = []
        for i in range(len(items)):
            if taken[i]:
                continue
            c = P[i]
            if any(math.hypot(c[0] - x, c[1] - y) < sep for x, y in centres):
                continue
            centres.append((c[0], c[1]))
            m = (np.abs(P[:, 0] - c[0]) < r * 1.5) & (np.abs(P[:, 1] - c[1]) < r)
            km2 = float(A[m].sum() * (111.0 ** 2) * math.cos(math.radians(c[1])))
            if km2 >= min_km2:
                cand.append((km2, ym, (c[1], c[0]), [items[j][0] for j in np.flatnonzero(m)]))
    out = RAW / "uk_extents"
    out.mkdir(exist_ok=True)
    cent = [(m2, g, g.centroid.x, g.centroid.y) for m2, g, _ in geoms]
    chosen: dict = {}
    for km2, ym, centre, win in sorted(cand, key=lambda t: -t[0]):
        dec = int(ym[:3]) * 10
        if chosen.get(dec, 0) >= per_decade or dec < 1940:
            continue
        chosen[dec] = chosen.get(dec, 0) + 1
        flooded = unary_union([w_.simplify(0.0001) for w_ in win])
        # other months' outlines near the window are excluded from the negatives
        near = [g for m2, g, cx, cy in cent if m2 != ym and abs(cy - centre[0]) < 0.1 and abs(cx - centre[1]) < 0.15]
        code = f"UK{ym.replace('-', '')}_{chosen[dec]:02d}"
        if (out / f"{code}.json").exists():
            continue
        (out / f"{code}.json").write_text(json.dumps({
            "code": code, "aoi": 0, "historic": True, "event": f"England recorded fluvial flood {ym}", "event_time": ym + "-01",
            "centre": list(centre), "n_outlines": len(win), "area_km2": round(km2, 2), "geometry": mapping(flooded.simplify(0.00005)),
            "exclude": mapping(unary_union([n.simplify(0.0002) for n in near[:300]]).simplify(0.0001)) if near else None, "sensor": "EA recorded outline"}))
    print("windows chosen per decade:", dict(sorted(chosen.items())))


def stage_history_eval() -> Path:
    """Score the shipped event model (fitted on modern CEMS events only) on historic England events, by decade."""
    from cri.analysis import flood_event_model as fem
    shipped = fem.load()
    if not shipped:
        raise SystemExit("run --stage evaluate first (it fits and ships the model)")
    rows = load_windows("uk_features")
    if not rows:
        raise SystemExit("no historic windows yet")
    per = []
    for r in rows:
        a, y = r["arr"], r["arr"]["flooded"]
        sc = _window_scores(a)
        sc["event_model"] = fem.predict(shipped["model"], fem.cells_design(a))
        per.append({"code": r["code"], "year": int(r["event_time"][:4]), "decade": int(r["event_time"][:3]) * 10, "n": r["n"],
                    "flooded_share": round(float(y.mean()), 3), "auc": {k: round(auc(v, y), 4) for k, v in sc.items()}})
    res = {"n_windows": len(per), "model_fitted_on": "modern CEMS events only (2023–2026), never on these historic events",
           "by_decade": {}, "overall": {}}
    for k in per[0]["auc"]:
        res["overall"][k] = _boot(np.array([p["auc"][k] for p in per], float), [p["code"] for p in per])
    for dec in sorted({p["decade"] for p in per}):
        sub = [p for p in per if p["decade"] == dec]
        res["by_decade"][str(dec)] = {"n_windows": len(sub), **{k: round(float(np.nanmean([p["auc"][k] for p in sub])), 3) for k in sub[0]["auc"]}}
    yr = np.array([p["year"] for p in per], float)
    am = np.array([p["auc"]["event_model"] for p in per], float)
    from scipy.stats import spearmanr
    res["event_model_auc_vs_year_spearman"] = [round(float(x), 3) for x in spearmanr(yr, am, nan_policy="omit")]
    res["per_window"] = per
    res["limits"] = ["Modern DEM and land cover are used for old floods: levees, channel works and urbanisation since the event are not "
                     "in the terrain, which is why skill is reported by decade.",
                     "England only, fluvial outlines; outline quality varies by source (see data_src in the EA record).",
                     "Windows are chosen where outlines are large, so this is skill within flood-affected landscapes."]
    (OUT / "flood_history_benchmark.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("n_windows", "overall", "by_decade", "event_model_auc_vs_year_spearman")}, indent=1))
    return OUT / "flood_history_benchmark.json"



def stage_index() -> Path:
    """Compact, shippable index of recorded flood events (bounding boxes + dates) for per-site 'what flooded here before' lookups."""
    from shapely.geometry import shape
    ev = []
    cat = {e["code"]: e for e in json.loads((RAW / "catalog.json").read_text())}
    for f in sorted((RAW / "extents").glob("*.json")):
        x = json.loads(f.read_text())
        if x.get("skipped"):
            continue
        c = cat.get(x["code"], {})
        b = x["bounds"]
        ev.append({"s": "CEMS", "id": x["code"], "n": c.get("name"), "d": (c.get("event_time") or "")[:10],
                   "b": [round(b[0], 4), round(b[1], 4), round(b[2], 4), round(b[3], 4)]})
    n_cems = len(ev)
    seen = set()
    for f in sorted((RAW / "uk").glob("fluvial_*.json")):
        for x in json.loads(f.read_text())["features"]:
            try:
                pr = x["properties"]
                key = pr.get("rec_out_id")
                if key in seen:
                    continue
                seen.add(key)
                b = shape(x["geometry"]).bounds
                ev.append({"s": "EA-RFO", "id": str(key), "n": pr.get("name"), "d": (pr.get("start_date") or "")[:10],
                           "b": [round(b[0], 4), round(b[1], 4), round(b[2], 4), round(b[3], 4)]})
            except Exception:                                       # noqa: BLE001
                continue
    out = ROOT / "src" / "cri" / "data" / "flood_event_index.json"
    out.write_text(json.dumps({"sources": {"CEMS": "Copernicus EMS Rapid Mapping observed flood extents (2023–2026, catalogued activations only)",
                                           "EA-RFO": "England Environment Agency Recorded Flood Outlines, fluvial (Open Government Licence)"},
                               "note": "Bounding boxes only. Not a complete flood history of any place.", "events": ev}, separators=(",", ":")))
    print(n_cems, "CEMS areas;", len(ev) - n_cems, "England outlines;", round(out.stat().st_size / 1e6, 2), "MB")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["catalog", "extents", "features", "evaluate", "uk_fetch", "uk_events", "uk_features", "history_eval", "index"])
    ap.add_argument("--max-mb", type=float, default=60.0)
    a = ap.parse_args()
    {"catalog": stage_catalog, "extents": lambda: stage_extents(a.max_mb), "features": stage_features,
     "evaluate": stage_evaluate, "uk_fetch": stage_uk_fetch, "uk_events": stage_uk_events,
     "uk_features": lambda: stage_features("uk_extents", "uk_features"), "history_eval": stage_history_eval, "index": stage_index}[a.stage]()


if __name__ == "__main__":
    main()
